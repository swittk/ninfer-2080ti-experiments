# HTTP serving

`build/apps/ninfer-serve` loads one registered artifact and exposes OpenAI- and
Anthropic-compatible HTTP endpoints over one resident NInfer Engine.

## Start the server

```bash
./build/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 16384 \
  --kv-capacity 32768 \
  --max-concurrency 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For the 35B-A3B artifact, select its artifact path; the public model ID follows the container
identity automatically:

```bash
./build/apps/ninfer-serve models/qwen3_6_35b_a3b.ninfer \
  --max-context 16384 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

When `--model-id` is omitted, the server advertises and accepts the loaded container's exact
`identity.model_id`. An explicit `--model-id` remains a public HTTP alias override and does not
select or alter the artifact.

Vision is disabled by default: its weights, Vision scratch phase, and frozen request-transient
buffer are not allocated, and media
requests and token-count requests fail with HTTP 400 `vision_disabled`. Add `--vision` when the
server must accept image or video input. Speculative residency is likewise frozen by
`--spec mtp|dflash` and `--draft-tokens`; omitting `--spec` loads neither backend.
`--lm-head-draft` additionally loads the optimized proposal head. On SM75 the legacy
optimized profile also permits the existing target-vocabulary shortcut. Add
`--full-target-head` to keep optimized draft selection while verifying with the
complete native target vocabulary; this does not relax the target acceptance test.
This explicit setting is frozen in the loaded runtime features and logged as
`engine.force_full_target_head`. The target-shortcut cache is neither allocated nor
used in that mode; its otherwise-required reservation is omitted rather than bypassed.
Existing defaults and the legacy optimized profile are unchanged. DFlash is 35B-A3B text-only and
cannot be combined with `--vision`. A later request cannot enable a capability omitted at startup.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /health` | process health |
| `GET /v1/models` | configured OpenAI model alias |
| `GET /v1/models/{id}` | lookup of the configured alias |
| `POST /v1/chat/completions` | OpenAI-style chat generation |
| `POST /v1/responses` | OpenAI Responses Core generation, state, typed Items, and SSE |
| `POST /v1/responses/input_tokens` | Responses prompt-token count without generation |
| `GET /v1/responses/{id}` | retrieve a locally stored terminal Response |
| `DELETE /v1/responses/{id}` | delete a locally stored Response |
| `GET /v1/responses/{id}/input_items` | list that Response's normalized input Items |
| `POST /v1/messages` | Anthropic-style message generation |
| `POST /v1/messages/count_tokens` | checkpoint-native expanded input-token count |

## OpenAI Chat Completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 128
  }'
```

The endpoint supports:

- `system`, `developer`, `user`, `assistant`, and `tool` history;
- string content and ordered text, `image_url`, and `video_url` parts;
- `max_completion_tokens` and the legacy `max_tokens` spelling;
- `temperature`, `top_p`, `top_k`, presence/frequency penalties, and a nonnegative `seed`;
- one stop string or an array of stop strings;
- non-streaming responses and server-sent event streams;
- `stream_options.include_usage`;
- function tools, tool choices, assistant tool-call history, and tool-result messages;
- the top-level `reasoning_effort` field;
- the `enable_thinking` extension;
- `chat_template_kwargs.preserve_thinking` and the top-level `preserve_thinking` alias.

The request `model` must equal the public model ID: the artifact `identity.model_id` by default, or
the explicit `--model-id` override. Reasoning is returned separately as `reasoning_content`; answer
text remains in `content`.

At startup, NInfer resolves prompt capabilities from the exact `frontend/chat_template.jinja`
resource embedded in the loaded artifact. It does not infer them from the request's `model` field,
the artifact identity, or a target profile. A recognized effort-capable template exposes `low`,
`medium`, and `xhigh`; omitting effort uses that template's declared default. An explicit effort
not exposed by the loaded template returns HTTP 400 with code
`reasoning_effort_not_supported` before prompt preparation.

For Chat Completions, `reasoning_effort: "none"` disables thinking. `low`, `medium`, and `xhigh`
select the corresponding template effort when available. The other OpenAI protocol values
`minimal`, `high`, and `max` are parsed but rejected when the loaded template does not expose them.
`enable_thinking` controls the same new-turn thinking switch; a contradictory combination with
`reasoning_effort` returns `conflicting_template_option`.

`preserve_thinking` controls whether reasoning from closed assistant turns remains in later
prompts. It defaults to the server setting, which is off unless `--preserve-thinking` is used. If
both OpenAI spellings are present they must carry the same boolean value. Unknown non-null
`chat_template_kwargs` are rejected.

Streaming begins with an assistant-role chunk, sends separate reasoning and content deltas, then a
finish-reason chunk and `[DONE]`. When `stream_options.include_usage` is true, a final empty
`choices` chunk contains completed usage.

### Multimodal request

Start the server with `--vision` before sending media:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 128
  }'
```

OpenAI image and video sources may be HTTP(S) URLs or base64 data URLs.

## OpenAI Responses Core

NInfer implements the typed-Item and semantic-event core of the OpenAI
[Responses API](https://developers.openai.com/api/reference/resources/responses/overview). All
registered artifact identities use this same adapter and Engine route. It is intentionally not
advertised as full parity with OpenAI-hosted tools, durable cloud storage, background jobs,
Conversations, or compaction.

### Create a Response

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "instructions": "Answer concisely.",
    "input": "What is speculative decoding?",
    "max_output_tokens": 128,
    "store": true
  }'
```

The same endpoint works with OpenAI SDKs by replacing their base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-secret")
response = client.responses.create(
    model="qwen3.6-27b",
    instructions="Answer concisely.",
    input="What is speculative decoding?",
    max_output_tokens=128,
)
print(response.output_text)  # SDK helper derived from response.output
```

`output_text` is an SDK convenience property. It is not emitted as a top-level wire field; the
wire response contains typed `output` Items.

### Create request fields

| Field | NInfer Responses Core contract |
|---|---|
| `model` | required non-empty string; must equal the artifact-derived public model ID or explicit `--model-id` override |
| `input` | required string or non-empty typed Item array |
| `instructions` | optional string, inserted before the reconstructed conversation for this request only |
| `previous_response_id` | optional ID of a retained local Response |
| `max_output_tokens` | integer at least `16`; default is `--default-max-tokens` |
| `stream` | boolean; `true` selects Responses SSE rather than a JSON body |
| `store` | boolean, default `true`; controls local retrieval and continuation state |
| `temperature` | finite number in `[0,2]` |
| `top_p` | finite number in `[0,1]` |
| `metadata` | at most 16 string pairs; keys at most 64 characters and values at most 512 |
| `reasoning.effort` | `none` disables thinking; `low`, `medium`, or `xhigh` selects an effort exposed by the loaded chat template; `minimal`, `high`, and `max` return `reasoning_effort_not_supported` for the registered templates |
| `chat_template_kwargs.preserve_thinking` | optional boolean controlling whether closed-turn reasoning remains in reconstructed prompts |
| `preserve_thinking` | top-level alias for the same option; conflicting values are rejected |
| `text.format` | omitted or `{"type":"text"}` only |
| `tools` | flat Responses function definitions; see below |
| `tool_choice` | `auto` or `none` |
| `parallel_tool_calls` | omitted or `true` |
| `truncation` | omitted or `disabled`; overlong input fails instead of silently dropping Items |
| `top_logprobs` | omitted or `0` |
| `service_tier` | omitted, `auto`, or `default`; the response reports `default` |
| `background` | omitted or `false` |
| `include` | omitted or an empty array |
| `stream_options` | omitted or `{"include_obfuscation":false}` |

Unknown top-level fields fail with `unknown_parameter`. Recognized but unsupported features fail
with a field-specific 400 error instead of being silently ignored.

### Input Item contract

String `input` is normalized to one user `message` with an `input_text` part. Array input accepts:

| Item | Supported form |
|---|---|
| `message` | roles `user`, `assistant`, `system`, and `developer`; string content or typed content array |
| `input_text` | message content part containing string `text` |
| `output_text` | assistant-message replay part containing string `text` |
| `input_image` | user-message part with HTTP(S) or data-URI `image_url`; detail omitted or `auto`; requires server `--vision` |
| `input_video` | NInfer extension with HTTP(S) or data-URI `video_url`; requires server `--vision` |
| `reasoning` | raw replay Item with an empty `summary` and `reasoning_text` content parts |
| `function_call` | completed assistant call with optional `id`, and required `call_id`, `name`, and JSON-object string `arguments` |
| `function_call_output` | completed tool result with required `call_id` and string `output` |

Adjacent function-call Items are grouped into one assistant history turn. A reasoning Item attaches
to the following assistant message or function call. Input Item IDs are preserved when supplied and
generated otherwise; duplicate IDs fail.

`input_file`, `input_audio`, image `file_id`, non-`auto` image detail, reasoning summaries or
encrypted reasoning, message `phase`, and other Item/content types are not supported. HTTP media
URLs stored in a response chain are fetched again when that chain is continued; use data URIs when
the historical media bytes must be immutable.

### Function tools

Responses function definitions are flat rather than Chat Completions' nested `function` object:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current weather",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  },
  "strict": false
}
```

NInfer renders these definitions in the Qwen prompt and parses model output into separate
`function_call` output Items. Each output has a protocol Item `id` (`fc_...`) and a distinct
`call_id` (`call_...`). The client executes the function and sends a `function_call_output` Item in
a later request. NInfer does not execute functions or enforce JSON Schema through constrained
decoding, so `strict:true`, `tool_choice:required`, named tool choice, hosted tools, MCP tools, and
custom free-form tools are rejected.

### Response object and usage

A terminal wire response has `object: "response"`, one of `completed`, `incomplete`, or
`cancelled` in `status`, and a typed `output` array. NInfer may emit:

- a `reasoning` Item containing raw `reasoning_text` and an empty summary;
- an assistant `message` containing an `output_text` part;
- one or more `function_call` Items.

Ordinary model/string stops produce `completed`. Output-token or context-capacity exhaustion
produces `incomplete` with `incomplete_details.reason: "max_output_tokens"`. Errors accepted after
an SSE response has started produce `response.failed`; validation and preparation errors remain
normal HTTP error responses.

Usage is checkpoint-native:

```json
{
  "input_tokens": 42,
  "input_tokens_details": {"cached_tokens": 17},
  "output_tokens": 12,
  "output_tokens_details": {"reasoning_tokens": 5},
  "total_tokens": 54
}
```

`input_tokens` includes the chat template and expanded media tokens. `cached_tokens` is the exact
resident prompt prefix reused by Engine. `output_tokens` is the count of accepted generated token
IDs, including a withheld stop token when applicable. `reasoning_tokens` is counted in the Qwen
output decoder while accepted tokens are still in the reasoning channel; it is not estimated by
re-tokenizing decoded text.

### Responses streaming

Set `stream:true` for semantic Server-Sent Events. Every frame uses both the SSE event name and a
matching JSON `type`, and every JSON event has a monotonically increasing `sequence_number`:

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":7,...}

```

The normal lifecycle is:

1. `response.created`, then `response.in_progress`;
2. `response.output_item.added` and `response.content_part.added`;
3. zero or more `response.reasoning_text.delta` or `response.output_text.delta` events;
4. matching `*.done`, `response.content_part.done`, and `response.output_item.done` events;
5. exactly one `response.completed`, `response.incomplete`, or `response.failed` terminal event.

Function arguments use `response.function_call_arguments.delta` and `.done`. IDs, output indices,
and content indices remain stable, and concatenated deltas equal the terminal Item. Responses SSE
does not emit the Chat Completions `[DONE]` sentinel. With tools enabled, ordinary answer text still
streams immediately; only an ambiguous `<tool_call>` suffix or the structured tool region is held.
Malformed tool markup is flushed back as ordinary text without losing bytes.

### Local response state and resources

`store` defaults to `true`. Stored Responses live only in this server process and are bounded by an
LRU store. They are lost on restart and are not OpenAI's durable cloud retention service.

`previous_response_id` reconstructs the complete stored input/output Item history before the new
input. The current `instructions` value is placed first but is not saved into the continuation
context, matching the Responses rule that previous top-level instructions do not carry forward.
Function definitions are request configuration rather than conversation Items and must be sent
again on tool-result turns. The reconstructed prompt follows the ordinary Engine path, so resident
prefix reuse applies naturally.

A stored Response also retains its resolved `preserve_thinking` value. A child which omits the
field inherits the parent value. An explicit different value creates a new semantic branch; prompt
identity then determines whether the Engine restores a turn checkpoint or performs a full reset.

Resource behavior:

| Endpoint | Contract |
|---|---|
| `GET /v1/responses/{id}` | returns the stored terminal object, or 404 `response_not_found` |
| `DELETE /v1/responses/{id}` | removes public retrieval and returns `response.deleted`; descendant contexts already retained by other Responses remain usable |
| `GET /v1/responses/{id}/input_items` | returns normalized Items supplied to that request; supports `after`, `limit` `1..100` (default `20`), and `order` `asc|desc` (default `desc`) |
| `POST /v1/responses/{id}/cancel` | explicitly fails because background execution is unsupported |
| `POST /v1/responses/compact` | explicitly fails with `compaction_not_supported` |

`store:false` Responses cannot be retrieved or used as `previous_response_id`. LRU eviction and
explicit deletion also make an ID unavailable. A single Response larger than the configured store
capacity fails with `response_store_capacity_exceeded` rather than silently pretending it was
stored.

### Responses input token count

`POST /v1/responses/input_tokens` accepts exactly `model` and `input`, performs the same typed Item,
template, and media expansion, and does not run generation:

```bash
curl http://127.0.0.1:8080/v1/responses/input_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","input":"Count this prompt."}'
```

```json
{"object":"response.input_tokens","input_tokens":11}
```

Unsupported Create fields include Conversations, prompt templates, context management, hosted
moderation, prompt-cache controls, safety/user identifiers, Structured Outputs/JSON mode,
non-empty `include`, background execution, compaction, files/audio, and OpenAI-hosted/MCP/custom
tools. These are compatibility boundaries, not silently accepted placeholders.

## Anthropic Messages

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Explain prefix reuse in one sentence."}
    ]
  }'
```

The endpoint supports system text, user/assistant history, text and image blocks, thinking blocks,
tool-use history, tool results, client-defined tools, non-streaming responses, and Anthropic SSE
events. `thinking.type: "disabled"` disables thinking; other supported values enable it.
The independent top-level `preserve_thinking` boolean controls closed-turn history and otherwise
uses the server default.

Anthropic `output_config.effort` accepts the protocol values `low`, `medium`, `high`, `xhigh`, and
`max`. The value is then checked against the loaded chat template in the same way as the OpenAI
endpoints; the registered effort-capable template exposes `low`, `medium`, and `xhigh`. Combining
an effort with `thinking.type: "disabled"` is rejected as contradictory.

Anthropic's `model` field is treated as a response label and does not select the loaded artifact.

`POST /v1/messages/count_tokens` uses the artifact's tokenizer, chat template, and media expansion
without running GPU generation:

```bash
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Count this prompt."}]
  }'
```

## Authentication and CORS

Pass `--api-key VALUE` to require the same value as an OpenAI bearer token or Anthropic
`x-api-key` header. `GET /health` and CORS preflight requests remain unauthenticated.

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```

`--cors` adds permissive browser CORS headers. It is disabled by default.

## Server options

| Option | Meaning | Default |
|---|---|---:|
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--api-key KEY` | required bearer or `x-api-key` value | unset |
| `--model-id ID` | override the public OpenAI model alias | artifact `identity.model_id` |
| `--max-context N` | logical context ceiling of each sequence | `8192` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `8192` |
| `--max-concurrency N` | maximum admitted requests; valid range `1..8` | `1` |
| `--max-pending-requests N` | additional requests allowed to wait for admission | `16` |
| `--pending-timeout-ms N` | maximum preparation-plus-admission wait | `30000` |
| `--prefill-chunk N` | text-prefill chunk | `1024` |
| `--max-prefill-lanes N` | simultaneous staged prefill owners per replica, bounded by `--max-concurrency` | `1` |
| `--log-stats-interval-ms N` | aggregate throughput report interval; `0` disables it | `5000` |
| `--device N` | CUDA device index | `0` |
| `--max-request-mib N` | body-size limit before JSON parsing | `384` |
| `--request-log-jsonl FILE` | append full-precision server/request records | disabled |
| `--response-store-max-records N` | maximum locally retained Responses objects | `1024` |
| `--response-store-max-mib N` | total local Response envelope/Item/context budget | `256` |
| `--kv-dtype bf16\|int8` | KV-cache storage | `bf16` |
| `--spec mtp\|dflash` | speculative backend | off |
| `--draft-tokens N` | MTP `1..5`; DFlash `1..15` | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--full-target-head` | pin full native target vocabulary independently of draft selection | off |
| `--fast-full-vocab-draft` | SM75 C1 draft-only accelerated full-vocabulary ranking; requires full target verification | off |
| `--full-vocab-draft-retained-target` | Explicit strict C1/D3 full-vocabulary proposals with the retained optimized target; requires `--lm-head-draft` and no full-target override | off |
| `--default-max-tokens N` | output limit when omitted by a request | `8192` |
| `--vision` | enable media input and load Vision GPU allocations | off |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-prefix-reuse` | disable compatible-prefix caching | prefix reuse on |
| `--no-thinking` | disable thinking by default | thinking on |
| `--preserve-thinking` | preserve closed-turn assistant reasoning by default | off |
| `--cors` | permissive browser CORS headers | off |
| `--temperature F` | process-level temperature override | unset |
| `--top-p F` | process-level top-p override | unset |
| `--top-k N` | process-level top-k override | unset |
| `--min-p F` | process-level min-p override | unset |
| `--presence-penalty F` | process-level presence-penalty override | unset |
| `--frequency-penalty F` | process-level frequency-penalty override | unset |
| `--seed N` | fixed seed when a request omits one | fresh random seed per request |
| `--greedy` | force exact argmax for all requests | off |

Engine selects sampling defaults from the loaded model and the request's resolved thinking mode.
Qwen3.6-27B and Qwen3.8-27B use `1.0/0.95/20/0/0` for
temperature/top-p/top-k/min-p/presence penalty in thinking mode and `0.7/0.80/20/0/1.5` in
non-thinking mode. Qwen3.6-35B-A3B differs only in its thinking presence penalty, which is `1.5`.
Frequency penalty is `0` for all registered presets. Process flags override registered values,
request fields override process flags, and `--greedy` finally forces temperature `0`.

Run `./build/apps/ninfer-serve --help` for the exact option contract.

## Structured request log

`--request-log-jsonl FILE` enables the machine-readable measurement log. The server opens `FILE`
in append mode and flushes every event, so successive model or MTP blocks may share one campaign
file. The parent directory must already exist. Failure to open the file aborts startup; the log path
is also rejected if it resolves to the model artifact.

```bash
./build/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --request-log-jsonl profiles/bench/run/server.requests.jsonl
```

Every line is one `ninfer_serve_request_log` schema-v8 JSON object. All events carry
`timestamp_unix_ms` and a process-unique `server_instance_id`; request IDs are monotonic only within
that server instance.

| Event | Contents |
|---|---|
| `server_start` | target/weights identity and artifact, resolved Engine, registered thinking/non-thinking sampler defaults plus process overrides, thinking-history defaults, weights/sequence/workspace/request-transient arenas, KV sizing ledger, CUDA Graph observed/allowance bytes, CUDA/GPU environment, and redacted argv |
| `request_start` | protocol, resolved sampler and seed, thinking modes, Responses semantic-change flag, output budget, stream/message/tool shape |
| `request_done` | finish reason, prompt/completion/cache/computed-prefill tokens, prefix reuse path, unrounded phase seconds, and complete speculative-decoding counters |
| `request_error` | the resolved request configuration and generation error message |
| `throughput` | interval token deltas and rates, scheduler occupancy, and decode-round batch statistics |

`request_done.timings_seconds` contains `prepare`, `ttft`, `vision`, `prefill`, `decode`, and `total`
as full-precision JSON numbers. Its `speculative` object contains `backend`, `draft_window`, `rounds`,
`drafted_tokens`, `accepted_tokens`, `fallback_steps`, and `accepted_per_position`. Rates can be
derived downstream from raw token counts and seconds instead of rounded stderr strings.

The JSONL file contains no generated response text and never records an API-key value; `argv`
replaces that value with `<redacted>`. The existing stderr summaries remain available for operators
but are rounded and are not the aggregation source. Console lines use local
`[YYYY-MM-DD HH:MM:SS.mmm] [level]` timestamps. Structured request events cover successfully
prepared OpenAI Responses, OpenAI Chat, and Anthropic generation requests and errors during their
generation; schema rejection
and token-count-only calls are not measurement requests and do not receive request IDs.

By default the server also reports aggregate activity every five seconds. `prefill` counts prompt
suffix tokens actually computed during the interval, excluding prefix-cache hits; `decode` counts
tokens finally committed by decode rounds, excluding the first token produced by prefill. For MTP
and DFlash this is the accepted committed output, not draft or rejected tokens.
`avg_decode_batch` is decode row-rounds divided by decode rounds during the same interval. The
`running`, `prefilling`, `decode_ready`, and `waiting` fields are the Engine scheduler snapshot at
the end of the interval. Fully idle zero intervals are omitted. The JSONL `throughput` event keeps
the raw token and round deltas as well as derived rates; downstream measurement should prefer those
raw values.

## Execution behavior

The server owns one resident Engine with a startup-fixed capacity of `1..8` active generation
requests. At each decode boundary, every decode-ready request is compacted into one batch and
processed by one model traversal and, when graphs are enabled, one exact-batch CUDA Graph replay. A
request joins that batch only after its single-request prefill finishes; when it completes or is
cancelled, the next boundary rebuilds the batch without an empty row.

`--max-pending-requests` bounds the requests waiting behind the active set. The total generation
request lifetime capacity is `max_concurrency + max_pending_requests`, including requests still in
CPU/media preparation and completed model results whose response has not yet been released. A full
capacity returns HTTP 429 with code `server_overloaded`. The absolute
`--pending-timeout-ms` deadline starts before preparation, covers media acquisition and Engine FIFO
waiting, and returns HTTP 503 with code `request_queue_timeout` if admission does not occur in time.
There is no admission ETA or unbounded overflow queue.

Input memory is bounded by the outstanding-request count and the per-request
`--max-request-mib` limit. Media requests additionally share one preparation permit, so a waiting
media request retains the same cancellation and timeout deadline. Model output is bounded by the
same finite request count and each request's effective output-token limit; output callbacks and
network serialization run outside the GPU executor and do not delay formation of the next batch.

`--max-context` and the resolved `--kv-capacity` are independent limits. The former is each
sequence's logical ceiling; the latter sizes the shared Main Text KV pool used by all active
requests and retained prefixes. Both are represented with 64-token pages internally, while a
sequence can never cross the exact `--max-context` frontier. `--kv-capacity N` requests an explicit
capacity; `--kv-capacity auto` chooses the largest legal capacity that fits the memory remaining
after weights are loaded while keeping 1 GiB of sizing headroom. When omitted it follows
`--max-context`, preserving one full-length request's capacity. The shared pool is fixed at startup
and is not divided evenly among request lanes.

Automatic sizing evaluates the complete target runtime layout for the chosen concurrency, KV
dtype, speculative backend, draft window, Vision setting, workspace, and CUDA Graph allowance. It
uses a direct page-capacity calculation rather than allocation probing. Startup reports the policy,
resolved capacity, runtime reservation, free memory after weights, automatic headroom, planned
slack, actual free memory after complete startup, and observed Graph memory. An explicit capacity
is never silently reduced, and neither policy permits request-time pool growth.

Admission reserves the full prompt-plus-effective-output page entitlement, so an admitted request
can finish within its declared bound. A later request waits in FIFO order when the remaining shared
pages cannot satisfy its complete entitlement; the Engine never admits it and later truncates an
older request to recover capacity. Startup rejects a KV pool smaller than one sequence, too small to
provide one page per configured lane, or larger than all configured lanes could use.

Compatible resident prefixes are reused for both text and multimodal histories unless the server is
started with `--no-prefix-reuse`. A multimodal hit requires matching token types, three-axis MRoPE
positions, encoded-media digest, grid, and consumer spans; changing an earlier image or video
therefore resets the prefix instead of reusing placeholder-token KV. Media wholly inside a matched
prefix skips Vision execution, while new suffix media is encoded normally. The completion log
reports the reused token count as `cache=`.

The shared family runtime distinguishes `full_reset`, `append_frontier`, and
`restore_turn_checkpoint`. A turn checkpoint includes the recurrent and selected
speculative-backend continuation state required to recompute a rewritten suffix; matching KV
tokens alone never authorize a partial hit. Stable `preserve_thinking=true` histories normally
append, while stable `false` histories restore the previous open-turn checkpoint when a new user
closes that turn. The JSONL completion record exposes the selected path as `prefix_reuse_path`.
Changing reasoning effort changes the rendered prompt and therefore does not reuse a prefix whose
effort instruction differs.

Speculative decoding is an engine option and does not change protocol output shapes, stop behavior,
or usage accounting. If a stop truncates a multi-token MTP or DFlash round, the Engine commits the
exact accepted target prefix so a following compatible turn can still reuse it. Output-limit and
context-capacity finishes map to `length`/ `max_tokens`; ordinary model or string stops map to
`stop`/ `end_turn`.

Function tools are rendered into the model prompt and generated calls are parsed into protocol
responses. NInfer does not execute tools and does not enforce client JSON Schema through constrained
decoding.

Prompt-token usage includes chat-template and expanded media tokens. Generated-token usage comes
from accepted output token IDs, including a stop token whose decoded text may be withheld.

## Independent GPU replicas

`ninfer-serve MODEL.ninfer --devices 0,1 --max-concurrency 4 --kv-capacity auto`
loads a full independent Engine on each selected CUDA device behind the same HTTP
endpoint. `--device N` retains single-device operation and is mutually exclusive with
`--devices`. Device indices must be distinct. Model, context, KV, vision and speculative
options apply to every replica; `--max-concurrency` is **per replica**, while
`--max-pending-requests` remains the **global** pending/response-lifetime allowance.
A request is assigned once, before preparation, and retains that Engine until completion
or cancellation. The router balances outstanding reservations, including preparation;
bounded transcript fingerprints prefer the longest recent turn-prefix owner while it
has a free active slot, and break equal-load ties thereafter. Appending assistant,
user or tool turns preserves earlier routing hints; role, media-source and tool-call
boundaries are included. Only the most recent 64 turn boundaries are searched. A full cached owner never forces queueing
while another replica is freer. These are locality hints only: each Engine still validates the exact full prefix and media
identity. Active KV is never migrated, and routing never substitutes for Engine admission.

Each replica retains the existing boundary scheduler, staged prefill/decode interleaving,
protected-head backfill, retained prefix lanes, and request-local output. Media preparation
permits are per replica. Abandoning a prepared request cancels its generation before its
routing reservation is released; model completion releases the reservation independently
of HTTP response lifetime. Global HTTP lifetime and queue limits remain enforced.

JSONL schema 9 adds `request.replica_index`, `request.device`,
`engine.replica_devices`, and `engine.total_max_concurrency`. Request IDs remain global
within the HTTP server instance. Periodic counters sum actual work across the replicas;
startup memory is logged separately for each device on the console. The server-start JSON
memory/environment describe the first device, not a pooled device. Throughput qualification
must use client-observed streaming intervals and wall time, not a sum of isolated rates.
This implementation is under physical serving qualification; no C8 performance or
multimodal certification is implied by enabling the topology.


### Adaptive prefill work units

`--prefill-chunk N` is the maximum prefill work unit and the size used for startup
workspace planning. `--prefill-live-chunk M` optionally selects a smaller unit when
clients were already streaming at the incoming request's Engine arrival. Zero
(the default) inherits N; a nonzero M must be a multiple of 128 and no larger than N.
The two limits do not change token positions, context capacity, checkpoint identity
or output limits. The admission/backfill service projection budgets the smaller
unit, so later small-step execution cannot consume an optimistic large-step budget.
Once established clients retire, subsequent chunks can use the larger bound again.
Initial all-cold cohorts use the larger unit; they do not receive a claim of
continuous per-user decode service while the other cold requests are still prefilling.

The measured 1024/128 profile reduced a roughly 7,600-input-token eight-request
cold cohort from approximately 124 to 66 seconds p95 first-content latency. The
subsequent admitted dense attention-input projection path measured about 55.5 seconds
on the same roughly 7.6K-input cohort. It keeps the small live unit used for
existing-client protection. These measurements
exclude initial model loading but include each request's preparation/admission.

### Bounded protection for established streams

`--prefill-decode-rounds N` permits up to N decode rounds between substantial prefill
chunks when a live request was already streaming before the prefilling request arrived
at the Engine. The default is 1; accepted values are 1 through 64. Requests in the same
all-cold ingress cohort retain one-round alternation, so they are not delayed by bursts
for decoders which only became ready after their arrival. Genuinely short cached suffixes keep the one-round policy. A substantial cold
suffix retains protection through its final chunk, avoiding a last-chunk stall. Protection ends when the established
clients finish or cancel. Cancellation and pending-deadline checks still run each round.


## Opt-in SM75 C4+C4 serving profile

The stock two-2080-Ti qualification profile uses independent full-model engines,
four request slots per engine, the full target head and one factory-MTP draft.
It is opt-in; the normal defaults and other hardware profiles are unchanged.

```bash
MODEL=.chatgpt/continuation_1605_20260821/p0_m_int4mtp_q5resident/native_q4g64_msefine_artifact/qwen3_8_27b_mtp_q4g64_msefine.ninfer
NINFER_MTP_Q4G64=1 NINFER_MTP_MARLIN_Q4=1 \
NINFER_DUAL_SM75_A8=both-mlp-stage1-all \
NINFER_SERVE_C4_A8=1 NINFER_SERVE_RESIDENT_PREFILL=1 \
NINFER_SERVE_PREFILL_BLAS=1 NINFER_SERVE_MEDIA_CACHE=1 \
build-sm75/apps/ninfer-serve "$MODEL" \
  --host 127.0.0.1 --port 18080 --model-id qwen3.8-27b \
  --devices 0,1 --max-concurrency 4 --max-context 8192 \
  --kv-capacity auto --kv-dtype int8 --prefill-chunk 1024 \
  --prefill-live-chunk 128 \
  --prefill-decode-rounds 24 --spec mtp --draft-tokens 1 \
  --default-max-tokens 2048 --vision --no-thinking --greedy
```

`NINFER_SERVE_PREFILL_BLAS=1` selects the 27B SM75 Program-owned cuBLASLt dense
prefill implementation for admitted shapes. Layer-local expansion, activations,
FP32 accumulators and library scratch come from the planned workspace. It does
not duplicate the resident model, change encoded artifact bytes or replace the
Vision tower. The MLP follows the existing native materialized/fused SwiGLU
rounding rule at the actual chunk width, including ragged tails; projected residuals
are qualified under the existing floating-point reduction criteria, not a claim
that every token trajectory must match the prior implementation. The library
handle and shape plans belong to one Program, never a process-global GPU cache.
The split Q4/Q5 attention-input projections now use the same admitted prefill
executor; Q/K normalization, MRoPE, KV publication and decode remain in the existing
runtime. Gate/up weights are expanded in bounded row sections into their complete
FP32 result, retaining the existing SwiGLU boundary without a second resident model.

`NINFER_SERVE_MEDIA_CACHE=1` retains validated image/video preprocessing in an
immutable CPU cache owned by each Frontend. Retained storage is capped at 64 MiB
and 64 entries per Frontend (up to 128 MiB retained storage for this two-engine
profile), accounting for encoded identity bytes and patch/metadata capacity.
Oversized items bypass retention. Lookup checks modality, every processor option,
SHA-256 and exact encoded bytes. Each request receives its own patch copy,
rebuilds prompt token spans and positions, and rechecks aggregate media/prompt
budgets. No generated answer, sampled token or GPU/KV state is cached here.
The retained patch samples are the exact resized RGB8 values in merger-friendly
order, not normalized FP32 arrays. Each request reconstructs the original FP32
normalization with the unchanged formula. This is lossless source-pixel retention,
not a new model/activation quantization. It uses one quarter of the patch payload
storage; encoded identity, metadata and vector capacities are still charged against
the same 64-MiB cap.
Added-token matching keeps the next occurrence of each loaded token within the
current encode call. Absent tokens are not repeatedly searched through the
remaining video-marker stream; consumed overlaps are rescanned from the new
position. The original earliest-position/token-ID ordering and ordinary NFC/BPE
segments are unchanged. State is local to each encode call, not shared across
requests. A real 40-second clip's repeated frontend preparation dropped from
13.1–13.2 s to 0.124–0.129 s in the isolated preparation check. This timing excludes
HTTP admission, GPU prefill and output generation.
The subsequent real HTTP test for that same 40-second clip measured warm first
content at 0.675 s and retry at 0.845 s, versus 14.62/14.24 s with RGB8 retention
but the old repeated-search tokenizer. The complete mixed quality cohort's warm
p95 was 0.854 s. Those short expected-fact replies qualify preparation latency
and correctness, not sustained eight-client throughput.

The eight-client performance check uses simultaneous client SSE boundaries,
not a sum of isolated rates. Cold TTFT and existing-client service during cold
joins must be reported separately. Context capacity includes both input and
output: an approximately 7,600-token prompt inside an 8,192-token capacity can
legitimately terminate at the capacity before its requested output limit. Longer
histories and output-completion checks must reserve explicit context headroom.
The current profile does not recertify the original 65K/262K workload gates.


The protected profile above uses 24 bounded decode rounds. In two physical fresh
image/video arrival repeats, six already-streaming text requests retained minimum
rates of 22.35 and 21.87 tok/s during the common cold-pending interval. All-cold
2K-request p95 first-content latency was 30.4–30.6 s in the original fixed-128
profile. The adaptive 1024/128 profile measured 15.9 s while preserving small live
units; these are different recorded profiles, not one combined result. Fresh image waits were about 8.1–8.2 s
and video waits 17.0–19.0 s. These are explicit latency/fairness tradeoffs, not a
universal service-level guarantee. Six-client cold-pending measurements are not
reported as eight-client throughput. The older 16-round profile reached 20.34
but only 19.76 on repeat and is not the protected-profile qualification.

### Ingress time and established-stream protection

The HTTP service passes its original accepted-request timestamp into `Engine::submit`,
before media acquisition or preprocessing. This timestamp controls only the established-
stream decision for cold/live chunk size and bounded decode priority. A request which
arrived with the initial cohort must not become a late arrival merely because media
preparation finishes after an earlier request starts streaming. Queue deadlines remain
absolute and unchanged; Engine timing remains measured from submission. Direct C++
callers may omit the timestamp to retain submission-time behavior. Future ingress
values are rejected before reserving queue capacity.

### Measuring genuine conversation growth

A repeated identical request is a prefix-replay measurement, not a new conversation
turn. `scripts/bench_serve_replicas.py --phases cold,followup,followup2` appends the
actual delivered assistant response and a new user message each time. A new long
document can be supplied with `--followup-text-file`; the resulting total prompt,
reused/computed tokens and old/new physical owners are recorded per request.
`repeat_current` explicitly repeats the latest input for a subsequent steady-decode
measurement. Do not label that repeat as another appended turn.
`churn_current,refill_current` cancel and retry that same continuing-history input,
rather than silently reverting to the first prompt. Two/four-client runs can supply
correspondingly sized message arrays; their shared-window results are labelled by
actual cohort size and cannot satisfy the eight-client gate.

`--max-context` is the per-request input-plus-output ceiling. `--kv-capacity` is the
aggregate resident KV capacity **per GPU replica**, not that ceiling multiplied by
four implicitly. Four growing 15.4K histories require more than 61K resident tokens
per replica before output allowance. An automatic pool smaller than this may queue
requests and evict completed prefixes correctly; routing affinity cannot create
missing capacity. Explicit capacity remains subject to the native startup memory
reservation checks, including Vision/workspace allocations. The previous whole-item
Vision workspace made a 65,536-token pool fail the measured 22-GiB memory budget.
The bounded-segment implementation described below now passes that native budget;
the original failed run remains evidence for the earlier implementation.

A separately measured longer-history profile uses `--max-context 12288
--kv-capacity 49152 --vision` with the same 1,024/128 prefill caps and full-head
D1 settings. The pool is 49,152 aggregate tokens on **each GPU**, sufficient for
four 12,288-token request ceilings. On the measured stock cards the native
explicit-capacity checks passed; free GPU memory after startup was approximately
1,013/902 MiB and observed peaks in the eight-client long-text run were
20,991/21,100 MiB out of 22,528 MiB. Explicit capacity does not retain the automatic
policy's extra 1-GiB sizing headroom; do not reuse this capacity on a smaller-memory
system without its startup checks.

This Vision-enabled server retained all eight 7.55–7.59K input prefixes through
real appended replies/questions (7.84–8.55K total histories). Warm p95 first content
was 0.885 s, actual follow-up p95 6.802 s, and a subsequently repeated latest input
0.854 s. The actual follow-ups ended too quickly for a qualified common eight-client
decode interval; the repeated latest-input interval measured 205.77 aggregate and
22.70 minimum tok/s. Distinguish the successful follow-up reuse from the separately
measured repeated-input decode rate. This does not establish four simultaneous
16K multimodal histories per GPU or recertify the original long-context gates.


### Bounded Vision workspace and the longer-history profile

The frontend's existing total attention-pair limit remains 128 Mi pairs. An admitted
independent Vision attention segment cannot exceed the integer square root of that
limit. The runtime batches complete temporal segments within that bound; ordinary
images and short videos retain the existing execution. Large videos keep every
previously sampled frame and all original output/scatter positions. Sampling,
spatial resolution, Vision weights and frontend budgets are unchanged. The complete
real-artifact encoder produced identical embeddings for the tested image, short
video and 40-segment extended video on both GPUs. This bounds workspace; it does
not introduce preemption inside a Vision encoding call.

With the opt-in flags above, the measured longer-history profile is:

```text
--devices 0,1 --max-concurrency 4 --max-context 16384 --kv-capacity 65536 --vision
--prefill-chunk 1024 --prefill-live-chunk 128 --prefill-decode-rounds 24
--spec mtp --draft-tokens 1
```

The 65,536-token capacity is aggregate **per GPU**, not per request. All native
allocation checks remain enforced. The measured runtime reservation is about
4.45 GiB per replica, with about 771/660 MiB free after startup on these specific
22,528-MiB GPUs. Explicit capacity does not retain automatic sizing's extra 1-GiB
headroom. Use the startup check and do not assume this fits smaller cards.

Eight real image/video histories grew from approximately 7.7K to 15.3K tokens,
retained all eight prefixes and returned their expected media facts. A separate
longer-answer serving workload at approximately 15.4K prompt tokens measured
214.24 aggregate / 24.94 slowest-client tok/s in its shared warm C8 interval;
post-cancellation refill measured 208.78 / 24.46. All non-cancelled outputs ended
naturally without runtime errors. These are bounded measured workloads, not a
universal semantic or latency guarantee. Full cold ingestion of those 15.4K
histories still took about 119 seconds p95; adding a new 7.6K-token document to
cached media histories still took about 59 seconds p95. Prefix reuse does not
eliminate work on newly appended input.

Evidence: `.chatgpt/production_serve_20260908/cold_memory/`, particularly
`media16k_growth`, `media15k_serving`, `vision_embeddings_final_gpu0.log`,
`vision_embeddings_final_gpu1.log` and `vision_plan_tests.log`.


### Raw single-client qualification

`bench_serve_replicas.py --devices 0 --concurrency 1 --sequential` can drive a fixed
corpus with exactly one live client on one GPU. Its `serial_c1.windows` record
independent client-correlated decode intervals; do not add those rates or call the
corpus a concurrent-throughput result. The 768-token profiling cap can truncate
requested answers and is not an answer-completion/semantic-quality certificate.

Keep inference policies explicit. The historical ~107 tok/s single-client frontier
used the earlier `loose_logit_delta=7` route; it is not directly comparable to the
current full-target, ordinary acceptance policy. Likewise an idle C4-configured D1
server is not a qualification of the dedicated C1 D3 profile. Record output hashes,
MTP yield, actual loaded clocks, client decode, Engine decode and whole-request time
separately. No hardware clocks or thermal policy are adjusted by the harness.


### Separate C1 full-vocabulary drafting profile

On the measured SM75 27B runtime, `--spec mtp --draft-tokens 3 --full-target-head
--fast-full-vocab-draft --max-concurrency 1` uses the existing accelerated full-
vocabulary ranking and native reranking for draft selection only. The original
W8 target weights and full target-vocabulary verification remain authoritative.
There is no relaxed acceptance margin and no artifact modification. This is an
explicit C1 profile, not a replacement for the physical C4+C4 D1 configuration.
It rejects incompatible backend, short draft-head, non-SM75 target and concurrency
settings rather than silently falling back to another performance contract.

The draft-only cache is charged in startup memory planning and released with its
source model, including constructor rollback. Target optimization guards are
independent of the cache's existence: a prepared draft cache cannot accidentally
enable the old target shortlist. The frozen setting is recorded in
`engine.fast_full_vocab_draft` alongside `engine.force_full_target_head`.

The current six-workload single-GPU/single-slot comparison measured 11.2–13.1%
higher client-correlated decode rates with identical delivered output bytes in all
six timed samples. Five answers reached the 768-token timing cap; Japanese ended
naturally at 672. This is output-prefix/performance evidence, not a complete-answer
certificate or recertification of the historical looser-acceptance ~107 frontier.
The short-vocabulary optimized draft head with full target verification was also
measured: it improved five workloads but regressed Thai by about 30%, so it is not
the recommended general C1 profile.

`NINFER_SERVE_Q5_BATCH=1` remains a separate opt-in dense-body optimization. Its
T2 native row mapping is bit-exact in the tested comparisons; its T8 grouped
accumulation is numerically qualified, not bit-exact. Recorded physical C4+C4
mixed, conversation-growth, long-history and cancellation/refill evidence lives
under `decode_architecture/`; do not attribute its batch gains to the C1 draft mode.

## Retained accepted C1 performance profile

Accepted performance routes are cumulative. The historical single-card D3
margin-7 profile remains a supported route, not replaced by newer strict controls.
The rebuilt CLI reproduced its exact 128-token output, digest and 33-round /
99-draft / 94-accepted trajectory at 108.06 and 107.95 tok/s. These are the original
fixed-endpoint qualification, not a universal sustained rate for every workload.

```bash
NINFER_MTP_Q4G64=1 NINFER_MTP_MARLIN_Q4=1 \
NINFER_DUAL_SM75_A8=both-mlp-stage1-all \
build-sm75/apps/ninfer-serve "$MODEL" \
  --host 127.0.0.1 --port 18080 --model-id qwen3.8-27b \
  --device 0 --max-concurrency 1 --max-context 8192 --kv-capacity 8192 \
  --prefill-chunk 128 --kv-dtype int8 --spec mtp --draft-tokens 3 \
  --lm-head-draft --mtp-logit-margin 7 --mtp-retained-packed-prefix \
  --default-max-tokens 2048 --greedy --no-thinking
```

`--mtp-logit-margin` is explicit, validated and logged. Omission remains strict
(-1); a supplied finite value must be in [0,128] and is restricted to the supported
SM75 27B C1 D3 runtime. Non-greedy sampling does not use the greedy margin. Dense
vocabulary scores and indexed candidate metadata have separate typed contracts;
existing dual/D4 callers retain their original semantics.

`--mtp-retained-packed-prefix` preserves the historical initial MTP query's
interpretation of the packed attention parent. That older route was selected by
pointer-only cache lookup. The present default lookup correctly requires exact
shape and scale identity, so the accepted numeric behavior is now a separate,
bounded draft-only operation with its exact prepared parent. It is **not** an
exact logical row slice. The selected target verifier remains the authority; no
cache safety or source-lifetime checks are weakened to emulate the old lookup.
This switch is explicit/default-off and is permitted only for the supported C1 D3
route with prepared MTP Marlin weights. It does not change stored model weights.

The same command with `--mtp-logit-margin 7` omitted is the retained strict-P0M
control. Current-source matching128-token checks reproduced its original output
and43/129/83 counts at80.51 tok/s (35/27/21 accepted by position). The historical
79.01 reference and the older distinct certified80.185 record remain separate.
`ninfer_retained_c1_real_test` exercises both strict and margin profiles after an
unrelated warm-up request, then repeats each golden fixture in the same Engine.

The canonical profile and newer strict/full-target controls remain separately
labelled. Do not enable either retained option in the C4+D1 serving profile.
The new full-target/fast-full-vocabulary combination with margin7 is not a
promoted replacement: it loses some workload throughput and a longer Japanese
answer repeats phrases. The retained route also shows Thai language drift in one
new768-token corpus. Preserve the existing128/256-token qualifications without
claiming they certify arbitrary longer answers. Omitting the margin retains the
separately verified strict full-target option and its six unchanged output hashes.
The existing single/dual/65K/262K and multimodal witnesses are listed in
`tests/retained_sm75_profiles.json`. Run `python3 scripts/check_retained_sm75_profiles.py`
to detect deleted/altered witnesses; that guard alone is not a new performance
certificate. Re-run an affected profile's real fixture before claiming preservation.


## Independent staged prefill owners

`--max-prefill-lanes 4` allows a four-slot replica to retain several unfinished
prompt ingestions and service them at chunk boundaries. The default remains `1`,
which retains the established single-owner scheduler. The maximum must be between
one and the replica's configured concurrency; it does not create additional slots.

This mode removes the requirement that a long cold prompt finish before another
request can be admitted. Existing staged owners receive round-robin work; new
admission alternates with that work so continuous short arrivals cannot starve an
older prompt. The existing bounded protection for streams established before a
new request's ingress still applies.

Each owner has a disjoint, stable slice of the **existing startup-frozen request
transient allocation**. No per-request GPU allocation, compaction or increased
startup memory reservation is introduced. Text-only prompts need no such slice;
media owners must fit the available bytes and alignment. A blocked reservation
waits rather than borrowing another request's storage. KV/GDN ownership, request
limits, cancellation and exact prefix identity remain per request. Every resumed
prefill step explicitly rebinds the current owner's text/backend KV selectors and
position state, not the last request admitted by the scheduler.

This is independent **staged** execution, not simultaneous GPU kernel execution
or packing several prompts into one dense matrix operation. Cross-request packed
prefill remains separate work. More owners can improve short-request responsiveness
and cohort completion, but can delay the first content of a large request while
others get service. Evaluate the actual input-length mix and report both sides of
that tradeoff rather than treating owner count as a universal throughput switch.
Keep canonical C1 and its retained strict/margin profiles at one owner.

The current owner gate is **strict C1 D3 >=100 useful tok/s with acceptable output
quality**. It remains open. The retained ~107 margin7 route is still available as
a distinct historical/experimental profile and does not satisfy that strict gate.
Staged-prefill latency results do not earn raw-C1 kernel or acceptance-policy credit.

## Historical C1 radix-projection qualification

`--sm75-radix-projections` is an opt-in numerical alternative for the SM75 27B
C1 MTP D3 profile with strict acceptance (`greedy_margin_delta=-1`). The section
below records its original C1 qualification; wider D3 support is described
separately. It is not a margin policy and remains rejected for C4/D1. It leaves the model artifact unchanged
and uses source-owned prepared Q4/Q5 views with two activation limbs.

The final prepared allocation is **2,080 MiB** beyond the ordinary model/runtime
allocation, requested and charged in whole2-MiB blocks. Native startup memory
checks remain enforced. Do not treat the flag as a default or enable it in the
physicalC4+C4 profile. On the matching128-token strict fixture it measured82.63
versus80.74tok/s in the final reverse pair, with exact historical output/counts.
The strict100 gate remains open. Longer outputs can differ numerically; the
second-card six-family corpus improved five measured rates but not Japanese,
and capped answers are not complete-answer certificates.

Use the existing C1D3 optimized-proposal/retained-packed-prefix command, omit
`--mtp-logit-margin`, and add this flag. The tested no-Vision8K profile and
focused Vision facts, exact command/binary receipts, independentFP64 tests,
cancellation and model-reload evidence are indexed in
`.chatgpt/production_serve_20260908/strict_projection_0947/serving_checkpoint.md`.

This candidate does not supersede strict79, legacySINGLE80, retainedmargin107,
or any existing dual/long-context/media/serving witness.


## Longer radix answers and explicit retained-target draft composition

Radix is not rejected merely because numerical differences change the wording. A
same-binary one-GPU/one-slot D3 test with a 3,072-token output budget produced natural
Thai/Japanese/Arabic completions of 1,057/800/1,265 tokens with radix, versus
1,037/723/1,189 in its control. The complete sampled prose remained in the requested
language with ordinary technical code-switching, without observed pathological
repetition. This is longer-answer evidence, not a universal quality certificate.
The separate generated-code answer had erroneous tests and malformed quoting; its
full delivered file is NOT a code-quality pass. The verbatim cache implementation
extracted before its tests passed 10,000 independent model-based operations in each
arm; do not substitute that library check for the full answer's correctness.

`--full-vocab-draft-retained-target` is a separate explicit composition: it ranks
proposals over the full native vocabulary while retaining the existing optimized
target policy and strict acceptance. It requires C1 MTP D3, `--lm-head-draft`, and
margin -1; `--full-target-head` and margin7 are incompatible. The older
`--fast-full-vocab-draft --full-target-head` policy remains unchanged. Cache readiness
never implicitly selects the target policy. The new flag is frozen in Engine
features and logged as `retained_target_full_vocab_draft`; the existing vocabulary
cache is charged once, not duplicated.

In a matched test with radix enabled in both arms, complete Thai prose improved
34.77 -> 54.28 client-correlated tok/s, Japanese 49.69 -> 54.41, and byte-identical
Arabic 54.85 -> 60.18. All ended naturally. Thai/Japanese answer bytes differed;
Arabic retained exactly 1,265 output tokens while verification rounds fell612->551.
This is proposal-yield improvement, not a pure kernel gain or strict100 certification.
The historical128-token strict fixture was slightly slower (81.15–81.24 versus
82.17–82.42), so this profile does not replace the original radix/strict winner.
Further current-source validation receipts are under the evidence path below.

Composing radix with the existing large-chunk dense prefill also reduced cold
first-content time on two populated7.55–7.57K inputs (~37s -> ~11.5s), but that
specific inventory/migration cohort did not pass full answer-quality review: the
inventory response mixed up transfer count and box quantity and the fast answer
ended mid-sentence despite a stop-token termination. Do not promote that composition
as qualified from timing alone. Cycling two unrelated conversations through one
retained slot produced cache misses; the harness's `warm` phase name is not proof
of cache reuse in that cohort.

Evidence: `.chatgpt/production_serve_20260908/long_quality_and_body_1106/`, especially
`long_quality_comparison.json`, `generated_code_review.json`, `stack_assessment.json`,
and `retained_target_composition/prose_comparison.json`. The existing strict79,
radix, margin107, dual, multimodal and long-context routes remain preserved.

The final-build repeat reproduced all three composed prose output hashes at
54.34/54.41/60.10 tok/s. Current-source cancellation/invalidation/deadline/lease,
three model reloads, old strict/margin goldens and16focused media facts passed.
Eighteen affected CTest targets and14streaming tests passed. The separate4,096-budget
code check parsed in the new profile but3of17generated tests had incorrect expected
values; both generated implementations passed10,000 independent cache operations.
Both JSON outputs satisfied every rule across40records. These are separate verdicts,
not a universal quality pass. All current test servers shut down normally.

## Timing-aware established-stream prefill budget

`--prefill-target-decode-tps F` optionally replaces the fixed decode-burst length
with a recent measured budget for each established stream. Zero (the default)
retains the existing fixed-round scheduling exactly. `--prefill-decode-rounds N`
remains an upper bound, so a slow or infeasible target cannot starve the incoming
request indefinitely. This is a best-effort scheduling target, not a throughput SLA.

For the physical two-C4/D1 profile, the tested setting is:

```text
--prefill-chunk 1024 --prefill-live-chunk 128 --max-prefill-lanes 4
--prefill-decode-rounds 24 --prefill-target-decode-tps 24
```

Each request owns a bounded window of actual committed decode progress. The
scheduler measures a complete decode/resolve/publish work unit and the recent peak
cost of protected prefill chunks. It selects the largest required burst across
individual established streams, not aggregate batch throughput. Unknown timings,
a changed batch size, or a requested rate above the measured decode rate retain
the configured fixed maximum until sufficient observations exist. Cancellation,
deadline, admission, prefix identity and chunk-boundary ownership remain unchanged.
No GPU allocation, target arithmetic, speculative acceptance or media input changes.

For a measured per-round token yield `u`, decode duration `d`, protected prefill
cost `p`, and configured target `r`, the budget is the bounded ceiling of
`r*p/(u-r*d)` when the denominator is positive. The input windows are small and
request-local; no model training, prompt tables or checkpoint calibration is used.
The measured policy only shortens the existing maximum burst; all-cold ingress and
short reusable suffixes retain their existing one-round alternation.

Initial real-HTTP qualification on September13 compared the same executable,
workload corpus, fresh media and arrival pattern, with two independent timed runs.
The late video reached first content in14.48/14.92s versus17.58s in the fixed-round
control. The late image remained around8.04/8.09s versus8.13s. The slowest established
client in the common six-stream cold-pending interval measured21.58/21.60 tok/s
versus23.22 in the control. This is a latency/fairness tradeoff, not a decode-kernel
speedup and not another eight-client throughput gate. Dynamic routing selected
slightly different warm prompt assignments across runs; do not call them byte-for-
byte identical GPU cohorts. Final repeat/control and validation receipts are under
`.chatgpt/production_serve_20260913/timed_prefill/`.

The target24 setting is not a claim that every client always achieves24 tok/s.
Observed client-boundary rates include interruption and conservative stream-edge
accounting; the owner20 tok/s floor remains the separate measured test criterion.
General video temporal descriptions and some inventory answers have existing
semantic limitations; this timing test does not certify those tasks universally.

The later fixed-round control measured19.30s video first content and20.90tok/s
minimum established rate. Across two fixed and two timed runs, video means were
18.441 ->14.700s (20.3% reduction); the same-corpus caveat above still applies.
Final timing-enabled mixed C8 measured252.88/29.55 warm,236.70/26.94 actual follow-up,
and239.33/27.45 cancellation/refill (aggregate/minimum tok/s), with8/8prefix hits.
The real overlapping-media cancellation test and retained strict/margin golden
fixtures passed. No test server remains deployed.


## Streamed prefill: memory option, not a new speed default

`NINFER_SERVE_PREFILL_STREAMED=1` enables process-frozen streamed projection
intermediates and aligned bounded workspace planning. It makes the tested4K
prefill/16K-per-request/64K-per-GPU/Vision C4 profile fit native checks. Existing
flags/defaults and the old dense route remain unchanged with flag0.

A September14 same-binary comparison against the stronger2K reference found cold
p95 39.965/42.136s for two2K controls,40.558s for streamed4K and41.440s for streamed2K
on eight~7.6K inputs. Do not promote4K as a speed winner from the earlier1K comparison.
At matched2K width, the streamed option saves128,247,808 planned bytes (~122.31MiB)
per GPU and122MiB of observed peak memory on each card, with latency inside the
observed control range. Numerical output differences remain; no universal bit-exact
or general-answer-quality claim follows from the memory saving.

The4K profile passed complete-versus-incomplete prefill cancellation, real concurrent
media isolation,16short focused facts and16~15.3K-history facts. A separately labelled
longer actual-follow-up scenario measured267.37 aggregate /30.63 minimum tok/s,
with8/8prefixes; it is not the earlier short-follow-up throughput sample, whose
1.776-second common interval was too short to qualify. The corrected cancellation
test uses actual token counts and validates complete-checkpoint reuse as well as
partial-state invalidation. No runtime cancellation guard was weakened.

Evidence and remaining limits: `.chatgpt/production_serve_20260914/resume_streamed/serving_checkpoint.md`.


## Packing protected text-prefill work units

`--pack-prefill` is opt-in and defaults to off. It allows compatible unfinished
text requests to share their dense projections while keeping attention, KV-cache
rows, convolution state, recurrent state and MTP alignment independent. Sampling,
last chunks, turn-checkpoint boundaries and requests containing media stay on the
ordinary path. This is not answer caching, cross-request attention, or a change to
speculative acceptance.

Only whole, equal-sized work units are packed, and their combined token count must
fit the existing `--prefill-chunk` reservation. Thus large all-cold requests are not
subdivided just to create a batch. The useful tested case is two established users
and two arriving text requests on each GPU: the arrivals' protected128-token work
units can share a256-token projection inside the already-planned2K workspace.
The combined elapsed time feeds the existing timing-aware decode budget.
The four established users are measured individually; aggregate throughput cannot
hide a slow user. `--prefill-target-decode-tps` remains a best-effort target, not an SLA.

The qualified model/hardware scope is the current SM75 dense27B MTP runtime, with
at least two prefill lanes and a maximum chunk of at least256tokens. The measured
profile keeps the existing dense/resident prefill implementation enabled:

```text
NINFER_SERVE_PREFILL_BLAS=1 NINFER_SERVE_RESIDENT_PREFILL=1
--devices 0,1 --max-concurrency 4 --max-prefill-lanes 4
--prefill-chunk 2048 --prefill-live-chunk 128
--prefill-decode-rounds 24 --prefill-target-decode-tps 24 --pack-prefill
```

It retains the normal D1/full-target serving profile, Vision and native capacity
checks. It does not require an additional GPU allocation or larger KV pool. All
existing non-packed paths remain callable. The source-preparation mechanism does
not use checkpoint hashes, calibrated prompt tables, training or learned adapters.
Only the existing checkpoint has been exercised at full-model level.

The startup record includes `pack_prefill`. Throughput records report cumulative
`packed_prefill_cumulative.rounds` and `.tokens`; these are actual executed work,
not queue-size estimates. The harness's optional `--cold-join-warm-per-gpu 2`
selects a separately labelled four-established/four-new scenario. Its established
stream window is NOT an eight-client throughput certificate. The original3+1
per-GPU arrival scenario remains the default and keeps its prior checks.

Initial all-cold packing was correctly state-isolated but slower than the retained
2K control, so the final policy does not use that strategy. Complete-operation
projection speedups are not substituted for real first-content latency. Current
same-binary comparisons, state-isolation proofs, qualification limits and final
source identities are recorded under
`.chatgpt/production_serve_20260914/packed_prefill/`.

The final same-binary live-join pair measured22.926 ->18.718s mean first content
for four newcomers (18.36% reduction). The slowest established-stream rate in the
common four-stream early cold-pending window changed24.04 ->22.17tok/s, remaining
above the measured20tok/s floor; maximum observed chunk gap increased.483 ->.616s.
This tradeoff is explicit. Dynamic warm owner assignments and boost clocks differ,
so the repeated results are not a universal SLA or isolated-kernel percentage.
Planned GPU runtime bytes remained4,999,614,720 in both arms. Final mixed C8 warm,
actual-followup and refill rates were266.38/31.69,262.61/30.91 and259.51/30.51
aggregate/minimum tok/s; all non-cancelled replies stopped naturally with zero errors.
The21CTest/15stream/11policy checks, actual packed cancellation on both GPUs, media
fallback and old strict/margin golden trajectories passed. See `serving_checkpoint.md`
in the evidence root for limits, exact identities and reproduction commands.


## Complete-context direct FP16 value arithmetic

`ninfer-dual --context-value-fp16` is an opt-in numerical alternative that requires
`--context-headscale-qk`, complete target computation, physical context-parallel
INT8 cache, one MTP D3 request, and a compatible canonical prefix snapshot.
It is disabled by default; the prior headscale and original group64 paths remain
separate. It is not a replacement for the native C4+C4 serving profile.

The value cache's existing signed INT8 codes and FP16 scales are untouched. The
kernel reconstructs pairs of values directly in FP16 instead of rounding through
BF16 before the existing FP16 tensor-core operation. It still reads all original
target keys and executes all target layers and the full target vocabulary.
No target weights, draft depth or acceptance margins change. The new intermediate
rounding can change outputs and acceptance trajectories; do not label it bit-exact.

The packed converter matches independently computed FP32-product-to-FP16 rounding
for every signed byte and every nonnegative finite FP16 scale: 8,126,464 cases per
GPU, including zero, subnormal and overflow outcomes. These arithmetic checks are
separate from model quality. No persistent cache allocation or format conversion
is needed for the value change. Existing headscale-QK conversion and canonical
snapshot protections remain unchanged.

The first paired genuine260,973-input test measured40.78 ->44.01 tok/s; reversing
order gave40.63 control versus43.98 candidate. The candidate produced241 tokens
versus227, stopped naturally, recovered all three distant records and repeated its
own output exactly. It used105 verification rounds in each mode, accepting136
rather than122 drafts. Consequently the observed8% gain combines faster work and
changed proposal yield; it is not an8% pure-kernel claim.

At64,172 input tokens the candidate was slightly slower (54.20/54.26 versus54.54/
54.38), so it is not a universal replacement. Fresh12K-context JSON and Thai tests
also compared canonical capture, headscale, and direct-FP16 modes. All three modes
satisfied every rule across40 JSONL records and finished naturally. The complete
Thai sample stayed coherent and in-language through its conclusion; this is not
universal factual or cross-checkpoint quality certification.

The startup summary reports `target V arithmetic`, and `context GQA value FP16
calls` proves actual execution. Incompatible CLI profiles are rejected before
loading a model; coordinator construction also validates the arithmetic contract.
Current evidence, final paired tests and remaining limits are under
`.chatgpt/production_serve_20260914/gqa_packed_half_values/`.
Neither strict-C1 nor complete-dual100 has been closed by these results.


## Active-width TP2 transport

The dual MLP coordinator sends only the active input columns and their FP16 G64
scales: 21,120 bytes for four columns, 26,400 for five. The scale offset follows
that width, while the reusable allocation remains large enough for either.
Producer-ready events, remote-copy ordering, engine completion and pinned-buffer
ownership are unchanged. Four/five/four reuse is checked against the old packet
layout through every real TP2 MLP shard, with unused allocation bytes poisoned.

Both owner and helper outputs remain BF16. The owner no longer expands its result
into an unused FP32 array, and remote staging is sized for the actual BF16 data.
The summary reports input and partial transfer bytes separately, each counting
both PCIe legs. This internal change does not alter target weights, verification,
KV precision, vocabulary coverage or the old four-/five-column interfaces.
Evidence and the small, clock-sensitive end-to-end comparison are under
`.chatgpt/production_serve_20260916/tp2_active_packet/`.


## Shared-weight radix projections for wider D3 verification

`--sm75-radix-projections` also supports two, three or four configured D3 slots
on the SM75 dense-27B architecture. It remains explicit and default-off. The
native production C4+D1 and physical C4+C4/D1 serving profile is unchanged.
The multi-slot tests use the full native target and full proposal vocabulary;
the C1-only retained packed-MTP option is not enabled by this feature.

The eight-/sixteen-column kernels load each original Q4/Q5 weight fragment once
and use it for two/four independent T4 radix groups. Each integer dot product,
radix-limb FMA, split-reduction order and BF16 output matches the independent
T4-group reference exactly. Compared with the original native wide fallback,
radix is still a numerical alternative: changed intermediate projection arithmetic
can change answers and acceptance. Target weights and strict acceptance do not change.

The explicit four-slot test uses4K context per request and16K total KV capacity.
It is a separate high-memory D3 profile, not the normal16K/65K native D1 profile.
Automatic sizing with its usual extra1GiB headroom may reject the wide D3 setup;
that headroom rule is not weakened. Explicit capacity still passes the full
runtime reservation check before any allocation.

Prepared source identity, stream, capacity and allocation remain fixed for graph
lifetime. Capacities are4,8,16; unhandled ragged widths retain their native fallback.
The memory planner reserves the same capacity as preparation. Same-address weight
reuse requires release/reprepare; duplicate preparation cannot grow existing storage.
Repeated graphs, inactive columns and padded output strides are tested on both GPUs.

The measured two-slot D3 target pass was45.98ms native versus40.21ms radix,
a5.77ms target-time reduction. This is not a single-client decode result or a
100tok/s certificate. The four-client bounded serving comparison was about19–21%
faster on the sampled workload, but those512-token measurements hit their caps.
Separate natural code, JSON, multilingual and media responses were checked rather
than treating a token cap or a natural stop as proof of semantic correctness.

The wider prepared weights consume additional VRAM. Startup reservation guards
remain authoritative; no capacity is silently reduced. In particular, attempting
this kernel in the normal16K-context/65K-total-cache/vision C4+D1 service exceeded
available memory. Direct canonical-source reads were then checked independently:
outputs were exact, but eight-column cost lost the speed advantage. That D1 admission
extension was removed rather than shipping an unqualified or smaller-capacity route.
No in-place mutation of the canonical model weights was introduced.

Final source-bound tests and qualified operating cases are recorded in
`.chatgpt/production_serve_20260916/wide_radix_final/`. The original operator/profile
results are in `wide_radix_weight_reuse/`; memory-failed D1 and direct-read experiments
remain separately archived. Generic strict C1 D3 and complete dual full-context
100+ remain open; multi-client or wide-verifier rates do not close those gates.


## Exact ragged vocabulary projection on SM75

The full-vocabulary W8 head now initializes every column of its existing eight-column
activation tile when five, six or seven columns are active. The unused columns are
zero. Valid inputs, weight/scaling values, matrix operations, reduction order and
output width remain unchanged. No extra model storage or allocation is introduced.
The T1–T4, full T8, wider shapes and non-SM75 execution policies are unchanged.

The original and revised kernels matched all tested full-vocabulary logits, padded
output guards and repeated graphs on both cards. The permanent regression also
changes source contents in place and checks zero inputs and strided outputs. The
five-to-seven-column operator was about1.52–1.60x faster in these tests. A full261K
experimental D4 comparison improved only about2.3% and remained slower than retained
D3; that D4 admission extension was not promoted. Operator speed is not whole-model
speed, and the generic C1/full-context100 gates remain open.
Evidence: `.chatgpt/production_serve_20260918/w8_single_tile_scale_ownership/`.


## Parallel immutable prefix restore

When the dual runner imports a canonical legacy prefix into sequence-sharded KV,
it reads and uploads the two independent stage snapshots concurrently. Each task
owns its source and helper buffers until the respective uploads have completed.
Both tasks join before prefix activation, decoding, or error propagation. If both
fail, the first-stage error is reported deterministically; neither task is detached.
Other snapshot layouts retain their existing sequential import order.

Snapshot input storage is allocated for overwrite rather than first zeroing bytes
that a successful complete file read replaces. Size, open and read checks still
reject missing, inconsistent or truncated files before exposing the buffer.
The same joined-task helper owns the already parallel stage-local attention-prefix
preparation. Snapshot contents, format, model arithmetic and decode policy do not
change, and no preparation or restore time is excluded from startup accounting.

On the recorded 260,973-token snapshot, the stage-concurrency comparison reduced
restore time from about12.9 to7.8seconds; eliminating redundant initialization
reduced a second matched comparison from about8.0 to7.2seconds. Warm decode remains
roughly50tokens/s with optional all-key projected attention and47.6with the native
path. These are snapshot-restoration/startup improvements, not fresh uncached
prefill results, a100tokens/s certificate, or new eight-client serving performance.
Evidence: `.chatgpt/production_serve_20260919/parallel_snapshot_restore/`.


## Exact joint-attention union compaction

The optional all-key joint-attention pipeline compacts selected union masks in
three ordered stages: tile-local prefix counts, per-head tile offsets, then
parallel emission of the selected indices. Ascending index order, counts, overflow
limits and preexisting fallback flags are unchanged. This scheduling change does
not change the numerical attention mode, selected-key thresholds, basis ranks,
query arithmetic or contribution of any key.

The pipeline reuses the per-query mask allocation only after query counting and
union-mask generation have finished on the same stream. That storage is cleared
before the next replay. No allocation or source-weight change is introduced.
Capacity, empty/full masks, failure flags, source reuse, ordered outputs and guard
bytes are covered by `ninfer_joint_ordered_union_test` on both GPUs.

The current261K comparison preserved its entire answer and acceptance trajectory.
Its approximately0.47% mean decode difference is small and clock-sensitive; the
larger isolated compaction ratio is not a whole-model speedup. The underlying
integer/floating coordinate attention modes remain numerical alternatives with
their own qualification limits. Full-context100 and generic strictC1 100 remain open.
Evidence: `.chatgpt/production_serve_20260919/exact_selection_1815/`.


## Direct expansion of resident prefill weights

The resident-radix serving profile expands GDN query/key and attention/GDN output
weights directly into the existing prefill BLAS operand. It no longer materializes
an intermediate canonical quantized matrix for these two qualified shapes. Original
codes, scales, BF16 weight rounding and FP16 operands are unchanged, as are the BLAS
plan, residual arithmetic and warm decoding. Other layouts and unsupported-width
native fallbacks still use their original paths. Canonical scratch reservation stays
available for those fallbacks; no memory-capacity relaxation is claimed.

Tests compare entire expanded operands, non-aligned row slices, all complete
operator outputs, repeated graphs with changed inputs, and invalid-layout rejection.
Both GPUs pass. Controlled same-device text/image/video requests preserve output
bytes and acceptance. Concurrent timing depends on replica/batch scheduling and
is not an exact-response or statistically established latency-improvement claim.
The initial isolated live-prefill operations save approximately3–10% of their own
cost; this is not a warm-decoding improvement or a100tok/s result.
Evidence: `.chatgpt/production_serve_20260920/resident_prefill_direct/`.


## Ready-only short-text prefill qualification

With the existing packed-prefill and resident SM75 serving paths enabled, the
executor can group already-ready, fresh short-text owners. The qualified path is
restricted to prompts below the existing 64-token BLAS boundary. It does not wait
for future arrivals, split good large cold chunks, or remove established-stream
protection. Request-specific attention, convolution, recurrent state, MTP context,
final-token processing and sampling remain separate. Images, videos and prefix
reuse retain their existing paths.

A fresh reverse-order four-run comparison on physical C4+C4/D1, at unchanged
16K-per-request context and 65K total KV capacity, reduced the eight-request burst
completion time from 2.726 to 2.334 seconds (14.4%) and p95 first-token time from
2.517 to 2.117 seconds (15.9%). Median first-token time increased from 1.664 to
1.741 seconds (4.6%). Every owner returned its expected private fact, naturally,
with the same 81 total output tokens. This is an explicit throughput/tail-latency
tradeoff for simultaneous short arrivals, not an across-the-board latency gain
or a warm single-request decode improvement. The earlier mixed-media suite stayed
within fairness/cancellation/prefix gates, but did not establish higher warm
throughput; its rates and timing variations are retained as separate evidence.

The September20 final numerical 261K check once returned a different answer.
It remains recorded and unexplained: the unchanged binary later reproduced the
retained 241-token answer twice, an original executable once, and both current
capture/control calls again. This is not a claim that universal numerical
reproducibility was proved or the earlier outlier was repaired.

Current qualification and exact source identities are under
`.chatgpt/production_serve_20260928/ready_prefill_qualification/` and
`full_context_resume/`. The inherited implementation remains uncommitted. Generic
strict C1 D3 and correct full-context dual 100+ remain unresolved requirements.


## Coalesced dual-attention packets

The optional joint-attention path transfers each query/position/optional-KV input
in a single aligned packet. The helper uses typed views into that packet rather
than unpacking it. Native attention, shorter contexts and target-guidance input
retain their direct copy-engine transfers: an input packing kernel competed with
native attention in the measured all-modes experiment. No new public flag or
numerical policy is added.

All dual-attention paths use contiguous FP32 numerator/maximum/normalizer output
state, requiring one DMA transfer in each direction instead of three. These are
layout/copy changes only. Existing host synchronization, cross-device event
ordering, global admission, native fallback, target-key coverage and arithmetic
remain unchanged. Fixed envelopes support one through six active columns, with
explicit position padding. Model prefix snapshots and canonical KV layout do not
change.

Joint input packing needs one 98,336-byte owner staging allocation per stage and
one small packing kernel per joint layer. The helper packet adds only eight bytes
of maximum-width alignment padding; output-state allocation size is unchanged.
When joint attention is disabled there is no owner staging allocation. For the
measured four-column request, transfer payload bytes are unchanged. This is fewer
transfer operations, not compression or a reduction in model weights.

The final actual 261K paired checks preserved both native and joint answers,
verification counts and fallback decisions. Native throughput remained essentially
flat; joint throughput improved about2.4% in the measured alternating runs.
Natural clocks varied and the experiment is not a statistical confidence bound.
The inherited joint numerical approximation retains its own quality limitations;
packet equivalence does not certify every generated answer. Generic strict C1
and complete-context dual100+ goals remain open.

`ninfer_attention_packets_test` checks bit patterns, widths1–6, optional KV,
position padding, source reuse, strided state regions and repeated graph guards
on both GPUs. Source-bound model/profile evidence is retained under
`.chatgpt/production_serve_20260928/coalesced_attention_packets/routed_inputs/`.


## Full-vocabulary MTP proposal-resource ownership

The legacy `ProposalHead::Optimized` selector controls target-head eligibility as
well as the older corpus-selected proposal path. Full-vocabulary MTP does not
consume that proposal payload: it ranks from the source-owned native W8 output
head instead. `StartupFeatures::requires_optimized_proposal_payload()` now makes
that resource distinction without changing target-head eligibility or numerical
execution.

Full-vocabulary MTP therefore keeps the two optional corpus-head objects as
schema-validated metadata, rather than materializing their weights and token IDs
on the GPU. Their B2 preparation and reservation are also omitted. The native W8
ranking cache remains prepared and reserved exactly once, whether used by target
ranking, full-vocabulary proposals, or both. Binder ownership, loaded-model
invariants, dual-shard loading, prefill configuration and memory planning all use
the same frozen feature decision. Legacy optimized-proposal and DFlash behavior
are unchanged; existing model artifacts are not rewritten.

The real-artifact ownership regression verifies that exactly the two unused
objects disappear and that every target/MTP/native-vocabulary object remains.
Their device payload is 357,040,128 bytes. In the measured C1 composition, removing
that payload and its unused cache reduced sampled peak memory from 20,347 to
19,917 MiB. The 261K dual output GPU fell from 18,110 to 17,704 MiB; the other GPU's
sampled peak rose from 18,307 to 18,351 MiB. These independently sampled peaks are
not a precise simultaneous total. Exact answers, verification trajectories and
snapshot metadata were preserved. Warm decoding did not materially accelerate.

The separately tested existing dynamic-target/live-draft-union profile completed
long answers faster than the full native target head, but its generated Python
module failed two of its own tests while the native output passed all 15. It is
not promoted as a new general-purpose default or a correctness-certified speed
profile. Full-vocabulary proposal search and unchanged strict acceptance settings
do not guarantee that approximate target selection equals full-native argmax.

The ownership fix is independent of that quality decision and changes neither
profile's output. The real text/image lifecycle regression covers cancellation,
matching-state prefix replay and invalidation; its known cold/warm image wording
difference is not asserted to be universal token equality. Generic strict C1 D3
and complete-context dual single-request 100+ remain unmet.
Evidence: `.chatgpt/production_serve_20260929/full_vocab_head_ownership/` and
`.chatgpt/production_serve_20260929/permitted_c1_composition/`.


## Exact full-vocabulary output-row parallelism

The experimental dual-driver flag `--parallel-vocab-head` applies only to complete
single-request D3 verification. It splits the native 248,320-row W8 output head
across the two devices without changing its per-row kernel, FP16 operands, group
scales, FP32 reduction, BF16 logits, or full-vocabulary argmax. The helper returns
all of its logits, not just a candidate list or local maximum. Existing optimized
heads and ordinary single-device/multi-user execution remain separate and unchanged.

The owner retains its canonical full output-head weights. The helper receives an
immutable source-owned copy of the upper half: 675,430,400 bytes. Persistent device
workspace adds 2,027,520 bytes across the cards, and pinned input/reply packets add
1,034,240 bytes. Initial copying uses an additional temporary 16 MiB pinned staging
buffer. Preparation is charged to prefill; the measured addition was about 0.37 s.
This optional memory/startup tradeoff is not a free or automatic optimization.

The operator joins owner/helper streams through event dependencies and captures
its own bounded graph. Changing source contents requires explicit preparation,
which invalidates the old graph and reloads helper weights. Changing input/output
addresses rebuilds the graph. Unsupported widths use the original local path;
outer stream capture likewise selects native fallback. Detachment clears the
borrowed runtime hook and joins work before the coordinator frees pinned buffers
or device allocations. Tests exercise both physical ownership assignments,
same-address source refresh, zero/nonzero inputs, output rebinding, invalid API
requests, outer graph fallback, output guards, and immediate destruction after
submission against every original full-vocabulary logit.

On the 260,973-token paired workload, the optional integer-coordinate joint route
measured 54.35–54.38 useful tok/s versus 53.50–53.58 without head splitting. The
non-joint headscale/direct-value route measured 48.11–48.25 versus 47.31–47.56.
Both changes are approximately 1.5% warm-decode improvement, not 100+. Every run
preserved its complete 241-token answer, natural stopping, verification trajectory,
distant records and snapshot metadata. These are four-run alternating comparisons,
not confidence intervals; current boost and host activity still affect timing.
An unrelated Android-emulator CUDA context was left untouched on one GPU; its
process metrics reported no positive sampled SM/memory usage, often unavailable
(`-`). Thus this is shared-host evidence, not certified exclusive-GPU measurement.

The head operation is exact relative to the selected original execution profile.
It does not remove the underlying joint-attention approximation's previously
recorded quality limitations. Higher peak memory and extra initialization can make
short end-to-end requests slower even though warm decoding improves. Both generic
strict C1 and complete-context dual single-request 100+ goals remain unmet.
Evidence: `.chatgpt/production_serve_20260929/frequent_exact_head_parallel/`.


## Exact compact greedy-result transfer

`--parallel-greedy-head` selects an additional complete dual C1 D3 mode. It uses
all original W8 vocabulary rows and the same BF16 logits as `--parallel-vocab-head`,
but each GPU reduces its local logits and the helper returns four (score, token)
pairs: 32 bytes instead of 993,280 bytes. Global tie-breaking still selects the
lower token ID. Invalid physical padding is excluded by the logical vocabulary
domain. No candidate shortlist, precision change or relaxed acceptance is involved.

This is a separate `HeadExecutionOverride::execute_greedy()` contract: success
means exact token IDs are available, not that caller-owned full logits are valid.
The full `execute()` contract still writes every logit. The output stage tracks
whether its current verification materialized logits; a subsequent full-logit read
reprojects the current stored residual when necessary. It never exports a previous
round's stale logits. Switching result contracts or changing token addresses/domain
invalidates the cached graph. The new workspace is 15,648 bytes; the existing
source-owned helper weight copy and preparation tradeoff remain unchanged.

The real-stage audit matched 7,946,240 bytes of full logits between direct full
return and on-demand materialization after compact verification, including repeated
reads and successive rounds. Permanent tests also cover source refresh, graph
replay, full/compact switching, logical domains, NaNs, guards and invalid requests
on both physical ownership assignments.

The prerequisite comparison uncovered a tiled-argmax replay defect: a NaN at row
zero failed to win against a finite stale seed, and an all-NaN input could leave an
old result untouched. A uniform row-zero check now enforces the existing documented
contract independently of prior output contents. A failing regression was recorded
before this correction, then passed on both GPUs; finite logits and tie ordering
are unchanged. Shared reduction helpers live in `ops/common/argmax_reduce.cuh` so
application users do not instantiate duplicate global launch kernels.

On one 260,973-token alternating comparison, the retained full-return head measured
54.61/54.36 useful tok/s versus 55.04/54.99 for compact transfer (about 0.97%). Each
run preserved the same 241-token naturally completed answer, strict acceptance
trajectory, distant records and snapshot metadata. These small paired measurements
are not confidence intervals or a 100+ result. The existing joint-attention
approximation and its quality limitations are not changed by the head optimization.
Both modes remain opt-in; ordinary C1 and existing full-logit callers remain separate.
Evidence: `.chatgpt/production_serve_20260929/compact_greedy_integration/`.


## Exact valid-query tiling for integer joint attention

The integer-coordinate all-key scan now assigns one head/partition to three native
SM75 eight-row warp tiles. Those 24 rows are exactly the real queries. The previous
two logical sixteen-row tiles performed work for eight padded rows. The new launch
is `4 * active_parts` blocks of 96 threads; callers must preserve that geometry.
No query, key, layer, selected-key correction, or fallback decision is omitted.

The same signed-byte matrix instructions, per-query contraction order, group
scales, threshold comparisons, union exclusion, online softmax, probability
quantization and value-accumulation order are retained. Each warp contributes to
a shared three-way exclusion union before accumulating values. Double-buffered
mask slots prevent premature overwrite between key tiles. Prepared coordinates,
source lifetimes and external output/state layouts are unchanged. There is no
extra device tensor, preparation step, runtime flag or new approximation.

On both physical 2080 Ti cards, the complete all-key operator improved from about
205–209 microseconds to 139–143 microseconds. Registers fell from 122 to 72 per
thread and static shared memory from 32 to 24 bytes; neither implementation used
local spills. These operator results are not the model-level speedup.

Actual 260,973-token same-binary alternating runs improved from 55.15/55.14 to
56.15/56.14 useful tokens/s, a 1.81% ratio-of-means increase. Every run retained the
same 241-token naturally completed answer, 96 verification rounds, 288 drafts,
145 accepted drafts, distant-record recovery and snapshot metadata. The final
rebuilt implementation confirmed the same behavior at 55.99 tok/s. Natural clocks
varied and these four alternating observations are not a confidence interval.

The permanent regression compares production with a frozen pre-change reference,
not with another call to the same kernel. It covers ten source/range/threshold/
empty/tail cases on each GPU, including nonzero owned-storage offsets, changed
query contents under captured graphs and output guards. Independent signed-byte
matrix checks cover all 65,536 input pairs per card. Ordinary single-GPU C1,
floating-coordinate attention, compact/full vocabulary heads and legacy routes
remain separate; the new kernel changes only the integer-coordinate all-key path.

This is exact scheduling relative to the existing numerical joint-attention
profile. It does not remove that profile's previous long-output quality limitations
or establish an unquantized-model or multimodal-quality certificate. No additional
buffer allocation was introduced; one candidate run showed a 24 MiB higher sampled
transient peak on GPU1, so perfectly identical sampled memory is not claimed.
Both generic strict C1 D3 and complete-context dual single-request 100+ goals remain
unmet. Evidence: `.chatgpt/production_serve_20261001/valid_query_integer_attention/`.


## Direct selection metadata in integer-coordinate attention

The integer-coordinate all-key pass already produces each query's selected mask
and each head's union while computing attention. It now publishes the head union
directly and accumulates per-query counts within its existing key partitions,
instead of materializing and rescanning the full 96-query mask array. A small
integer count reduction preserves the same per-query counts and overflow flags.
The existing mask allocation holds those partial counts and is then reused by
ordered union compaction. Every active count is overwritten on replay; only the
smaller head-union allocation needs clearing for unowned blocks.

The direct pass retains the selected union before the tail's all-excluded softmax
rule. Because every tail score is excluded from this projected pass, old-prefix
probability mass has the same recurrence as the denominator; reusing that result
removes a duplicate live accumulator. The selected full-dimensional correction
continues to handle the original selected and tail keys. Selection thresholds,
rank, source precision, key coverage, probability arithmetic and verification
rules are unchanged. No device allocation is added or resized.

The original query-mask specialization remains available for reference and
existing operator callers; floating-coordinate attention retains its old mask
pipeline. Tests compare native state/output bits, query counts, head unions and
failed flags, with independent CPU metadata checks, output guards and changed-input
graph replays. Cases include empty and shifted ownership, short/tail ranges, zero
queries, nonfinite scales/thresholds and all-selected/no-selected states.

On four alternating runs of the same genuinely populated 260,973-token request,
the direct pipeline measured 56.44–56.47 useful tok/s versus 56.14–56.17 before it,
about 0.53% faster. All runs kept the same 241-token naturally completed answer,
verification trajectory, fallback decisions and snapshot metadata. This is a
small warm-decoding gain on the existing numerical-attention route, not a generic
single-GPU gain or a new quality certificate. The underlying joint approximation
keeps its previously documented quality limitations. Four runs with varying
natural GPU clocks are not a statistical confidence interval.
Evidence: `.chatgpt/production_serve_20261001/direct_attention_metadata_2017/`.


## Query-owned integer-attention completion

The integer-coordinate joint-attention mode now completes each query in one
256-thread block instead of separately reducing coordinate partitions, normalizing
the exact selected-value subset, and reconstructing the final value. The existing
float-coordinate mode keeps its original implementation.

The new block preserves the original eight-warp reduction order, sequential
exact-subset denominator FMA, selected-value FMA order and final multiply/FMA
contraction. Independent split factors are computed in parallel without changing
their formula or the denominator's accumulation order. The final value basis is
transposed once into its consumer's layout after all source-coordinate preparation.
This is a bit-preserving permutation using the existing 64 KiB preparation scratch;
there is no added persistent device allocation or changed inference-buffer address.
The prepared-layer invariant prevents mixing the original and transposed layouts.

This does not change key coverage, coordinate rank, quantization, selection,
verification, fallback conditions or snapshot contents. It is exact relative to
the existing integer-coordinate profile, whose previously documented numerical
approximation and language-quality limitations remain. It is not a new universal
model-quality certificate.

In the final same-binary eight-run comparison at 260,973 populated tokens, original
completion averaged 56.3925 useful tok/s and query-owned completion 56.5725, a
0.3192% improvement. Each adjacent A/B pair favored the candidate. All runs
preserved the exact 241-token naturally completed answer, 96 verification rounds,
288 drafts, 145 accepted drafts, three distant records, per-layer fallback counts
and canonical snapshot metadata. Natural GPU clocks varied; this small measured
gain is not a confidence interval, a total-request latency claim, or either 100+
single-request goal. The private A/B environment switch was removed after validation.

The permanent regression compares with the original three-stage implementation on
96 scenarios per card, including source updates at graph-stable addresses, exact
basis permutations, empty and failed subsets, nonfinite values, output guards,
and an independent simple-case scalar formula. The legacy kernels remain available
as the float-mode implementation and numerical reference.
Evidence: `.chatgpt/production_serve_20261002/query_owned_attention_finish/`.


## Producer-owned integer query preparation

The integer-coordinate joint-attention mode now prepares each query in one block.
Eight warps perform the original four-part basis projection; one producer warp
handles the original full-query quantization, four ordered mean dot products, and
projected-query quantization. This removes divergent mean work from the basis
projection loop without changing the per-output arithmetic order. The original
32-lane maximum tree, pairwise partial sum, FP16 projection boundary, mean shift,
and final signed-byte rounding remain unchanged.

The same producer grid initializes all 104 metadata integers and the complete
head-union bitset before subsequent kernels can consume them. Disjoint ownership
replaces two separate memset nodes, not initialization coverage. No persistent
GPU buffer, allocation size, coordinate rank, target key coverage, acceptance rule
or inference-buffer address changes. Float-coordinate mode retains its original
preparation and initialization path. The private A/B switch was removed after
validation; this runs automatically within the already-selected integer mode.

The final eight-run comparison on 260,973 populated input tokens averaged 56.52
useful tok/s for the original path and 56.67 for the new path (+0.2654%). Adjacent
candidate-minus-control differences were +0.13, +0.23, +0.17 and +0.07 tok/s.
Natural clocks varied; this is a small measured gain, not a universal confidence
interval or a claim to have met either single-request 100+ goal. All outputs,
verification trajectories, fallback counts and snapshot metadata were unchanged.
The underlying integer-attention approximation retains its existing quality limits.

Tests compare every projected value/code/scale/shift and guard against the original
three-stage path on both GPUs, including nonfinite/zero/extreme values, changed
query/basis contents at graph-stable addresses and cross-query independence.
Independent CPU calculations cover quantization, ordered basis/mean projection
and final quantization. Initialization tests cover empty, short, tail and full
union extents with stale data and guard storage. The earlier mixed-timing basic
fusion and slower two-thread TP2 submission are archived, not retained as options.
Evidence: `.chatgpt/production_serve_20261002/query_input_lifetime_fusion/`.
