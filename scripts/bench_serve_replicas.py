#!/usr/bin/env python3
"""Physical HTTP/SSE serving measurement. Does not simulate or sum replica rates.

Starts the supplied production server, drives bounded cold/warm/staggered/churn
requests, writes raw SSE timestamps and seed-correlated server records, then
terminates only its own child. No hardware settings or model changes.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request

FAMILIES = ('triage', 'medication', 'appointments', 'laboratory', 'billing',
            'inventory', 'ophthalmology', 'migration')

def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]

def phase_arrival_delay(phase: str, index: int, prime_gap: float = 0.0) -> float:
    if index < 0 or not math.isfinite(prime_gap) or prime_gap < 0:
        raise ValueError('priming gap and request index must be nonnegative and finite')
    if phase in ('staggered','churn','churn_current'):
        return index * .5
    return index * prime_gap if phase == 'cold' else 0.0


def stream_request(url: str, messages: list[dict], seed: int, maximum: int,
                   barrier: threading.Barrier, delay: float = 0,
                   cancel_after: int = 0) -> dict:
    body = {'model': 'qwen3.8-27b', 'messages': messages, 'max_tokens': maximum,
            'temperature': 0, 'seed': seed, 'stream': True,
            'stream_options': {'include_usage': True}}
    request = urllib.request.Request(url + '/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    barrier.wait()
    if delay:
        time.sleep(delay)
    started = time.perf_counter()
    result = {'seed': seed, 'started_monotonic': started, 'started_unix': time.time(),
              'delay_seconds': delay, 'events': [], 'content': '', 'reasoning': '',
              'usage': {}, 'finish_reason': None, 'cancelled': False}
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            for raw in response:
                if not raw.startswith(b'data: '):
                    continue
                payload = raw[6:].strip()
                if payload == b'[DONE]':
                    result['done_marker'] = True
                    break
                event = json.loads(payload)
                if 'error' in event:
                    raise RuntimeError(str(event['error']))
                if event.get('usage'):
                    result['usage'] = event['usage']
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    content = delta.get('content') or ''
                    reasoning = delta.get('reasoning_content') or ''
                    if content or reasoning:
                        result['events'].append({'seconds': time.perf_counter()-started,
                                                 'content': content, 'reasoning': reasoning})
                        result['content'] += content
                        result['reasoning'] += reasoning
                    if choice.get('finish_reason'):
                        result['finish_reason'] = choice['finish_reason']
                if cancel_after and len(result['events']) >= cancel_after:
                    result['cancelled'] = True
                    break
    except Exception as error:
        result['error'] = repr(error)
    result['wall_seconds'] = time.perf_counter() - started
    visible = [event['seconds'] for event in result['events'] if event['content']]
    result['ttft_seconds'] = visible[0] if visible else None
    result['last_content_seconds'] = visible[-1] if visible else None
    gaps = [b-a for a, b in zip(visible, visible[1:])]
    result['inter_chunk_p95_seconds'] = percentile(gaps, .95)
    result['inter_chunk_max_seconds'] = max(gaps, default=None)
    tokens = result['usage'].get('completion_tokens', 0)
    result['completion_tokens'] = tokens
    # End-to-end is exact. Decode excludes first content arrival and conservatively
    # excludes a whole MTP round (up to four tokens), not merely one presumed token.
    result['end_to_end_tps'] = tokens/result['wall_seconds']
    duration = (visible[-1]-visible[0]) if len(visible)>1 else 0
    result['visible_decode_tps_conservative'] = max(0, tokens-4)/duration if duration else None
    result['content_sha256'] = hashlib.sha256(result['content'].encode()).hexdigest()
    return result

def summarize(rows: list[dict], makespan: float) -> dict:
    successful = [row for row in rows if not row.get('error') and not row['cancelled']]
    ttfts = [r['ttft_seconds'] for r in successful if r['ttft_seconds'] is not None]
    rates = [r['visible_decode_tps_conservative'] for r in successful
             if r['visible_decode_tps_conservative'] is not None]
    first = [r['started_monotonic']+r['ttft_seconds'] for r in successful
             if r['ttft_seconds'] is not None]
    last = [r['started_monotonic']+r['last_content_seconds'] for r in successful
            if r['last_content_seconds'] is not None]
    overlap = max(0, min(last)-max(first)) if len(first)==len(rows) and last else 0
    gaps = [r['inter_chunk_max_seconds'] for r in successful
            if r['inter_chunk_max_seconds'] is not None]
    tokens = sum(r['completion_tokens'] for r in successful)
    return {'classification': 'MEASURED_NOT_CERTIFIED', 'requests': len(rows),
            'completed': len(successful), 'errors': sum('error' in r for r in rows),
            'cancelled': sum(r['cancelled'] for r in rows),
            'length_limited': sum(r['finish_reason']=='length' for r in successful),
            'makespan_seconds': makespan, 'completion_tokens': tokens,
            'physical_end_to_end_tps': tokens/makespan,
            'all_request_visible_overlap_seconds': overlap,
            'ttft_p50_seconds': percentile(ttfts, .5),
            'ttft_p95_seconds': percentile(ttfts, .95),
            'ttft_max_seconds': max(ttfts, default=None),
            'minimum_visible_decode_tps_conservative': min(rates, default=None),
            'maximum_inter_chunk_gap_seconds': max(gaps, default=None),
            'jain_visible_decode_fairness': sum(rates)**2/(len(rates)*sum(x*x for x in rates))
                if rates and any(rates) else None}

def common_stream_window(rows: list[dict], minimum_seconds: float = 2.0, *,
                         expected_count: int = 8, replica_count: int = 2, not_before: float | None = None,
                         not_after: float | None = None) -> dict:
    """Count actual accepted tokens delivered during ONE shared physical interval.

    Pair raw client SSE content events with Engine-licensed cumulative counts by
    UTF-8 content byte boundary. Multiple events belonging to the same accepted
    round are placed at the LAST delivered event of that round, never the first.
    Subtract one whole four-token MTP round from each request as a conservative
    boundary/control-token allowance. No extrapolation or sum of disjoint rates.
    """
    if (replica_count not in (1, 2) or expected_count not in (1, 2, 4, 6, 8) or
            expected_count % replica_count != 0 or len(rows) != expected_count) or any(
            r.get('error') or r['cancelled'] or r.get('reasoning') for r in rows):
        return {'qualified_measurement': False,
                'reason': f'requires {expected_count} successful content-only streams'}
    timelines = []
    for row in rows:
        trace = (row.get('server_record') or {}).get('stream_trace')
        if not trace:
            return {'qualified_measurement': False, 'reason': 'missing licensed stream trace'}
        by_bytes = {point['content_bytes']: point['committed_tokens'] for point in trace}
        if len(by_bytes) != len(trace):
            return {'qualified_measurement': False, 'reason': 'duplicate server byte boundary'}
        bytes_seen = 0
        rounds = {}
        previous = 0
        for event in row['events']:
            if not event['content']:
                continue
            bytes_seen += len(event['content'].encode('utf-8'))
            count = by_bytes.get(bytes_seen)
            if count is None or count < previous or count > row['completion_tokens']:
                return {'qualified_measurement': False, 'reason': 'unmatched or invalid token/byte boundary'}
            previous = count
            rounds[count] = row['started_monotonic'] + event['seconds']
        if bytes_seen != trace[-1]['content_bytes'] or len(rounds) < 2:
            return {'qualified_measurement': False, 'reason': 'incomplete stream correlation'}
        timelines.append(sorted((timestamp, count) for count, timestamp in rounds.items()))
    start = max(points[0][0] for points in timelines)
    end = min(points[-1][0] for points in timelines)
    if not_before is not None:
        start = max(start, not_before)
    if not_after is not None:
        end = min(end, not_after)
    duration = end-start
    if duration < minimum_seconds:
        return {'qualified_measurement': False, 'reason': f'insufficient common {expected_count}-request overlap',
                'seconds': max(0, duration)}
    window_rows = []
    devices = []
    for row, points in zip(rows, timelines):
        before = max((count for stamp, count in points if stamp <= start), default=0)
        after = max((count for stamp, count in points if stamp <= end), default=0)
        licensed = max(0, after-before-4)
        device = row['server_record']['request']['device']
        devices.append(device)
        window_rows.append({'seed':row['seed'], 'device':device,
            'tokens_before':before, 'tokens_after':after,
            'conservative_tokens':licensed, 'tokens_per_second':licensed/duration})
    rates = [row['tokens_per_second'] for row in window_rows]
    physical_cohort = len(set(devices)) == replica_count and all(
        devices.count(d) == expected_count // replica_count for d in set(devices))
    physical_c4_c4 = expected_count == 8 and replica_count == 2 and physical_cohort
    aggregate = sum(row['conservative_tokens'] for row in window_rows)/duration
    return {'qualified_measurement':physical_cohort, 'streams':expected_count,
            'streams_per_gpu':expected_count//replica_count, 'replicas':replica_count,
            'common_start_monotonic':start,
            'common_end_monotonic':end, 'seconds':duration,
            'conservative_boundary_allowance_per_request':4,
            'physical_c4_c4':physical_c4_c4, 'aggregate_tokens_per_second':aggregate,
            'minimum_tokens_per_second':min(rates), 'rows':window_rows,
            'jain_fairness':sum(rates)**2/(len(rates)*sum(r*r for r in rates)) if any(rates) else None,
            'numeric_decode_gate_met':physical_c4_c4 and aggregate >= 160 and min(rates) >= 20,
            'semantic_quality_certified':False}

def serial_stream_windows(rows: list[dict]) -> dict:
    """A corpus of one-client measurements, never an aggregate concurrent claim."""
    successful = [r for r in rows if not r.get('error') and not r.get('cancelled')]
    ordered = sorted(successful, key=lambda r: r['started_monotonic'])
    no_overlap = all(a['started_monotonic'] + a['wall_seconds'] <= b['started_monotonic']
                     for a, b in zip(ordered, ordered[1:]))
    windows = [common_stream_window([r], expected_count=1, replica_count=1) for r in rows]
    devices = {(r.get('server_record') or {}).get('request',{}).get('device') for r in rows}
    one_device = len(devices)==1 and None not in devices
    return {'actual_clients':1, 'serial_requests':len(rows), 'no_request_overlap':no_overlap,
            'one_physical_device':one_device,
            'qualified_measurement':bool(rows) and len(successful)==len(rows) and no_overlap and one_device and
                all(w['qualified_measurement'] for w in windows),
            'numeric_decode_gate_met':False, 'role':'independent C1 corpus; do not sum rates',
            'windows':windows}


def followup_workload(messages: list[list[dict]], previous: list[dict], text: str) -> list[list[dict]]:
    """Append real delivered assistant replies, never replay the unchanged old prompt."""
    if not messages or len(messages) != len(previous):
        raise ValueError('followup requires a matching previous conversation cohort')
    workload = []
    for history, row in zip(messages, previous):
        if row.get('error') or row.get('cancelled') or not row.get('content'):
            raise ValueError('followup requires successful prior content')
        prompt = copy.deepcopy(history)
        assistant = {'role':'assistant', 'content':row['content']}
        if row.get('reasoning'):
            assistant['reasoning_content'] = row['reasoning']
        prompt += [assistant, {'role':'user', 'content':text}]
        workload.append(prompt)
    return workload


def summarize_followup(rows: list[dict], previous: list[dict]) -> dict:
    details = []
    for row, prior in zip(rows, previous):
        record, old = row.get('server_record') or {}, prior.get('server_record') or {}
        result, old_result = record.get('result', {}), old.get('result', {})
        details.append({'seed':row['seed'], 'previous_device':old.get('request', {}).get('device'),
            'device':record.get('request', {}).get('device'),
            'previous_prompt_tokens':old_result.get('prompt_tokens'),
            'prompt_tokens':result.get('prompt_tokens'),
            'reused_tokens':result.get('prefix_cache_hit_tokens'),
            'computed_prefill_tokens':result.get('computed_prefill_tokens')})
    return {'role':'appended real conversation turn, not same-prompt replay',
        'owner_preserved':sum(d['device'] is not None and d['device']==d['previous_device'] for d in details),
        'prefix_hits':sum((d['reused_tokens'] or 0)>0 for d in details), 'requests':details}


def cold_join_workload(messages: list[list[dict]], previous: list[dict], nonce: str,
                       cold_messages: list[list[dict]] | None = None, warm_per_gpu: int = 3) -> tuple[list, list]:
    """Select three known cached owners per GPU, then one fresh prefix per GPU.

    Ownership here is an expectation checked against actual server receipts; the
    client never simulates a replica or migrates an in-flight request.
    """
    if warm_per_gpu not in (2,3):
        raise ValueError('cold_join supports two or three established streams per GPU')
    cold_count=2*(4-warm_per_gpu)
    if cold_messages is not None and len(cold_messages) != cold_count:
        raise ValueError(f'cold_join requires exactly {cold_count} replacement cold inputs')
    if len(messages) != 8 or len(previous) != 8:
        raise ValueError('cold_join requires a preceding complete eight-request cohort')
    owners: dict[int, list[int]] = {}
    for i, row in enumerate(previous):
        if row.get('error') or row.get('cancelled') or not row.get('server_record'):
            raise ValueError('cold_join requires successful prior owner receipts')
        device = row['server_record']['request']['device']
        owners.setdefault(device, []).append(i)
    if len(owners) != 2 or any(len(indices) != 4 for indices in owners.values()):
        raise ValueError('cold_join requires actual prior physical C4+C4 placement')
    order = [(i, device, False) for device, indices in sorted(owners.items()) for i in indices[:warm_per_gpu]]
    order += [(i, device, True) for device, indices in sorted(owners.items()) for i in indices[warm_per_gpu:]]
    workload, plan = [], []
    cold_index = 0
    for source, device, cold in order:
        if cold and cold_messages is not None:
            prompt = copy.deepcopy(cold_messages[cold_index])
            cold_index += 1
        else:
            prompt = copy.deepcopy(messages[source])
        if cold:
            user = next((m for m in prompt if m.get('role') == 'user'), None)
            if user is None:
                raise ValueError('cold_join requires user messages')
            marker = f'[Independent request {nonce}-{source}]\n'
            if isinstance(user.get('content'), str):
                user['content'] = marker + user['content']
            elif isinstance(user.get('content'), list):
                user['content'].insert(0, {'type':'text', 'text':marker})
            else:
                raise ValueError('cold_join requires text or structured user content')
        workload.append(prompt)
        plan.append({'source_index':source, 'expected_device':device, 'cold':cold,
                     'delay_seconds':2.0 if cold else 0.0})
    return workload, plan


def summarize_cold_join(rows: list[dict], plan: list[dict]) -> dict:
    cold_count=sum(p['cold'] for p in plan)
    if len(rows) != 8 or len(plan) != 8 or cold_count not in (2,4):
        return {'qualified_scenario':False, 'reason':'requires balanced established/cold cohorts on two GPUs'}
    warm_count=8-cold_count
    warm_per_gpu=warm_count//2
    if any(r.get('error') or r.get('cancelled') for r in rows):
        return {'qualified_scenario':False, 'reason':'requires successful complete streams'}
    details = []
    placement = True
    active = True
    for row, spec in zip(rows, plan):
        record = row.get('server_record') or {}
        device = record.get('request', {}).get('device')
        if not spec['cold']:
            placement = placement and device == spec['expected_device']
        result = record.get('result', {})
        details.append({'seed':row['seed'], 'device':device, 'cold':spec['cold'],
            'ttft_seconds':row['ttft_seconds'], 'prompt_tokens':result.get('prompt_tokens'),
            'reused_tokens':result.get('prefix_cache_hit_tokens'),
            'computed_prefill_tokens':result.get('computed_prefill_tokens')})
        if spec['cold']:
            join = row['started_monotonic']
            live = [r for r, p in zip(rows, plan) if not p['cold'] and
                    r.get('server_record', {}).get('request', {}).get('device') == device and
                    r['ttft_seconds'] is not None and r['last_content_seconds'] is not None and
                    r['started_monotonic'] + r['ttft_seconds'] <= join <
                    r['started_monotonic'] + r['last_content_seconds']]
            active = active and len(live) == warm_per_gpu
    expected_devices = {p['expected_device'] for p in plan if not p['cold']}
    cold_devices = [d['device'] for d in details if d['cold']]
    placement = placement and set(cold_devices) == expected_devices and len(set(cold_devices)) == 2 and all(cold_devices.count(d)==cold_count//2 for d in expected_devices)
    warm = [r for r, p in zip(rows, plan) if not p['cold']]
    misses = all((d['computed_prefill_tokens'] or 0) >= .9 * (d['prompt_tokens'] or 1)
                 for d in details if d['cold'])
    hits = all((d['reused_tokens'] or 0) > 0 for d in details if not d['cold'])
    cold = [r for r, p in zip(rows, plan) if p['cold']]
    cold_pending_window = {'qualified_measurement':False, 'reason':'missing cold first-token times'}
    if all(r['ttft_seconds'] is not None for r in cold):
        cold_pending_window = common_stream_window(warm, expected_count=warm_count,
            not_before=max(r['started_monotonic'] for r in cold),
            not_after=min(r['started_monotonic']+r['ttft_seconds'] for r in cold))
        cold_pending_window['role'] = f'{warm_count} established streams while all {cold_count} fresh requests await first content'
    return {'qualified_scenario':placement and active and misses and hits,
            'warm_window_while_cold_pending':cold_pending_window,
            'actual_owner_placement_matches':placement,
            'three_warm_streams_active_at_each_join':active if warm_per_gpu==3 else None,
            'established_streams_per_gpu':warm_per_gpu,
            'established_streams_active_at_each_join':active,
            'warm_prefix_hits':hits, 'fresh_prefix_misses':misses,
            'minimum_warm_visible_decode_tps':min((r['visible_decode_tps_conservative']
                for r in warm if r['visible_decode_tps_conservative'] is not None), default=None),
            'maximum_warm_inter_chunk_gap_seconds':max((r['inter_chunk_max_seconds']
                for r in warm if r['inter_chunk_max_seconds'] is not None), default=None),
            'requests':details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18080)
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--kv-capacity', default='auto', help='Native aggregate KV capacity per replica: token count or auto')
    parser.add_argument('--vision', action='store_true', help='Reserve the native multimodal runtime even for a text-only cohort')
    parser.add_argument('--prefill-chunk', type=int, default=128)
    parser.add_argument('--prefill-live-chunk', type=int, default=0)
    parser.add_argument('--max-prefill-lanes', type=int, default=1)
    parser.add_argument('--pack-prefill', action='store_true')
    parser.add_argument('--prefill-decode-rounds', type=int, choices=range(1, 65), default=1)
    parser.add_argument('--prefill-target-decode-tps', type=float, default=0.0)
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--draft-tokens', type=int, choices=(1, 3), default=3)
    parser.add_argument('--spec', choices=('none','mtp'), default='mtp')
    parser.add_argument('--sequential', action='store_true', help='One actual active client, sequential corpus; never a concurrent gate')
    parser.add_argument('--prime-tokens', type=int, help='optional cold-phase-only output bound')
    parser.add_argument('--prime-gap', type=float, default=0.0,
                        help='Optional cold-priming arrival spacing only; warm cohorts remain simultaneous')
    parser.add_argument('--prompt-size', choices=('smoke', '2k', '8k'), default='2k')
    parser.add_argument('--phases', default='cold,warm,staggered,churn')
    parser.add_argument('--followup-text-file', type=Path, help='Append this document/question to each continuing conversation')
    parser.add_argument('--followup-text', default='Using the facts already provided, identify the single most important next verification step and explain why. Keep your answer under 100 words.')
    parser.add_argument('--no-prefix-reuse', action='store_true')
    parser.add_argument('--preserve-thinking', action='store_true', help='Use the existing native history-rendering option; records the changed prompt semantics')
    parser.add_argument('--optimized-proposal', action='store_true')
    parser.add_argument('--full-vocab-draft-retained-target', action='store_true')
    parser.add_argument('--mtp-logit-margin', type=float)
    parser.add_argument('--mtp-retained-packed-prefix', action='store_true')
    parser.add_argument('--sm75-radix-projections', action='store_true')
    parser.add_argument('--sm75-radix-resident', action='store_true')
    parser.add_argument('--sm75-nu3-mlp', action='store_true')
    parser.add_argument('--target-draft-union', action='store_true')
    parser.add_argument('--session-copy-draft', action='store_true')
    parser.add_argument('--full-target-head', action='store_true', help='Keep full target verification with an optimized draft head')
    parser.add_argument('--fast-full-vocab-draft', action='store_true', help='C1 full-vocabulary draft-only projection; requires --full-target-head')
    parser.add_argument('--trace-stream', action='store_true')
    parser.add_argument('--messages-json', type=Path,
                        help='JSON array of per-request message arrays; enables vision')
    parser.add_argument('--cold-join-warm-per-gpu', type=int, choices=(2,3), default=3)
    parser.add_argument('--cold-join-messages-json', type=Path,
                        help='Two or four fresh inputs, matching cold-join-warm-per-gpu, optionally including media')
    args = parser.parse_args()
    if not math.isfinite(args.prime_gap) or args.prime_gap < 0:
        parser.error('--prime-gap must be finite and nonnegative')
    if args.followup_text_file:
        args.followup_text = args.followup_text_file.read_text(encoding='utf-8')
    if args.sequential and (len(args.devices.split(',')) != 1 or args.prime_gap or any(
            phase not in ('cold','warm','followup','followup2','followup3','repeat_current')
            for phase in args.phases.split(','))):
        parser.error('--sequential requires one device, no arrival delay, and no concurrent/churn phases')
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    url = f'http://127.0.0.1:{args.port}'
    binary = args.binary.resolve(); artifact = args.artifact.resolve()
    command = [str(binary), str(artifact), '--host', '127.0.0.1', '--port', str(args.port),
        '--model-id', 'qwen3.8-27b', '--devices', args.devices,
        '--max-context', str(args.context), '--kv-capacity', args.kv_capacity,
        '--max-concurrency', str(args.concurrency), '--max-pending-requests', '8',
        '--pending-timeout-ms', '600000', '--prefill-chunk', str(args.prefill_chunk),
        '--prefill-live-chunk', str(args.prefill_live_chunk),
        '--max-prefill-lanes', str(args.max_prefill_lanes),
               '--prefill-decode-rounds', str(args.prefill_decode_rounds),
               '--prefill-target-decode-tps', str(args.prefill_target_decode_tps),
        '--kv-dtype', 'int8',
        '--default-max-tokens', str(args.tokens), '--no-thinking', '--greedy',
        '--request-log-jsonl', str(out/'server.jsonl'), '--log-stats-interval-ms', '1000']
    if args.spec == 'mtp':
        command += ['--spec', 'mtp', '--draft-tokens', str(args.draft_tokens)]
    if args.pack_prefill:
        command.append('--pack-prefill')
    if args.vision:
        command.append('--vision')
    if args.preserve_thinking:
        command.append('--preserve-thinking')
    if args.no_prefix_reuse:
        command.append('--no-prefix-reuse')
    if args.optimized_proposal:
        command.append('--lm-head-draft')
    if args.mtp_retained_packed_prefix:
        command.append('--mtp-retained-packed-prefix')
    if args.sm75_radix_projections:
        command.append('--sm75-radix-projections')
    if args.sm75_radix_resident:
        command.append('--sm75-radix-resident')
    if args.sm75_nu3_mlp:
        command.append('--sm75-nu3-mlp')
    if args.target_draft_union:
        command.append('--target-draft-union')
    if args.session_copy_draft:
        command.append('--session-copy-draft')
    if args.mtp_logit_margin is not None:
        command += ['--mtp-logit-margin', str(args.mtp_logit_margin)]
    if args.full_target_head:
        command.append('--full-target-head')
    if args.fast_full_vocab_draft:
        command.append('--fast-full-vocab-draft')
    if args.full_vocab_draft_retained_target:
        command.append('--full-vocab-draft-retained-target')
    if args.trace_stream:
        command.append('--trace-stream')
    if args.messages_json:
        messages = json.loads(args.messages_json.read_text())
        requires_vision = any(
            isinstance(message.get('content'), list) and any(
                part.get('type') in ('image_url', 'video_url')
                for part in message['content'])
            for request_messages in messages for message in request_messages)
        if requires_vision and '--vision' not in command:
            command.append('--vision')
    elif args.prompt_size == 'smoke':
        messages = [[{'role':'user', 'content':f'Reply with exactly the word READY{i} and nothing else.'}]
                    for i in range(8)]
    else:
        prompt_dir = root/'.chatgpt/post_reboot_20260821/p0_q73_multisession_baseline/prompts'
        messages = [[{'role':'user', 'content':(prompt_dir/f'{args.prompt_size}_s{i}_{family}.txt').read_text()}]
                    for i, family in enumerate(FAMILIES)]
    cold_join_messages = None
    if args.cold_join_messages_json:
        cold_join_messages = json.loads(args.cold_join_messages_json.read_text())
        if len(cold_join_messages) != 2*(4-args.cold_join_warm_per_gpu) or 'cold_join' not in args.phases.split(','):
            raise ValueError('replacement cold input count must match the cold_join owner split')
        if '--vision' not in command and any(
                isinstance(message.get('content'), list) and any(
                    part.get('type') in ('image_url', 'video_url') for part in message['content'])
                for request_messages in cold_join_messages for message in request_messages):
            command.append('--vision')
    env = os.environ.copy()
    env.update({'NINFER_MTP_Q4G64':'1', 'NINFER_MTP_MARLIN_Q4':'1',
                'NINFER_DUAL_SM75_A8':'both-mlp-stage1-all', 'NINFER_MTP_TRAJECTORY_DIGEST':'1'})
    # Both physical devices must remain visible to the production process.
    env.pop('CUDA_VISIBLE_DEVICES', None)
    (out/'command.json').write_text(json.dumps({'command':command,
        'env':{k:v for k,v in env.items() if k.startswith('NINFER_')},
        'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
        'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256':hashlib.sha256(json.dumps(messages,ensure_ascii=False).encode()).hexdigest(),
        'sequential':args.sequential,
        'cold_join_input_sha256':hashlib.sha256(args.cold_join_messages_json.read_bytes()).hexdigest()
            if args.cold_join_messages_json else None}, indent=2)+'\n')
    stop = threading.Event()
    def telemetry() -> None:
        with (out/'gpu.csv').open('w') as log:
            while not stop.is_set():
                try:
                    value = subprocess.run(['nvidia-smi',
                        '--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.current.sm,temperature.gpu',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                    for line in value.stdout.splitlines():
                        log.write(f'{time.time():.6f}, {line}\n')
                    log.flush()
                except Exception as error:
                    log.write(f'telemetry_error,{error!r}\n')
                stop.wait(1)
    thread = threading.Thread(target=telemetry)
    phases = []
    with (out/'server.stdout').open('w') as stdout, (out/'server.stderr').open('w') as stderr:
        process = subprocess.Popen(command,cwd=root,env=env,stdout=stdout,stderr=stderr)
        thread.start()
        try:
            deadline = time.monotonic()+240
            while time.monotonic()<deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'server exited during startup: {process.returncode}')
                try:
                    with urllib.request.urlopen(url+'/health',timeout=2) as response:
                        if json.load(response)=={'status':'ok'}:
                            break
                except (OSError, urllib.error.URLError):
                    time.sleep(.2)
            else:
                raise TimeoutError('production server health deadline')
            print('Physical production server ready',flush=True)
            previous_rows = []
            previous_messages = messages
            for phase_index, phase in enumerate(args.phases.split(',')):
                is_followup = phase in ('followup', 'followup2', 'followup3')
                if not is_followup and phase not in ('cold','warm','staggered','churn','refill','cold_join','repeat_current','churn_current','refill_current'):
                    raise ValueError(f'unknown phase {phase}')
                phase_messages = messages
                if phase in ('repeat_current','churn_current','refill_current'):
                    if not previous_rows:
                        raise ValueError('repeat_current requires a prior completed cohort')
                    phase_messages = previous_messages
                join_plan = None
                if is_followup:
                    phase_messages = followup_workload(previous_messages, previous_rows, args.followup_text)
                    (out/f'{phase}_messages.json').write_text(json.dumps(phase_messages, ensure_ascii=False)+'\n')
                if phase == 'cold_join':
                    phase_messages, join_plan = cold_join_workload(
                        previous_messages, previous_rows, f'phase-{phase_index}', cold_join_messages, args.cold_join_warm_per_gpu)
                    (out/'cold_join_plan.json').write_text(json.dumps(join_plan, indent=2)+'\n')
                if args.sequential:
                    started = time.perf_counter()
                    rows = []
                    for index, prompt in enumerate(phase_messages):
                        seed = 960000 + phase_index*100 + index
                        maximum = args.prime_tokens if phase=='cold' and args.prime_tokens else args.tokens
                        rows.append(stream_request(url,prompt,seed,maximum,threading.Barrier(1)))
                    makespan = time.perf_counter()-started
                else:
                    barrier = threading.Barrier(len(phase_messages)+1)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(messages)) as pool:
                        futures = []
                        for index, prompt in enumerate(phase_messages):
                            seed = 960000 + phase_index*100 + index
                            delay = (join_plan[index]['delay_seconds'] if join_plan else
                                     phase_arrival_delay(phase, index, args.prime_gap))
                            cancel = 3 if phase in ('churn','churn_current') and index%3==0 else 0
                            maximum = args.prime_tokens if phase=='cold' and args.prime_tokens else args.tokens
                            futures.append(pool.submit(stream_request,url,prompt,seed,maximum,
                                                       barrier,delay,cancel))
                        started = time.perf_counter(); barrier.wait()
                        rows = [future.result() for future in futures]
                        makespan = time.perf_counter()-started
                time.sleep(.1)
                records = [json.loads(line) for line in (out/'server.jsonl').read_text().splitlines()
                           if line.strip()]
                by_seed = {r['request']['sampling']['seed']:r for r in records
                           if r.get('event')=='request_done'}
                for row in rows:
                    row['server_record'] = by_seed.get(row['seed'])
                    if not row['cancelled'] and not row.get('error') and not row['server_record']:
                        row['error'] = 'missing seed-correlated server completion evidence'
                summary = summarize(rows,makespan);summary['phase']=phase
                if phase == 'cold' and args.prime_gap:
                    summary['cold_priming_gap_seconds'] = args.prime_gap
                summary['workload_kind'] = ('appended_conversation' if is_followup else
                    'repeat_previous_input' if phase == 'repeat_current' else
                    'cancel_previous_input' if phase == 'churn_current' else
                    'refill_previous_input' if phase == 'refill_current' else phase)
                if phase == 'cold' and args.prime_gap:
                    summary['workload_kind'] = 'staggered_cache_priming'
                summary['common_c8_stream_window'] = common_stream_window(rows)
                replica_count = len(args.devices.split(','))
                if len(rows) != 8 or replica_count != 2:
                    summary['common_client_stream_window'] = common_stream_window(
                        rows, expected_count=len(rows), replica_count=replica_count)
                if args.sequential:
                    summary['workload_kind'] = 'serial_c1_' + summary['workload_kind']
                    summary['serial_c1'] = serial_stream_windows(rows)
                if join_plan is not None:
                    summary['cold_join'] = summarize_cold_join(rows, join_plan)
                if is_followup:
                    summary['conversation_followup'] = summarize_followup(rows, previous_rows)
                previous_rows = rows
                previous_messages = phase_messages
                (out/f'{phase}.json').write_text(json.dumps({'summary':summary,'requests':rows},
                                                           indent=2,ensure_ascii=False)+'\n')
                phases.append(summary)
                print(json.dumps(summary),flush=True)
                if summary['errors']:
                    raise RuntimeError('request failure; see per-request evidence')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait(timeout=20)
            stop.set();thread.join(timeout=10)
            (out/'result.json').write_text(json.dumps({'classification':'MEASURED_NOT_CERTIFIED',
                'server_exit_code':process.returncode,'phases':phases},indent=2)+'\n')

if __name__=='__main__':
    main()
