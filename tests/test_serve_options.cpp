#include "serve/serve_options.h"
#include "serve/translate.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main() {
    int failures = 0;

    const ServeOptions defaults = parse({"ninfer-serve", "model.ninfer"});
    const auto radix = parse({"ninfer-serve","model.ninfer","--max-concurrency","1",
                              "--spec","mtp","--draft-tokens","3","--sm75-radix-projections"});
    failures += check(radix.speculative.sm75_radix_projections && !defaults.speculative.sm75_radix_projections,
                       "radix projection policy must remain explicit/default-off");

    for(const char* slots:{"2","3","4"}) {
        const auto wide=parse({"ninfer-serve","model.ninfer","--max-concurrency",slots,
                              "--spec","mtp","--draft-tokens","3","--sm75-radix-projections"});
        failures+=check(wide.speculative.sm75_radix_projections && wide.speculative.greedy_margin_delta==-1.0F,
                        "wide radix changed strict policy");
    }
    for(const auto& invalid:std::vector<std::vector<std::string>>{
        {"--max-concurrency","5"},{"--draft-tokens","1"},{"--mtp-logit-margin","7"}}) {
        std::vector<std::string> args{"ninfer-serve","model.ninfer","--spec","mtp","--draft-tokens","3","--sm75-radix-projections"};
        args.insert(args.end(),invalid.begin(),invalid.end());bool rejected=false;
        try{(void)parse(args);}catch(const std::invalid_argument&){rejected=true;}
        failures+=check(rejected,"unsupported wide radix policy accepted");
    }
    for (const char* depth : {"1", "3"}) {
        const auto resident = parse({"ninfer-serve", "model.ninfer", "--max-concurrency", "4",
            "--spec", "mtp", "--draft-tokens", depth, "--sm75-radix-resident"});
        failures += check(resident.speculative.sm75_resident_radix &&
                          resident.speculative.sm75_radix_projections &&
                          resident.speculative.greedy_margin_delta == -1.0F,
                          "resident radix did not preserve strict explicit policy");
    }
    for (const char* slots : {"1", "2", "3", "5"}) {
        bool rejected=false;
        try {(void)parse({"ninfer-serve","model.ninfer","--max-concurrency",slots,
                         "--spec","mtp","--draft-tokens","1","--sm75-radix-resident"});}
        catch(const std::invalid_argument&){rejected=true;}
        failures += check(rejected,"unsupported resident D1 topology accepted");
    }
    failures+=check(defaults.max_prefill_lanes==1,"existing single prefill default changed");
    auto concurrent_prefill=parse({"ninfer-serve","model.ninfer","--max-concurrency","4","--max-prefill-lanes","4"});
    failures+=check(concurrent_prefill.max_prefill_lanes==4,"concurrent prefill not parsed");
    for(const char* bad:{"0","5","-1"}){
        bool rejected=false;
        try{(void)parse({"ninfer-serve","model.ninfer","--max-concurrency","4","--max-prefill-lanes",bad});}
        catch(const std::invalid_argument&){rejected=true;}
        failures+=check(rejected,"invalid prefill owner capacity accepted");
    }

    failures += check(defaults.speculative.greedy_margin_delta == -1.0F,
                      "margin acceptance activated without an explicit request");
    const auto margin = parse({"ninfer-serve","model.ninfer","--spec","mtp","--draft-tokens","3",
                               "--mtp-logit-margin","7","--lm-head-draft"});
    failures += check(margin.speculative.greedy_margin_delta == 7.0F &&
                      margin.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                      "historical C1 profile lost its explicit margin/head settings");
    const auto retained = parse({"ninfer-serve","model.ninfer","--spec","mtp","--draft-tokens","3",
                                  "--mtp-logit-margin","7","--mtp-retained-packed-prefix","--lm-head-draft"});
    failures += check(retained.speculative.retained_mtp_packed_prefix &&
                      !defaults.speculative.retained_mtp_packed_prefix,
                      "retained packed-prefix policy not explicit/default-off");
    for(const char* bad : {"nan","inf","-1","-0.1","129","7junk"}) {
        bool rejected=false;
        try {(void)parse({"ninfer-serve","model.ninfer","--spec","mtp","--draft-tokens","3","--mtp-logit-margin",bad});}
        catch(const std::invalid_argument&){rejected=true;}
        failures += check(rejected,"invalid explicit margin accepted");
    }
    for(const auto extra : std::vector<std::vector<std::string>>{
            {"--mtp-logit-margin","7"},
            {"--spec","mtp","--draft-tokens","1","--mtp-logit-margin","7"},
            {"--spec","dflash","--draft-tokens","3","--mtp-logit-margin","7"},
            {"--spec","mtp","--draft-tokens","3","--max-concurrency","4","--mtp-logit-margin","7"}}) {
        bool rejected=false;auto args=std::vector<std::string>{"ninfer-serve","model.ninfer"};
        args.insert(args.end(),extra.begin(),extra.end());
        try{(void)parse(args);}catch(const std::invalid_argument&){rejected=true;}
        failures+=check(rejected,"margin policy escaped its C1/D3 startup scope");
    }

    failures += check(defaults.prefill_live_chunk == 0, "live prefill default unexpectedly enabled");
    const auto adaptive = parse({"ninfer-serve", "model.ninfer", "--prefill-chunk", "1024",
                                "--prefill-live-chunk", "128"});
    failures += check(adaptive.prefill_chunk == 1024 && adaptive.prefill_live_chunk == 128,
                      "independent prefill caps were not parsed");
    for (const char* bad : {"1", "129", "2048"}) {
        bool rejected = false;
        try { (void)parse({"ninfer-serve", "model.ninfer", "--prefill-chunk", "1024",
                          "--prefill-live-chunk", bad}); }
        catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "invalid live prefill bound was accepted");
    }

    failures += check(!defaults.speculative.force_full_target_head,
                      "legacy head profile default changed");
    for (const bool force_first : {false, true}) {
        std::vector<std::string> argv{"ninfer-serve", "model.ninfer", "--spec", "mtp",
                                      "--draft-tokens", "3"};
        argv.push_back(force_first ? "--full-target-head" : "--lm-head-draft");
        argv.push_back(force_first ? "--lm-head-draft" : "--full-target-head");
        const auto explicit_heads = parse(std::move(argv));
        failures += check(explicit_heads.speculative.force_full_target_head &&
                              explicit_heads.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                          "independent draft/target policies depend on argument order");
    }
    const auto full_vocab = parse({"ninfer-serve", "model.ninfer", "--spec", "mtp",
        "--draft-tokens", "3", "--full-target-head", "--fast-full-vocab-draft"});
    failures += check(full_vocab.speculative.fast_full_vocab_draft &&
        full_vocab.speculative.force_full_target_head &&
        full_vocab.speculative.proposal_head == ninfer::ProposalHead::Full,
        "full-vocabulary drafting did not remain independent of the target shortcut");
    for (const auto& extra : std::vector<std::vector<std::string>>{
            {}, {"--full-target-head", "--lm-head-draft"},
            {"--full-target-head", "--max-concurrency", "4"}}) {
        std::vector<std::string> argv{"ninfer-serve", "model.ninfer", "--spec", "mtp",
                                     "--draft-tokens", "3", "--fast-full-vocab-draft"};
        argv.insert(argv.end(),extra.begin(),extra.end());
        bool rejected = false;
        try { (void)parse(std::move(argv)); } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected,"unsafe full-vocabulary draft profile was accepted");
    }
    const auto composed = parse({"ninfer-serve", "model.ninfer", "--spec", "mtp",
        "--draft-tokens", "3", "--lm-head-draft", "--full-vocab-draft-retained-target"});
    failures += check(composed.speculative.retained_target_full_vocab_draft &&
        composed.speculative.fast_full_vocab_draft && !composed.speculative.force_full_target_head &&
        composed.speculative.greedy_margin_delta == -1.0F, "composition changed retained strict target policy");
    for (const auto& extra : std::vector<std::vector<std::string>>{
            {"--full-target-head"}, {"--max-concurrency", "4"}, {"--mtp-logit-margin", "7"},
            {"--draft-tokens", "1"}}) {
        std::vector<std::string> argv{"ninfer-serve", "model.ninfer", "--spec", "mtp",
            "--draft-tokens", "3", "--lm-head-draft", "--full-vocab-draft-retained-target"};
        argv.insert(argv.end(),extra.begin(),extra.end());bool rejected=false;
        try {(void)parse(argv);}catch(const std::invalid_argument&){rejected=true;}
        failures += check(rejected,"composition admitted an incompatible target/acceptance/topology");
    }
    failures += check(defaults.prefill_target_decode_tps==0.0,"timed prefill unexpectedly enabled by default");
    const auto timed=parse({"ninfer-serve","model.ninfer","--prefill-decode-rounds","24",
                            "--prefill-target-decode-tps","24"});
    failures += check(timed.prefill_target_decode_tps==24.0 && timed.prefill_decode_rounds==24,
                      "timed target and maximum burst were not independent");
    for(const char* invalid : {"-1","1001","NaN","inf"}) {
        bool rejected=false;
        try{(void)parse({"ninfer-serve","model.ninfer","--prefill-target-decode-tps",invalid});}
        catch(const std::invalid_argument&){rejected=true;}
        failures += check(rejected,"invalid timing target was accepted");
    }
    failures += check(!defaults.pack_prefill,"packed prefill changed the default");
    const auto packed=parse({"ninfer-serve","model.ninfer","--pack-prefill","--spec","mtp",
        "--draft-tokens","1","--max-concurrency","4","--max-prefill-lanes","4","--prefill-chunk","1024"});
    failures += check(packed.pack_prefill,"explicit packed prefill was not parsed");
    for(const auto& extra:std::vector<std::vector<std::string>>{{},{"--max-prefill-lanes","1"},{"--prefill-chunk","128"}}){
        std::vector<std::string> args{"ninfer-serve","model.ninfer","--pack-prefill"};
        if(!extra.empty()){args.insert(args.end(),{"--spec","mtp","--draft-tokens","1","--max-concurrency","4","--max-prefill-lanes","4"});args.insert(args.end(),extra.begin(),extra.end());}
        bool rejected=false;try{(void)parse(args);}catch(const std::invalid_argument&){rejected=true;}
        failures += check(rejected,"unsupported packed prefill bounds accepted");
    }
    failures += check(!defaults.speculative.session_copy_draft,"session-copy changed a default");
    const auto copy_options=parse({"ninfer-serve","model.ninfer","--spec","mtp","--draft-tokens","3",
        "--full-target-head","--session-copy-draft"});
    failures += check(copy_options.speculative.session_copy_draft &&
        copy_options.speculative.force_full_target_head && copy_options.speculative.greedy_margin_delta==-1.f,
        "session-copy changed target verification or acceptance");
    for(const auto& tail:std::vector<std::vector<std::string>>{
        {"--max-concurrency","4"},{"--draft-tokens","1"},{"--mtp-logit-margin","7"}}) {
        std::vector<std::string> args{"ninfer-serve","model.ninfer","--spec","mtp","--draft-tokens","3",
                                     "--full-target-head","--session-copy-draft"};
        args.insert(args.end(),tail.begin(),tail.end());bool rejected=false;
        try{(void)parse(args);}catch(const std::invalid_argument&){rejected=true;}
        failures += check(rejected,"unqualified session-copy configuration accepted");
    }
    failures += check(!defaults.speculative.sm75_nu3_mlp,"NU3 unexpectedly changed default precision");
    const auto nu3=parse({"ninfer-serve","model.ninfer","--spec","mtp","--draft-tokens","3","--sm75-nu3-mlp"});
    failures += check(nu3.speculative.sm75_nu3_mlp && nu3.speculative.greedy_margin_delta==-1.0F,
                      "NU3 did not retain strict acceptance");
    for(const auto& invalid:std::vector<std::vector<std::string>>{
            {"--draft-tokens","1"},{"--mtp-logit-margin","7"},{"--max-concurrency","4"},{"--spec","none"}}){
        std::vector<std::string> args{"ninfer-serve","model.ninfer","--spec","mtp","--draft-tokens","3","--sm75-nu3-mlp"};
        args.insert(args.end(),invalid.begin(),invalid.end());bool rejected=false;
        try{(void)parse(args);}catch(const std::invalid_argument&){rejected=true;}
        failures += check(rejected,"NU3 accepted incompatible MTP, acceptance or ownership shape");
    }
    failures += check(!defaults.speculative.target_draft_union,"target union unexpectedly enabled");
    const auto union_option=parse({"ninfer-serve","model.ninfer","--spec","mtp","--draft-tokens","3","--lm-head-draft","--target-draft-union"});
    failures += check(union_option.speculative.target_draft_union && union_option.speculative.greedy_margin_delta==-1.0F,"explicit target union changed acceptance");
    for(const auto& tail:std::vector<std::vector<std::string>>{{"--full-target-head"},{"--mtp-logit-margin","7"},{"--draft-tokens","1"},{"--max-concurrency","4"}}){
        std::vector<std::string> args{"ninfer-serve","model.ninfer","--spec","mtp","--draft-tokens","3","--lm-head-draft","--target-draft-union"};
        args.insert(args.end(),tail.begin(),tail.end());bool rejected=false;
        try{(void)parse(args);}catch(const std::invalid_argument&){rejected=true;}
        failures += check(rejected,"incompatible target union settings accepted");
    }
    failures += check(defaults.allow_prefix_reuse, "prefix reuse is not enabled by default");
    failures +=
        check(!defaults.preserve_thinking, "thinking history is unexpectedly preserved by default");
    failures += check(!defaults.enable_vision, "Vision is not disabled by default");
    failures += check(defaults.request_log_jsonl.empty(),
                      "request JSONL logging is not disabled by default");
    failures += check(defaults.log_stats_interval_ms == 5000,
                      "periodic throughput interval default mismatch");
    failures += check(defaults.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          defaults.kv_capacity.explicit_tokens == defaults.max_context,
                      "default KV capacity does not follow max context");
    failures += check(defaults.speculative.backend == ninfer::SpeculativeBackend::None,
                      "speculative decoding is not disabled by default");
    failures += check(defaults.response_store_max_records == kDefaultResponseStoreRecords &&
                          defaults.response_store_max_bytes == kDefaultResponseStoreBytes,
                      "Responses store defaults mismatch");
    failures += check(!defaults.model_id_override.has_value(),
                      "model id override is unexpectedly configured by default");
    failures += check(
        !defaults.sampling_overrides.temperature && !defaults.sampling_overrides.top_p &&
            !defaults.sampling_overrides.top_k && !defaults.sampling_overrides.presence_penalty &&
            !defaults.sampling_overrides.frequency_penalty,
        "server defaults unexpectedly override registered model sampling");
    failures += check(resolve_public_model_id(defaults, "artifact-model") == "artifact-model",
                      "artifact model id was not selected by default");

    failures += check(defaults.prefill_decode_rounds == 1,
                      "default prefill/decode alternation changed");
    const auto interactive = parse({"ninfer-serve", "model.ninfer",
                                    "--prefill-decode-rounds", "16"});
    failures += check(interactive.prefill_decode_rounds == 16,
                      "prefill decode budget did not reach serving options");
    for (const char* bad : {"0", "65", "-1", "abc"}) {
        bool rejected = false;
        try { (void)parse({"ninfer-serve", "model.ninfer", "--prefill-decode-rounds", bad}); }
        catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "invalid prefill decode budget accepted");
    }

    const ServeOptions model_alias =
        parse({"ninfer-serve", "model.ninfer", "--model-id", "deployment-alias"});
    failures +=
        check(model_alias.model_id_override == "deployment-alias" &&
                  resolve_public_model_id(model_alias, "artifact-model") == "deployment-alias",
              "explicit model id did not override the artifact identity");

    bool empty_model_id_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--model-id", ""});
    } catch (const std::invalid_argument&) { empty_model_id_rejected = true; }
    failures += check(empty_model_id_rejected, "empty --model-id was accepted");

    const ServeOptions dflash = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash",
                                       "--draft-tokens", "15", "--lm-head-draft"});
    failures += check(dflash.speculative.backend == ninfer::SpeculativeBackend::DFlash,
                      "--spec dflash did not select DFlash");
    failures += check(dflash.speculative.draft_tokens == 15,
                      "--draft-tokens did not preserve the DFlash window");
    failures += check(dflash.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                      "--lm-head-draft did not select the optimized proposal head");

    bool dflash_vision_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--spec", "dflash", "--draft-tokens", "15",
                     "--vision"});
    } catch (const std::invalid_argument&) { dflash_vision_rejected = true; }
    failures += check(dflash_vision_rejected, "DFlash and Vision were accepted together");

    bool implicit_backend_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--draft-tokens", "3"});
    } catch (const std::invalid_argument&) { implicit_backend_rejected = true; }
    failures += check(implicit_backend_rejected, "--draft-tokens selected a backend implicitly");

    const ServeOptions configured = parse(
        {"ninfer-serve", "model.ninfer", "--no-prefix-reuse", "--vision", "--max-concurrency", "4",
         "--max-pending-requests", "12", "--pending-timeout-ms", "2500", "--max-context", "4096",
         "--kv-capacity", "8192", "--log-stats-interval-ms", "0", "--preserve-thinking"});
    failures += check(!configured.allow_prefix_reuse,
                      "--no-prefix-reuse did not disable server prefix reuse");
    failures += check(configured.enable_vision, "--vision did not enable Vision");
    failures +=
        check(configured.preserve_thinking, "--preserve-thinking did not reach serving options");
    failures +=
        check(configured.max_concurrency == 4, "--max-concurrency did not reach serving options");
    failures += check(configured.max_context == 4096 &&
                          configured.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          configured.kv_capacity.explicit_tokens == 8192,
                      "context and KV capacity options were not kept distinct");
    failures += check(configured.max_pending_requests == 12,
                      "--max-pending-requests did not reach serving options");
    failures += check(configured.pending_timeout_ms == 2500,
                      "--pending-timeout-ms did not reach serving options");
    failures += check(configured.log_stats_interval_ms == 0,
                      "--log-stats-interval-ms did not disable periodic reporting");

    const ServeOptions response_store =
        parse({"ninfer-serve", "model.ninfer", "--response-store-max-records", "42",
               "--response-store-max-mib", "8"});
    failures += check(response_store.response_store_max_records == 42 &&
                          response_store.response_store_max_bytes == (8ULL << 20),
                      "Responses store limits did not reach serving options");

    const ServeOptions sampling =
        parse({"ninfer-serve", "model.ninfer", "--temperature", "0", "--top-p", "0.9", "--top-k",
               "40", "--min-p", "0.1", "--presence-penalty", "1.25", "--frequency-penalty", "-0.5",
               "--seed", "0"});
    failures += check(sampling.sampling_overrides.temperature == 0.0F &&
                          sampling.sampling_overrides.top_p == 0.9F &&
                          sampling.sampling_overrides.top_k == 40 &&
                          sampling.sampling_overrides.min_p == 0.1F &&
                          sampling.sampling_overrides.presence_penalty == 1.25F &&
                          sampling.sampling_overrides.frequency_penalty == -0.5F &&
                          sampling.sampling_overrides.seed == 0,
                      "server sampling flags did not preserve explicit values and zeros");

    GenerationRequest request;
    request.max_tokens = 1;
    ninfer::PromptCapabilities prompt_capabilities;
    prompt_capabilities.enable_thinking = true;
    failures += check(to_request_options(request, defaults).execution.allow_prefix_reuse,
                      "default server policy did not reach Engine options");
    failures += check(!to_request_options(request, configured).execution.allow_prefix_reuse,
                      "disabled server policy did not reach Engine options");
    const ninfer::RequestOptions inherited_sampling = to_request_options(request, sampling);
    failures += check(inherited_sampling.execution.sampling.temperature == 0.0F &&
                          inherited_sampling.execution.sampling.top_p == 0.9F &&
                          inherited_sampling.execution.sampling.seed == 0,
                      "server sampling overrides did not reach Engine options");
    request.sampling.temperature = 1.1;
    failures += check(to_request_options(request, sampling).execution.sampling.temperature == 1.1F,
                      "request sampling override did not win over the server override");
    failures +=
        check(resolve_prompt_semantics(request, configured, prompt_capabilities).preserve_thinking,
              "server preserve-thinking default was not resolved");
    request.preserve_thinking = false;
    failures +=
        check(!resolve_prompt_semantics(request, configured, prompt_capabilities).preserve_thinking,
              "request preserve-thinking override did not win");

    failures +=
        check(serve_usage_text("ninfer-serve").find("--no-prefix-reuse") != std::string::npos,
              "serve help omits --no-prefix-reuse");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--preserve-thinking") != std::string::npos,
              "serve help omits --preserve-thinking");
    failures += check(serve_usage_text("ninfer-serve").find("--vision") != std::string::npos,
                      "serve help omits --vision");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--log-stats-interval-ms") != std::string::npos,
              "serve help omits --log-stats-interval-ms");
    failures += check(serve_usage_text("ninfer-serve").find("--kv-capacity") != std::string::npos,
                      "serve help omits --kv-capacity");
    failures += check(serve_usage_text("ninfer-serve").find("--response-store-max-mib") !=
                          std::string::npos,
                      "serve help omits Responses store limits");
    failures +=
        check(serve_usage_text("ninfer-serve").find("identity.model_id") != std::string::npos,
              "serve help omits the artifact-derived model id default");

    const ServeOptions inherited =
        parse({"ninfer-serve", "model.ninfer", "--max-context", "16384"});
    failures += check(inherited.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          inherited.kv_capacity.explicit_tokens == 16384,
                      "omitted --kv-capacity did not follow --max-context");

    const ServeOptions automatic = parse({"ninfer-serve", "model.ninfer", "--kv-capacity", "auto"});
    failures += check(automatic.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          automatic.kv_capacity.explicit_tokens == 0 &&
                          automatic.kv_capacity.automatic_headroom_bytes ==
                              ninfer::kDefaultKvCapacityHeadroomBytes,
                      "--kv-capacity auto did not select automatic sizing");

    const ServeOptions logged = parse({"ninfer-serve", "model.ninfer", "--request-log-jsonl",
                                       "requests.jsonl", "--api-key", "do-not-log"});
    failures += check(logged.request_log_jsonl == "requests.jsonl",
                      "--request-log-jsonl did not preserve its path");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--request-log-jsonl") != std::string::npos,
              "serve help omits --request-log-jsonl");
    bool secret_present    = false;
    bool redaction_present = false;
    for (const std::string& argument : logged.startup_argv) {
        secret_present    = secret_present || argument == "do-not-log";
        redaction_present = redaction_present || argument == "<redacted>";
    }
    failures += check(!secret_present, "startup argv retained the API key");
    failures += check(redaction_present, "startup argv omitted the API-key redaction marker");

    const auto replicas = parse({"ninfer-serve", "model.ninfer", "--devices", "0,1",
                                  "--max-concurrency", "4"});
    failures += check(replicas.devices == std::vector<int>({0,1}) &&
                      replicas.replica_count() == 2 && replicas.total_concurrency() == 8,
                      "dual C4 topology was not preserved");
    failures += check(defaults.replica_count() == 1 && defaults.total_concurrency() == 1,
                      "single-device admission changed");
    for (const std::string& devices : {"", "0,", ",1", "0,0", "-1,0", "0,x"}) {
        bool rejected = false;
        try { (void)parse({"ninfer-serve", "model.ninfer", "--devices", devices}); }
        catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "invalid replica list was accepted");
    }
    for (const auto& args : std::vector<std::vector<std::string>>{
             {"ninfer-serve", "model.ninfer", "--device", "0", "--devices", "0,1"},
             {"ninfer-serve", "model.ninfer", "--devices", "0,1", "--device", "0"},
             {"ninfer-serve", "model.ninfer", "--devices", "0,1", "--devices", "1,0"}}) {
        bool rejected = false;
        try { (void)parse(args); } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "ambiguous replica device selection was accepted");
    }
    failures += check(!defaults.trace_stream &&
                      parse({"ninfer-serve", "model.ninfer", "--trace-stream"}).trace_stream,
                      "stream tracing must be explicitly enabled");
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
