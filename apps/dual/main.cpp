#include "dual/prefix_snapshot_io.h"
#include "dual/joined_stage_tasks.h"
#include "targets/qwen3_6_27b/impl/dual_stage.h"
#include "dual/context_parallel_gqa.h"
#include "dual/tp2_mlp.h"
#include "dual/parallel_vocab_head.h"
#include "product/prompt_input/prompt_input.h"
#include "core/arena.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
namespace qwen3_6 = ninfer::targets::qwen3_6;
using Clock = std::chrono::steady_clock;
using ninfer::KvCacheStorage;
using ninfer::TokenId;
using ninfer::targets::qwen3_6_27b::detail::Mtp3RoundResult;
using ninfer::targets::qwen3_6_27b::detail::Mtp3SessionState;
using ninfer::targets::qwen3_6_27b::detail::Mtp3VerifyInput;
using ninfer::targets::qwen3_6_27b::detail::kDualMtpMaximumDrafts;
using ninfer::targets::qwen3_6_27b::detail::kDualMtpMaximumWidth;
using ninfer::targets::qwen3_6_27b::detail::TextPromptSequence;
using ninfer::targets::qwen3_6_27b::detail::TextShardSpec;
using ninfer::targets::qwen3_6_27b::detail::TextShardStage;
using ninfer::targets::qwen3_6_27b::detail::ContextParallelGqaCoordinator;
using ninfer::targets::qwen3_6_27b::detail::ContextParallelGqaStats;
using ninfer::targets::qwen3_6_27b::detail::SequenceShardGeometry;
using ninfer::targets::qwen3_6_27b::detail::TextKvPageOwnership;
using ninfer::targets::qwen3_6_27b::detail::plan_sequence_sharded_kv;
using ninfer::apps::dual::Tp2MlpCoordinator;
using ninfer::apps::dual::Tp2MlpStats;

struct DualTiming {
    double prefill_stage0 = 0.0;
    double prefill_stage1 = 0.0;
    double prefill_head   = 0.0;
    double decode_stage0  = 0.0;
    double decode_stage1  = 0.0;
    double decode_head    = 0.0;
};

struct Options {
    std::string artifact;
    std::vector<std::string> prompts;
    std::optional<std::filesystem::path> messages_path;
    bool vision = false;
    std::uint32_t max_context = 8192;
    std::uint32_t max_new = 16;
    std::uint32_t prefill_chunk = 128;
    KvCacheStorage kv_cache = KvCacheStorage::Int8Group64;
    std::uint32_t sessions = 1;
    bool prefill_pipeline = true;
    int device0 = 0;
    int device1 = 1;
    int split_layer = 36;
    std::optional<int> single_device;
    bool thinking = false;
    bool print_token_ids = false;
    bool target_verify_oracle = false;
    bool mtp3 = false;
    std::uint32_t mtp_depth = 3;
    bool mtp_media_throughput = false;
    bool mtp_batch = false;
    bool mtp_proposal_stage0 = false;
    std::uint32_t mtp_attention_window=0;
    bool mtp_full_vocab_draft=false;
    std::optional<std::filesystem::path> tp2_mlp_dir;
    bool context_parallel_gqa = false;
    bool parallel_vocab_head = false;
    bool parallel_greedy_head = false;
    bool context_headscale_qk = false;
    bool context_value_fp16 = false;
    qwen3_6::LongContextComputation computation = qwen3_6::LongContextComputation::RetainedThroughput;
    std::uint32_t context_parallel_threshold = 32768;
    std::optional<std::filesystem::path> prefix_snapshot_dir;
};

[[noreturn]] void usage(std::string_view error = {}) {
    if (!error.empty()) std::cerr << "error: " << error << "\n\n";
    std::cerr
        << "usage: ninfer-dual <model.ninfer> (--prompt <text>|--prompt-file <path>|--messages <path>) ... [options]\n"
        << "  --max-context N    per-device context capacity (default 8192)\n"
        << "  --max-new N        greedy output tokens (default 16)\n"
        << "  --prefill-chunk N  prompt chunk width per shard (default 128)\n"
        << "  --kv-dtype bf16|int8  KV cache storage (default int8)\n"
        << "  --prompt-file PATH  read one session prompt from UTF-8 file (repeatable)\n"
        << "  --sessions N       shared-weight independent sessions (default 1)\n"
        << "  --no-prefill-pipeline  serialize prompt chunks for diagnostics\n"
        << "  --device0 N        first-stage CUDA device (default 0)\n"
        << "  --device1 N        second-stage CUDA device (default 1)\n"
        << "  --split-layer N    multiple-of-4 boundary (default 36)\n"
        << "  --single-device N  all 64 layers on one GPU, exact T1 oracle\n"
        << "  --thinking         enable thinking prompt template\n"
        << "  --print-token-ids  duplicate generated IDs to stderr\n"
        << "  --target-verify-oracle  run full-vs-split width-4 verifier/fold qualification\n"
        << "  --spec none|mtp    speculative backend\n"
        << "  --mtp-depth N      fixed/adaptive MTP draft budget 0..5 (default 3)\n"
        << "  --mtp-media-throughput  explicit D3 media-description profile; permits the\n"
        << "                          qualified target-margin policy (not for exact replies)\n"
        << "  --mtp-batch        packed B=2 target verification; sessions=4 enables rolling C4\n"
        << "  --mtp-proposal-stage0  diagnostic stage-0 canonical proposal sidecar\n"
        << "  --tp2-mlp-dir PATH  opt-in block69 TP2 target MLP weights (C1 MTP only)\n"
        << "  --full-context-compute  all target layers/keys, full target head and strictly verified drafts\n"
        << "  --parallel-vocab-head  exact native T4 output-row split; complete dual C1 D3 only\n"
        << "  --parallel-greedy-head same exact rows, return greedy IDs only; logits materialized on demand\n"
        << "  --context-headscale-qk experimental per-head Q/K quantization; complete context-parallel INT8 only\n"
        << "  --context-value-fp16  experimental direct FP16 value arithmetic with headscale-QK; cache unchanged\n"
        << "  --mtp-attention-window N  proposal-only recent tokens, with128-token sink; target remains complete\n"
        << "  --mtp-full-vocab-draft  optional full-vocabulary proposals with the explicit window\n"
        << "  --context-parallel-gqa  physical 60/40 sequence-sharded C1 KV decode\n"
        << "  --context-parallel-threshold N  minimum populated keys (default 32768)\n"
        << "  --prefix-snapshot-dir PATH  capture or load a C1 MTP prefix snapshot\n";
    std::exit(error.empty() ? 0 : 2);
}

std::uint32_t u32(std::string_view text, std::string_view label) {
    if (text.empty()) usage(std::string(label) + " requires a value");
    std::uint64_t value = 0;
    for (char ch : text) {
        if (ch < '0' || ch > '9') usage(std::string(label) + " must be an integer");
        value = value * 10U + static_cast<unsigned>(ch - '0');
        if (value > std::numeric_limits<std::uint32_t>::max())
            usage(std::string(label) + " is out of range");
    }
    return static_cast<std::uint32_t>(value);
}

int integer(std::string_view text, std::string_view label) {
    auto value = u32(text, label);
    if (value > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
        usage(std::string(label) + " is out of range");
    return static_cast<int>(value);
}

Options parse(int argc, char** argv) {
    if (argc < 2) usage("model artifact is required");
    Options out;
    out.artifact = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string_view arg = argv[i];
        auto take = [&](std::string_view label) -> std::string_view {
            if (++i >= argc) usage(std::string(label) + " requires a value");
            return argv[i];
        };
        if (arg == "--prompt") out.prompts.emplace_back(take(arg));
        else if (arg == "--prompt-file") {
            const std::filesystem::path path(take(arg));
            std::ifstream input(path, std::ios::binary);
            if (!input) usage("failed to open --prompt-file");
            std::ostringstream buffer;
            buffer << input.rdbuf();
            if (!input.good() && !input.eof()) usage("failed to read --prompt-file");
            out.prompts.push_back(buffer.str());
        }
        else if (arg == "--messages") out.messages_path = std::filesystem::path(take(arg));
        else if (arg == "--vision") out.vision = true;
        else if (arg == "--max-context") out.max_context = u32(take(arg), arg);
        else if (arg == "--max-new") out.max_new = u32(take(arg), arg);
        else if (arg == "--prefill-chunk") out.prefill_chunk = u32(take(arg), arg);
        else if (arg == "--kv-dtype") {
            const std::string_view value = take(arg);
            if (value == "bf16") out.kv_cache = KvCacheStorage::BFloat16;
            else if (value == "int8") out.kv_cache = KvCacheStorage::Int8Group64;
            else usage("--kv-dtype must be bf16 or int8");
        }
        else if (arg == "--sessions") out.sessions = u32(take(arg), arg);
        else if (arg == "--no-prefill-pipeline") out.prefill_pipeline = false;
        else if (arg == "--device0") out.device0 = integer(take(arg), arg);
        else if (arg == "--device1") out.device1 = integer(take(arg), arg);
        else if (arg == "--split-layer") out.split_layer = integer(take(arg), arg);
        else if (arg == "--single-device") out.single_device = integer(take(arg), arg);
        else if (arg == "--thinking") out.thinking = true;
        else if (arg == "--print-token-ids") out.print_token_ids = true;
        else if (arg == "--target-verify-oracle") out.target_verify_oracle = true;
        else if (arg == "--mtp-depth") out.mtp_depth = u32(take(arg), arg);
        else if (arg == "--mtp-media-throughput") out.mtp_media_throughput = true;
        else if (arg == "--mtp-batch") out.mtp_batch = true;
        else if (arg == "--mtp-proposal-stage0") out.mtp_proposal_stage0 = true;
        else if (arg == "--tp2-mlp-dir") out.tp2_mlp_dir = std::filesystem::path(take(arg));
        else if (arg == "--full-context-compute") out.computation=qwen3_6::LongContextComputation::Complete;
        else if (arg == "--mtp-attention-window") out.mtp_attention_window=u32(take(arg),arg);
        else if (arg == "--mtp-full-vocab-draft") out.mtp_full_vocab_draft=true;
        else if (arg == "--context-headscale-qk") out.context_headscale_qk = true;
        else if (arg == "--context-value-fp16") out.context_value_fp16 = true;
        else if (arg == "--context-parallel-gqa") out.context_parallel_gqa = true;
        else if (arg == "--parallel-vocab-head") out.parallel_vocab_head = true;
        else if (arg == "--parallel-greedy-head") { out.parallel_vocab_head = true; out.parallel_greedy_head = true; }
        else if (arg == "--context-parallel-threshold")
            out.context_parallel_threshold = u32(take(arg), arg);
        else if (arg == "--prefix-snapshot-dir")
            out.prefix_snapshot_dir = std::filesystem::path(take(arg));
        else if (arg == "--spec") {
            const std::string_view value = take(arg);
            if (value == "none") out.mtp3 = false;
            else if (value == "mtp") out.mtp3 = true;
            else usage("--spec must be none or mtp");
        }
        else if (arg == "--help" || arg == "-h") usage();
        else usage("unknown option '" + std::string(arg) + "'");
    }
    if (out.prompts.empty() == !out.messages_path.has_value())
        usage("provide exactly one prompt source: --prompt/--prompt-file or --messages");
    if (out.messages_path && !std::filesystem::is_regular_file(*out.messages_path))
        usage("--messages is not a file");
    if (out.messages_path && out.sessions != 1)
        usage("--messages currently requires --sessions 1");
    if (out.max_context == 0 || out.max_new == 0 || out.prefill_chunk == 0)
        usage("context, output, and prefill chunk must be positive");
    if (out.prefill_chunk > out.max_context)
        usage("--prefill-chunk cannot exceed --max-context");
    if (out.prefill_chunk > 128)
        usage("--prefill-chunk must be in [1,128]");
    if (out.split_layer <= 0 || out.split_layer >= 64 || out.split_layer % 4 != 0)
        usage("--split-layer must be a multiple of 4 in [4,60]");
    if (out.sessions == 0 || out.sessions > 8) usage("--sessions must be in [1,8]");
    if (!out.messages_path && out.prompts.size() != 1 && out.prompts.size() != out.sessions)
        usage("provide one shared --prompt or exactly one --prompt per session");
    if (out.vision && !out.messages_path)
        usage("--vision currently requires --messages");
    if(out.context_value_fp16 && (!out.context_headscale_qk || !out.mtp3 || out.sessions!=1))
        usage("packed FP16 values require the explicit complete headscale-QK profile");
    if(out.context_headscale_qk && (!out.context_parallel_gqa ||
        out.computation!=qwen3_6::LongContextComputation::Complete || out.mtp_depth!=3 ||
        out.kv_cache!=KvCacheStorage::Int8Group64 || !out.prefix_snapshot_dir))
        usage("headscale-QK requires complete target, physical context parallel INT8, D3 and a canonical prefix snapshot");
    if (out.parallel_vocab_head && (!out.mtp3 || out.mtp_depth != 3 || out.sessions != 1 ||
        out.single_device || out.target_verify_oracle || out.mtp_batch ||
        out.computation != qwen3_6::LongContextComputation::Complete))
        usage("--parallel-vocab-head requires complete dual C1 D3 verification");
    if (out.target_verify_oracle && (out.sessions != 1 || out.single_device))
        usage("--target-verify-oracle requires dual mode with --sessions 1");
    if(out.computation==qwen3_6::LongContextComputation::Complete &&
       (out.single_device || out.sessions!=1 || out.mtp_media_throughput)) {
        usage("--full-context-compute requires dual single-request mode without media-throughput acceptance");
    }
    if(out.mtp_full_vocab_draft && out.mtp_attention_window==0)
        usage("full-vocabulary draft composition requires an explicit proposal window");
    if(!qwen3_6::complete_proposal_window_allowed(out.computation,out.mtp3,out.sessions,
        out.mtp_depth,out.mtp_proposal_stage0,out.mtp_attention_window)) {
        usage("proposal attention window requires complete dual C1 D3 and page-aligned tokens in [128,262144]");
    }
    if (out.mtp_depth > kDualMtpMaximumDrafts) usage("--mtp-depth must be in [0,5]");
    if (!out.mtp3 && out.mtp_depth != 3) usage("--mtp-depth requires --spec mtp");
    if (out.mtp3 && out.sessions > 1 && out.mtp_depth != 3)
        usage("multi-session MTP currently requires --mtp-depth 3");
    if (out.mtp_media_throughput &&
        (!out.mtp3 || out.mtp_depth != 3 || !out.vision || !out.messages_path ||
         out.sessions != 1 || out.single_device || out.mtp_batch || out.target_verify_oracle)) {
        usage("--mtp-media-throughput requires dual C1 --vision --messages --spec mtp --mtp-depth 3");
    }
    if (out.mtp_batch &&
        (!out.mtp3 || (out.sessions != 2 && out.sessions != 4 && out.sessions != 8) ||
         out.single_device))
        usage("--mtp-batch requires dual --spec mtp with --sessions 2, 4, or 8");
    if (out.mtp_proposal_stage0 && (!out.mtp3 || out.sessions != 2 || out.single_device))
        usage("--mtp-proposal-stage0 requires dual --spec mtp --sessions 2");
    if (out.tp2_mlp_dir &&
        (!out.mtp3 || out.sessions != 1 || out.mtp_batch || out.single_device ||
         out.target_verify_oracle)) {
        usage("--tp2-mlp-dir requires dual --spec mtp --sessions 1 without --mtp-batch");
    }
    if (out.tp2_mlp_dir && !std::filesystem::is_directory(*out.tp2_mlp_dir))
        usage("--tp2-mlp-dir is not a directory");
    if (out.tp2_mlp_dir && out.mtp_depth != 3 && out.mtp_depth != 4)
        usage("--tp2-mlp-dir currently requires --mtp-depth 3 or 4");
    if (out.context_parallel_gqa &&
        (out.single_device || out.sessions != 1 || out.target_verify_oracle)) {
        usage("--context-parallel-gqa requires dual C1 without verifier oracle");
    }
    if (out.context_parallel_gqa && out.context_parallel_threshold == 0)
        usage("--context-parallel-threshold must be positive");
    if (out.context_parallel_gqa && out.split_layer % 4 != 0)
        usage("--context-parallel-gqa requires a split layer divisible by 4");
    if (out.prefix_snapshot_dir &&
        (!out.mtp3 || out.sessions != 1 || out.single_device || out.target_verify_oracle))
        usage("--prefix-snapshot-dir requires dual C1 MTP mode");
    if (out.mtp3 && out.single_device)
        usage("dual-driver MTP currently requires two sharded devices");
    if (out.single_device && out.sessions != 1)
        usage("multi-session mode currently requires dual devices");
    if (!out.single_device && out.device0 == out.device1)
        usage("dual mode requires distinct CUDA devices");
    return out;
}

std::string bytes(std::size_t value) {
    constexpr double MiB = 1024.0 * 1024.0, GiB = 1024.0 * MiB;
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (value >= static_cast<std::size_t>(GiB)) out << value / GiB << " GiB";
    else out << value / MiB << " MiB";
    return out.str();
}

std::string seconds(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << value << " s";
    return out.str();
}

std::string rate(std::size_t tokens, double elapsed) {
    if (tokens == 0 || elapsed <= 0.0) return "n/a";
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << tokens / elapsed << " tok/s";
    return out.str();
}

std::string percentage(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) return "n/a";
    std::ostringstream out;
    out << std::fixed << std::setprecision(2)
        << (100.0 * static_cast<double>(numerator) / static_cast<double>(denominator)) << '%';
    return out.str();
}

std::string joined_counts(std::span<const std::uint64_t> values, std::size_t count) {
    count = std::min(count, values.size());
    std::ostringstream out;
    for (std::size_t index = 0; index < count; ++index) {
        if (index != 0) { out << ','; }
        out << values[index];
    }
    return out.str();
}

std::uint64_t digest(const std::vector<TokenId>& tokens) {
    std::uint64_t out = 14695981039346656037ULL;
    for (TokenId token : tokens) {
        std::uint32_t value = static_cast<std::uint32_t>(token);
        for (int i = 0; i < 4; ++i) {
            out ^= value & 0xffU;
            out *= 1099511628211ULL;
            value >>= 8;
        }
    }
    return out;
}

std::string hex_digest(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

constexpr std::uint64_t kPrefixSnapshotMagic = 0x315846504e494eULL;
constexpr std::uint32_t kPrefixSnapshotVersion = 1;

struct PrefixSnapshotManifest {
    std::uint64_t magic = kPrefixSnapshotMagic;
    std::uint32_t version = kPrefixSnapshotVersion;
    std::uint32_t split_layer = 0;
    std::uint32_t max_context = 0;
    std::uint32_t prefill_chunk = 0;
    std::uint32_t kv_cache = 0;
    std::uint32_t reserved = 0;
    std::uint64_t prompt_tokens = 0;
    std::uint64_t prompt_digest = 0;
    std::int32_t rope_delta = 0;
    TokenId anchor = 0;
    std::array<TokenId, 3> drafts{};
    std::uint32_t padding = 0;
    std::uint64_t stage0_bytes = 0;
    std::uint64_t stage1_bytes = 0;
    std::uint64_t context_bytes = 0;
};

struct PrefixSnapshotInitial {
    TokenId anchor = 0;
    std::array<TokenId, 3> drafts{};
};

std::filesystem::path prefix_snapshot_file(const std::filesystem::path& directory,
                                           std::string_view name) {
    return directory / std::string(name);
}

PrefixSnapshotManifest read_prefix_snapshot_manifest(const std::filesystem::path& directory) {
    PrefixSnapshotManifest manifest;
    std::ifstream input(prefix_snapshot_file(directory, "manifest.bin"), std::ios::binary);
    input.read(reinterpret_cast<char*>(&manifest), sizeof(manifest));
    if (!input || input.peek() != std::ifstream::traits_type::eof() ||
        manifest.magic != kPrefixSnapshotMagic || manifest.version != kPrefixSnapshotVersion) {
        throw std::runtime_error("invalid prefix snapshot manifest");
    }
    return manifest;
}

bool prefix_snapshot_available(const std::filesystem::path& directory) {
    if (!std::filesystem::is_regular_file(prefix_snapshot_file(directory, "manifest.bin")) ||
        !std::filesystem::is_regular_file(prefix_snapshot_file(directory, "stage0.bin")) ||
        !std::filesystem::is_regular_file(prefix_snapshot_file(directory, "stage1.bin"))) {
        return false;
    }
    try {
        const PrefixSnapshotManifest manifest = read_prefix_snapshot_manifest(directory);
        return manifest.context_bytes == 0 ||
               std::filesystem::is_regular_file(prefix_snapshot_file(directory, "context.bin"));
    } catch (...) {
        return false;
    }
}

void write_snapshot_file(const std::filesystem::path& path, const void* data,
                         std::size_t bytes) {
    const std::filesystem::path temporary = path.string() + ".part";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    output.close();
    if (!output) throw std::runtime_error("failed to write prefix snapshot " + path.string());
    std::error_code error;
    std::filesystem::remove(path, error);
    error.clear();
    std::filesystem::rename(temporary, path, error);
    if (error) throw std::runtime_error("failed to publish prefix snapshot " + path.string());
}

using ninfer::apps::dual::detail::run_joined_stage_tasks;
using ninfer::apps::dual::detail::read_prefix_snapshot_bytes;

void save_prefix_snapshot(const Options& opt, const std::filesystem::path& directory,
                          TextShardStage& first, TextShardStage& second,
                          ContextParallelGqaCoordinator* context,
                          const TextPromptSequence& prompt, TokenId anchor,
                          std::array<TokenId, 3> drafts) {
    std::filesystem::create_directories(directory);
    PrefixSnapshotManifest manifest;
    manifest.reserved = static_cast<std::uint32_t>(opt.computation);
    manifest.split_layer = static_cast<std::uint32_t>(opt.split_layer);
    manifest.max_context = opt.max_context;
    manifest.prefill_chunk = opt.prefill_chunk;
    manifest.kv_cache = static_cast<std::uint32_t>(opt.kv_cache);
    manifest.prompt_tokens = prompt.token_ids.size();
    manifest.prompt_digest = digest(prompt.token_ids);
    manifest.rope_delta = prompt.rope_delta;
    manifest.anchor = anchor;
    manifest.drafts = drafts;
    manifest.stage0_bytes = first.prefix_snapshot_bytes();
    manifest.stage1_bytes = second.prefix_snapshot_bytes();
    manifest.context_bytes = context != nullptr ? context->prefix_snapshot_bytes() : 0;

    auto save_blob = [&](std::string_view name, std::size_t bytes,
                         auto&& copy_to_host) {
        if (bytes == 0) return;
        std::vector<std::uint8_t> buffer(bytes);
        copy_to_host(buffer.data(), buffer.size());
        write_snapshot_file(prefix_snapshot_file(directory, name), buffer.data(), buffer.size());
    };
    save_blob("stage0.bin", manifest.stage0_bytes,
              [&](void* data, std::size_t bytes) {
                  first.copy_prefix_snapshot_to_host(data, bytes);
              });
    save_blob("stage1.bin", manifest.stage1_bytes,
              [&](void* data, std::size_t bytes) {
                  second.copy_prefix_snapshot_to_host(data, bytes);
              });
    if (context != nullptr) {
        save_blob("context.bin", manifest.context_bytes,
                  [&](void* data, std::size_t bytes) {
                      context->copy_prefix_snapshot_to_host(data, bytes);
                  });
    }
    write_snapshot_file(prefix_snapshot_file(directory, "manifest.bin"),
                        &manifest, sizeof(manifest));
}

PrefixSnapshotInitial load_prefix_snapshot(
    const Options& opt, const std::filesystem::path& directory,
    TextShardStage& first, TextShardStage& second,
    ContextParallelGqaCoordinator* context, const TextPromptSequence& prompt) {
    const PrefixSnapshotManifest manifest = read_prefix_snapshot_manifest(directory);
    const bool common_contract =
        manifest.reserved == static_cast<std::uint32_t>(opt.computation) &&
        manifest.split_layer == static_cast<std::uint32_t>(opt.split_layer) &&
        manifest.max_context == opt.max_context &&
        manifest.prefill_chunk == opt.prefill_chunk &&
        manifest.kv_cache == static_cast<std::uint32_t>(opt.kv_cache) &&
        manifest.prompt_tokens == prompt.token_ids.size() &&
        manifest.prompt_digest == digest(prompt.token_ids) &&
        manifest.rope_delta == prompt.rope_delta;
    if (!common_contract) {
        throw std::runtime_error("prefix snapshot does not match prompt/runtime geometry");
    }

    const bool legacy_physical_import = context != nullptr &&
        manifest.stage0_bytes >= first.legacy_full_prefix_snapshot_bytes() &&
        manifest.stage1_bytes >= second.legacy_full_prefix_snapshot_bytes();
    if (legacy_physical_import) {
        // Each stage and its helper own disjoint cache planes and CUDA streams.
        // The immutable files can be read concurrently. Keep each buffer alive
        // through its completed upload, and join both tasks before activation.
        auto restore_stage = [&](std::uint32_t index, TextShardStage& stage,
                                 const char* file_name, std::uint64_t file_bytes) {
            auto snapshot = read_prefix_snapshot_bytes(
                prefix_snapshot_file(directory, file_name), file_bytes);
            const TextKvPageOwnership helper = context->helper_ownership(index);
            std::vector<std::uint8_t> helper_pages = stage.pack_legacy_text_kv_range(
                snapshot.data(), snapshot.size(), helper.logical_page_begin,
                helper.logical_page_count);
            stage.copy_legacy_full_prefix_snapshot_from_host(snapshot.data(), snapshot.size());
            context->load_legacy_helper_pages(index, helper_pages);
        };
        run_joined_stage_tasks(
            [&] { restore_stage(0, first, "stage0.bin", manifest.stage0_bytes); },
            [&] { restore_stage(1, second, "stage1.bin", manifest.stage1_bytes); });
        context->activate_legacy_prefix(static_cast<std::uint32_t>(manifest.prompt_tokens));
        return {.anchor = manifest.anchor, .drafts = manifest.drafts};
    }

    const std::uint64_t context_bytes = context != nullptr ? context->prefix_snapshot_bytes() : 0;
    if (manifest.stage0_bytes != first.prefix_snapshot_bytes() ||
        manifest.stage1_bytes != second.prefix_snapshot_bytes() ||
        (context != nullptr && manifest.context_bytes != context_bytes)) {
        throw std::runtime_error("prefix snapshot does not match current state layout");
    }
    auto buffer = read_prefix_snapshot_bytes(
        prefix_snapshot_file(directory, "stage0.bin"), manifest.stage0_bytes);
    first.copy_prefix_snapshot_from_host(buffer.data(), buffer.size());
    buffer = read_prefix_snapshot_bytes(prefix_snapshot_file(directory, "stage1.bin"),
                                manifest.stage1_bytes);
    second.copy_prefix_snapshot_from_host(buffer.data(), buffer.size());
    if (context != nullptr) {
        buffer = read_prefix_snapshot_bytes(prefix_snapshot_file(directory, "context.bin"),
                                    manifest.context_bytes);
        context->copy_prefix_snapshot_from_host(buffer.data(), buffer.size());
    }
    return {.anchor = manifest.anchor, .drafts = manifest.drafts};
}

void metric(std::string_view label, std::string_view value) {
    std::cerr << std::left << std::setw(12) << "summary" << std::setw(30) << label << value << '\n';
}

void stage_metrics(std::string_view label, const TextShardStage& stage) {
    metric(std::string(label) + " device", std::to_string(stage.device_id()));
    metric(std::string(label) + " weights", bytes(stage.weights_bytes()));
    metric(std::string(label) + " state", bytes(stage.state_bytes()));
    metric(std::string(label) + " KV payload", bytes(stage.kv_payload_bytes()));
    metric(std::string(label) + " workspace", bytes(stage.workspace_bytes()));
    metric(std::string(label) + " sessions", std::to_string(stage.session_count()));
}

std::vector<TokenId> single(const Options& opt, TextShardStage& stage,
                            const TextPromptSequence& prompt, double& prefill, double& decode) {
    auto start = Clock::now();
    const std::span<const TokenId> prompt_tokens(prompt.token_ids);
    for (std::size_t begin = 0; begin < prompt_tokens.size();) {
        const std::size_t nominal =
            std::min<std::size_t>(opt.prefill_chunk, prompt_tokens.size() - begin);
        const std::size_t tokens = prompt.has_media()
            ? stage.prefill_multimodal_tokens(
                  prompt, static_cast<std::uint32_t>(begin),
                  static_cast<std::uint32_t>(nominal))
            : nominal;
        if (!prompt.has_media()) {
            stage.prefill_tokens(prompt_tokens.subspan(begin, tokens),
                                 static_cast<std::int32_t>(begin),
                                 static_cast<std::int32_t>(begin) + prompt.rope_delta);
        }
        begin += tokens;
    }
    std::vector<TokenId> out;
    out.reserve(opt.max_new);
    out.push_back(stage.project_token());
    auto first = Clock::now();
    prefill = std::chrono::duration<double>(first - start).count();
    start = Clock::now();
    for (std::uint32_t i = 1; i < opt.max_new; ++i) {
        auto pos = static_cast<std::int32_t>(prompt.token_ids.size() + i - 1U);
        stage.forward_token(out.back(), pos, pos + prompt.rope_delta);
        out.push_back(stage.project_token());
    }
    decode = std::chrono::duration<double>(Clock::now() - start).count();
    return out;
}

std::vector<TokenId> dual(const Options& opt, TextShardStage& first, TextShardStage& second,
                          const TextPromptSequence& prompt, double& prefill, double& decode,
                          DualTiming& timing) {
    const std::size_t boundary_capacity = TextShardStage::residual_bytes(opt.prefill_chunk);
    std::array<ninfer::PinnedHostBuffer, 2> boundaries{
        ninfer::PinnedHostBuffer(boundary_capacity), ninfer::PinnedHostBuffer(boundary_capacity)};
    const auto feed = [&](TokenId token, std::int32_t pos, double& stage0, double& stage1) {
        const auto rope = pos + prompt.rope_delta;
        auto started    = Clock::now();
        first.forward_token(token, pos, rope);
        first.copy_residual_to_host(boundaries[0].data(), TextShardStage::residual_bytes());
        stage0 += std::chrono::duration<double>(Clock::now() - started).count();

        started = Clock::now();
        second.copy_residual_from_host(boundaries[0].data(), TextShardStage::residual_bytes());
        second.forward_hidden(pos, rope);
        second.synchronize();
        stage1 += std::chrono::duration<double>(Clock::now() - started).count();
    };
    auto start = Clock::now();
    const std::span<const TokenId> prompt_tokens(prompt.token_ids);
    bool stage1_inflight = false;
    std::size_t chunk_index = 0;
    for (std::size_t begin = 0; begin < prompt_tokens.size(); ++chunk_index) {
        const std::size_t nominal =
            std::min<std::size_t>(opt.prefill_chunk, prompt_tokens.size() - begin);
        const std::size_t tokens = prompt.has_media()
            ? first.prefill_multimodal_tokens(
                  prompt, static_cast<std::uint32_t>(begin),
                  static_cast<std::uint32_t>(nominal))
            : nominal;
        auto& boundary = boundaries[chunk_index % boundaries.size()];
        auto started   = Clock::now();
        if (!prompt.has_media()) {
            first.prefill_tokens(prompt_tokens.subspan(begin, tokens),
                                 static_cast<std::int32_t>(begin),
                                 static_cast<std::int32_t>(begin) + prompt.rope_delta);
        }
        first.copy_prefill_residual_to_host(boundary.data(), tokens);
        timing.prefill_stage0 += std::chrono::duration<double>(Clock::now() - started).count();

        if (opt.prefill_pipeline && stage1_inflight) {
            started = Clock::now();
            second.synchronize();
            timing.prefill_stage1 += std::chrono::duration<double>(Clock::now() - started).count();
        }
        started = Clock::now();
        second.copy_prefill_residual_from_host(boundary.data(), tokens);
        if (prompt.has_media()) {
            second.prefill_multimodal_hidden(
                prompt, static_cast<std::uint32_t>(begin), tokens);
        } else {
            second.prefill_hidden(tokens, static_cast<std::int32_t>(begin),
                                  static_cast<std::int32_t>(begin) + prompt.rope_delta);
        }
        begin += tokens;
        stage1_inflight = true;
        if (!opt.prefill_pipeline) {
            second.synchronize();
            timing.prefill_stage1 += std::chrono::duration<double>(Clock::now() - started).count();
            stage1_inflight = false;
        }
    }
    if (stage1_inflight) {
        const auto started = Clock::now();
        second.synchronize();
        timing.prefill_stage1 += std::chrono::duration<double>(Clock::now() - started).count();
    }
    std::vector<TokenId> out;
    out.reserve(opt.max_new);
    auto head_started = Clock::now();
    out.push_back(second.project_token());
    timing.prefill_head += std::chrono::duration<double>(Clock::now() - head_started).count();
    auto first_done = Clock::now();
    prefill = std::chrono::duration<double>(first_done - start).count();
    start   = Clock::now();
    for (std::uint32_t i = 1; i < opt.max_new; ++i) {
        const auto pos = static_cast<std::int32_t>(prompt.token_ids.size() + i - 1U);
        feed(out.back(), pos, timing.decode_stage0, timing.decode_stage1);
        head_started = Clock::now();
        out.push_back(second.project_token());
        timing.decode_head += std::chrono::duration<double>(Clock::now() - head_started).count();
    }
    decode = std::chrono::duration<double>(Clock::now() - start).count();
    return out;
}

std::vector<std::vector<TokenId>>
dual_multi(const Options& opt, TextShardStage& first, TextShardStage& second,
           std::span<const TextPromptSequence> prompts, double& prefill, double& decode) {
    const std::size_t prefill_boundary_bytes = TextShardStage::residual_bytes(opt.prefill_chunk);
    std::array<ninfer::PinnedHostBuffer, 2> prefill_boundaries{
        ninfer::PinnedHostBuffer(prefill_boundary_bytes),
        ninfer::PinnedHostBuffer(prefill_boundary_bytes)};
    std::vector<ninfer::PinnedHostBuffer> decode_boundaries;
    decode_boundaries.reserve(opt.sessions);
    for (std::uint32_t session = 0; session < opt.sessions; ++session) {
        decode_boundaries.emplace_back(TextShardStage::residual_bytes());
    }

    if (prompts.size() != opt.sessions) {
        throw std::invalid_argument("prepared prompt count does not match sessions");
    }
    std::vector<std::vector<TokenId>> output(opt.sessions);
    auto started = Clock::now();
    for (std::uint32_t session = 0; session < opt.sessions; ++session) {
        const TextPromptSequence& prompt = prompts[session];
        const std::span<const TokenId> prompt_tokens(prompt.token_ids);
        bool stage1_inflight = false;
        std::size_t chunk_index = 0;
        for (std::size_t begin = 0; begin < prompt_tokens.size();
             begin += opt.prefill_chunk, ++chunk_index) {
            const std::size_t tokens =
                std::min<std::size_t>(opt.prefill_chunk, prompt_tokens.size() - begin);
            auto& boundary = prefill_boundaries[chunk_index % prefill_boundaries.size()];
            first.prefill_tokens(prompt_tokens.subspan(begin, tokens),
                                 static_cast<std::int32_t>(begin),
                                 static_cast<std::int32_t>(begin) + prompt.rope_delta, session);
            first.copy_prefill_residual_to_host(boundary.data(), tokens);
            if (opt.prefill_pipeline && stage1_inflight) { second.synchronize(); }
            second.copy_prefill_residual_from_host(boundary.data(), tokens);
            second.prefill_hidden(tokens, static_cast<std::int32_t>(begin),
                                  static_cast<std::int32_t>(begin) + prompt.rope_delta, session);
            stage1_inflight = true;
            if (!opt.prefill_pipeline) {
                second.synchronize();
                stage1_inflight = false;
            }
        }
        if (stage1_inflight) { second.synchronize(); }
        output[session].reserve(opt.max_new);
        output[session].push_back(second.project_token(session));
    }
    prefill = std::chrono::duration<double>(Clock::now() - started).count();

    started = Clock::now();
    const auto enqueue_stage0 = [&](std::uint32_t session, std::uint32_t step) {
        const TextPromptSequence& prompt = prompts[session];
        const auto pos = static_cast<std::int32_t>(prompt.token_ids.size() + step - 1U);
        first.forward_token(output[session].back(), pos, pos + prompt.rope_delta, session);
        first.copy_residual_to_host_async(decode_boundaries[session].data(),
                                          TextShardStage::residual_bytes(), session);
    };
    if (opt.max_new > 1) {
        for (std::uint32_t session = 0; session < opt.sessions; ++session) {
            enqueue_stage0(session, 1);
        }
        first.synchronize();
        const std::uint64_t jobs =
            static_cast<std::uint64_t>(opt.max_new - 1U) * opt.sessions;
        for (std::uint64_t job = 0; job < jobs; ++job) {
            const auto session = static_cast<std::uint32_t>(job % opt.sessions);
            const auto step = static_cast<std::uint32_t>(job / opt.sessions) + 1U;
            first.synchronize();
            const TextPromptSequence& prompt = prompts[session];
            const auto pos = static_cast<std::int32_t>(prompt.token_ids.size() + step - 1U);
            second.copy_residual_from_host(decode_boundaries[session].data(),
                                           TextShardStage::residual_bytes(), session);
            second.forward_hidden(pos, pos + prompt.rope_delta, session);

            if (job != 0) {
                const auto previous = static_cast<std::uint32_t>((job - 1U) % opt.sessions);
                const auto previous_step =
                    static_cast<std::uint32_t>((job - 1U) / opt.sessions) + 1U;
                if (previous_step + 1U < opt.max_new) {
                    enqueue_stage0(previous, previous_step + 1U);
                }
            }
            output[session].push_back(second.project_token(session));
        }
    }
    decode = std::chrono::duration<double>(Clock::now() - started).count();
    return output;
}


struct DualMtpTiming {
    double prefill_stage0 = 0.0;
    double prefill_stage1 = 0.0;
    double target_stage0 = 0.0;
    double target_stage1 = 0.0;
    double proposal = 0.0;
    double fold = 0.0;
    std::uint64_t rounds = 0;
    std::uint64_t fallback_steps = 0;
    std::uint64_t drafted = 0;
    std::uint64_t accepted = 0;
    std::array<std::uint64_t, kDualMtpMaximumDrafts> accepted_by_position{};
    std::array<std::uint64_t, 8> session_rounds{};
    std::array<std::uint64_t, 8> session_fallback_steps{};
    std::array<std::uint64_t, 8> session_drafted{};
    std::array<std::uint64_t, 8> session_accepted{};
    std::array<std::array<std::uint64_t, kDualMtpMaximumDrafts>, 8> session_accepted_by_position{};
    std::array<std::uint64_t, 8> session_scalar_tail{};
};

void record_mtp_round(DualMtpTiming& timing, std::uint32_t session,
                      std::int32_t draft_extent, const Mtp3RoundResult& round) {
    if (session >= timing.session_rounds.size() || draft_extent < 0 || draft_extent > static_cast<std::int32_t>(kDualMtpMaximumDrafts)) {
        throw std::out_of_range("MTP timing session or extent is out of range");
    }
    if (draft_extent == 0) {
        if (round.accepted_drafts != 0) {
            throw std::logic_error("no-spec MTP fallback accepted a draft");
        }
        timing.fallback_steps += 1;
        timing.session_fallback_steps[session] += 1;
        return;
    }
    timing.rounds += 1;
    timing.drafted += static_cast<std::uint64_t>(draft_extent);
    timing.accepted += static_cast<std::uint64_t>(round.accepted_drafts);
    timing.session_rounds[session] += 1;
    timing.session_drafted[session] += static_cast<std::uint64_t>(draft_extent);
    timing.session_accepted[session] += static_cast<std::uint64_t>(round.accepted_drafts);
    for (std::int32_t position = 0; position < round.accepted_drafts; ++position) {
        timing.accepted_by_position[static_cast<std::size_t>(position)] += 1;
        timing.session_accepted_by_position[session][static_cast<std::size_t>(position)] += 1;
    }
}

std::int32_t mtp_verify_columns(const Mtp3SessionState& state) {
    if (state.draft_extent < 0 ||
        state.draft_extent > static_cast<std::int32_t>(kDualMtpMaximumDrafts)) {
        throw std::invalid_argument("MTP draft extent exceeds the supported maximum");
    }
    return state.draft_extent + 1;
}

bool can_run_mtp_round(const Mtp3SessionState& state, std::uint32_t max_context) {
    const std::int32_t columns = mtp_verify_columns(state);
    return state.remaining_budget >= columns && state.frontier >= 0 &&
           static_cast<std::uint64_t>(state.frontier) +
                   static_cast<std::uint32_t>(columns) <=
               max_context;
}

void promote_fixed_mtp_extent(Mtp3SessionState& state, const Mtp3RoundResult& round,
                              std::uint32_t configured_depth) {
    const std::int32_t target = static_cast<std::int32_t>(configured_depth);
    if (state.draft_extent < target && round.next_extent >= target) {
        // A reusable prefix snapshot may carry fewer initial drafts than the verifier width
        // selected for this run. Once the canonical proposal recursion produces a complete
        // next draft window, honor the fixed configured budget for every later full round.
        state.draft_extent = target;
    }
}

// The execution hook is borrowed; detach and drain it before its storage dies,
// including exception unwinding. The two stage objects outlive this binding.
struct BoundVocabHead {
    TextShardStage& stage;
    std::unique_ptr<ninfer::apps::dual::ParallelVocabHead> head;
    explicit BoundVocabHead(TextShardStage& output) : stage(output) {}
    ~BoundVocabHead() {
        if (head) {
            try { stage.set_head_execution_override(nullptr); }
            catch (...) { /* Coordinator teardown also synchronizes outstanding work. */ }
        }
    }
};

std::vector<TokenId> dual_mtp3(const Options& opt, TextShardStage& first,
                               TextShardStage& second, ContextParallelGqaCoordinator* context,
                               const TextPromptSequence& prompt, double& prefill, double& decode,
                               DualMtpTiming& timing) {
    const std::uint32_t verify_width = std::max<std::uint32_t>(4U, opt.mtp_depth + 1U);
    if (opt.sessions != 1 || first.verify_width() != verify_width ||
        second.verify_width() != verify_width) {
        throw std::invalid_argument("dual one-session MTP verify width is inconsistent");
    }
    const std::span<const TokenId> prompt_tokens(prompt.token_ids);
    TokenId anchor = 0;
    std::array<TokenId, kDualMtpMaximumDrafts> initial_drafts{};
    std::int32_t initial_extent = static_cast<std::int32_t>(opt.mtp_depth);
    auto started = Clock::now();
    const bool load_snapshot = opt.prefix_snapshot_dir &&
                               prefix_snapshot_available(*opt.prefix_snapshot_dir);
    if (context != nullptr && !load_snapshot) {
        throw std::invalid_argument(
            "physical sequence-sharded KV currently requires --prefix-snapshot-dir");
    }
    if (load_snapshot) {
        const auto restore_started = Clock::now();
        const PrefixSnapshotInitial loaded = load_prefix_snapshot(
            opt, *opt.prefix_snapshot_dir, first, second, context, prompt);
        metric("prefix snapshot restore", seconds(
            std::chrono::duration<double>(Clock::now() - restore_started).count()));
        anchor = loaded.anchor;
        std::copy(loaded.drafts.begin(), loaded.drafts.end(), initial_drafts.begin());
        initial_extent = std::min<std::int32_t>(
            static_cast<std::int32_t>(opt.mtp_depth),
            static_cast<std::int32_t>(loaded.drafts.size()));
    } else {
        std::array<ninfer::PinnedHostBuffer, 2> prefill_boundaries{
            ninfer::PinnedHostBuffer(TextShardStage::residual_bytes(opt.prefill_chunk)),
            ninfer::PinnedHostBuffer(TextShardStage::residual_bytes(opt.prefill_chunk))};
        std::size_t final_begin = 0;
        std::size_t final_count = 0;
        bool stage1_inflight = false;
        std::size_t chunk_index = 0;
        for (std::size_t begin = 0; begin < prompt_tokens.size(); ++chunk_index) {
            const std::size_t nominal =
                std::min<std::size_t>(opt.prefill_chunk, prompt_tokens.size() - begin);
            const std::size_t count = prompt.has_media()
                ? first.prefill_multimodal_tokens(
                      prompt, static_cast<std::uint32_t>(begin),
                      static_cast<std::uint32_t>(nominal))
                : nominal;
            const bool final_chunk = begin + count == prompt_tokens.size();
            auto& boundary = prefill_boundaries[chunk_index % prefill_boundaries.size()];
            auto phase = Clock::now();
            if (!prompt.has_media()) {
                first.prefill_tokens(prompt_tokens.subspan(begin, count),
                                     static_cast<std::int32_t>(begin),
                                     static_cast<std::int32_t>(begin) + prompt.rope_delta);
            }
            first.copy_prefill_residual_to_host(boundary.data(), count);
            timing.prefill_stage0 +=
                std::chrono::duration<double>(Clock::now() - phase).count();

            if (opt.prefill_pipeline && stage1_inflight) {
                phase = Clock::now();
                second.synchronize();
                timing.prefill_stage1 +=
                    std::chrono::duration<double>(Clock::now() - phase).count();
                stage1_inflight = false;
            }
            phase = Clock::now();
            second.copy_prefill_residual_from_host(boundary.data(), count);
            if (prompt.has_media()) {
                second.prefill_multimodal_hidden(
                    prompt, static_cast<std::uint32_t>(begin), count);
            } else {
                second.prefill_hidden(count, static_cast<std::int32_t>(begin),
                                      static_cast<std::int32_t>(begin) + prompt.rope_delta);
            }
            if (!final_chunk) {
                if (prompt.has_media()) {
                    second.mtp_prefill_multimodal_prompt_chunk(
                        prompt, static_cast<std::uint32_t>(begin), count, false, 0);
                } else {
                    second.mtp_prefill_prompt_chunk(
                        prompt_tokens, static_cast<std::uint32_t>(begin), count, false, 0);
                }
            } else {
                final_begin = begin;
                final_count = count;
            }
            begin += count;
            stage1_inflight = true;
            if (!opt.prefill_pipeline) {
                second.synchronize();
                timing.prefill_stage1 +=
                    std::chrono::duration<double>(Clock::now() - phase).count();
                stage1_inflight = false;
            }
        }
        if (stage1_inflight) {
            const auto phase = Clock::now();
            second.synchronize();
            timing.prefill_stage1 +=
                std::chrono::duration<double>(Clock::now() - phase).count();
        }
        anchor = second.project_token();
        if (prompt.has_media()) {
            second.mtp_prefill_multimodal_prompt_chunk(
                prompt, static_cast<std::uint32_t>(final_begin), final_count, true, anchor);
        } else {
            second.mtp_prefill_prompt_chunk(prompt_tokens,
                                            static_cast<std::uint32_t>(final_begin),
                                            final_count, true, anchor);
        }
        const std::vector<TokenId> drafts = second.mtp_initial_drafts();
        if (drafts.size() != verify_width - 1U || drafts.size() > initial_drafts.size()) {
            throw std::logic_error("MTP prefill draft count disagrees with verify width");
        }
        std::copy(drafts.begin(), drafts.end(), initial_drafts.begin());
        initial_extent = static_cast<std::int32_t>(opt.mtp_depth);
        if (opt.prefix_snapshot_dir) {
            if (opt.mtp_depth != 3) {
                throw std::invalid_argument(
                    "creating a prefix snapshot currently requires --mtp-depth 3");
            }
            const std::array<TokenId, 3> snapshot_drafts{
                initial_drafts[0], initial_drafts[1], initial_drafts[2]};
            save_prefix_snapshot(opt, *opt.prefix_snapshot_dir, first, second, context,
                                 prompt, anchor, snapshot_drafts);
        }
    }
    prefill = std::chrono::duration<double>(Clock::now() - started).count();
    first.prepare_sm75_a8_decode_layout();
    second.prepare_sm75_a8_decode_layout();
    if (context && context->requires_prefix_preparation()) {
        const auto prepare_started = Clock::now();
        // Each stage owns distinct prefix layers, cache planes, streams and
        // preparation handles. Prepare both independently; join before decode
        // or propagating either error so no worker outlives its source state.
        run_joined_stage_tasks(
            [&] { first.prepare_gqa_prefix(*context, static_cast<std::uint32_t>(prompt_tokens.size())); },
            [&] { second.prepare_gqa_prefix(*context, static_cast<std::uint32_t>(prompt_tokens.size())); });
        const auto preparation = std::chrono::duration<double>(Clock::now() - prepare_started).count();
        prefill += preparation; // All representation construction is charged before decode.
        metric("attention prefix preparation", seconds(preparation));
    }

    BoundVocabHead head_binding(second);
    if (opt.parallel_vocab_head) {
        const auto head_started = Clock::now();
        head_binding.head = std::make_unique<ninfer::apps::dual::ParallelVocabHead>(first.device_id());
        second.set_head_execution_override(head_binding.head.get(),opt.parallel_greedy_head);
        const auto elapsed = std::chrono::duration<double>(Clock::now()-head_started).count();
        prefill += elapsed;
        metric("parallel vocabulary preparation",seconds(elapsed));
        const auto resources=head_binding.head->stats();
        metric("parallel vocabulary helper weights",std::to_string(resources.helper_weight_bytes));
        metric("parallel vocabulary device workspace",std::to_string(resources.owner_workspace_bytes+resources.helper_workspace_bytes));
    }

    constexpr TokenId kQwenImEnd = 248046;
    constexpr TokenId kQwenEndOfText = 248044;
    const auto is_default_stop = [&](TokenId token) {
        return qwen3_6::stop_at_message_end(opt.computation,prompt.has_media()) &&
               (token == kQwenImEnd || token == kQwenEndOfText);
    };
    bool stopped = is_default_stop(anchor);
    std::vector<TokenId> output;
    output.reserve(opt.max_new);
    output.push_back(anchor);
    Mtp3SessionState state{
        .anchor = anchor,
        .drafts = initial_drafts,
        .draft_extent = initial_extent,
        .frontier = static_cast<std::int32_t>(prompt_tokens.size()),
        .remaining_budget = static_cast<std::int32_t>(opt.max_new - 1U),
        .rope_delta = prompt.rope_delta,
    };
    ninfer::PinnedHostBuffer verify_boundary(first.verify_residual_bytes());
    std::size_t measured_rounds = 0, warm_initial_outputs = 0;
    Clock::time_point warm_started{};
    started = Clock::now();
    while (!stopped && can_run_mtp_round(state, opt.max_context)) {
        if (measured_rounds == 1) {
            warm_started = Clock::now();
            warm_initial_outputs = output.size();
        }
        ++measured_rounds;
        const Mtp3VerifyInput verify = second.mtp_prepare_verify(state);
        if (verify.valid_columns != mtp_verify_columns(state)) {
            throw std::logic_error("MTP verify extent disagrees with selected depth");
        }
        for (std::size_t column = 0;
             column < static_cast<std::size_t>(verify.valid_columns); ++column) {
            const auto expected = state.frontier + std::min<std::int32_t>(
                static_cast<std::int32_t>(column), state.draft_extent);
            if (verify.positions[column] != expected ||
                verify.rope_positions[column] != expected + state.rope_delta) {
                throw std::logic_error("MTP verify matrix disagrees with selected depth");
            }
        }
        auto phase = Clock::now();
        first.target_verify_tokens(verify);
        first.copy_verify_residual_to_host(verify_boundary.data(), first.verify_residual_bytes());
        timing.target_stage0 += std::chrono::duration<double>(Clock::now() - phase).count();

        phase = Clock::now();
        second.copy_verify_residual_from_host(verify_boundary.data(), second.verify_residual_bytes());
        (void)second.target_verify_hidden(verify);
        timing.target_stage1 += std::chrono::duration<double>(Clock::now() - phase).count();

        phase = Clock::now();
        Mtp3RoundResult round = second.mtp_accept_round(state);
        std::int32_t commit_count = round.licensed_count;
        for (std::int32_t index = 0; index < round.licensed_count; ++index) {
            if (is_default_stop(round.licensed_tokens[static_cast<std::size_t>(index)])) {
                commit_count = index + 1;
                stopped = true;
                break;
            }
        }
        if (!stopped) {
            round.next_drafts = second.mtp_propose_round(state, round, nullptr, 0);
        }
        timing.proposal += std::chrono::duration<double>(Clock::now() - phase).count();
        phase = Clock::now();
        first.fold_verify_records(commit_count);
        second.fold_verify_records(commit_count);
        first.synchronize();
        second.synchronize();
        timing.fold += std::chrono::duration<double>(Clock::now() - phase).count();

        for (std::int32_t index = 0; index < commit_count; ++index) {
            output.push_back(round.licensed_tokens[static_cast<std::size_t>(index)]);
        }
        Mtp3RoundResult visible_round = round;
        visible_round.licensed_count = commit_count;
        visible_round.accepted_drafts =
            std::min<std::int32_t>(round.accepted_drafts, commit_count);
        record_mtp_round(timing, 0, state.draft_extent, visible_round);
        if (stopped) { break; }
        state.anchor = round.next_anchor;
        state.drafts = round.next_drafts;
        // --mtp-depth is a fixed verifier budget. A smaller next_extent only means the
        // remaining output budget cannot support another full-width round; the loop must
        // fall back to its scalar tail rather than silently becoming an adaptive scheduler.
        state.frontier = round.next_frontier;
        state.remaining_budget -= round.licensed_count;
        promote_fixed_mtp_extent(state, round, opt.mtp_depth);
    }

    ninfer::PinnedHostBuffer scalar_boundary(TextShardStage::residual_bytes());
    while (!stopped && state.remaining_budget > 0) {
        const auto position = state.frontier;
        first.forward_token(state.anchor, position, position + state.rope_delta);
        first.copy_residual_to_host(scalar_boundary.data(), TextShardStage::residual_bytes());
        second.copy_residual_from_host(scalar_boundary.data(), TextShardStage::residual_bytes());
        second.forward_hidden(position, position + state.rope_delta);
        state.anchor = second.project_token();
        output.push_back(state.anchor);
        stopped = is_default_stop(state.anchor);
        timing.session_scalar_tail[0] += 1;
        state.frontier += 1;
        state.remaining_budget -= 1;
    }
    const auto finished = Clock::now();
    decode = std::chrono::duration<double>(finished - started).count();
    if (measured_rounds > 1 && std::getenv("NINFER_RECORD_STEADY_DECODE") != nullptr) {
        const double warm_seconds = std::chrono::duration<double>(finished - warm_started).count();
        const auto warm_tokens = output.size() - warm_initial_outputs;
        std::cerr << "WARM_DECODE excluded_rounds=1 generated_tokens=" << warm_tokens
                  << " elapsed_seconds=" << std::setprecision(9) << warm_seconds
                  << " tokens_per_second=" << warm_tokens / warm_seconds << '\n';
    }
    return output;
}



std::vector<std::vector<TokenId>>
dual_mtp3_two(const Options& opt, TextShardStage& first, TextShardStage& second,
              std::span<const TextPromptSequence> prompts, double& prefill, double& decode,
              DualMtpTiming& timing) {
    if (opt.sessions != 2 || prompts.size() != 2 || first.verify_width() != 4 ||
        second.verify_width() != 4) {
        throw std::invalid_argument("dual MTP3 pipeline currently requires B2");
    }
    std::array<ninfer::PinnedHostBuffer, 2> prefill_boundaries{
        ninfer::PinnedHostBuffer(TextShardStage::residual_bytes(opt.prefill_chunk)),
        ninfer::PinnedHostBuffer(TextShardStage::residual_bytes(opt.prefill_chunk))};
    std::vector<std::vector<TokenId>> output(2);
    std::array<Mtp3SessionState, 2> states{};
    auto started = Clock::now();
    for (std::uint32_t session = 0; session < 2; ++session) {
        const TextPromptSequence& prompt = prompts[session];
        const std::span<const TokenId> prompt_tokens(prompt.token_ids);
        std::size_t final_begin = 0;
        std::size_t final_count = 0;
        for (std::size_t begin = 0; begin < prompt_tokens.size(); begin += opt.prefill_chunk) {
            const std::size_t count =
                std::min<std::size_t>(opt.prefill_chunk, prompt_tokens.size() - begin);
            const bool final_chunk = begin + count == prompt_tokens.size();
            auto phase = Clock::now();
            first.prefill_tokens(prompt_tokens.subspan(begin, count),
                                 static_cast<std::int32_t>(begin),
                                 static_cast<std::int32_t>(begin) + prompt.rope_delta, session);
            first.copy_prefill_residual_to_host(prefill_boundaries[session].data(), count);
            timing.prefill_stage0 += std::chrono::duration<double>(Clock::now() - phase).count();
            phase = Clock::now();
            second.copy_prefill_residual_from_host(prefill_boundaries[session].data(), count);
            second.prefill_hidden(count, static_cast<std::int32_t>(begin),
                                  static_cast<std::int32_t>(begin) + prompt.rope_delta, session);
            if (opt.mtp_proposal_stage0) {
                second.copy_prefill_residual_to_host(prefill_boundaries[session].data(), count);
                first.copy_prefill_residual_from_host(prefill_boundaries[session].data(), count);
                if (!final_chunk) {
                    first.mtp_prefill_prompt_chunk(prompt_tokens,
                                                   static_cast<std::uint32_t>(begin), count,
                                                   false, 0, session);
                }
                first.synchronize();
            } else if (!final_chunk) {
                second.mtp_prefill_prompt_chunk(prompt_tokens,
                                                static_cast<std::uint32_t>(begin), count,
                                                false, 0, session);
            }
            if (final_chunk) {
                final_begin = begin;
                final_count = count;
            }
            second.synchronize();
            timing.prefill_stage1 += std::chrono::duration<double>(Clock::now() - phase).count();
        }
        const TokenId anchor = second.project_token(session);
        TextShardStage& proposal_stage = opt.mtp_proposal_stage0 ? first : second;
        proposal_stage.mtp_prefill_prompt_chunk(prompt_tokens,
                                                static_cast<std::uint32_t>(final_begin),
                                                final_count, true, anchor, session);
        const std::vector<TokenId> drafts = proposal_stage.mtp_initial_drafts(session);
        if (drafts.size() != 3) { throw std::logic_error("MTP3 B2 prefill draft count failed"); }
        output[session].reserve(opt.max_new);
        output[session].push_back(anchor);
        states[session] = {
            .anchor = anchor,
            .drafts = {drafts[0], drafts[1], drafts[2]},
            .draft_extent = static_cast<std::int32_t>(opt.mtp_depth),
            .frontier = static_cast<std::int32_t>(prompt_tokens.size()),
            .remaining_budget = static_cast<std::int32_t>(opt.max_new - 1U),
            .rope_delta = prompt.rope_delta,
        };
    }
    prefill = std::chrono::duration<double>(Clock::now() - started).count();
    first.prepare_sm75_a8_decode_layout();
    second.prepare_sm75_a8_decode_layout();

    std::array<Mtp3VerifyInput, 2> verify{};
    std::array<ninfer::PinnedHostBuffer, 2> boundaries{
        ninfer::PinnedHostBuffer(first.verify_residual_bytes()),
        ninfer::PinnedHostBuffer(first.verify_residual_bytes())};
    std::array<ninfer::PinnedHostBuffer, 2> target_hidden{
        ninfer::PinnedHostBuffer(second.verify_residual_bytes()),
        ninfer::PinnedHostBuffer(second.verify_residual_bytes())};
    const auto full_round = [&](std::uint32_t session) {
        return can_run_mtp_round(states[session], opt.max_context);
    };
    const auto run_stage0 = [&](std::uint32_t session) {
        auto phase = Clock::now();
        first.target_verify_tokens(verify[session], session);
        first.copy_verify_residual_to_host(boundaries[session].data(),
                                           first.verify_residual_bytes(), session);
        timing.target_stage0 += std::chrono::duration<double>(Clock::now() - phase).count();
    };
    const auto launch_stage1 = [&](std::uint32_t session) {
        second.copy_verify_residual_from_host(boundaries[session].data(),
                                              second.verify_residual_bytes(), session);
        second.launch_target_verify_hidden(verify[session], session);
    };
    const auto commit_round = [&](std::uint32_t session, Mtp3RoundResult& round,
                                  bool update_drafts) {
        auto phase = Clock::now();
        first.fold_verify_records(round.licensed_count, session);
        second.fold_verify_records(round.licensed_count, session);
        timing.fold += std::chrono::duration<double>(Clock::now() - phase).count();
        for (std::int32_t index = 0; index < round.licensed_count; ++index) {
            output[session].push_back(round.licensed_tokens[static_cast<std::size_t>(index)]);
        }
        record_mtp_round(timing, session, states[session].draft_extent, round);
        states[session].anchor = round.next_anchor;
        if (update_drafts) { states[session].drafts = round.next_drafts; }
        states[session].frontier = round.next_frontier;
        states[session].remaining_budget -= round.licensed_count;
        promote_fixed_mtp_extent(states[session], round, opt.mtp_depth);
    };

    started = Clock::now();
    if (!opt.mtp_proposal_stage0) {
        const auto prepare_verify = [&](std::uint32_t session) {
            verify[session] = second.mtp_prepare_verify(states[session], session);
        };
        const auto finish_session = [&](std::uint32_t session) {
            auto phase = Clock::now();
            Mtp3RoundResult round = second.mtp_finish_round(states[session], session);
            timing.proposal += std::chrono::duration<double>(Clock::now() - phase).count();
            commit_round(session, round, true);
        };
        if (full_round(0) && full_round(1)) {
            prepare_verify(0);
            prepare_verify(1);
            run_stage0(0);
        }
        while (full_round(0) && full_round(1)) {
            auto phase = Clock::now();
            launch_stage1(0);
            run_stage0(1);
            (void)second.collect_verify_tokens(0);
            timing.target_stage1 += std::chrono::duration<double>(Clock::now() - phase).count();
            finish_session(0);
            if (full_round(0)) { prepare_verify(0); }

            phase = Clock::now();
            launch_stage1(1);
            if (full_round(0)) { run_stage0(0); }
            (void)second.collect_verify_tokens(1);
            timing.target_stage1 += std::chrono::duration<double>(Clock::now() - phase).count();
            finish_session(1);
            if (full_round(1)) { prepare_verify(1); }
        }
    } else {
        struct PendingRound {
            bool valid = false;
            Mtp3RoundResult round{};
        };
        std::array<PendingRound, 2> pending{};
        std::array<bool, 2> boundary_ready{false, false};
        std::array<bool, 2> stage1_prepared{false, false};
        const auto prepare_stage0 = [&](std::uint32_t session) {
            verify[session] = first.mtp_prepare_verify(states[session], session);
        };
        const auto prepare_stage1 = [&](std::uint32_t session) {
            const Mtp3VerifyInput stage1_verify =
                second.mtp_prepare_verify(states[session], session);
            if (stage1_verify.ids != verify[session].ids ||
                stage1_verify.positions != verify[session].positions ||
                stage1_verify.rope_positions != verify[session].rope_positions ||
                stage1_verify.valid_columns != verify[session].valid_columns) {
                throw std::runtime_error(
                    "overlapped MTP3 round preparation disagrees across shards");
            }
            verify[session] = stage1_verify;
            stage1_prepared[session] = true;
        };
        const auto accept_stage1 = [&](std::uint32_t session) {
            second.copy_verify_hidden_to_host_async(target_hidden[session].data(),
                                                    second.verify_residual_bytes(), session);
            auto phase = Clock::now();
            pending[session].round = second.mtp_accept_round(states[session], session);
            timing.proposal += std::chrono::duration<double>(Clock::now() - phase).count();
            pending[session].valid = true;
            boundary_ready[session] = false;
            stage1_prepared[session] = false;
        };
        const auto propose_commit = [&](std::uint32_t session) {
            if (!pending[session].valid) {
                throw std::logic_error("overlapped MTP3 proposal has no accepted round");
            }
            auto phase = Clock::now();
            pending[session].round.next_drafts = first.mtp_propose_round(
                states[session], pending[session].round, target_hidden[session].data(),
                second.verify_residual_bytes(), session);
            timing.proposal += std::chrono::duration<double>(Clock::now() - phase).count();
            commit_round(session, pending[session].round, true);
            pending[session].valid = false;
        };

        if (full_round(0) && full_round(1)) {
            prepare_stage0(0);
            prepare_stage1(0);
            prepare_stage0(1);
            prepare_stage1(1);
            run_stage0(0);
            boundary_ready[0] = true;
        }
        std::uint32_t current = 0;
        while (full_round(current) && boundary_ready[current]) {
            const std::uint32_t other = 1U - current;
            auto phase = Clock::now();
            launch_stage1(current);
            if (pending[other].valid) {
                propose_commit(other);
                if (full_round(other)) {
                    prepare_stage0(other);
                    run_stage0(other);
                    boundary_ready[other] = true;
                    stage1_prepared[other] = false;
                }
            } else if (!boundary_ready[other] && full_round(other)) {
                run_stage0(other);
                boundary_ready[other] = true;
            }
            accept_stage1(current);
            timing.target_stage1 += std::chrono::duration<double>(Clock::now() - phase).count();
            if (boundary_ready[other] && !stage1_prepared[other]) {
                prepare_stage1(other);
            }
            current = other;
        }
        for (std::uint32_t session = 0; session < 2; ++session) {
            if (pending[session].valid) {
                commit_round(session, pending[session].round, false);
                pending[session].valid = false;
            }
        }
    }
    first.synchronize();
    second.synchronize();

    std::array<ninfer::PinnedHostBuffer, 2> scalar_boundaries{
        ninfer::PinnedHostBuffer(TextShardStage::residual_bytes()),
        ninfer::PinnedHostBuffer(TextShardStage::residual_bytes())};
    for (std::uint32_t session = 0; session < 2; ++session) {
        while (states[session].remaining_budget > 0) {
            const auto position = states[session].frontier;
            first.forward_token(states[session].anchor, position,
                                position + states[session].rope_delta, session);
            first.copy_residual_to_host(scalar_boundaries[session].data(),
                                        TextShardStage::residual_bytes(), session);
            second.copy_residual_from_host(scalar_boundaries[session].data(),
                                           TextShardStage::residual_bytes(), session);
            second.forward_hidden(position, position + states[session].rope_delta, session);
            states[session].anchor = second.project_token(session);
            output[session].push_back(states[session].anchor);
            timing.session_scalar_tail[session] += 1;
            states[session].frontier += 1;
            states[session].remaining_budget -= 1;
        }
    }
    decode = std::chrono::duration<double>(Clock::now() - started).count();
    return output;
}

std::vector<std::vector<TokenId>>
dual_mtp3_two_batch(const Options& opt, TextShardStage& first, TextShardStage& second,
              std::span<const TextPromptSequence> prompts, double& prefill, double& decode,
              DualMtpTiming& timing) {
    if (opt.sessions != 2 || prompts.size() != 2 || first.verify_width() != 4 ||
        second.verify_width() != 4) {
        throw std::invalid_argument("dual MTP3 multi-session path currently requires B2");
    }
    std::array<ninfer::PinnedHostBuffer, 2> prefill_boundaries{
        ninfer::PinnedHostBuffer(TextShardStage::residual_bytes(opt.prefill_chunk)),
        ninfer::PinnedHostBuffer(TextShardStage::residual_bytes(opt.prefill_chunk))};
    std::vector<std::vector<TokenId>> output(2);
    std::array<Mtp3SessionState, 2> states{};
    auto started = Clock::now();
    for (std::uint32_t session = 0; session < 2; ++session) {
        const TextPromptSequence& prompt = prompts[session];
        const std::span<const TokenId> prompt_tokens(prompt.token_ids);
        std::size_t final_begin = 0;
        std::size_t final_count = 0;
        for (std::size_t begin = 0; begin < prompt_tokens.size(); begin += opt.prefill_chunk) {
            const std::size_t count =
                std::min<std::size_t>(opt.prefill_chunk, prompt_tokens.size() - begin);
            const bool final_chunk = begin + count == prompt_tokens.size();
            auto phase = Clock::now();
            first.prefill_tokens(prompt_tokens.subspan(begin, count),
                                 static_cast<std::int32_t>(begin),
                                 static_cast<std::int32_t>(begin) + prompt.rope_delta, session);
            first.copy_prefill_residual_to_host(prefill_boundaries[session].data(), count);
            timing.prefill_stage0 += std::chrono::duration<double>(Clock::now() - phase).count();
            phase = Clock::now();
            second.copy_prefill_residual_from_host(prefill_boundaries[session].data(), count);
            second.prefill_hidden(count, static_cast<std::int32_t>(begin),
                                  static_cast<std::int32_t>(begin) + prompt.rope_delta, session);
            if (!final_chunk) {
                second.mtp_prefill_prompt_chunk(prompt_tokens, static_cast<std::uint32_t>(begin),
                                                count, false, 0, session);
            } else {
                final_begin = begin;
                final_count = count;
            }
            second.synchronize();
            timing.prefill_stage1 += std::chrono::duration<double>(Clock::now() - phase).count();
        }
        const TokenId anchor = second.project_token(session);
        second.mtp_prefill_prompt_chunk(prompt_tokens, static_cast<std::uint32_t>(final_begin),
                                        final_count, true, anchor, session);
        const std::vector<TokenId> drafts = second.mtp_initial_drafts(session);
        if (drafts.size() != 3) { throw std::logic_error("MTP3 B2 prefill draft count failed"); }
        output[session].reserve(opt.max_new);
        output[session].push_back(anchor);
        states[session] = {
            .anchor = anchor,
            .drafts = {drafts[0], drafts[1], drafts[2]},
            .draft_extent = static_cast<std::int32_t>(opt.mtp_depth),
            .frontier = static_cast<std::int32_t>(prompt_tokens.size()),
            .remaining_budget = static_cast<std::int32_t>(opt.max_new - 1U),
            .rope_delta = prompt.rope_delta,
        };
    }
    prefill = std::chrono::duration<double>(Clock::now() - started).count();
    first.prepare_sm75_a8_decode_layout();
    second.prepare_sm75_a8_decode_layout();

    const std::size_t boundary_bytes = first.verify_residual_bytes() * 2U;
    ninfer::PinnedHostBuffer verify_boundary(boundary_bytes);
    started = Clock::now();
    while (can_run_mtp_round(states[0], opt.max_context) &&
           can_run_mtp_round(states[1], opt.max_context)) {
        std::array<Mtp3VerifyInput, 2> verify{
            second.mtp_prepare_verify(states[0], 0),
            second.mtp_prepare_verify(states[1], 1),
        };
        auto phase = Clock::now();
        first.target_verify_tokens_batch(verify);
        first.copy_verify_residual_batch_to_host(verify_boundary.data(), boundary_bytes);
        timing.target_stage0 += std::chrono::duration<double>(Clock::now() - phase).count();
        phase = Clock::now();
        second.copy_verify_residual_batch_from_host(verify_boundary.data(), boundary_bytes);
        (void)second.target_verify_hidden_batch(verify);
        timing.target_stage1 += std::chrono::duration<double>(Clock::now() - phase).count();

        std::array<Mtp3RoundResult, 2> rounds;
        phase = Clock::now();
        rounds[0] = second.mtp_finish_round(states[0], 0);
        rounds[1] = second.mtp_finish_round(states[1], 1);
        timing.proposal += std::chrono::duration<double>(Clock::now() - phase).count();
        const std::array<std::int32_t, 2> commits{
            rounds[0].licensed_count, rounds[1].licensed_count};
        phase = Clock::now();
        first.fold_verify_records_batch(commits);
        second.fold_verify_records_batch(commits);
        first.synchronize();
        second.synchronize();
        timing.fold += std::chrono::duration<double>(Clock::now() - phase).count();

        for (std::uint32_t session = 0; session < 2; ++session) {
            const Mtp3RoundResult& round = rounds[session];
            for (std::int32_t index = 0; index < round.licensed_count; ++index) {
                output[session].push_back(round.licensed_tokens[static_cast<std::size_t>(index)]);
            }
            record_mtp_round(timing, session, states[session].draft_extent, round);
            states[session].anchor = round.next_anchor;
            states[session].drafts = round.next_drafts;
            states[session].frontier = round.next_frontier;
            states[session].remaining_budget -= round.licensed_count;
            promote_fixed_mtp_extent(states[session], round, opt.mtp_depth);
        }
    }

    std::array<ninfer::PinnedHostBuffer, 2> scalar_boundaries{
        ninfer::PinnedHostBuffer(TextShardStage::residual_bytes()),
        ninfer::PinnedHostBuffer(TextShardStage::residual_bytes())};
    for (std::uint32_t session = 0; session < 2; ++session) {
        while (states[session].remaining_budget > 0) {
            const auto position = states[session].frontier;
            first.forward_token(states[session].anchor, position,
                                position + states[session].rope_delta, session);
            first.copy_residual_to_host(scalar_boundaries[session].data(),
                                        TextShardStage::residual_bytes(), session);
            second.copy_residual_from_host(scalar_boundaries[session].data(),
                                           TextShardStage::residual_bytes(), session);
            second.forward_hidden(position, position + states[session].rope_delta, session);
            states[session].anchor = second.project_token(session);
            output[session].push_back(states[session].anchor);
            timing.session_scalar_tail[session] += 1;
            states[session].frontier += 1;
            states[session].remaining_budget -= 1;
        }
    }
    decode = std::chrono::duration<double>(Clock::now() - started).count();
    return output;
}

std::vector<std::vector<TokenId>>
dual_mtp3_four_batch_group(const Options& opt, TextShardStage& first, TextShardStage& second,
                     std::span<const TextPromptSequence> prompts, double& prefill,
                     double& decode, DualMtpTiming& timing, std::uint32_t session_base) {
    constexpr std::uint32_t Sessions = 4;
    constexpr std::uint32_t PairSize = 2;
    if (prompts.size() != Sessions || session_base + Sessions > opt.sessions ||
        first.verify_width() != 4 || second.verify_width() != 4) {
        throw std::invalid_argument("dual MTP3 C4 group requires in-range sessions and T4 verification");
    }
    const auto absolute_session = [&](std::uint32_t local) { return session_base + local; };

    std::array<ninfer::PinnedHostBuffer, Sessions> prefill_boundaries{
        ninfer::PinnedHostBuffer(TextShardStage::residual_bytes(opt.prefill_chunk)),
        ninfer::PinnedHostBuffer(TextShardStage::residual_bytes(opt.prefill_chunk)),
        ninfer::PinnedHostBuffer(TextShardStage::residual_bytes(opt.prefill_chunk)),
        ninfer::PinnedHostBuffer(TextShardStage::residual_bytes(opt.prefill_chunk))};
    std::vector<std::vector<TokenId>> output(Sessions);
    std::array<Mtp3SessionState, Sessions> states{};
    auto started = Clock::now();
    for (std::uint32_t session = 0; session < Sessions; ++session) {
        const TextPromptSequence& prompt = prompts[session];
        const std::span<const TokenId> prompt_tokens(prompt.token_ids);
        std::size_t final_begin = 0;
        std::size_t final_count = 0;
        for (std::size_t begin = 0; begin < prompt_tokens.size(); begin += opt.prefill_chunk) {
            const std::size_t count =
                std::min<std::size_t>(opt.prefill_chunk, prompt_tokens.size() - begin);
            const bool final_chunk = begin + count == prompt_tokens.size();
            auto phase = Clock::now();
            first.prefill_tokens(prompt_tokens.subspan(begin, count),
                                 static_cast<std::int32_t>(begin),
                                 static_cast<std::int32_t>(begin) + prompt.rope_delta, absolute_session(session));
            first.copy_prefill_residual_to_host(prefill_boundaries[session].data(), count);
            timing.prefill_stage0 += std::chrono::duration<double>(Clock::now() - phase).count();
            phase = Clock::now();
            second.copy_prefill_residual_from_host(prefill_boundaries[session].data(), count);
            second.prefill_hidden(count, static_cast<std::int32_t>(begin),
                                  static_cast<std::int32_t>(begin) + prompt.rope_delta, absolute_session(session));
            if (!final_chunk) {
                second.mtp_prefill_prompt_chunk(prompt_tokens, static_cast<std::uint32_t>(begin),
                                                count, false, 0, absolute_session(session));
            } else {
                final_begin = begin;
                final_count = count;
            }
            second.synchronize();
            timing.prefill_stage1 += std::chrono::duration<double>(Clock::now() - phase).count();
        }
        const TokenId anchor = second.project_token(absolute_session(session));
        second.mtp_prefill_prompt_chunk(prompt_tokens, static_cast<std::uint32_t>(final_begin),
                                        final_count, true, anchor, absolute_session(session));
        const std::vector<TokenId> drafts = second.mtp_initial_drafts(absolute_session(session));
        if (drafts.size() != 3) { throw std::logic_error("MTP3 C4 prefill draft count failed"); }
        output[session].reserve(opt.max_new);
        output[session].push_back(anchor);
        states[session] = {
            .anchor = anchor,
            .drafts = {drafts[0], drafts[1], drafts[2]},
            .draft_extent = static_cast<std::int32_t>(opt.mtp_depth),
            .frontier = static_cast<std::int32_t>(prompt_tokens.size()),
            .remaining_budget = static_cast<std::int32_t>(opt.max_new - 1U),
            .rope_delta = prompt.rope_delta,
        };
    }
    prefill = std::chrono::duration<double>(Clock::now() - started).count();
    first.prepare_sm75_a8_decode_layout();
    second.prepare_sm75_a8_decode_layout();

    const std::size_t pair_boundary_bytes = first.verify_residual_bytes() * PairSize;
    std::array<ninfer::PinnedHostBuffer, 2> verify_boundaries{
        ninfer::PinnedHostBuffer(pair_boundary_bytes),
        ninfer::PinnedHostBuffer(pair_boundary_bytes)};

    const auto can_verify = [&]() {
        return std::all_of(states.begin(), states.end(), [&](const Mtp3SessionState& state) {
            return can_run_mtp_round(state, opt.max_context);
        });
    };
    const auto can_verify_pair = [&](std::uint32_t session_begin) {
        for (std::uint32_t local = 0; local < PairSize; ++local) {
            if (!can_run_mtp_round(states[session_begin + local], opt.max_context)) {
                return false;
            }
        }
        return true;
    };
    const auto pair_guarantees_next = [&](std::uint32_t session_begin) {
        for (std::uint32_t local = 0; local < PairSize; ++local) {
            const Mtp3SessionState& state = states[session_begin + local];
            const std::int32_t columns = mtp_verify_columns(state);
            const std::uint32_t two_rounds = static_cast<std::uint32_t>(2 * columns);
            if (state.remaining_budget < static_cast<std::int32_t>(two_rounds) ||
                static_cast<std::uint64_t>(state.frontier) + two_rounds > opt.max_context) {
                return false;
            }
        }
        return true;
    };
    bool pair0_preloaded = false;
    std::array<Mtp3VerifyInput, PairSize> preloaded_pair0{};
    const auto commit_session = [&](std::uint32_t session,
                                    const Mtp3RoundResult& round) {
        for (std::int32_t index = 0; index < round.licensed_count; ++index) {
            output[session].push_back(round.licensed_tokens[static_cast<std::size_t>(index)]);
        }
        record_mtp_round(timing, absolute_session(session), states[session].draft_extent, round);
        states[session].anchor = round.next_anchor;
        states[session].drafts = round.next_drafts;
        states[session].frontier = round.next_frontier;
        states[session].remaining_budget -= round.licensed_count;
        promote_fixed_mtp_extent(states[session], round, opt.mtp_depth);
    };
    const auto commit_pair = [&](std::uint32_t session_begin,
                                 const std::array<Mtp3RoundResult, PairSize>& rounds) {
        for (std::uint32_t local = 0; local < PairSize; ++local) {
            commit_session(session_begin + local, rounds[local]);
        }
    };

    started = Clock::now();
    while (can_verify()) {
        std::array<Mtp3VerifyInput, Sessions> verify{};
        const bool pair0_was_preloaded = pair0_preloaded;
        pair0_preloaded = false;
        if (pair0_was_preloaded) {
            verify[0] = preloaded_pair0[0];
            verify[1] = preloaded_pair0[1];
        } else {
            verify[0] = second.mtp_prepare_verify(states[0], absolute_session(0));
            verify[1] = second.mtp_prepare_verify(states[1], absolute_session(1));
        }
        verify[2] = second.mtp_prepare_verify(states[2], absolute_session(2));
        verify[3] = second.mtp_prepare_verify(states[3], absolute_session(3));
        std::array<Mtp3RoundResult, PairSize> rounds0{};
        std::array<Mtp3RoundResult, PairSize> rounds1{};
        double target1_pair0 = 0.0;
        double target1_pair1 = 0.0;
        double proposal_pair0 = 0.0;
        double proposal_pair1 = 0.0;
        std::exception_ptr pair0_error;
        std::exception_ptr pair1_error;

        auto phase = Clock::now();
        if (!pair0_was_preloaded) {
            first.target_verify_tokens_batch(
                std::span<const Mtp3VerifyInput>(verify.data(), PairSize), session_base);
            first.copy_verify_residual_batch_to_host(verify_boundaries[0].data(),
                                                     pair_boundary_bytes, session_base);
            timing.target_stage0 +=
                std::chrono::duration<double>(Clock::now() - phase).count();
        }

        std::thread pair0_worker([&]() {
            try {
                auto local = Clock::now();
                second.copy_verify_residual_batch_from_host(verify_boundaries[0].data(),
                                                            pair_boundary_bytes, session_base);
                (void)second.target_verify_hidden_batch(
                    std::span<const Mtp3VerifyInput>(verify.data(), PairSize), session_base);
                target1_pair0 = std::chrono::duration<double>(Clock::now() - local).count();
                local = Clock::now();
                rounds0[0] = second.mtp_finish_round(states[0], absolute_session(0));
                rounds0[1] = second.mtp_finish_round(states[1], absolute_session(1));
                proposal_pair0 = std::chrono::duration<double>(Clock::now() - local).count();
            } catch (...) {
                pair0_error = std::current_exception();
            }
        });

        phase = Clock::now();
        first.target_verify_tokens_batch(
            std::span<const Mtp3VerifyInput>(verify.data() + PairSize, PairSize), session_base + PairSize);
        first.copy_verify_residual_batch_to_host(verify_boundaries[1].data(),
                                                 pair_boundary_bytes, session_base + PairSize);
        timing.target_stage0 += std::chrono::duration<double>(Clock::now() - phase).count();
        pair0_worker.join();
        if (pair0_error) { std::rethrow_exception(pair0_error); }
        timing.target_stage1 += target1_pair0;
        timing.proposal += proposal_pair0;
        commit_pair(0, rounds0);

        const std::array<std::int32_t, PairSize> commits0{
            rounds0[0].licensed_count, rounds0[1].licensed_count};
        const bool preload_next_pair0 =
            can_verify_pair(0) && pair_guarantees_next(PairSize);
        std::array<Mtp3VerifyInput, PairSize> next_pair0{};
        if (preload_next_pair0) {
            next_pair0[0] = second.mtp_prepare_verify(states[0], absolute_session(0));
            next_pair0[1] = second.mtp_prepare_verify(states[1], absolute_session(1));
        }
        std::thread pair1_worker([&]() {
            try {
                auto local = Clock::now();
                second.copy_verify_residual_batch_from_host(verify_boundaries[1].data(),
                                                            pair_boundary_bytes, session_base + PairSize);
                (void)second.target_verify_hidden_batch(
                    std::span<const Mtp3VerifyInput>(verify.data() + PairSize, PairSize), session_base + PairSize);
                target1_pair1 = std::chrono::duration<double>(Clock::now() - local).count();
                local = Clock::now();
                rounds1[0] = second.mtp_finish_round(states[2], absolute_session(2));
                rounds1[1] = second.mtp_finish_round(states[3], absolute_session(3));
                proposal_pair1 = std::chrono::duration<double>(Clock::now() - local).count();
            } catch (...) {
                pair1_error = std::current_exception();
            }
        });

        auto fold_phase = Clock::now();
        first.fold_verify_records_batch(commits0, session_base);
        first.synchronize();
        timing.fold += std::chrono::duration<double>(Clock::now() - fold_phase).count();
        if (preload_next_pair0) {
            phase = Clock::now();
            first.target_verify_tokens_batch(
                std::span<const Mtp3VerifyInput>(next_pair0.data(), PairSize), session_base);
            first.copy_verify_residual_batch_to_host(verify_boundaries[0].data(),
                                                     pair_boundary_bytes, session_base);
            timing.target_stage0 +=
                std::chrono::duration<double>(Clock::now() - phase).count();
            preloaded_pair0 = next_pair0;
            pair0_preloaded = true;
        }
        pair1_worker.join();
        if (pair1_error) { std::rethrow_exception(pair1_error); }
        timing.target_stage1 += target1_pair1;
        timing.proposal += proposal_pair1;

        const std::array<std::int32_t, PairSize> commits1{
            rounds1[0].licensed_count, rounds1[1].licensed_count};
        fold_phase = Clock::now();
        std::exception_ptr stage1_fold_error;
        std::thread stage1_fold([&]() {
            try {
                second.fold_verify_records_batch(commits0, session_base);
                second.fold_verify_records_batch(commits1, session_base + PairSize);
                second.synchronize();
            } catch (...) {
                stage1_fold_error = std::current_exception();
            }
        });
        first.fold_verify_records_batch(commits1, session_base + PairSize);
        first.synchronize();
        stage1_fold.join();
        if (stage1_fold_error) { std::rethrow_exception(stage1_fold_error); }
        timing.fold += std::chrono::duration<double>(Clock::now() - fold_phase).count();

        commit_pair(PairSize, rounds1);
    }
    if (pair0_preloaded) {
        throw std::logic_error("C4 scheduler exited with an unconsumed preloaded verifier round");
    }

    const auto drain_pair = [&](std::uint32_t session_begin, std::size_t boundary_index) {
        while (true) {
            std::array<bool, PairSize> active{};
            int donor = -1;
            for (std::uint32_t local = 0; local < PairSize; ++local) {
                const std::uint32_t session = session_begin + local;
                active[local] = can_run_mtp_round(states[session], opt.max_context);
                if (active[local] && donor < 0) { donor = static_cast<int>(local); }
            }
            if (donor < 0) { break; }

            std::array<Mtp3VerifyInput, PairSize> verify{};
            for (std::uint32_t local = 0; local < PairSize; ++local) {
                if (active[local]) {
                    verify[local] = second.mtp_prepare_verify(
                        states[session_begin + local],
                        absolute_session(session_begin + local));
                }
            }
            for (std::uint32_t local = 0; local < PairSize; ++local) {
                if (!active[local]) {
                    verify[local] = verify[static_cast<std::size_t>(donor)];
                    verify[local].valid_columns = 0;
                }
            }

            auto phase = Clock::now();
            first.target_verify_tokens_batch(verify, absolute_session(session_begin));
            first.copy_verify_residual_batch_to_host(verify_boundaries[boundary_index].data(),
                                                     pair_boundary_bytes, absolute_session(session_begin));
            timing.target_stage0 +=
                std::chrono::duration<double>(Clock::now() - phase).count();

            phase = Clock::now();
            second.copy_verify_residual_batch_from_host(verify_boundaries[boundary_index].data(),
                                                        pair_boundary_bytes, absolute_session(session_begin));
            (void)second.target_verify_hidden_batch(verify, absolute_session(session_begin));
            timing.target_stage1 +=
                std::chrono::duration<double>(Clock::now() - phase).count();

            std::array<Mtp3RoundResult, PairSize> rounds{};
            std::array<std::int32_t, PairSize> commits{};
            phase = Clock::now();
            for (std::uint32_t local = 0; local < PairSize; ++local) {
                if (!active[local]) { continue; }
                const std::uint32_t session = session_begin + local;
                rounds[local] = second.mtp_finish_round(states[session], absolute_session(session));
                commits[local] = rounds[local].licensed_count;
            }
            timing.proposal += std::chrono::duration<double>(Clock::now() - phase).count();

            phase = Clock::now();
            first.fold_verify_records_batch(commits, absolute_session(session_begin));
            second.fold_verify_records_batch(commits, absolute_session(session_begin));
            first.synchronize();
            second.synchronize();
            timing.fold += std::chrono::duration<double>(Clock::now() - phase).count();

            for (std::uint32_t local = 0; local < PairSize; ++local) {
                if (active[local]) {
                    commit_session(session_begin + local, rounds[local]);
                }
            }
        }
    };
    drain_pair(0, 0);
    drain_pair(PairSize, 1);

    std::array<ninfer::PinnedHostBuffer, Sessions> scalar_boundaries{
        ninfer::PinnedHostBuffer(TextShardStage::residual_bytes()),
        ninfer::PinnedHostBuffer(TextShardStage::residual_bytes()),
        ninfer::PinnedHostBuffer(TextShardStage::residual_bytes()),
        ninfer::PinnedHostBuffer(TextShardStage::residual_bytes())};
    for (std::uint32_t session = 0; session < Sessions; ++session) {
        while (states[session].remaining_budget > 0) {
            const auto position = states[session].frontier;
            first.forward_token(states[session].anchor, position,
                                position + states[session].rope_delta, absolute_session(session));
            first.copy_residual_to_host(scalar_boundaries[session].data(),
                                        TextShardStage::residual_bytes(), absolute_session(session));
            second.copy_residual_from_host(scalar_boundaries[session].data(),
                                           TextShardStage::residual_bytes(), absolute_session(session));
            second.forward_hidden(position, position + states[session].rope_delta, absolute_session(session));
            states[session].anchor = second.project_token(absolute_session(session));
            output[session].push_back(states[session].anchor);
            timing.session_scalar_tail[absolute_session(session)] += 1;
            states[session].frontier += 1;
            states[session].remaining_budget -= 1;
        }
    }
    decode = std::chrono::duration<double>(Clock::now() - started).count();
    return output;
}


std::vector<std::vector<TokenId>>
dual_mtp3_four_batch(const Options& opt, TextShardStage& first, TextShardStage& second,
                     std::span<const TextPromptSequence> prompts, double& prefill,
                     double& decode, DualMtpTiming& timing) {
    if (opt.sessions != 4 || prompts.size() != 4) {
        throw std::invalid_argument("dual MTP3 C4 path requires four sessions");
    }
    return dual_mtp3_four_batch_group(
        opt, first, second, prompts, prefill, decode, timing, 0);
}

std::vector<std::vector<TokenId>>
dual_mtp3_eight_batch(const Options& opt, TextShardStage& first, TextShardStage& second,
                      std::span<const TextPromptSequence> prompts, double& prefill,
                      double& decode, DualMtpTiming& timing) {
    if (opt.sessions != 8 || prompts.size() != 8 || first.verify_width() != 4 ||
        second.verify_width() != 4) {
        throw std::invalid_argument("dual MTP3 C8 path requires eight sessions and T4 verification");
    }
    double p0=0.0,d0=0.0,p1=0.0,d1=0.0;
    auto out0=dual_mtp3_four_batch_group(opt,first,second,prompts.first<4>(),p0,d0,timing,0);
    auto out1=dual_mtp3_four_batch_group(opt,first,second,prompts.subspan(4,4),p1,d1,timing,4);
    prefill=p0+p1; decode=d0+d1;
    out0.insert(out0.end(),std::make_move_iterator(out1.begin()),
                std::make_move_iterator(out1.end()));
    return out0;
}

struct Bf16Comparison {
    std::size_t bit_mismatches = 0;
    double max_abs             = 0.0;
    double max_rel             = 0.0;
};

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
    float value = 0.0F;
    static_assert(sizeof(value) == sizeof(word));
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

Bf16Comparison compare_bf16(std::span<const std::uint16_t> lhs,
                            std::span<const std::uint16_t> rhs) {
    if (lhs.size() != rhs.size()) { throw std::invalid_argument("BF16 comparison size mismatch"); }
    Bf16Comparison out;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) ++out.bit_mismatches;
        const double a = bf16_to_float(lhs[i]);
        const double b = bf16_to_float(rhs[i]);
        const double error = std::abs(a - b);
        const double scale = std::max({1.0, std::abs(a), std::abs(b)});
        out.max_abs = std::max(out.max_abs, error);
        out.max_rel = std::max(out.max_rel, error / scale);
    }
    return out;
}

TokenId prefill_oracle_full(TextShardStage& stage, const TextPromptSequence& prompt,
                            std::uint32_t prefill_chunk) {
    const std::span<const TokenId> tokens(prompt.token_ids);
    for (std::size_t begin = 0; begin < tokens.size(); begin += prefill_chunk) {
        const std::size_t count = std::min<std::size_t>(prefill_chunk, tokens.size() - begin);
        stage.prefill_tokens(tokens.subspan(begin, count), static_cast<std::int32_t>(begin),
                             static_cast<std::int32_t>(begin) + prompt.rope_delta);
    }
    return stage.project_token();
}

TokenId prefill_oracle_split(TextShardStage& first, TextShardStage& second,
                             const TextPromptSequence& prompt, std::uint32_t prefill_chunk) {
    ninfer::PinnedHostBuffer boundary(TextShardStage::residual_bytes(prefill_chunk));
    const std::span<const TokenId> tokens(prompt.token_ids);
    for (std::size_t begin = 0; begin < tokens.size(); begin += prefill_chunk) {
        const std::size_t count = std::min<std::size_t>(prefill_chunk, tokens.size() - begin);
        first.prefill_tokens(tokens.subspan(begin, count), static_cast<std::int32_t>(begin),
                             static_cast<std::int32_t>(begin) + prompt.rope_delta);
        first.copy_prefill_residual_to_host(boundary.data(), count);
        second.copy_prefill_residual_from_host(boundary.data(), count);
        second.prefill_hidden(count, static_cast<std::int32_t>(begin),
                              static_cast<std::int32_t>(begin) + prompt.rope_delta);
        second.synchronize();
    }
    return second.project_token();
}

TokenId feed_oracle_split(TextShardStage& first, TextShardStage& second, TokenId token,
                          std::int32_t position, std::int32_t rope_position) {
    ninfer::PinnedHostBuffer boundary(TextShardStage::residual_bytes());
    first.forward_token(token, position, rope_position);
    first.copy_residual_to_host(boundary.data(), TextShardStage::residual_bytes());
    second.copy_residual_from_host(boundary.data(), TextShardStage::residual_bytes());
    second.forward_hidden(position, rope_position);
    return second.project_token();
}

struct VerifyEvidence {
    TokenId anchor = 0;
    std::vector<TokenId> targets;
    std::vector<std::uint16_t> hidden;
    std::vector<std::uint16_t> logits;
    std::vector<std::uint64_t> replay;
    TokenId continuation = 0;
};

int run_target_verify_oracle(const Options& opt, ninfer::artifact::LoadProgress& progress,
                             std::string& load_stage) {
    constexpr std::uint32_t verify_width = 4;
    constexpr std::int32_t commit_columns = 2;
    TextPromptSequence prompt;
    std::array<TokenId, 4> ordinary{};

    {
        load_stage = "oracle-reference";
        TextShardStage reference(
            opt.artifact, opt.device1, opt.max_context,
            TextShardSpec{0, 64, true, true}, &progress, opt.prefill_chunk, opt.kv_cache, 1, 0);
        prompt = reference.prepare_text_prompt(opt.prompts.front(), opt.thinking);
        if (prompt.token_ids.empty() || prompt.token_ids.size() + 8U > opt.max_context) {
            throw std::invalid_argument("oracle prompt is empty or too close to capacity");
        }
        ordinary[0] = prefill_oracle_full(reference, prompt, opt.prefill_chunk);
        TokenId current = ordinary[0];
        for (std::size_t step = 1; step < ordinary.size(); ++step) {
            const auto position =
                static_cast<std::int32_t>(prompt.token_ids.size() + step - 1U);
            reference.forward_token(current, position, position + prompt.rope_delta);
            ordinary[step] = reference.project_token();
            current = ordinary[step];
        }
    }

    const TokenId wrong1 = ordinary[2] == 1 ? 2 : 1;
    const TokenId wrong2 = ordinary[3] == 3 ? 4 : 3;
    const std::array<TokenId, verify_width> verify_input{
        ordinary[0], ordinary[1], wrong1, wrong2};
    const auto verify_position = static_cast<std::int32_t>(prompt.token_ids.size());
    const auto verify_rope = verify_position + prompt.rope_delta;

    VerifyEvidence full;
    {
        load_stage = "oracle-full";
        TextShardStage stage(
            opt.artifact, opt.device1, opt.max_context,
            TextShardSpec{0, 64, true, true}, &progress, opt.prefill_chunk, opt.kv_cache, 1,
            verify_width);
        full.anchor = prefill_oracle_full(stage, prompt, opt.prefill_chunk);
        stage.target_verify_tokens(verify_input, verify_position, verify_rope);
        full.targets = stage.project_verify_tokens();
        full.hidden.resize(stage.verify_residual_bytes() / sizeof(std::uint16_t));
        full.logits.resize(stage.verify_logits_bytes() / sizeof(std::uint16_t));
        stage.copy_verify_hidden_to_host(full.hidden.data(), stage.verify_residual_bytes());
        stage.copy_verify_logits_to_host(full.logits.data(), stage.verify_logits_bytes());
        full.replay = stage.verify_record_layer_digests();
        stage.fold_verify_records(commit_columns);
        const auto continuation_position = verify_position + commit_columns;
        stage.forward_token(ordinary[2], continuation_position,
                            continuation_position + prompt.rope_delta);
        full.continuation = stage.project_token();
    }

    VerifyEvidence split;
    {
        load_stage = "oracle-stage0";
        TextShardStage first(
            opt.artifact, opt.device0, opt.max_context,
            TextShardSpec{0, static_cast<std::size_t>(opt.split_layer), true, false}, &progress,
            opt.prefill_chunk, opt.kv_cache, 1, verify_width);
        load_stage = "oracle-stage1";
        TextShardStage second(
            opt.artifact, opt.device1, opt.max_context,
            TextShardSpec{static_cast<std::size_t>(opt.split_layer), 64, false, true}, &progress,
            opt.prefill_chunk, opt.kv_cache, 1, verify_width);
        split.anchor = prefill_oracle_split(first, second, prompt, opt.prefill_chunk);
        ninfer::PinnedHostBuffer boundary(first.verify_residual_bytes());
        first.target_verify_tokens(verify_input, verify_position, verify_rope);
        first.copy_verify_residual_to_host(boundary.data(), first.verify_residual_bytes());
        second.copy_verify_residual_from_host(boundary.data(), second.verify_residual_bytes());
        split.targets = second.target_verify_hidden(verify_position, verify_rope);
        split.hidden.resize(second.verify_residual_bytes() / sizeof(std::uint16_t));
        split.logits.resize(second.verify_logits_bytes() / sizeof(std::uint16_t));
        second.copy_verify_hidden_to_host(split.hidden.data(), second.verify_residual_bytes());
        second.copy_verify_logits_to_host(split.logits.data(), second.verify_logits_bytes());
        split.replay = first.verify_record_layer_digests();
        const std::vector<std::uint64_t> second_replay = second.verify_record_layer_digests();
        split.replay.insert(split.replay.end(), second_replay.begin(), second_replay.end());
        first.fold_verify_records(commit_columns);
        second.fold_verify_records(commit_columns);
        const auto continuation_position = verify_position + commit_columns;
        split.continuation = feed_oracle_split(
            first, second, ordinary[2], continuation_position,
            continuation_position + prompt.rope_delta);
    }

    const Bf16Comparison hidden = compare_bf16(full.hidden, split.hidden);
    const Bf16Comparison logits = compare_bf16(full.logits, split.logits);
    const bool anchors_match =
        full.anchor == ordinary[0] && split.anchor == ordinary[0];
    const bool target_reference =
        full.targets.size() == verify_width && full.targets[0] == ordinary[1] &&
        full.targets[1] == ordinary[2];
    const bool target_match = full.targets == split.targets;
    const bool replay_match = full.replay == split.replay;
    const bool continuation_match =
        full.continuation == ordinary[3] && split.continuation == ordinary[3];
    const bool numerics_ok = hidden.max_abs <= 0.03125 && hidden.max_rel <= 0.01 &&
                             logits.max_abs <= 0.125 && logits.max_rel <= 0.01;

    metric("oracle anchors", anchors_match ? "PASS" : "FAIL");
    metric("oracle target reference", target_reference ? "PASS" : "FAIL");
    metric("oracle target tokens", target_match ? "PASS" : "FAIL");
    metric("oracle replay layers", std::to_string(full.replay.size()) +
                                      (replay_match ? " PASS" : " FAIL"));
    metric("oracle hidden bit mismatches", std::to_string(hidden.bit_mismatches));
    metric("oracle hidden max abs", std::to_string(hidden.max_abs));
    metric("oracle hidden max rel", std::to_string(hidden.max_rel));
    metric("oracle logits bit mismatches", std::to_string(logits.bit_mismatches));
    metric("oracle logits max abs", std::to_string(logits.max_abs));
    metric("oracle logits max rel", std::to_string(logits.max_rel));
    metric("oracle continuation", continuation_match ? "PASS" : "FAIL");
    metric("oracle full target digest", hex_digest(digest(full.targets)));
    metric("oracle split target digest", hex_digest(digest(split.targets)));

    if (!anchors_match || !target_reference || !target_match || !replay_match ||
        !continuation_match || !numerics_ok) {
        throw std::runtime_error("split target-verifier/replay-fold oracle failed");
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        Options opt = parse(argc, argv);
        auto load_start = Clock::now();
        std::string load_stage;
        ninfer::artifact::LoadProgress progress;
        progress.callback = [&](std::string_view detail, std::uint64_t done, std::uint64_t total) {
            if (done == total)
                std::cerr << "load        " << load_stage << ' ' << detail
                          << (total ? " " + bytes(total) : std::string{}) << '\n';
        };
        if (opt.target_verify_oracle) {
            return run_target_verify_oracle(opt, progress, load_stage);
        }

        std::vector<TextPromptSequence> prepared_prompts;
        std::vector<TokenId> generated;
        std::vector<std::vector<TokenId>> generated_sessions;
        double prefill = 0.0, decode = 0.0;
        DualTiming dual_timing;
        DualMtpTiming mtp_timing;
        std::unique_ptr<Tp2MlpCoordinator> tp2_mlp;
        std::optional<Tp2MlpStats> tp2_stats;
        std::optional<ContextParallelGqaStats> context_gqa_stats;

        if (opt.single_device) {
            load_stage = "single";
            TextShardStage stage(opt.artifact, *opt.single_device, opt.max_context,
                TextShardSpec{0, 64, true, true}, &progress, opt.prefill_chunk, opt.kv_cache, 1,
                0, false, {}, opt.vision);
            prepared_prompts.push_back(opt.messages_path
                ? stage.prepare_prompt(ninfer::product::prompt_from_messages(
                      *opt.messages_path, opt.thinking, opt.vision))
                : stage.prepare_text_prompt(opt.prompts.front(), opt.thinking));
            const TextPromptSequence& prompt = prepared_prompts.front();
            if (prompt.token_ids.empty()) throw std::runtime_error("prepared prompt is empty");
            if (prompt.token_ids.size() + opt.max_new > opt.max_context)
                throw std::invalid_argument("prompt plus output exceeds --max-context");
            stage_metrics("single", stage);
            generated = single(opt, stage, prompt, prefill, decode);
            generated_sessions.push_back(generated);
        } else {
            const std::uint32_t verifier_width = opt.mtp3
                ? (opt.sessions == 1 ? std::max<std::uint32_t>(4U, opt.mtp_depth + 1U) : 4U)
                : 0U;
            const std::uint32_t context_width = verifier_width == 0 ? 4U : verifier_width;
            std::optional<SequenceShardGeometry> sequence_geometry;
            if (opt.context_parallel_gqa) {
                sequence_geometry =
                    plan_sequence_sharded_kv(opt.max_context, opt.kv_cache, 0.60, context_width);
            }
            const TextKvPageOwnership stage0_pages = sequence_geometry
                ? sequence_geometry->fast_pages : TextKvPageOwnership{};
            const TextKvPageOwnership stage1_pages = sequence_geometry
                ? sequence_geometry->slow_pages : TextKvPageOwnership{};
            load_stage = "stage0";
            TextShardStage first(opt.artifact, opt.device0, opt.max_context,
                TextShardSpec{0, static_cast<std::size_t>(opt.split_layer), true, false}, &progress,
                opt.prefill_chunk, opt.kv_cache, opt.sessions,
                verifier_width, opt.mtp_proposal_stage0, stage0_pages, opt.vision, opt.computation);
            prepared_prompts.reserve(opt.sessions);
            for (std::uint32_t session = 0; session < opt.sessions; ++session) {
                TextPromptSequence prepared;
                if (opt.messages_path) {
                    prepared = first.prepare_prompt(ninfer::product::prompt_from_messages(
                        *opt.messages_path, opt.thinking, opt.vision));
                } else {
                    const std::string& source =
                        opt.prompts.size() == 1 ? opt.prompts.front() : opt.prompts[session];
                    prepared = first.prepare_text_prompt(source, opt.thinking);
                }
                if (prepared.token_ids.empty()) throw std::runtime_error("prepared prompt is empty");
                if (prepared.token_ids.size() + opt.max_new > opt.max_context)
                    throw std::invalid_argument("prompt plus output exceeds --max-context");
                prepared_prompts.push_back(std::move(prepared));
            }
            load_stage = "stage1";
            TextShardStage second(opt.artifact, opt.device1, opt.max_context,
                TextShardSpec{static_cast<std::size_t>(opt.split_layer), 64, false, true}, &progress,
                opt.prefill_chunk, opt.kv_cache, opt.sessions,
                verifier_width, opt.mtp3, stage1_pages, opt.vision, opt.computation,
                opt.mtp_attention_window,opt.mtp_full_vocab_draft);
            constexpr float kQualifiedMediaGreedyMargin = 10.0F;
            const float greedy_margin =
                opt.mtp_media_throughput ? kQualifiedMediaGreedyMargin : -1.0F;
            first.set_mtp_greedy_margin(greedy_margin);
            second.set_mtp_greedy_margin(greedy_margin);
            stage_metrics("stage0", first);
            stage_metrics("stage1", second);
            std::unique_ptr<ContextParallelGqaCoordinator> context_gqa;
            if (opt.context_parallel_gqa) {
                load_stage = "context-parallel-gqa";
                context_gqa = std::make_unique<ContextParallelGqaCoordinator>(
                    opt.device0, opt.device1, opt.max_context, opt.split_layer, opt.kv_cache,
                    opt.context_parallel_threshold, context_width, 0.60, opt.computation,opt.context_headscale_qk,opt.context_value_fp16);
                metric("target QK precision",opt.context_headscale_qk ? "head256-numerical" : "retained-group64");
                metric("target V arithmetic",opt.context_value_fp16 ? "direct-fp16-numerical" : "retained-bf16-rounded");
                const SequenceShardGeometry& geometry = context_gqa->geometry();
                metric("sequence fast pages", std::to_string(geometry.fast_pages.logical_page_count));
                metric("sequence slow pages", std::to_string(geometry.slow_pages.logical_page_count));
                metric("sequence duplicate pages",
                       std::to_string(geometry.duplicate_boundary_pages));
                metric("sequence boundary key", std::to_string(geometry.boundary_key));
                metric("sequence helper bytes",
                       bytes(context_gqa->stats().physical_helper_bytes));
                first.set_gqa_execution_override(context_gqa.get());
                second.set_gqa_execution_override(context_gqa.get());
            }
            if (opt.tp2_mlp_dir) {
                load_stage = "tp2-mlp";
                tp2_mlp = std::make_unique<Tp2MlpCoordinator>(
                    *opt.tp2_mlp_dir, opt.device0, opt.device1, opt.split_layer);
                first.set_mlp_execution_override(tp2_mlp.get());
                second.set_mlp_execution_override(tp2_mlp.get());
            }
            if (opt.mtp3) {
                if (opt.sessions == 1) {
                    generated = dual_mtp3(opt, first, second, context_gqa.get(),
                                          prepared_prompts.front(), prefill, decode,
                                          mtp_timing);
                    generated_sessions.push_back(generated);
                } else if (opt.sessions == 2) {
                    generated_sessions = opt.mtp_batch
                        ? dual_mtp3_two_batch(opt, first, second, prepared_prompts, prefill, decode,
                                              mtp_timing)
                        : dual_mtp3_two(opt, first, second, prepared_prompts, prefill, decode,
                                        mtp_timing);
                } else if (opt.sessions == 4 && opt.mtp_batch) {
                    generated_sessions = dual_mtp3_four_batch(
                        opt, first, second, prepared_prompts, prefill, decode, mtp_timing);
                } else if (opt.sessions == 8 && opt.mtp_batch) {
                    generated_sessions = dual_mtp3_eight_batch(
                        opt, first, second, prepared_prompts, prefill, decode, mtp_timing);
                } else {
                    throw std::invalid_argument(
                        "dual MTP3 supports one or two scalar sessions, or four with --mtp-batch");
                }
            } else if (opt.sessions == 1) {
                generated = dual(opt, first, second, prepared_prompts.front(), prefill, decode,
                                 dual_timing);
                generated_sessions.push_back(generated);
            } else {
                generated_sessions = dual_multi(opt, first, second, prepared_prompts, prefill,
                                                decode);
            }
            if (tp2_mlp) { tp2_stats = tp2_mlp->stats(); }
            if (context_gqa) { context_gqa_stats = context_gqa->stats(); }
        }

        double load = std::chrono::duration<double>(Clock::now() - load_start).count() - prefill - decode;
        const std::size_t session_count = generated_sessions.size();
        std::size_t aggregate_generated = 0;
        std::size_t aggregate_decode_tokens = 0;
        std::size_t aggregate_prompt_tokens = 0;
        for (const auto& prompt : prepared_prompts) {
            aggregate_prompt_tokens += prompt.token_ids.size();
        }
        for (const auto& session_tokens : generated_sessions) {
            aggregate_generated += session_tokens.size();
            if (!session_tokens.empty()) aggregate_decode_tokens += session_tokens.size() - 1;
        }
        for (std::size_t session = 0; session < generated_sessions.size(); ++session) {
            if (generated_sessions.size() > 1) std::cout << "session " << session << ':';
            for (TokenId token : generated_sessions[session]) std::cout << ' ' << token;
            std::cout << '\n';
        }

        const std::string_view mode = opt.single_device ? "single-oracle" :
            (opt.mtp3 && opt.mtp_batch && session_count == 8 ? "dual-layer-shard-mtp3-c8" :
             opt.mtp3 && opt.mtp_batch && session_count == 4 ? "dual-layer-shard-mtp3-c4" :
             opt.mtp3 && opt.mtp_batch ? "dual-layer-shard-mtp3-b2" :
             opt.mtp3 ? "dual-layer-shard-mtp3" :
             session_count > 1 ? "dual-shared-multisession" : "dual-layer-shard");
        metric("mode", mode);
        metric("sessions", std::to_string(session_count));
        metric("max context", std::to_string(opt.max_context));
        metric("KV dtype", opt.kv_cache == KvCacheStorage::BFloat16 ? "bf16" : "int8-group64");
        metric("prefill chunk", std::to_string(opt.prefill_chunk));
        metric("prefill pipeline", opt.single_device ? "n/a" :
               (opt.prefill_pipeline ? "enabled" : "disabled"));
        metric("split layer", opt.single_device ? "n/a" : std::to_string(opt.split_layer));
        metric("aggregate prompt tokens", std::to_string(aggregate_prompt_tokens));
        for (std::size_t session = 0; session < prepared_prompts.size(); ++session) {
            metric("session " + std::to_string(session) + " prompt tokens",
                   std::to_string(prepared_prompts[session].token_ids.size()));
        }
        metric("generated/session", generated_sessions.empty() ? "0" :
               std::to_string(generated_sessions.front().size()));
        metric("aggregate generated", std::to_string(aggregate_generated));
        for (std::size_t session = 0; session < generated_sessions.size(); ++session) {
            metric("session " + std::to_string(session) + " digest",
                   hex_digest(digest(generated_sessions[session])));
        }
        metric("load elapsed", seconds(load));
        metric("prefill elapsed", seconds(prefill));
        metric("prefill aggregate speed",
               rate(aggregate_prompt_tokens, prefill));
        metric("decode elapsed", seconds(decode));
        metric("decode per-user speed",
               generated_sessions.empty() ? "n/a" :
               rate(generated_sessions.front().empty() ? 0 : generated_sessions.front().size() - 1,
                    decode));
        metric("decode aggregate speed", rate(aggregate_decode_tokens, decode));
        if (tp2_stats) {
            metric("tp2 MLP calls", std::to_string(tp2_stats->calls));
            metric("tp2 MLP width4 calls", std::to_string(tp2_stats->width4_calls));
            metric("tp2 MLP width5 calls", std::to_string(tp2_stats->width5_calls));
            metric("tp2 MLP input wire bytes",std::to_string(tp2_stats->input_transfer_bytes));
            metric("tp2 MLP partial wire bytes",std::to_string(tp2_stats->partial_transfer_bytes));
            metric("tp2 MLP input staging", seconds(tp2_stats->input_seconds));
            metric("tp2 MLP dual engine", seconds(tp2_stats->engine_seconds));
            metric("tp2 MLP reduction", seconds(tp2_stats->reduction_seconds));
            metric("tp2 MLP fast bytes", bytes(tp2_stats->fast_device_bytes));
            metric("tp2 MLP slow bytes", bytes(tp2_stats->slow_device_bytes));
            metric("tp2 MLP fast alias bytes",
                   bytes(tp2_stats->fast_canonical_alias_bytes));
            metric("tp2 MLP slow alias bytes",
                   bytes(tp2_stats->slow_canonical_alias_bytes));
        }
        if (context_gqa_stats) {
            metric("context GQA calls", std::to_string(context_gqa_stats->calls));
            metric("context GQA parallel calls",
                   std::to_string(context_gqa_stats->parallel_calls));
            metric("context GQA headscale calls",std::to_string(context_gqa_stats->headscale_calls));
            metric("context GQA value FP16 calls",std::to_string(context_gqa_stats->value_fp16_calls));
            metric("context GQA routed tokens",
                   std::to_string(context_gqa_stats->routed_tokens));
            metric("context GQA state bytes", bytes(context_gqa_stats->state_bytes));
            metric("context GQA q bytes", bytes(context_gqa_stats->q_bytes));
            metric("context GQA helper bytes",
                   bytes(context_gqa_stats->physical_helper_bytes));
            metric("context GQA fast pages", std::to_string(context_gqa_stats->fast_pages));
            metric("context GQA slow pages", std::to_string(context_gqa_stats->slow_pages));
            metric("context GQA duplicate pages",
                   std::to_string(context_gqa_stats->duplicate_boundary_pages));
            metric("context GQA packet", seconds(context_gqa_stats->packet_seconds));
            metric("context GQA helper", seconds(context_gqa_stats->helper_seconds));
            metric("context GQA state transfer",
                   seconds(context_gqa_stats->state_transfer_seconds));
            metric("context GQA merge", seconds(context_gqa_stats->merge_seconds));
        }
        metric("boundary bytes/token", opt.single_device ? "0" :
               std::to_string(TextShardStage::residual_bytes()));
        if (opt.mtp3) {
            metric("mtp selected depth", std::to_string(opt.mtp_depth));
            metric("mtp full vocabulary draft",opt.mtp_full_vocab_draft?"enabled":"disabled");
            metric("mtp proposal window",opt.mtp_attention_window?std::to_string(opt.mtp_attention_window)+" +128 sink (target unchanged)":"complete");
            metric("long-context computation",opt.computation==qwen3_6::LongContextComputation::Complete
                ? "complete-layers-all-keys-full-head" : "retained-throughput");
            metric("mtp acceptance policy",
                   opt.computation==qwen3_6::LongContextComputation::Complete ? "strict-full-context" :
                   opt.mtp_media_throughput ? "media-throughput"
                                            : (opt.mtp_depth == 4 ? "licensed-d4" : "natural"));
            metric("mtp rounds", std::to_string(mtp_timing.rounds));
            metric("mtp fallback steps", std::to_string(mtp_timing.fallback_steps));
            metric("mtp drafted tokens", std::to_string(mtp_timing.drafted));
            metric("mtp accepted tokens", std::to_string(mtp_timing.accepted));
            metric("mtp acceptance rate",
                   percentage(mtp_timing.accepted, mtp_timing.drafted));
            metric("mtp accepted by pos",
                   joined_counts(mtp_timing.accepted_by_position, opt.mtp_depth));
            for (std::size_t session = 0; session < session_count; ++session) {
                metric("session " + std::to_string(session) + " mtp rounds",
                       std::to_string(mtp_timing.session_rounds[session]));
                metric("session " + std::to_string(session) + " mtp fallback",
                       std::to_string(mtp_timing.session_fallback_steps[session]));
                metric("session " + std::to_string(session) + " mtp accepted",
                       std::to_string(mtp_timing.session_accepted[session]) + "/" +
                       std::to_string(mtp_timing.session_drafted[session]));
                metric("session " + std::to_string(session) + " mtp acceptance",
                       percentage(mtp_timing.session_accepted[session],
                                  mtp_timing.session_drafted[session]));
                metric("session " + std::to_string(session) + " mtp accepted by pos",
                       joined_counts(mtp_timing.session_accepted_by_position[session],
                                     opt.mtp_depth));
                metric("session " + std::to_string(session) + " scalar tail",
                       std::to_string(mtp_timing.session_scalar_tail[session]));
            }
            metric("mtp target stage0", seconds(mtp_timing.target_stage0));
            metric("mtp target stage1/head", seconds(mtp_timing.target_stage1));
            metric("mtp align/proposal", seconds(mtp_timing.proposal));
            metric("mtp replay fold", seconds(mtp_timing.fold));
            if (!opt.single_device && session_count == 1) {
                metric("prefill stage0 elapsed", seconds(mtp_timing.prefill_stage0));
                metric("prefill stage1 critical wait", seconds(mtp_timing.prefill_stage1));
            }
        }
        if (!opt.single_device && session_count == 1 && !opt.mtp3) {
            metric("prefill stage0 elapsed", seconds(dual_timing.prefill_stage0));
            metric("prefill stage1 critical wait", seconds(dual_timing.prefill_stage1));
            metric("prefill head elapsed", seconds(dual_timing.prefill_head));
            metric("decode stage0 elapsed", seconds(dual_timing.decode_stage0));
            metric("decode stage1 elapsed", seconds(dual_timing.decode_stage1));
            metric("decode head elapsed", seconds(dual_timing.decode_head));
        }
        if (opt.print_token_ids) {
            for (std::size_t session = 0; session < generated_sessions.size(); ++session) {
                std::cerr << "tokens      session " << session << " generated ids     ";
                for (std::size_t i = 0; i < generated_sessions[session].size(); ++i) {
                    if (i) std::cerr << ' ';
                    std::cerr << generated_sessions[session][i];
                }
                std::cerr << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
