#include "targets/qwen3_6_27b/impl/dual_stage.h"
#include "core/arena.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using ninfer::TokenId;
using ninfer::targets::qwen3_6_27b::detail::TextPromptSequence;
using ninfer::targets::qwen3_6_27b::detail::TextShardSpec;
using ninfer::targets::qwen3_6_27b::detail::TextShardStage;

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
    std::string prompt;
    std::uint32_t max_context = 8192;
    std::uint32_t max_new = 16;
    std::uint32_t prefill_chunk = 128;
    bool prefill_pipeline = true;
    int device0 = 0;
    int device1 = 1;
    int split_layer = 36;
    std::optional<int> single_device;
    bool thinking = false;
    bool print_token_ids = false;
};

[[noreturn]] void usage(std::string_view error = {}) {
    if (!error.empty()) std::cerr << "error: " << error << "\n\n";
    std::cerr
        << "usage: ninfer-dual <model.ninfer> --prompt <text> [options]\n"
        << "  --max-context N    per-device context capacity (default 8192)\n"
        << "  --max-new N        greedy output tokens (default 16)\n"
        << "  --prefill-chunk N  prompt chunk width per shard (default 128)\n"
        << "  --no-prefill-pipeline  serialize prompt chunks for diagnostics\n"
        << "  --device0 N        first-stage CUDA device (default 0)\n"
        << "  --device1 N        second-stage CUDA device (default 1)\n"
        << "  --split-layer N    multiple-of-4 boundary (default 36)\n"
        << "  --single-device N  all 64 layers on one GPU, exact T1 oracle\n"
        << "  --thinking         enable thinking prompt template\n"
        << "  --print-token-ids  duplicate generated IDs to stderr\n";
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
        if (arg == "--prompt") out.prompt.assign(take(arg));
        else if (arg == "--max-context") out.max_context = u32(take(arg), arg);
        else if (arg == "--max-new") out.max_new = u32(take(arg), arg);
        else if (arg == "--prefill-chunk") out.prefill_chunk = u32(take(arg), arg);
        else if (arg == "--no-prefill-pipeline") out.prefill_pipeline = false;
        else if (arg == "--device0") out.device0 = integer(take(arg), arg);
        else if (arg == "--device1") out.device1 = integer(take(arg), arg);
        else if (arg == "--split-layer") out.split_layer = integer(take(arg), arg);
        else if (arg == "--single-device") out.single_device = integer(take(arg), arg);
        else if (arg == "--thinking") out.thinking = true;
        else if (arg == "--print-token-ids") out.print_token_ids = true;
        else if (arg == "--help" || arg == "-h") usage();
        else usage("unknown option '" + std::string(arg) + "'");
    }
    if (out.prompt.empty()) usage("--prompt is required");
    if (out.max_context == 0 || out.max_new == 0 || out.prefill_chunk == 0)
        usage("context, output, and prefill chunk must be positive");
    if (out.prefill_chunk > out.max_context)
        usage("--prefill-chunk cannot exceed --max-context");
    if (out.prefill_chunk > 128)
        usage("--prefill-chunk must be in [1,128]");
    if (out.split_layer <= 0 || out.split_layer >= 64 || out.split_layer % 4 != 0)
        usage("--split-layer must be a multiple of 4 in [4,60]");
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

void metric(std::string_view label, std::string_view value) {
    std::cerr << std::left << std::setw(12) << "summary" << std::setw(30) << label << value << '\n';
}

void stage_metrics(std::string_view label, const TextShardStage& stage) {
    metric(std::string(label) + " device", std::to_string(stage.device_id()));
    metric(std::string(label) + " weights", bytes(stage.weights_bytes()));
    metric(std::string(label) + " state", bytes(stage.state_bytes()));
    metric(std::string(label) + " workspace", bytes(stage.workspace_bytes()));
}

std::vector<TokenId> single(const Options& opt, TextShardStage& stage,
                            const TextPromptSequence& prompt, double& prefill, double& decode) {
    auto start = Clock::now();
    const std::span<const TokenId> prompt_tokens(prompt.token_ids);
    for (std::size_t begin = 0; begin < prompt_tokens.size(); begin += opt.prefill_chunk) {
        const std::size_t tokens =
            std::min<std::size_t>(opt.prefill_chunk, prompt_tokens.size() - begin);
        stage.prefill_tokens(prompt_tokens.subspan(begin, tokens), static_cast<std::int32_t>(begin),
                             static_cast<std::int32_t>(begin) + prompt.rope_delta);
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
    for (std::size_t begin = 0; begin < prompt_tokens.size(); begin += opt.prefill_chunk, ++chunk_index) {
        const std::size_t tokens =
            std::min<std::size_t>(opt.prefill_chunk, prompt_tokens.size() - begin);
        auto& boundary = boundaries[chunk_index % boundaries.size()];
        auto started   = Clock::now();
        first.prefill_tokens(prompt_tokens.subspan(begin, tokens), static_cast<std::int32_t>(begin),
                             static_cast<std::int32_t>(begin) + prompt.rope_delta);
        first.copy_prefill_residual_to_host(boundary.data(), tokens);
        timing.prefill_stage0 += std::chrono::duration<double>(Clock::now() - started).count();

        if (opt.prefill_pipeline && stage1_inflight) {
            started = Clock::now();
            second.synchronize();
            timing.prefill_stage1 += std::chrono::duration<double>(Clock::now() - started).count();
        }
        started = Clock::now();
        second.copy_prefill_residual_from_host(boundary.data(), tokens);
        second.prefill_hidden(tokens, static_cast<std::int32_t>(begin),
                              static_cast<std::int32_t>(begin) + prompt.rope_delta);
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

        TextPromptSequence prompt;
        std::vector<TokenId> generated;
        double prefill = 0.0, decode = 0.0;
        DualTiming dual_timing;

        if (opt.single_device) {
            load_stage = "single";
            TextShardStage stage(opt.artifact, *opt.single_device, opt.max_context,
                TextShardSpec{0, 64, true, true}, &progress, opt.prefill_chunk);
            prompt = stage.prepare_text_prompt(opt.prompt, opt.thinking);
            if (prompt.token_ids.empty()) throw std::runtime_error("prepared prompt is empty");
            if (prompt.token_ids.size() + opt.max_new > opt.max_context)
                throw std::invalid_argument("prompt plus output exceeds --max-context");
            stage_metrics("single", stage);
            generated = single(opt, stage, prompt, prefill, decode);
        } else {
            load_stage = "stage0";
            TextShardStage first(opt.artifact, opt.device0, opt.max_context,
                TextShardSpec{0, static_cast<std::size_t>(opt.split_layer), true, false}, &progress,
                opt.prefill_chunk);
            prompt = first.prepare_text_prompt(opt.prompt, opt.thinking);
            if (prompt.token_ids.empty()) throw std::runtime_error("prepared prompt is empty");
            if (prompt.token_ids.size() + opt.max_new > opt.max_context)
                throw std::invalid_argument("prompt plus output exceeds --max-context");
            load_stage = "stage1";
            TextShardStage second(opt.artifact, opt.device1, opt.max_context,
                TextShardSpec{static_cast<std::size_t>(opt.split_layer), 64, false, true}, &progress,
                opt.prefill_chunk);
            stage_metrics("stage0", first);
            stage_metrics("stage1", second);
            generated = dual(opt, first, second, prompt, prefill, decode, dual_timing);
        }

        double load = std::chrono::duration<double>(Clock::now() - load_start).count() - prefill - decode;
        for (std::size_t i = 0; i < generated.size(); ++i) {
            if (i) std::cout << ' ';
            std::cout << generated[i];
        }
        std::cout << '\n';

        metric("mode", opt.single_device ? "single-oracle" : "dual-layer-shard");
        metric("max context", std::to_string(opt.max_context));
        metric("prefill chunk", std::to_string(opt.prefill_chunk));
        metric("prefill pipeline", opt.single_device ? "n/a" :
               (opt.prefill_pipeline ? "enabled" : "disabled"));
        metric("split layer", opt.single_device ? "n/a" : std::to_string(opt.split_layer));
        metric("prompt tokens", std::to_string(prompt.token_ids.size()));
        metric("generated tokens", std::to_string(generated.size()));
        metric("generated token digest", hex_digest(digest(generated)));
        metric("load elapsed", seconds(load));
        metric("prefill elapsed", seconds(prefill));
        metric("prefill speed", rate(prompt.token_ids.size(), prefill));
        metric("decode elapsed", seconds(decode));
        metric("decode speed", rate(generated.size() > 0 ? generated.size() - 1 : 0, decode));
        metric("boundary bytes/token", opt.single_device ? "0" :
               std::to_string(TextShardStage::residual_bytes()));
        if (!opt.single_device) {
            metric("prefill stage0 elapsed", seconds(dual_timing.prefill_stage0));
            metric("prefill stage1 critical wait", seconds(dual_timing.prefill_stage1));
            metric("prefill head elapsed", seconds(dual_timing.prefill_head));
            metric("decode stage0 elapsed", seconds(dual_timing.decode_stage0));
            metric("decode stage1 elapsed", seconds(dual_timing.decode_stage1));
            metric("decode head elapsed", seconds(dual_timing.decode_head));
        }
        if (opt.print_token_ids) {
            std::cerr << "tokens      generated ids             ";
            for (std::size_t i = 0; i < generated.size(); ++i) {
                if (i) std::cerr << ' ';
                std::cerr << generated[i];
            }
            std::cerr << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
