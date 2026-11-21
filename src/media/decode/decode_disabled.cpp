#include "media/decode/decode.h"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#include <array>
#include <cerrno>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::media::decode {
#ifndef _WIN32
namespace {

class TempFile {
public:
    explicit TempFile(std::span<const std::uint8_t> bytes) {
        std::array<char, 32> pattern{};
        std::snprintf(pattern.data(), pattern.size(), "/tmp/ninfer-media-XXXXXX");
        const int fd = ::mkstemp(pattern.data());
        if (fd < 0) { throw std::runtime_error("failed to create media temp file"); }
        path_ = pattern.data();
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            const ssize_t amount = ::write(fd, bytes.data() + offset, bytes.size() - offset);
            if (amount < 0 && errno == EINTR) { continue; }
            if (amount <= 0) {
                ::close(fd); ::unlink(path_.c_str());
                throw std::runtime_error("failed to write media temp file");
            }
            offset += static_cast<std::size_t>(amount);
        }
        if (::close(fd) != 0) {
            ::unlink(path_.c_str());
            throw std::runtime_error("failed to close media temp file");
        }
    }
    ~TempFile() { if (!path_.empty()) { ::unlink(path_.c_str()); } }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    [[nodiscard]] const std::string& path() const noexcept { return path_; }
private:
    std::string path_;
};

std::vector<std::uint8_t> pipe_bytes(const std::string& command, std::size_t limit) {
    FILE* pipe = ::popen(command.c_str(), "r");
    if (pipe == nullptr) { throw std::runtime_error("failed to launch FFmpeg"); }
    std::vector<std::uint8_t> out;
    std::array<std::uint8_t, 64 * 1024> buffer{};
    for (;;) {
        const std::size_t amount = std::fread(buffer.data(), 1, buffer.size(), pipe);
        if (amount != 0) {
            if (amount > limit - std::min(limit, out.size())) {
                ::pclose(pipe);
                throw Error(ErrorKind::BudgetExceeded, "decoded media exceeds output budget");
            }
            out.insert(out.end(), buffer.begin(), buffer.begin() + amount);
        }
        if (amount < buffer.size()) {
            if (std::ferror(pipe)) {
                ::pclose(pipe);
                throw std::runtime_error("failed to read FFmpeg output");
            }
            break;
        }
    }
    const int status = ::pclose(pipe);
    if (status != 0) { throw std::invalid_argument("FFmpeg failed to decode media input"); }
    return out;
}

std::string pipe_text(const std::string& command) {
    const std::vector<std::uint8_t> bytes = pipe_bytes(command, 1U << 20);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

std::pair<int, int> dimensions(const std::string& path) {
    const std::string text = pipe_text(
        "ffprobe -v error -select_streams v:0 -show_entries stream=width,height "
        "-of csv=p=0:s=x " + path + " 2>/dev/null");
    const std::size_t separator = text.find('x');
    if (separator == std::string::npos) {
        throw std::invalid_argument("media dimensions are unavailable");
    }
    const int width = std::stoi(text.substr(0, separator));
    const int height = std::stoi(text.substr(separator + 1U));
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("media dimensions are invalid");
    }
    return {width, height};
}

void validate(std::span<const std::uint8_t> bytes, const Policy& policy) {
    if (bytes.empty()) { throw std::invalid_argument("media input is empty"); }
    if (bytes.size() > policy.max_bytes) {
        throw Error(ErrorKind::BudgetExceeded, "media input exceeds byte limit");
    }
}
} // namespace

Image decode_image(std::span<const std::uint8_t> bytes, const Policy& policy) {
    validate(bytes, policy);
    TempFile input(bytes);
    const auto [width, height] = dimensions(input.path());
    const std::uint64_t pixels = static_cast<std::uint64_t>(width) * height;
    if (pixels > policy.max_decoded_pixels ||
        pixels > std::numeric_limits<std::size_t>::max() / 3U) {
        throw Error(ErrorKind::BudgetExceeded, "decoded image exceeds pixel limit");
    }
    Image out;
    out.width = width;
    out.height = height;
    const std::size_t expected = static_cast<std::size_t>(pixels) * 3U;
    out.rgb = pipe_bytes("ffmpeg -v error -i " + input.path() +
                             " -frames:v 1 -f rawvideo -pix_fmt rgb24 pipe:1 2>/dev/null",
                         expected);
    if (out.rgb.size() != expected) {
        throw std::runtime_error("FFmpeg produced an incomplete image frame");
    }
    return out;
}

Video decode_video(std::span<const std::uint8_t> bytes, const Policy& policy,
                   double target_fps, int min_frames, int max_frames) {
    validate(bytes, policy);
    if (!(target_fps > 0.0) || !std::isfinite(target_fps)) {
        throw std::invalid_argument("video target_fps must be finite and positive");
    }
    if (min_frames <= 0 || max_frames < min_frames) {
        throw std::invalid_argument("video frame bounds are invalid");
    }

    TempFile input(bytes);
    const std::string probe = pipe_text(
        "ffprobe -v error -select_streams v:0 "
        "-show_entries stream=width,height,avg_frame_rate,nb_frames,duration "
        "-of default=noprint_wrappers=1 " + input.path());
    std::unordered_map<std::string, std::string> values;
    std::istringstream lines(probe);
    std::string line;
    while (std::getline(lines, line)) {
        const std::size_t equal = line.find('=');
        if (equal != std::string::npos) {
            values.emplace(line.substr(0, equal), line.substr(equal + 1));
        }
    }
    const auto require_value = [&](const char* key) -> const std::string& {
        const auto it = values.find(key);
        if (it == values.end() || it->second.empty() || it->second == "N/A") {
            throw std::runtime_error(std::string("ffprobe omitted video field: ") + key);
        }
        return it->second;
    };
    const auto parse_int = [&](const std::string& text, const char* field) {
        char* end = nullptr;
        const long long value = std::strtoll(text.c_str(), &end, 10);
        if (end == text.c_str() || *end != '\0' || value <= 0 ||
            value > std::numeric_limits<int>::max()) {
            throw std::runtime_error(std::string("invalid ffprobe integer field: ") + field);
        }
        return static_cast<int>(value);
    };
    const auto parse_rate = [&](const std::string& text) {
        const std::size_t slash = text.find('/');
        double value = 0.0;
        if (slash == std::string::npos) {
            value = std::strtod(text.c_str(), nullptr);
        } else {
            const double numerator = std::strtod(text.substr(0, slash).c_str(), nullptr);
            const double denominator = std::strtod(text.substr(slash + 1).c_str(), nullptr);
            if (denominator != 0.0) { value = numerator / denominator; }
        }
        if (!(value > 0.0) || !std::isfinite(value)) {
            throw std::runtime_error("invalid ffprobe video frame rate");
        }
        return value;
    };

    const int source_width = parse_int(require_value("width"), "width");
    const int source_height = parse_int(require_value("height"), "height");
    const double fps = parse_rate(require_value("avg_frame_rate"));
    int total_frames = 0;
    const auto frames_it = values.find("nb_frames");
    if (frames_it != values.end() && frames_it->second != "N/A") {
        total_frames = parse_int(frames_it->second, "nb_frames");
    }
    double duration = 0.0;
    const auto duration_it = values.find("duration");
    if (duration_it != values.end() && duration_it->second != "N/A") {
        duration = std::strtod(duration_it->second.c_str(), nullptr);
    }
    if (total_frames <= 0) {
        const std::string counted = pipe_text(
            "ffprobe -v error -count_frames -select_streams v:0 "
            "-show_entries stream=nb_read_frames -of default=nw=1:nk=1 " + input.path());
        std::istringstream count_stream(counted);
        std::string count_text;
        std::getline(count_stream, count_text);
        total_frames = parse_int(count_text, "nb_read_frames");
    }
    if (!(duration > 0.0) || !std::isfinite(duration)) {
        duration = static_cast<double>(total_frames) / fps;
    }
    if (total_frames > policy.max_video_source_frames) {
        throw Error(ErrorKind::BudgetExceeded, "video source frame budget exceeded");
    }
    if (duration > policy.max_video_duration_seconds) {
        throw Error(ErrorKind::BudgetExceeded, "video duration budget exceeded");
    }

    // A short source cannot supply min_frames distinct samples. Keep the clamp
    // interval valid; temporal padding belongs to the processor, after decoding.
    const double desired = std::round(
        static_cast<double>(total_frames) * target_fps / fps);
    const int sample_count = std::clamp(
        static_cast<int>(std::clamp(
            desired, 1.0, static_cast<double>(std::numeric_limits<int>::max()))),
        std::min(min_frames, total_frames), std::min(max_frames, total_frames));
    std::vector<int> indices(static_cast<std::size_t>(sample_count));
    if (sample_count == 1) {
        indices[0] = 0;
    } else {
        const double scale = static_cast<double>(total_frames - 1) /
                             static_cast<double>(sample_count - 1);
        for (int i = 0; i < sample_count; ++i) {
            indices[static_cast<std::size_t>(i)] = std::clamp(
                static_cast<int>(std::llround(static_cast<double>(i) * scale)),
                0, total_frames - 1);
        }
        indices.back() = total_frames - 1;
    }

    std::ostringstream expression;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (i != 0) { expression << '+'; }
        expression << "eq(n," << indices[i] << ')';
    }
    const std::uint64_t source_pixels =
        static_cast<std::uint64_t>(source_width) * static_cast<std::uint64_t>(source_height);
    if (source_pixels == 0 ||
        source_pixels > policy.max_decoded_video_pixels /
                            static_cast<std::uint64_t>(sample_count)) {
        throw Error(ErrorKind::BudgetExceeded, "decoded video pixel budget exceeded");
    }
    const std::uint64_t payload_limit_u64 =
        source_pixels * static_cast<std::uint64_t>(sample_count) * 3ULL;
    if (payload_limit_u64 > std::numeric_limits<std::size_t>::max() -
                                static_cast<std::size_t>(sample_count) * 128ULL) {
        throw Error(ErrorKind::BudgetExceeded, "decoded video output size overflow");
    }
    const std::size_t output_limit = static_cast<std::size_t>(payload_limit_u64) +
                                     static_cast<std::size_t>(sample_count) * 128ULL;
    const std::vector<std::uint8_t> ppm = pipe_bytes(
        "ffmpeg -v error -nostdin -i " + input.path() +
        " -vf \"select='" + expression.str() +
        "'\" -vsync 0 -f image2pipe -vcodec ppm -",
        output_limit);

    std::size_t cursor = 0;
    auto next_token = [&]() -> std::string {
        while (cursor < ppm.size()) {
            const unsigned char ch = ppm[cursor];
            if (std::isspace(ch)) { ++cursor; continue; }
            if (ch == '#') {
                while (cursor < ppm.size() && ppm[cursor] != '\n') { ++cursor; }
                continue;
            }
            break;
        }
        const std::size_t begin = cursor;
        while (cursor < ppm.size() && !std::isspace(static_cast<unsigned char>(ppm[cursor]))) {
            ++cursor;
        }
        if (begin == cursor) { throw std::runtime_error("truncated FFmpeg PPM stream"); }
        return std::string(reinterpret_cast<const char*>(ppm.data() + begin), cursor - begin);
    };

    Video video;
    video.total_frames = total_frames;
    video.fps = fps;
    video.duration = duration;
    video.indices = indices;
    video.frames.reserve(indices.size());
    std::uint64_t decoded_pixels = 0;
    for (std::size_t frame_index = 0; frame_index < indices.size(); ++frame_index) {
        if (next_token() != "P6") { throw std::runtime_error("FFmpeg did not emit P6 PPM"); }
        const int width = parse_int(next_token(), "decoded width");
        const int height = parse_int(next_token(), "decoded height");
        if (next_token() != "255") { throw std::runtime_error("unsupported FFmpeg PPM range"); }
        if (cursor >= ppm.size() || !std::isspace(static_cast<unsigned char>(ppm[cursor]))) {
            throw std::runtime_error("invalid FFmpeg PPM header terminator");
        }
        const unsigned char separator = ppm[cursor++];
        if (separator == '\r' && cursor < ppm.size() && ppm[cursor] == '\n') { ++cursor; }
        const std::uint64_t frame_pixels =
            static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
        if (frame_pixels == 0 ||
            frame_pixels > policy.max_decoded_video_pixels - decoded_pixels) {
            throw Error(ErrorKind::BudgetExceeded, "decoded video pixel budget exceeded");
        }
        if (frame_pixels > std::numeric_limits<std::size_t>::max() / 3ULL) {
            throw Error(ErrorKind::BudgetExceeded, "decoded video byte size overflow");
        }
        const std::size_t frame_bytes = static_cast<std::size_t>(frame_pixels * 3ULL);
        if (frame_bytes > ppm.size() - cursor) {
            throw std::runtime_error("truncated FFmpeg PPM frame payload");
        }
        Image image;
        image.width = width;
        image.height = height;
        image.rgb.assign(ppm.begin() + static_cast<std::ptrdiff_t>(cursor),
                         ppm.begin() + static_cast<std::ptrdiff_t>(cursor + frame_bytes));
        cursor += frame_bytes;
        decoded_pixels += frame_pixels;
        if (video.frames.empty()) {
            video.width = width;
            video.height = height;
        } else if (video.width != width || video.height != height) {
            throw std::runtime_error("video frame dimensions changed after decoding");
        }
        video.frames.push_back(std::move(image));
    }
    if (video.frames.size() != indices.size()) {
        throw std::runtime_error("FFmpeg emitted an unexpected video frame count");
    }
    return video;
}

#else
Image decode_image(std::span<const std::uint8_t>, const Policy&) {
    throw std::runtime_error("media decoding requires FFmpeg libraries on Windows");
}
Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    throw std::runtime_error("media decoding requires FFmpeg libraries on Windows");
}
#endif

} // namespace ninfer::media::decode
