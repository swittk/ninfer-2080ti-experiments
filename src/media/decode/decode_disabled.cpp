#include "media/decode/decode.h"

#include <stdexcept>

namespace ninfer::media::decode {
namespace {

[[noreturn]] void media_decode_disabled() {
    throw std::runtime_error(
        "image/video decoding is unavailable in this NInfer build; rebuild with "
        "NINFER_ENABLE_MEDIA_DECODE=ON and FFmpeg development libraries");
}

} // namespace

Image decode_image(std::span<const std::uint8_t>, const Policy&) { media_decode_disabled(); }

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    media_decode_disabled();
}

} // namespace ninfer::media::decode
