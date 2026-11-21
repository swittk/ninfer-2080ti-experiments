#include "product/media_acquire/acquire.h"

#include <stdexcept>

namespace ninfer::product::media_acquire {

std::vector<std::uint8_t> acquire_bytes(const Source&, const Policy&) {
    throw std::runtime_error(
        "media acquisition is unavailable in this NInfer build; rebuild with "
        "NINFER_ENABLE_MEDIA_ACQUIRE=ON and libcurl >= 7.85");
}

} // namespace ninfer::product::media_acquire
