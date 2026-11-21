#pragma once
#include <cstddef>
#include <stdexcept>

namespace ninfer::apps::dual::detail {
// Input codes and their FP16 G64 scales form one contiguous pinned-DMA packet.
// Scale offset follows the active columns, not the maximum allocation width.
struct Tp2InputPacketLayout {
    std::size_t code_bytes;
    std::size_t scale_offset;
    std::size_t scale_bytes;
    std::size_t packet_bytes;
};
inline constexpr int kTp2Hidden = 5120;
inline constexpr int kTp2MaximumTokens = 5;
inline constexpr std::size_t kTp2ScaleBytes = 2;
constexpr Tp2InputPacketLayout tp2_input_packet_layout(int tokens) {
    if (tokens != 4 && tokens != 5)
        throw std::invalid_argument("TP2 input packet requires four or five active columns");
    const auto code = static_cast<std::size_t>(tokens) * kTp2Hidden;
    const auto offset = (code + 15U) & ~std::size_t{15U};
    const auto scales = static_cast<std::size_t>(tokens) * (kTp2Hidden / 64) * kTp2ScaleBytes;
    return {code,offset,scales,offset + scales};
}
inline constexpr auto kTp2MaximumInputPacket = tp2_input_packet_layout(kTp2MaximumTokens);
} // namespace ninfer::apps::dual::detail
