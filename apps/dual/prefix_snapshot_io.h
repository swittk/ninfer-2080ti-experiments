#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace ninfer::apps::dual::detail {

// A snapshot is immutable input, not scratch storage. Allocate without first
// zeroing gigabytes that a successful read must overwrite completely. No bytes
// escape on a short/failed read. The owner must outlive every consuming DMA.
class PrefixSnapshotBytes {
public:
    explicit PrefixSnapshotBytes(std::size_t size)
        : storage_(std::make_unique_for_overwrite<std::uint8_t[]>(size)), size_(size) {}
    [[nodiscard]] const std::uint8_t* data() const noexcept { return storage_.get(); }
    [[nodiscard]] std::uint8_t* data() noexcept { return storage_.get(); }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
private:
    std::unique_ptr<std::uint8_t[]> storage_;
    std::size_t size_;
};

inline PrefixSnapshotBytes read_prefix_snapshot_bytes(
    const std::filesystem::path& path, std::size_t expected_bytes) {
    if (expected_bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::length_error("prefix snapshot exceeds the supported stream size");
    }
    std::error_code error;
    if (std::filesystem::file_size(path, error) != expected_bytes || error) {
        throw std::runtime_error("prefix snapshot size mismatch " + path.string());
    }
    PrefixSnapshotBytes buffer(expected_bytes);
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("failed to open prefix snapshot " + path.string()); }
    if (expected_bytes != 0) {
        input.read(reinterpret_cast<char*>(buffer.data()),
                   static_cast<std::streamsize>(expected_bytes));
    }
    if (!input) { throw std::runtime_error("failed to read prefix snapshot " + path.string()); }
    return buffer;
}

} // namespace ninfer::apps::dual::detail
