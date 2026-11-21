#include "ops/linear/w8/w8_vocab_b2_partial.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

int main() {
    using Geometry = ninfer::ops::detail::W8VocabB2PartialGeometry;
    int failures   = 0;

    for (int partial = 0; partial < Geometry::kPartialBlocks - 1; ++partial) {
        for (int rank = 0; rank < Geometry::kCandidateCount; ++rank) {
            if (!Geometry::rank_written(partial, rank)) { ++failures; }
        }
    }

    int valid_tail_ranks = 0;
    std::array<std::uint64_t, Geometry::kCandidateCount> tail{};
    for (int rank = 0; rank < Geometry::kCandidateCount; ++rank) {
        if (Geometry::rank_written(Geometry::kPartialBlocks - 1, rank)) {
            tail[rank] = static_cast<std::uint64_t>(rank + 1);
            ++valid_tail_ranks;
        } else {
            // Simulate stale data from a prior graph replay. It must never participate.
            tail[rank] = std::numeric_limits<std::uint64_t>::max();
        }
    }
    if (valid_tail_ranks != Geometry::kLastPartialRanks) { ++failures; }

    std::uint64_t unmasked_best = 0;
    std::uint64_t masked_best   = 0;
    for (int rank = 0; rank < Geometry::kCandidateCount; ++rank) {
        if (tail[rank] > unmasked_best) { unmasked_best = tail[rank]; }
        if (Geometry::rank_written(Geometry::kPartialBlocks - 1, rank) &&
            tail[rank] > masked_best) {
            masked_best = tail[rank];
        }
    }
    // The old unmasked reader consumes stale tail entries; the fixed reader cannot.
    if (unmasked_best != std::numeric_limits<std::uint64_t>::max()) { ++failures; }
    if (masked_best != static_cast<std::uint64_t>(Geometry::kLastPartialRanks)) { ++failures; }

    std::cout << (failures ? "FAIL" : "OK") << " W8 B2 partial tail mask\n";
    return failures;
}
