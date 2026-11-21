#pragma once

namespace ninfer::ops::detail {

#if defined(__CUDACC__)
#define NINFER_W8_B2_HOST_DEVICE __host__ __device__
#else
#define NINFER_W8_B2_HOST_DEVICE
#endif

// Geometry of the local-top2 B2 vocabulary shortlist. Every complete partial receives eight
// producer CTAs, each publishing two candidates from each of eight warps. The final partial has
// only four CTAs, so only its first 64 of 128 candidate slots are written.
struct W8VocabB2PartialGeometry {
    static constexpr int kVocabularyRows = 248320;
    static constexpr int kRowsPerWarp    = 16;
    static constexpr int kWarpsPerCta    = 8;
    static constexpr int kRowsPerCta     = kRowsPerWarp * kWarpsPerCta;
    static constexpr int kCandidateCount = 128;
    static constexpr int kCtasPerPartial = 8;
    static constexpr int kProjectCtas     = kVocabularyRows / kRowsPerCta;
    static constexpr int kPartialBlocks   =
        (kProjectCtas + kCtasPerPartial - 1) / kCtasPerPartial;
    static constexpr int kLastPartialCtas =
        kProjectCtas - (kPartialBlocks - 1) * kCtasPerPartial;
    static constexpr int kLastPartialRanks = kLastPartialCtas * 2 * kWarpsPerCta;

    NINFER_W8_B2_HOST_DEVICE static constexpr bool rank_written(int absolute_partial,
                                                                  int rank) noexcept {
        return absolute_partial >= 0 && absolute_partial < kPartialBlocks && rank >= 0 &&
               rank < kCandidateCount &&
               (absolute_partial != kPartialBlocks - 1 || rank < kLastPartialRanks);
    }
};

static_assert(W8VocabB2PartialGeometry::kProjectCtas == 1940);
static_assert(W8VocabB2PartialGeometry::kPartialBlocks == 243);
static_assert(W8VocabB2PartialGeometry::kLastPartialCtas == 4);
static_assert(W8VocabB2PartialGeometry::kLastPartialRanks == 64);

#undef NINFER_W8_B2_HOST_DEVICE

} // namespace ninfer::ops::detail
