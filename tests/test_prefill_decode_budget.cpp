#include "runtime/engine/prefill_timing_budget.h"
#include <limits>
#include "runtime/engine/prefill_decode_budget.h"
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main() {
    using ninfer::runtime::DecodeProgressWindow;
    using ninfer::runtime::PrefillDurationWindow;
    // Binary-exact times test the target equation without decimal ceil noise.
    DecodeProgressWindow progress;
    if (progress.rounds(.25,24,24)!=24) return 50;
    for(int i=0;i<4;++i) progress.observe(2,.0625,3);
    if (progress.rounds(.25,24,24)!=12) return 51;
    if (progress.rounds(.2,40,24)!=24) return 52;
    if (progress.rounds(.001,24,24)!=1) return 53;
    if (progress.rounds(10,24,24)!=24) return 54;
    if (progress.rounds(.2,0,24)!=24) return 55;
    progress.observe(2,.0625,4);
    if (progress.rounds(.25,24,24)!=24) return 56; // New batch size discards old rate.
    for(int i=0;i<8;++i) progress.observe(1,.0625,4);
    if (progress.rounds(.25,24,24)!=24) return 57; // Slowest stream cannot be averaged away.
    for(int i=0;i<8;++i) progress.observe(2,.0625,4);
    if (progress.rounds(.25,24,24)!=12) return 58; // Bounded window forgets the old rate.
    progress.observe(9,std::numeric_limits<double>::quiet_NaN(),4);
    progress.observe(0,.0625,4);
    if (progress.rounds(.25,24,24)!=12) return 59;
    PrefillDurationWindow duration;
    if(duration.peak()!=0) return 60;
    duration.observe(.8);duration.observe(.2);duration.observe(.1);duration.observe(.2);
    if(duration.peak()!=.8) return 61;
    duration.observe(.2);
    if(duration.peak()!=.2) return 62;
    for(double bad : {-1.0,1001.0,std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
        bool rejected=false;
        try {ninfer::runtime::validate_prefill_target_tps(bad);}
        catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)return 63;
    }
    ninfer::runtime::validate_prefill_target_tps(0);
    ninfer::runtime::validate_prefill_target_tps(24);
    ninfer::runtime::PrefillDecodeBudget timed(24);
    timed.prefilled(true,6);
    for(int i=0;i<6;++i){if(!timed.should_decode(true))return 64;timed.decoded();}
    if(timed.should_decode(true))return 65;
    timed.prefilled(true,100);
    for(int i=0;i<24;++i){if(!timed.should_decode(true))return 66;timed.decoded();}
    if(timed.should_decode(true))return 67;
    timed.prefilled(false,6);timed.decoded();
    if(timed.should_decode(true))return 68;

    using ninfer::runtime::PrefillDecodeBudget;
    for (const std::uint32_t limit : {1U, 4U, 16U, 64U}) {
        PrefillDecodeBudget budget(limit);
        for (int cycle = 0; cycle < 100; ++cycle) {
            if (budget.should_decode(false)) { return 1; }
            for (std::uint32_t step = 0; step < limit; ++step) {
                if (!budget.should_decode(true)) { return 2; }
                // A temporarily empty decode cohort must not block a prefill step.
                if (budget.should_decode(false)) { return 3; }
                budget.decoded();
            }
            if (budget.should_decode(true)) { return 4; }
            budget.decoded(); // Saturation must not wrap the unsigned counter.
            if (budget.should_decode(true)) { return 5; }
            budget.prefilled();
        }
    }
    PrefillDecodeBudget warm(64);
    warm.prefilled(false);
    if (!warm.should_decode(true)) { return 7; }
    warm.decoded();
    if (warm.should_decode(true)) { return 8; }
    warm.prefilled(true);
    for (int step = 0; step < 64; ++step) {
        if (!warm.should_decode(true)) { return 9; }
        warm.decoded();
    }
    using Clock = std::chrono::steady_clock;
    const auto arrival = Clock::time_point{} + std::chrono::seconds(10);
    const auto earlier = arrival - std::chrono::seconds(1);
    const auto later = arrival + std::chrono::seconds(1);
    if (!PrefillDecodeBudget::protects_existing_stream(true, earlier, arrival) ||
        !PrefillDecodeBudget::protects_existing_stream(true, arrival, arrival) ||
        PrefillDecodeBudget::protects_existing_stream(true, later, arrival) ||
        PrefillDecodeBudget::protects_existing_stream(false, earlier, arrival) ||
        PrefillDecodeBudget::protects_existing_stream(true, std::nullopt, arrival)) { return 10; }
    if (!PrefillDecodeBudget::has_substantial_suffix(128,319,128) ||
        !PrefillDecodeBudget::has_substantial_suffix(384,63,128) ||
        !PrefillDecodeBudget::has_substantial_suffix(446,1,128) ||
        PrefillDecodeBudget::has_substantial_suffix(447,0,128) ||
        PrefillDecodeBudget::has_substantial_suffix(0,4,128) ||
        PrefillDecodeBudget::has_substantial_suffix(3,1,128) ||
        PrefillDecodeBudget::has_substantial_suffix(0,128,128)) { return 15; }
    // Two requests arrive together, but the second spends time preparing media.
    // The first stream begins during that preparation, not before the arrival.
    const auto ingress = arrival;
    const auto began_while_preparing = ingress + std::chrono::milliseconds(250);
    const auto submitted_after_preparing = ingress + std::chrono::seconds(1);
    if (PrefillDecodeBudget::protects_existing_stream(true, began_while_preparing, ingress)) {
        return 18; // Same cold cohort must not become a protected late arrival.
    }
    if (!PrefillDecodeBudget::protects_existing_stream(true, began_while_preparing,
                                                      submitted_after_preparing)) {
        return 19; // This is the old, incorrect Engine-submission interpretation.
    }
    if (!PrefillDecodeBudget::protects_existing_stream(true, earlier, ingress)) {
        return 20; // Truly established streams remain protected through preparation.
    }
    PrefillDecodeBudget retiring(16);
    retiring.decoded(); retiring.limit_to_one();
    if (!retiring.should_decode(true)) { return 11; }
    retiring.decoded(); retiring.limit_to_one();
    if (retiring.should_decode(true)) { return 12; } // No new credit / starvation.
    retiring.prefilled(false);
    if (!retiring.should_decode(true)) { return 13; }
    retiring.decoded();
    if (retiring.should_decode(true)) { return 14; } // Cold cohorts keep one-round alternation.
    for (const std::uint32_t bad : {0U, 65U, UINT32_MAX}) {
        bool rejected = false;
        try { PrefillDecodeBudget budget(bad); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) { return 6; }
    }
    std::cout << "PASS bounded decode turns, mandatory prefill progress, empty cohorts, reset and bounds\n";
    return 0;
}
