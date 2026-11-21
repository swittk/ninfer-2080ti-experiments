#pragma once
#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::runtime {
/** Request-owned exact-history proposal index. It never licenses output tokens.
 * The caller supplies its current append-only committed ledger, including the
 * current anchor. No token payload or index is shared between requests.
 */
class SessionCopyIndex {
public:
    static constexpr std::size_t match_tokens=32;
    static constexpr std::size_t history_limit=65536;
    static constexpr std::size_t bucket_count=16384;
    static constexpr std::size_t ways=4;

    void reset() noexcept {
        for(auto& bucket:buckets_)bucket={};
        indexed_end_=0;seen_size_=0;rolling_=0;
    }

    /** Return only already-observed tokens after an exact 32-token match.
     * Hashes merely locate candidates; both the suffix and source bounds are
     * checked against the current ledger before any proposal is returned.
     */
    bool propose(std::span<const std::int32_t> history,std::span<std::int32_t> draft) {
        if(draft.empty() || draft.size()>7 || history.size()<2*match_tokens+draft.size())return false;
        if(history.size()<seen_size_)reset();
        seen_size_=history.size();
        const auto limit=history.size()-match_tokens-draft.size();
        if(buckets_.empty())buckets_.resize(bucket_count);
        if(indexed_end_==0 || limit>indexed_end_+history_limit) {
            indexed_end_=std::max(match_tokens,limit-history_limit_safe(limit)+1);
            rolling_=hash(history.subspan(indexed_end_-match_tokens,match_tokens));
            insert(rolling_,indexed_end_);
        }
        while(indexed_end_<limit) {
            rolling_=(rolling_-word(history[indexed_end_-match_tokens])*power())*base+
                     word(history[indexed_end_]);
            ++indexed_end_;insert(rolling_,indexed_end_);
        }
        const auto suffix=history.last(match_tokens);
        const auto key=hash(suffix);
        const auto& bucket=buckets_[key&(bucket_count-1)];
        const Entry* selected=nullptr;
        for(const auto& entry:bucket) {
            if(entry.end<match_tokens || entry.key!=key || entry.end>limit ||
               entry.end+history_limit<history.size())continue;
            if(!std::equal(suffix.begin(),suffix.end(),history.begin()+entry.end-match_tokens))continue;
            if(selected==nullptr || entry.end>selected->end)selected=&entry;
        }
        if(selected==nullptr)return false;
        std::copy_n(history.begin()+selected->end,draft.size(),draft.begin());
        return true;
    }

    [[nodiscard]] std::size_t storage_bytes() const noexcept {
        return buckets_.capacity()*sizeof(Bucket);
    }
private:
    struct Entry {std::uint64_t key=0;std::size_t end=0;};
    using Bucket=std::array<Entry,ways>;
    static constexpr std::uint64_t base=0x100000001b3ULL;
    static constexpr std::uint64_t word(std::int32_t token) noexcept {
        return static_cast<std::uint32_t>(token)+1ULL;
    }
    static constexpr std::uint64_t power() noexcept {
        std::uint64_t p=1;for(std::size_t i=1;i<match_tokens;++i)p*=base;return p;
    }
    static std::uint64_t hash(std::span<const std::int32_t> tokens) noexcept {
        std::uint64_t h=0;for(auto token:tokens)h=h*base+word(token);return h;
    }
    static constexpr std::size_t history_limit_safe(std::size_t end) noexcept {
        return std::min(end-match_tokens+1,history_limit);
    }
    void insert(std::uint64_t key,std::size_t end) noexcept {
        auto& bucket=buckets_[key&(bucket_count-1)];
        auto* oldest=&bucket[0];
        for(auto& entry:bucket)if(entry.end<oldest->end)oldest=&entry;
        *oldest={key,end};
    }
    std::vector<Bucket> buckets_;
    std::size_t indexed_end_=0,seen_size_=0;
    std::uint64_t rolling_=0;
};
} // namespace ninfer::runtime
