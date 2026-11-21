#pragma once
#include <array>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>

namespace ninfer::runtime {
// Pure address planner for one startup-frozen allocation. No compaction: a live
// GPU request retains its exact address until the lane is released.
template<std::size_t Slots>
class TransientSlots {
public:
    struct Slice { std::size_t offset=0, bytes=0, alignment=1; bool active=false; };
    explicit TransientSlots(std::size_t capacity) : capacity_(capacity) {}
    std::optional<std::size_t> find(std::size_t lane,std::size_t bytes,std::size_t alignment) const {
        if(lane>=Slots || alignment==0 || (alignment&(alignment-1))!=0 || alignment>256 ||
           (bytes==0 && alignment!=1)) throw std::invalid_argument("invalid transient lane/alignment");
        if(slices_[lane].active || bytes>capacity_)return {};
        if(bytes==0)return 0;
        std::size_t at=0;
        for(std::size_t step=0;step<=Slots;++step) {
            const auto padding=(alignment-at%alignment)%alignment;
            if(at>capacity_ || padding>capacity_-at)return {};
            at+=padding;
            if(bytes>capacity_-at)return {};
            bool overlaps=false;
            for(const auto&s:slices_) {
                if(s.active && s.bytes && at<s.offset+s.bytes && s.offset<at+bytes) {
                    at=s.offset+s.bytes;overlaps=true;break;
                }
            }
            if(!overlaps)return at;
        }
        return {};
    }
    Slice activate(std::size_t lane,std::size_t bytes,std::size_t alignment) {
        const auto offset=find(lane,bytes,alignment);
        if(!offset)throw std::invalid_argument("transient request does not fit frozen free regions");
        slices_[lane]={*offset,bytes,alignment,true};used_+=bytes;peak_=std::max(peak_,used_);
        return slices_[lane];
    }
    void release(std::size_t lane) noexcept {
        if(lane<Slots){used_-=slices_[lane].bytes;slices_[lane]={};}
    }
    void clear() noexcept {for(std::size_t lane=0;lane<Slots;++lane)release(lane);}
    Slice slice(std::size_t lane) const noexcept {return lane<Slots?slices_[lane]:Slice{};}
    std::size_t used() const noexcept{return used_;}
    std::size_t peak() const noexcept{return peak_;}
    void reset_peak() noexcept{peak_=used_;}
private:
    std::size_t capacity_=0,used_=0,peak_=0;
    std::array<Slice,Slots> slices_{};
};

// Bounded round-robin selection; a stalled/cancelled lane never monopolizes all
// prefill work, and removing a lane cannot invalidate any other lane's storage.
template<std::size_t Slots>
class PrefillWorkset {
public:
    void add(std::size_t lane){if(lane>=Slots || live_[lane])throw std::logic_error("prefill owner already active or invalid");live_[lane]=true;++count_;}
    void remove(std::size_t lane) noexcept{if(lane<Slots&&live_[lane]){live_[lane]=false;--count_;}}
    bool contains(std::size_t lane) const noexcept{return lane<Slots&&live_[lane];}
    std::size_t size() const noexcept{return count_;}
    bool empty() const noexcept{return count_==0;}
    void clear() noexcept{live_.fill(false);count_=0;next_=0;}
    std::size_t take_next(){
        for(std::size_t n=0;n<Slots;++n){auto lane=next_;next_=(next_+1)%Slots;if(live_[lane])return lane;}
        throw std::logic_error("no staged prefill owner");
    }
private:
    std::array<bool,Slots> live_{};std::size_t count_=0,next_=0;
};
} // namespace ninfer::runtime
