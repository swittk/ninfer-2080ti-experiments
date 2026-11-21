#include "targets/qwen3_6_27b/impl/variant.h"
#include <ninfer/targets/qwen3_6/vision_limits.h>
#define NINFER_QWEN36_VARIANT ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

int main() {
    namespace family = ninfer::targets::qwen3_6;
    using Vision = family::detail::qwen3_6_27b_runtime::schedule::VisionContext;
    const auto capacity = Vision::workspace_capacity_bytes(32768,384);
    const int shapes[][3]={{1,16,724},{1,32,32},{5,20,44},{40,20,44},
                          {128,32,32},{384,16,16},{3,48,64},{9,24,48}};
    for(const auto& shape:shapes) {
        family::VisionItemControl item;
        item.grid={shape[0],shape[1],shape[2]};
        item.segment_count=shape[0];item.segment_length=shape[1]*shape[2];
        item.patch_count=static_cast<std::size_t>(item.segment_count)*item.segment_length;
        item.merged_count=item.patch_count/4;
        const auto request = Vision::workspace_bytes(item);
        const auto bounded = Vision::workspace_capacity_bytes(item.merged_count,item.segment_count);
        if(request>capacity||request>bounded) return 1;
    }
    const int invalid[][3]={{0,32,32},{1,0,32},{1,16,726},{129,32,32},{1,3,3}};
    for(const auto& shape:invalid) {
        family::VisionItemControl item;
        item.grid={shape[0],shape[1],shape[2]};
        item.segment_count=shape[0];item.segment_length=shape[1]*shape[2];
        item.patch_count=static_cast<std::size_t>(item.segment_count)*item.segment_length;
        item.merged_count=item.patch_count/4;
        bool rejected=false;
        try { (void)Vision::workspace_bytes(item); }
        catch(const std::invalid_argument&) {rejected=true;}
        if(!rejected) return 2;
    }
    if(Vision::workspace_capacity_bytes(16384,384)!=capacity) return 3;
    if(Vision::workspace_capacity_bytes(256,1)>=capacity) return 4;
    std::cout<<"PASS unchanged attention budget, maximum segment, long video, request/capacity bounds; capacity="<<capacity<<'\n';
    return 0;
}
