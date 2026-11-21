#pragma once

#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

namespace ninfer::targets::qwen3_6 {

class FrontendTestAccess {
public:
    [[nodiscard]] static Frontend create_component(const FrontendResources& resources,
                                                   bool vision_enabled = true);
    [[nodiscard]] static const PreparedPromptData& inspect(const PreparedPrompt& prompt);
    [[nodiscard]] static std::size_t retained_media_bytes(const Frontend& frontend);
};

} // namespace ninfer::targets::qwen3_6
