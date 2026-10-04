#pragma once
// resolves the prompt's `executeVulkan(resources, -> { /* Vulkan work */ })` sketch

#include <vulkan/vulkan.h>
#include <cstdint>
#include <unordered_map>
#include "SharedTexture.h"  // PixelFormat

namespace interop {

enum class InteropOperation : int32_t {
    TrivialInvert = 0,

};

// one registered operation's compiled Vulkan pipeline(s) + the plumbing needed to bind resources to
// them
struct OperationPipeline {
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;  // shared across all format variants
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;  // shared across all format variants

    struct FormatVariant {
        VkShaderModule shaderModule = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
    };
    std::unordered_map<int32_t, FormatVariant> variants;  // keyed by (int32_t)PixelFormat

    const FormatVariant* variantFor(PixelFormat format) const {
        auto it = variants.find(static_cast<int32_t>(format));
        return it == variants.end() ? nullptr : &it->second;
    }

    uint32_t localSizeX = 16;
    uint32_t localSizeY = 16;
};

}  // namespace interop

