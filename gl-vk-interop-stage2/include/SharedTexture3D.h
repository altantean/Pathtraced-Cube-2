#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <windows.h>
#include "SharedTexture.h"  // reuses the same PixelFormat enum, no need for a parallel one

namespace interop {

class InteropContext;

class SharedTexture3D {
public:
    explicit SharedTexture3D(InteropContext& ctx);
    ~SharedTexture3D();

    SharedTexture3D(const SharedTexture3D&) = delete;
    SharedTexture3D& operator=(const SharedTexture3D&) = delete;

    bool create(uint32_t width, uint32_t height, uint32_t depth, PixelFormat format);

    VkImage vkImage() const { return image_; }
    VkFormat vkFormat() const { return format_; }
    PixelFormat pixelFormat() const { return pixelFormat_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    uint32_t depth() const { return depth_; }
    uint32_t glTextureId() const { return glTexture_; }
    uint32_t glMemoryObject() const { return glMemoryObject_; }
    VkDeviceMemory vkMemory() const { return memory_; }

private:
    bool createVulkanImage();
    bool allocateExportableDedicatedMemory();
    bool exportWin32Handle();
    bool importIntoOpenGL();

    InteropContext& ctx_;

    VkImage image_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    PixelFormat pixelFormat_ = PixelFormat::Rgba16Float;
    uint32_t glInternalFormat_ = 0;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize allocationSize_ = 0;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t depth_ = 0;

    HANDLE win32MemoryHandle_ = nullptr;  // same ownership/closing rule as SharedTexture's own identical field

    uint32_t glMemoryObject_ = 0;
    uint32_t glTexture_ = 0;
};

}  // namespace interop
