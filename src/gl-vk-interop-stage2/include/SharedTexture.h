#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <windows.h>

namespace interop {

class InteropContext;

enum class PixelFormat : int32_t {
    Rgba8Unorm = 0,
    Rgba16Float = 1,  // VK_FORMAT_R16G16B16A16_SFLOAT / GL_RGBA16F, HDR scene color
    Rg16Float = 2,  // VK_FORMAT_R16G16_SFLOAT / GL_RG16F, motion-vector-style data
    Depth32Float = 3,  // VK_FORMAT_D32_SFLOAT / GL_DEPTH_COMPONENT32F, DLSS depth
    Rgb16Float = 4,  // VK_FORMAT_R16G16B16_SFLOAT / GL_RGB16F, G-buffer normals
    R8Unorm = 5,  // VK_FORMAT_R8_UNORM / GL_R8, RTAO output
    Rgba32Float = 6,
    // the froxel sun-visibility grid is a single [0,1] scalar, temporally blended over many frames
    R16Float = 7,  // VK_FORMAT_R16_SFLOAT / GL_R16F
};

VkFormat toVkFormat(PixelFormat format);
uint32_t toGlInternalFormat(PixelFormat format);  // returns a GLenum value

class SharedTexture {
public:
    explicit SharedTexture(InteropContext& ctx);
    ~SharedTexture();

    SharedTexture(const SharedTexture&) = delete;
    SharedTexture& operator=(const SharedTexture&) = delete;

    // validates format support via ctx.queryFormatSupport before creating anything
    bool create(uint32_t width, uint32_t height, PixelFormat format);

    VkImage vkImage() const { return image_; }
    VkFormat vkFormat() const { return format_; }
    PixelFormat pixelFormat() const { return pixelFormat_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    uint32_t glTextureId() const { return glTexture_; }
    uint32_t glMemoryObject() const { return glMemoryObject_; }
    VkDeviceMemory vkMemory() const { return memory_; }
    bool isDepth() const { return pixelFormat_ == PixelFormat::Depth32Float; }

private:
    bool createVulkanImage();
    bool allocateExportableDedicatedMemory();
    bool exportWin32Handle();
    bool importIntoOpenGL();

    InteropContext& ctx_;

    VkImage image_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    PixelFormat pixelFormat_ = PixelFormat::Rgba8Unorm;
    uint32_t glInternalFormat_ = 0;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize allocationSize_ = 0;
    uint32_t width_ = 0;
    uint32_t height_ = 0;

    HANDLE win32MemoryHandle_ = nullptr;

    uint32_t glMemoryObject_ = 0;
    uint32_t glTexture_ = 0;
};

}  // namespace interop
