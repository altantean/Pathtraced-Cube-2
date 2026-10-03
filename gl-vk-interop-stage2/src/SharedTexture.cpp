#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "SharedTexture.h"
#include "InteropContext.h"
#include "gl_ext_loader.h"
#include <iostream>

namespace interop {

namespace {
using PFN_vkGetMemoryWin32HandleKHR_ = VkResult (VKAPI_PTR*)(
    VkDevice, const VkMemoryGetWin32HandleInfoKHR*, HANDLE*);
}  // namespace

VkFormat toVkFormat(PixelFormat format) {
    switch (format) {
        case PixelFormat::Rgba8Unorm: return VK_FORMAT_R8G8B8A8_UNORM;
        case PixelFormat::Rgba16Float: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case PixelFormat::Rg16Float: return VK_FORMAT_R16G16_SFLOAT;
        case PixelFormat::Depth32Float: return VK_FORMAT_D32_SFLOAT;
        case PixelFormat::Rgb16Float: return VK_FORMAT_R16G16B16_SFLOAT;
        case PixelFormat::R8Unorm: return VK_FORMAT_R8_UNORM;
        case PixelFormat::Rgba32Float: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case PixelFormat::R16Float: return VK_FORMAT_R16_SFLOAT;
    }
    return VK_FORMAT_UNDEFINED;
}

uint32_t toGlInternalFormat(PixelFormat format) {
    switch (format) {
        case PixelFormat::Rgba8Unorm: return GL_RGBA8_;
        case PixelFormat::Rgba16Float: return GL_RGBA16F_;
        case PixelFormat::Rg16Float: return GL_RG16F_;
        case PixelFormat::Depth32Float: return GL_DEPTH_COMPONENT32F_;
        case PixelFormat::Rgb16Float: return GL_RGB16F_;
        case PixelFormat::R8Unorm: return GL_R8_;
        case PixelFormat::Rgba32Float: return GL_RGBA32F_;
        case PixelFormat::R16Float: return GL_R16F_;
    }
    return 0;
}

SharedTexture::SharedTexture(InteropContext& ctx) : ctx_(ctx) {}

SharedTexture::~SharedTexture() {
    if (ctx_.device() != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(ctx_.device());
    }

    if (glTexture_ != 0 || glMemoryObject_ != 0) {
        GlExtFunctions gl{};
        std::string ignored;
        if (loadGlExtFunctions(gl, &ignored)) {
            if (glTexture_ != 0) gl.glDeleteTextures(1, &glTexture_);
            if (glMemoryObject_ != 0) gl.glDeleteMemoryObjectsEXT(1, &glMemoryObject_);
        }
    }
    if (win32MemoryHandle_ != nullptr) {
        CloseHandle(win32MemoryHandle_);
    }
    if (image_ != VK_NULL_HANDLE) {
        vkDestroyImage(ctx_.device(), image_, nullptr);
    }
    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(ctx_.device(), memory_, nullptr);
    }
}

bool SharedTexture::create(uint32_t width, uint32_t height, PixelFormat format) {
    width_ = width;
    height_ = height;
    format_ = toVkFormat(format);
    pixelFormat_ = format;
    glInternalFormat_ = toGlInternalFormat(format);

    const FormatSupport& support = ctx_.queryFormatSupport(format_);
    if (!support.exportable) {
        ctx_.setError("SharedTexture::create: VkFormat " + std::to_string(format_) +
                       " is not confirmed exportable on this device (see InteropContext::"
                       "queryFormatSupport's log output) -- refusing to proceed rather than fail "
                       "deep inside vkAllocateMemory with a less specific error.");
        return false;
    }
    if (!createVulkanImage()) return false;
    if (!allocateExportableDedicatedMemory()) return false;
    if (!exportWin32Handle()) return false;
    if (!importIntoOpenGL()) return false;
    if (!ctx_.transitionImageToGeneral(image_, pixelFormat_ == PixelFormat::Depth32Float ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT)) return false;
    return true;
}

bool SharedTexture::createVulkanImage() {
    VkExternalMemoryImageCreateInfo externalInfo{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    externalInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.pNext = &externalInfo;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format_;
    imageInfo.extent = {width_, height_, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    // not every externally shareable format supports storage
    if (pixelFormat_ == PixelFormat::Depth32Float) {
        imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                          VK_IMAGE_USAGE_SAMPLED_BIT |
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                          VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    } else if (pixelFormat_ == PixelFormat::Rgb16Float) {
        imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    } else if (pixelFormat_ == PixelFormat::R8Unorm || pixelFormat_ == PixelFormat::R16Float) {
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    } else {
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;  // see InteropFrame, first-use transition happens there, once

    VkResult result = vkCreateImage(ctx_.device(), &imageInfo, nullptr, &image_);
    if (result != VK_SUCCESS) {
        ctx_.setError("vkCreateImage failed with VkResult " + std::to_string(result));
        return false;
    }
    return true;
}

bool SharedTexture::allocateExportableDedicatedMemory() {
    VkMemoryDedicatedRequirements dedicatedReqs{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 memReqs2{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    memReqs2.pNext = &dedicatedReqs;

    VkImageMemoryRequirementsInfo2 reqInfo{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2};
    reqInfo.image = image_;
    vkGetImageMemoryRequirements2(ctx_.device(), &reqInfo, &memReqs2);

    std::cerr << "[interop] SharedTexture(" << width_ << "x" << height_ << ", format=" << format_
              << "): requiresDedicatedAllocation=" << dedicatedReqs.requiresDedicatedAllocation
              << " (allocating dedicated either way) - see stage 1 documentation."
              << std::endl;

    uint32_t memoryTypeIndex = UINT32_MAX;
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(ctx_.physicalDevice(), &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        const bool typeAllowed = (memReqs2.memoryRequirements.memoryTypeBits & (1u << i)) != 0;
        const bool deviceLocal =
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        if (typeAllowed && deviceLocal) {
            memoryTypeIndex = i;
            break;
        }
    }
    if (memoryTypeIndex == UINT32_MAX) {
        ctx_.setError("SharedTexture: no DEVICE_LOCAL memory type is compatible with this image's "
                       "memoryTypeBits requirements.");
        return false;
    }

    VkExportMemoryAllocateInfo exportInfo{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkMemoryDedicatedAllocateInfo dedicatedAllocInfo{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicatedAllocInfo.image = image_;
    dedicatedAllocInfo.pNext = &exportInfo;

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.pNext = &dedicatedAllocInfo;
    allocInfo.allocationSize = memReqs2.memoryRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    allocationSize_ = allocInfo.allocationSize;

    VkResult result = vkAllocateMemory(ctx_.device(), &allocInfo, nullptr, &memory_);
    if (result != VK_SUCCESS) {
        ctx_.setError("vkAllocateMemory (exportable, dedicated) failed with VkResult " +
                       std::to_string(result));
        return false;
    }

    result = vkBindImageMemory(ctx_.device(), image_, memory_, 0);
    if (result != VK_SUCCESS) {
        ctx_.setError("vkBindImageMemory failed with VkResult " + std::to_string(result));
        return false;
    }
    return true;
}

bool SharedTexture::exportWin32Handle() {
    auto vkGetMemoryWin32HandleKHR_ = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR_>(
        ctx_.loadDeviceProc("vkGetMemoryWin32HandleKHR"));
    if (vkGetMemoryWin32HandleKHR_ == nullptr) return false;

    VkMemoryGetWin32HandleInfoKHR getInfo{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    getInfo.memory = memory_;
    getInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkResult result = vkGetMemoryWin32HandleKHR_(ctx_.device(), &getInfo, &win32MemoryHandle_);
    if (result != VK_SUCCESS) {
        ctx_.setError("vkGetMemoryWin32HandleKHR failed with VkResult " + std::to_string(result));
        return false;
    }
    return true;
}

bool SharedTexture::importIntoOpenGL() {
    GlExtFunctions gl{};
    std::string error;
    if (!loadGlExtFunctions(gl, &error)) {
        ctx_.setError("SharedTexture::importIntoOpenGL: " + error);
        return false;
    }

    drainPendingGlErrors(gl, "SharedTexture import");
    gl.glCreateMemoryObjectsEXT(1, &glMemoryObject_);
    gl.glImportMemoryWin32HandleEXT(glMemoryObject_, allocationSize_,
                                     GL_HANDLE_TYPE_OPAQUE_WIN32_EXT_, win32MemoryHandle_);
    if (checkGlError(gl, "glImportMemoryWin32HandleEXT")) {
        ctx_.setError("SharedTexture::importIntoOpenGL: glImportMemoryWin32HandleEXT reported "
                       "a GL error -- see stderr for specifics.");
        return false;
    }

    CloseHandle(win32MemoryHandle_);
    win32MemoryHandle_ = nullptr;

    gl.glCreateTextures(GL_TEXTURE_2D_, 1, &glTexture_);
    gl.glTextureStorageMem2DEXT(glTexture_, /*levels=*/1, glInternalFormat_,
                                 static_cast<GLsizei>(width_), static_cast<GLsizei>(height_),
                                 glMemoryObject_, /*offset=*/0);
    if (checkGlError(gl, "glTextureStorageMem2DEXT")) {
        ctx_.setError("SharedTexture::importIntoOpenGL: glTextureStorageMem2DEXT reported a GL "
                       "error -- see stderr for specifics. If GL_INVALID_OPERATION, check that "
                       "the requested PixelFormat's GL internalformat is actually what this "
                       "driver expects for a memory object created from this VkFormat.");
        return false;
    }

    return true;
}

}  // namespace interop
