#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "SharedTexture3D.h"
#include "InteropContext.h"
#include "gl_ext_loader.h"
#include <iostream>

namespace interop {

// toVkFormat()/toGlInternalFormat() are already declared in SharedTexture.h and defined in
// SharedTexture.cpp

namespace {
using PFN_vkGetMemoryWin32HandleKHR_ = VkResult (VKAPI_PTR*)(
    VkDevice, const VkMemoryGetWin32HandleInfoKHR*, HANDLE*);
}  // namespace

SharedTexture3D::SharedTexture3D(InteropContext& ctx) : ctx_(ctx) {}

SharedTexture3D::~SharedTexture3D() {
    // same ordering requirement as SharedTexture::~SharedTexture()
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

bool SharedTexture3D::create(uint32_t width, uint32_t height, uint32_t depth, PixelFormat format) {
    width_ = width;
    height_ = height;
    depth_ = depth;
    format_ = toVkFormat(format);
    pixelFormat_ = format;
    glInternalFormat_ = toGlInternalFormat(format);

    const FormatSupport& support = ctx_.queryFormatSupport(format_);
    if (!support.exportable) {
        ctx_.setError("SharedTexture3D::create: VkFormat " + std::to_string(format_) +
                       " is not confirmed exportable on this device (see InteropContext::"
                       "queryFormatSupport's log output) -- refusing to proceed rather than fail "
                       "deep inside vkAllocateMemory with a less specific error.");
        return false;
    }
    if (!createVulkanImage()) return false;
    if (!allocateExportableDedicatedMemory()) return false;
    if (!exportWin32Handle()) return false;
    if (!importIntoOpenGL()) return false;
    if (!ctx_.transitionImageToGeneral(image_, VK_IMAGE_ASPECT_COLOR_BIT)) return false;
    return true;
}

bool SharedTexture3D::createVulkanImage() {
    VkExternalMemoryImageCreateInfo externalInfo{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    externalInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.pNext = &externalInfo;
    imageInfo.imageType = VK_IMAGE_TYPE_3D;
    imageInfo.format = format_;
    imageInfo.extent = {width_, height_, depth_};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkResult result = vkCreateImage(ctx_.device(), &imageInfo, nullptr, &image_);
    if (result != VK_SUCCESS) {
        ctx_.setError("SharedTexture3D: vkCreateImage failed with VkResult " + std::to_string(result));
        return false;
    }
    return true;
}

bool SharedTexture3D::allocateExportableDedicatedMemory() {
    VkMemoryDedicatedRequirements dedicatedReqs{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 memReqs2{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    memReqs2.pNext = &dedicatedReqs;

    VkImageMemoryRequirementsInfo2 reqInfo{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2};
    reqInfo.image = image_;
    vkGetImageMemoryRequirements2(ctx_.device(), &reqInfo, &memReqs2);

    std::cerr << "[interop] SharedTexture3D(" << width_ << "x" << height_ << "x" << depth_
              << ", format=" << format_ << "): requiresDedicatedAllocation="
              << dedicatedReqs.requiresDedicatedAllocation
              << " (allocating dedicated either way)"
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
        ctx_.setError("SharedTexture3D: no DEVICE_LOCAL memory type is compatible with this image's "
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
        ctx_.setError("SharedTexture3D: vkAllocateMemory (exportable, dedicated) failed with VkResult " +
                       std::to_string(result));
        return false;
    }

    result = vkBindImageMemory(ctx_.device(), image_, memory_, 0);
    if (result != VK_SUCCESS) {
        ctx_.setError("SharedTexture3D: vkBindImageMemory failed with VkResult " + std::to_string(result));
        return false;
    }
    return true;
}

bool SharedTexture3D::exportWin32Handle() {
    auto vkGetMemoryWin32HandleKHR_ = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR_>(
        ctx_.loadDeviceProc("vkGetMemoryWin32HandleKHR"));
    if (vkGetMemoryWin32HandleKHR_ == nullptr) return false;

    VkMemoryGetWin32HandleInfoKHR getInfo{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    getInfo.memory = memory_;
    getInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkResult result = vkGetMemoryWin32HandleKHR_(ctx_.device(), &getInfo, &win32MemoryHandle_);
    if (result != VK_SUCCESS) {
        ctx_.setError("SharedTexture3D: vkGetMemoryWin32HandleKHR failed with VkResult " + std::to_string(result));
        return false;
    }
    return true;
}

bool SharedTexture3D::importIntoOpenGL() {
    GlExtFunctions gl{};
    std::string error;
    if (!loadGlExtFunctions(gl, &error)) {
        ctx_.setError("SharedTexture3D::importIntoOpenGL: " + error);
        return false;
    }
    if (gl.glTextureStorageMem3DEXT == nullptr) {
        ctx_.setError("SharedTexture3D::importIntoOpenGL: glTextureStorageMem3DEXT could not be "
                       "resolved -- this driver may not expose the 3D-texture entry point of "
                       "GL_EXT_memory_object (confirmed present for the 2D case, since "
                       "loadGlExtFunctions succeeded above).");
        return false;
    }

    drainPendingGlErrors(gl, "SharedTexture3D import");
    gl.glCreateMemoryObjectsEXT(1, &glMemoryObject_);
    gl.glImportMemoryWin32HandleEXT(glMemoryObject_, allocationSize_,
                                     GL_HANDLE_TYPE_OPAQUE_WIN32_EXT_, win32MemoryHandle_);
    if (checkGlError(gl, "glImportMemoryWin32HandleEXT (3D)")) {
        ctx_.setError("SharedTexture3D::importIntoOpenGL: glImportMemoryWin32HandleEXT reported "
                       "a GL error -- see stderr for specifics.");
        return false;
    }

    CloseHandle(win32MemoryHandle_);
    win32MemoryHandle_ = nullptr;

    gl.glCreateTextures(GL_TEXTURE_3D_, 1, &glTexture_);
    gl.glTextureStorageMem3DEXT(glTexture_, /*levels=*/1, glInternalFormat_,
                                 static_cast<GLsizei>(width_), static_cast<GLsizei>(height_),
                                 static_cast<GLsizei>(depth_), glMemoryObject_, /*offset=*/0);
    if (checkGlError(gl, "glTextureStorageMem3DEXT")) {
        ctx_.setError("SharedTexture3D::importIntoOpenGL: glTextureStorageMem3DEXT reported a GL "
                       "error -- see stderr for specifics.");
        return false;
    }

    if (gl.glTextureParameteri != nullptr) {
        gl.glTextureParameteri(glTexture_, GL_TEXTURE_MIN_FILTER_, static_cast<GLint>(GL_LINEAR_));
        gl.glTextureParameteri(glTexture_, GL_TEXTURE_MAG_FILTER_, static_cast<GLint>(GL_LINEAR_));
        gl.glTextureParameteri(glTexture_, GL_TEXTURE_WRAP_S_, static_cast<GLint>(GL_CLAMP_TO_EDGE_));
        gl.glTextureParameteri(glTexture_, GL_TEXTURE_WRAP_T_, static_cast<GLint>(GL_CLAMP_TO_EDGE_));
        gl.glTextureParameteri(glTexture_, GL_TEXTURE_WRAP_R_, static_cast<GLint>(GL_CLAMP_TO_EDGE_));
        if (checkGlError(gl, "glTextureParameteri (volfog filtering)")) {
            ctx_.setError("SharedTexture3D::importIntoOpenGL: glTextureParameteri reported a GL "
                           "error -- see stderr for specifics.");
            return false;
        }
    } else {
        ctx_.setError("SharedTexture3D::importIntoOpenGL: glTextureParameteri could not be "
                       "resolved -- refusing to hand back an unfiltered (GL_NEAREST-degraded) "
                       "froxel texture rather than silently produce visible banding.");
        return false;
    }

    return true;
}

}  // namespace interop
