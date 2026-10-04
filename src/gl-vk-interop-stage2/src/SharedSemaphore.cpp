#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "SharedSemaphore.h"
#include "InteropContext.h"
#include "gl_ext_loader.h"

namespace interop {

namespace {
using PFN_vkGetSemaphoreWin32HandleKHR_ = VkResult (VKAPI_PTR*)(
    VkDevice, const VkSemaphoreGetWin32HandleInfoKHR*, HANDLE*);
}  // namespace

SharedSemaphore::SharedSemaphore(InteropContext& ctx) : ctx_(ctx) {}

SharedSemaphore::~SharedSemaphore() {
    if (glSemaphore_ != 0) {
        GlExtFunctions gl{};
        std::string ignored;
        if (loadGlExtFunctions(gl, &ignored)) {
            gl.glDeleteSemaphoresEXT(1, &glSemaphore_);
        }
    }
    if (win32SemaphoreHandle_ != nullptr) {
        CloseHandle(win32SemaphoreHandle_);
    }
    if (semaphore_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(ctx_.device(), semaphore_, nullptr);
    }
}

bool SharedSemaphore::create() {
    if (!createVulkanSemaphore()) return false;
    if (!exportWin32Handle()) return false;
    if (!importIntoOpenGL()) return false;
    return true;
}

bool SharedSemaphore::createVulkanSemaphore() {
    // flags this semaphore as exportable at creation time
    VkExportSemaphoreCreateInfo exportInfo{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkSemaphoreCreateInfo createInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    createInfo.pNext = &exportInfo;

    VkResult result = vkCreateSemaphore(ctx_.device(), &createInfo, nullptr, &semaphore_);
    if (result != VK_SUCCESS) {
        ctx_.setError("vkCreateSemaphore (exportable) failed with VkResult " + std::to_string(result));
        return false;
    }
    return true;
}

bool SharedSemaphore::exportWin32Handle() {
    auto vkGetSemaphoreWin32HandleKHR_ = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR_>(
        ctx_.loadDeviceProc("vkGetSemaphoreWin32HandleKHR"));
    if (vkGetSemaphoreWin32HandleKHR_ == nullptr) return false;

    VkSemaphoreGetWin32HandleInfoKHR getInfo{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
    getInfo.semaphore = semaphore_;
    getInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkResult result = vkGetSemaphoreWin32HandleKHR_(ctx_.device(), &getInfo, &win32SemaphoreHandle_);
    if (result != VK_SUCCESS) {
        ctx_.setError("vkGetSemaphoreWin32HandleKHR failed with VkResult " + std::to_string(result));
        return false;
    }
    return true;
}

bool SharedSemaphore::importIntoOpenGL() {
    GlExtFunctions gl{};
    std::string error;
    if (!loadGlExtFunctions(gl, &error)) {
        ctx_.setError("SharedSemaphore::importIntoOpenGL: " + error);
        return false;
    }

    gl.glGenSemaphoresEXT(1, &glSemaphore_);
    gl.glImportSemaphoreWin32HandleEXT(glSemaphore_, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT_,
                                        win32SemaphoreHandle_);

    CloseHandle(win32SemaphoreHandle_);
    win32SemaphoreHandle_ = nullptr;

    return true;
}

}  // namespace interop
