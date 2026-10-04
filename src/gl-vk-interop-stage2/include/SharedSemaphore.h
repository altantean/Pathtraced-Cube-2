#pragma once

#include <vulkan/vulkan.h>
#include <windows.h>

namespace interop {

class InteropContext;

class SharedSemaphore {
public:
    explicit SharedSemaphore(InteropContext& ctx);
    ~SharedSemaphore();

    SharedSemaphore(const SharedSemaphore&) = delete;
    SharedSemaphore& operator=(const SharedSemaphore&) = delete;

    bool create();

    VkSemaphore vkSemaphore() const { return semaphore_; }
    uint32_t glSemaphore() const { return glSemaphore_; }

private:
    bool createVulkanSemaphore();
    bool exportWin32Handle();
    bool importIntoOpenGL();

    InteropContext& ctx_;

    VkSemaphore semaphore_ = VK_NULL_HANDLE;
    HANDLE win32SemaphoreHandle_ = nullptr;
    uint32_t glSemaphore_ = 0;
};

}  // namespace interop
