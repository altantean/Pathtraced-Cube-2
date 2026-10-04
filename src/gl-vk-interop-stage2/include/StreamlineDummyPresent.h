#pragma once

#include <windows.h>
#include <vulkan/vulkan.h>
#include <string>
#include <vector>

namespace interop {

class StreamlineContext;

class StreamlineDummyPresent {
public:
    StreamlineDummyPresent() = default;
    ~StreamlineDummyPresent();
    StreamlineDummyPresent(const StreamlineDummyPresent&) = delete;
    StreamlineDummyPresent& operator=(const StreamlineDummyPresent&) = delete;

    // called once, from InteropContext::createLogicalDeviceAndQueue(), right after queue_ is fetched
    bool initialize(StreamlineContext& sl, VkInstance instance, VkPhysicalDevice physicalDevice,
                     VkDevice device, VkQueue queue, uint32_t queueFamily, std::string* error);

    bool presentOnce(std::string* error);

    void shutdown();
    bool ready() const { return swapchain_ != VK_NULL_HANDLE; }

private:
    bool createSwapchain(VkSurfaceKHR surface, std::string* error);
    bool createSyncObjects(std::string* error);
    void destroySyncObjects();

    VkDevice device_ = VK_NULL_HANDLE;  // non-owning, borrowed from InteropContext
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;  // non-owning
    VkQueue queue_ = VK_NULL_HANDLE;  // non-owning
    uint32_t queueFamily_ = UINT32_MAX;

    HINSTANCE hInstance_ = nullptr;
    HWND hwnd_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;  // non-owning, borrowed from InteropContext
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> images_;  // borrowed from the swapchain, not individually owned

    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> cmdBuffers_;
    std::vector<VkFence> inFlightFences_;
    std::vector<VkSemaphore> imageAvailableSemaphores_;  // indexed by currentFrame_
    std::vector<VkSemaphore> renderFinishedSemaphores_;  // indexed by the acquired image index
    uint32_t currentFrame_ = 0;

    PFN_vkCreateWin32SurfaceKHR vkCreateWin32SurfaceKHRProxy_ = nullptr;
    PFN_vkDestroySurfaceKHR vkDestroySurfaceKHRProxy_ = nullptr;
    PFN_vkCreateSwapchainKHR vkCreateSwapchainKHRProxy_ = nullptr;
    PFN_vkDestroySwapchainKHR vkDestroySwapchainKHRProxy_ = nullptr;
    PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHRProxy_ = nullptr;
    PFN_vkAcquireNextImageKHR vkAcquireNextImageKHRProxy_ = nullptr;
    PFN_vkQueuePresentKHR vkQueuePresentKHRProxy_ = nullptr;
};

}  // namespace interop
