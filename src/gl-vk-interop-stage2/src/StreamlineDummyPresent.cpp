#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "StreamlineDummyPresent.h"
#include "StreamlineContext.h"
#include <cstring>

namespace interop {

namespace {

const wchar_t* kWindowClassName = L"SLDummyPresentWindow";

HWND createHiddenWindow() {
    HINSTANCE hInstance = GetModuleHandleW(nullptr);
    WNDCLASSW wc{};
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = hInstance;
    wc.lpszClassName = kWindowClassName;
    // harmless if already registered (ERROR_CLASS_ALREADY_EXISTS), fine to call every time this class
    // is constructed
    RegisterClassW(&wc);
    return CreateWindowExW(0, kWindowClassName, L"", WS_POPUP,
                            0, 0, 4, 4, nullptr, nullptr, hInstance, nullptr);
}

}  // namespace

StreamlineDummyPresent::~StreamlineDummyPresent() {
    shutdown();
}

bool StreamlineDummyPresent::initialize(StreamlineContext& sl, VkInstance instance, VkPhysicalDevice physicalDevice,
                                         VkDevice device, VkQueue queue, uint32_t queueFamily, std::string* error) {
    auto fail = [&](const std::string& msg) {
        if (error) *error = "StreamlineDummyPresent::initialize: " + msg;
        return false;
    };
    if (!sl.enabled()) return fail("Streamline is not enabled.");

    PFN_vkGetInstanceProcAddr getInst = sl.instanceProcAddrProxy();
    PFN_vkGetDeviceProcAddr getDev = sl.deviceProcAddrProxy();
    if (!getInst || !getDev) return fail("Streamline's proxy vkGetInstanceProcAddr/vkGetDeviceProcAddr are unavailable.");

    // mandatory hooks (sl_hooks.h's own "Mandatory - Vulkan" list)
    vkCreateWin32SurfaceKHRProxy_ = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(getInst(instance, "vkCreateWin32SurfaceKHR"));
    vkDestroySurfaceKHRProxy_ = reinterpret_cast<PFN_vkDestroySurfaceKHR>(getInst(instance, "vkDestroySurfaceKHR"));
    vkCreateSwapchainKHRProxy_ = reinterpret_cast<PFN_vkCreateSwapchainKHR>(getDev(device, "vkCreateSwapchainKHR"));
    vkDestroySwapchainKHRProxy_ = reinterpret_cast<PFN_vkDestroySwapchainKHR>(getDev(device, "vkDestroySwapchainKHR"));
    vkGetSwapchainImagesKHRProxy_ = reinterpret_cast<PFN_vkGetSwapchainImagesKHR>(getDev(device, "vkGetSwapchainImagesKHR"));
    vkAcquireNextImageKHRProxy_ = reinterpret_cast<PFN_vkAcquireNextImageKHR>(getDev(device, "vkAcquireNextImageKHR"));
    vkQueuePresentKHRProxy_ = reinterpret_cast<PFN_vkQueuePresentKHR>(getDev(device, "vkQueuePresentKHR"));
    if (!vkCreateWin32SurfaceKHRProxy_ || !vkDestroySurfaceKHRProxy_ || !vkCreateSwapchainKHRProxy_ ||
        !vkDestroySwapchainKHRProxy_ || !vkGetSwapchainImagesKHRProxy_ || !vkAcquireNextImageKHRProxy_ ||
        !vkQueuePresentKHRProxy_) {
        return fail("one or more mandatory-hook Vulkan entry points could not be resolved through Streamline's proxy "
                    "(VK_KHR_surface/VK_KHR_win32_surface/VK_KHR_swapchain must be enabled on the real instance/device).");
    }

    device_ = device;
    physicalDevice_ = physicalDevice;
    queue_ = queue;
    queueFamily_ = queueFamily;
    instance_ = instance;

    hInstance_ = GetModuleHandleW(nullptr);
    hwnd_ = createHiddenWindow();
    if (!hwnd_) return fail("CreateWindowExW failed for the dummy present window.");

    VkWin32SurfaceCreateInfoKHR surfaceInfo{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    surfaceInfo.hinstance = hInstance_;
    surfaceInfo.hwnd = hwnd_;
    if (vkCreateWin32SurfaceKHRProxy_(instance_, &surfaceInfo, nullptr, &surface_) != VK_SUCCESS) {
        return fail("vkCreateWin32SurfaceKHR (mandatory hook) failed.");
    }

    VkBool32 presentSupported = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice_, queueFamily_, surface_, &presentSupported);
    if (!presentSupported) {
        return fail("the real production queue family cannot present to a Win32 surface on this device/driver.");
    }

    if (!createSwapchain(surface_, error)) return false;

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = queueFamily_;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device_, &poolInfo, nullptr, &cmdPool_) != VK_SUCCESS) return fail("vkCreateCommandPool failed.");

    if (!createSyncObjects(error)) return false;

    return true;
}

bool StreamlineDummyPresent::createSyncObjects(std::string* error) {
    auto fail = [&](const std::string& msg) {
        if (error) *error = "StreamlineDummyPresent::createSyncObjects: " + msg;
        return false;
    };
    const uint32_t framesInFlight = static_cast<uint32_t>(images_.size());

    VkCommandBufferAllocateInfo cbInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbInfo.commandPool = cmdPool_;
    cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = framesInFlight;
    cmdBuffers_.resize(framesInFlight);
    if (vkAllocateCommandBuffers(device_, &cbInfo, cmdBuffers_.data()) != VK_SUCCESS) return fail("vkAllocateCommandBuffers failed.");

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    inFlightFences_.resize(framesInFlight);
    for (auto& f : inFlightFences_) {
        if (vkCreateFence(device_, &fenceInfo, nullptr, &f) != VK_SUCCESS) return fail("vkCreateFence failed.");
    }

    VkSemaphoreCreateInfo semInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    imageAvailableSemaphores_.resize(framesInFlight);
    for (auto& s : imageAvailableSemaphores_) {
        if (vkCreateSemaphore(device_, &semInfo, nullptr, &s) != VK_SUCCESS) return fail("vkCreateSemaphore (imageAvailable) failed.");
    }
    renderFinishedSemaphores_.resize(images_.size());
    for (auto& s : renderFinishedSemaphores_) {
        if (vkCreateSemaphore(device_, &semInfo, nullptr, &s) != VK_SUCCESS) return fail("vkCreateSemaphore (renderFinished) failed.");
    }
    currentFrame_ = 0;
    return true;
}

void StreamlineDummyPresent::destroySyncObjects() {
    if (device_ == VK_NULL_HANDLE) return;
    for (auto s : imageAvailableSemaphores_) if (s != VK_NULL_HANDLE) vkDestroySemaphore(device_, s, nullptr);
    for (auto s : renderFinishedSemaphores_) if (s != VK_NULL_HANDLE) vkDestroySemaphore(device_, s, nullptr);
    for (auto f : inFlightFences_) if (f != VK_NULL_HANDLE) vkDestroyFence(device_, f, nullptr);
    if (cmdPool_ != VK_NULL_HANDLE && !cmdBuffers_.empty()) {
        vkFreeCommandBuffers(device_, cmdPool_, static_cast<uint32_t>(cmdBuffers_.size()), cmdBuffers_.data());
    }
    imageAvailableSemaphores_.clear();
    renderFinishedSemaphores_.clear();
    inFlightFences_.clear();
    cmdBuffers_.clear();
}

bool StreamlineDummyPresent::createSwapchain(VkSurfaceKHR surface, std::string* error) {
    auto fail = [&](const std::string& msg) {
        if (error) *error = "StreamlineDummyPresent::createSwapchain: " + msg;
        return false;
    };
    VkSurfaceCapabilitiesKHR caps{};
    // not proxied, see initialize()'s identical note on vkGetPhysicalDeviceSurfaceSupportKHR
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_, surface, &caps);
    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    if (formatCount > 0) vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface, &formatCount, formats.data());
    VkSurfaceFormatKHR chosen = formats.empty() ? VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR} : formats[0];
    for (auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM) { chosen = f; break; }
    }

    uint32_t desiredCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && desiredCount > caps.maxImageCount) desiredCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = surface;
    sci.minImageCount = desiredCount;
    sci.imageFormat = chosen.format;
    sci.imageColorSpace = chosen.colorSpace;
    sci.imageExtent = {4, 4};
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;  // always supported by spec
    sci.clipped = VK_TRUE;
    if (vkCreateSwapchainKHRProxy_(device_, &sci, nullptr, &swapchain_) != VK_SUCCESS) {
        return fail("vkCreateSwapchainKHR (mandatory hook) failed.");
    }

    uint32_t imageCount = 0;
    vkGetSwapchainImagesKHRProxy_(device_, swapchain_, &imageCount, nullptr);
    images_.resize(imageCount);
    vkGetSwapchainImagesKHRProxy_(device_, swapchain_, &imageCount, images_.data());
    return true;
}

bool StreamlineDummyPresent::presentOnce(std::string* error) {
    if (!ready()) return true;

    MSG msg;
    while (PeekMessageW(&msg, hwnd_, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    VkFence frameFence = inFlightFences_[currentFrame_];
    vkWaitForFences(device_, 1, &frameFence, VK_TRUE, UINT64_MAX);

    uint32_t imageIndex = 0;
    VkSemaphore imageAvailable = imageAvailableSemaphores_[currentFrame_];
    VkResult acquireResult = vkAcquireNextImageKHRProxy_(device_, swapchain_, UINT64_MAX, imageAvailable, VK_NULL_HANDLE, &imageIndex);
    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        // this window is never resized in practice, but handle it the same way any swapchain consumer
        // must
        vkDeviceWaitIdle(device_);
        destroySyncObjects();
        vkDestroySwapchainKHRProxy_(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
        std::string recreateError;
        if (!createSwapchain(surface_, &recreateError) || !createSyncObjects(&recreateError)) {
            if (error) *error = "presentOnce: swapchain out-of-date, recreate failed: " + recreateError;
            return false;
        }
        return true;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        if (error) *error = "vkAcquireNextImageKHR (mandatory hook) failed, VkResult " + std::to_string(acquireResult);
        return false;
    }

    vkResetFences(device_, 1, &frameFence);

    VkCommandBuffer cmd = cmdBuffers_[currentFrame_];
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkImageMemoryBarrier toClear{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toClear.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toClear.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toClear.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toClear.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toClear.image = images_[imageIndex];
    toClear.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    toClear.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          0, 0, nullptr, 0, nullptr, 1, &toClear);

    VkClearColorValue clearColor{};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, images_[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &range);

    VkImageMemoryBarrier toPresent = toClear;
    toPresent.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    toPresent.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toPresent.dstAccessMask = 0;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                          0, 0, nullptr, 0, nullptr, 1, &toPresent);

    vkEndCommandBuffer(cmd);

    VkSemaphore renderFinished = renderFinishedSemaphores_[imageIndex];
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &imageAvailable;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &renderFinished;
    if (vkQueueSubmit(queue_, 1, &submitInfo, frameFence) != VK_SUCCESS) {
        if (error) *error = "vkQueueSubmit failed for the dummy present frame.";
        return false;
    }

    VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinished;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain_;
    presentInfo.pImageIndices = &imageIndex;
    VkResult presentResult = vkQueuePresentKHRProxy_(queue_, &presentInfo);
    if (presentResult != VK_SUCCESS && presentResult != VK_SUBOPTIMAL_KHR && presentResult != VK_ERROR_OUT_OF_DATE_KHR) {
        if (error) *error = "vkQueuePresentKHR (mandatory hook) failed, VkResult " + std::to_string(presentResult);
        return false;
    }
    currentFrame_ = (currentFrame_ + 1) % static_cast<uint32_t>(imageAvailableSemaphores_.size());
    return true;
}

void StreamlineDummyPresent::shutdown() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        destroySyncObjects();
        if (cmdPool_ != VK_NULL_HANDLE) { vkDestroyCommandPool(device_, cmdPool_, nullptr); cmdPool_ = VK_NULL_HANDLE; }
        if (swapchain_ != VK_NULL_HANDLE && vkDestroySwapchainKHRProxy_) {
            vkDestroySwapchainKHRProxy_(device_, swapchain_, nullptr);
            swapchain_ = VK_NULL_HANDLE;
        }
    }
    if (surface_ != VK_NULL_HANDLE && vkDestroySurfaceKHRProxy_) {
        vkDestroySurfaceKHRProxy_(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }
    if (hwnd_) { DestroyWindow(hwnd_); hwnd_ = nullptr; }
    images_.clear();
    device_ = VK_NULL_HANDLE;
    physicalDevice_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    instance_ = VK_NULL_HANDLE;
}

}  // namespace interop
