#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <unordered_map>
#include "InteropOperation.h"
#include "StreamlineContext.h"
#include "StreamlineDummyPresent.h"

namespace interop {

struct Capabilities {
    bool hasExternalMemoryWin32 = false;
    bool hasExternalSemaphoreWin32 = false;
    bool hasDedicatedAllocation = false;
    bool hasRayTracing = false;
    bool hasRayQuery = false;
    bool hasShaderClock = false;
    bool hasPipelineStats = false;
    bool hasRtPipeline = false;
    bool hasInvocationReorder = false;
    bool hasInvocationReorderNV = false;
};

// cached result of one VkFormat's external-memory capability query, see
// InteropContext::queryFormatSupport
struct FormatSupport {
    bool queried = false;
    bool exportable = false;
    bool importable = false;
};

class InteropContext {
public:
    InteropContext() = default;
    ~InteropContext();

    InteropContext(const InteropContext&) = delete;
    InteropContext& operator=(const InteropContext&) = delete;

    bool initialize(const std::string& shaderDirectory, const std::wstring& streamlinePluginsDir = L"", const std::wstring& streamlineLogDir = L"");

    const Capabilities& capabilities() const { return caps_; }
    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    VkDevice device() const { return device_; }
    VkQueue queue() const { return queue_; }
    uint32_t queueFamilyIndex() const { return queueFamilyIndex_; }
    VkCommandPool setupCommandPool() const { return setupCommandPool_; }

    const std::string& lastError() const { return lastError_; }
    void setError(const std::string& message) { lastError_ = message; }

    void recordValidationMessage(const std::string& message);
    uint32_t validationErrorCount() const { return validationErrorCount_; }
    const std::string& lastValidationMessage() const { return lastValidationMessage_; }

    PFN_vkVoidFunction loadDeviceProc(const char* name);

    // returns cached support info for `format`, querying and caching it on first request
    const FormatSupport& queryFormatSupport(VkFormat format);

    // transitions `image` from VK_IMAGE_LAYOUT_UNDEFINED to VK_IMAGE_LAYOUT_GENERAL once
    bool transitionImageToGeneral(VkImage image, VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT);

    // registered pipelines for each InteropOperation
    const OperationPipeline* operationPipeline(InteropOperation op) const;

    StreamlineContext* streamline() { return streamlineEnabled_ ? &streamline_ : nullptr; }
    bool configureDlss(uint32_t outputWidth, uint32_t outputHeight, sl::DLSSMode mode, uint32_t* renderWidth, uint32_t* renderHeight);
    bool configureDlssRr(uint32_t outputWidth, uint32_t outputHeight, sl::DLSSMode mode, uint32_t* renderWidth, uint32_t* renderHeight);

    bool presentStreamlineDummyFrame(std::string* error) {
        if (!streamlineEnabled_) return true;
        return streamlineDummyPresent_.presentOnce(error);
    }

private:
    bool createInstance();
    bool selectPhysicalDevice();
    bool createLogicalDeviceAndQueue();
    // builds every currently-registered InteropOperation's VkPipeline
    bool buildOperationRegistry(const std::string& shaderDirectory);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex_ = UINT32_MAX;

    Capabilities caps_;
    std::string lastError_;
    uint32_t validationErrorCount_ = 0;
    std::string lastValidationMessage_;
    std::unordered_map<int32_t, FormatSupport> formatSupportCache_;
    std::unordered_map<int32_t, OperationPipeline> operations_;
    VkCommandPool setupCommandPool_ = VK_NULL_HANDLE;

    StreamlineContext streamline_;
    bool streamlineEnabled_ = false;
    StreamlineDummyPresent streamlineDummyPresent_;

#ifndef NDEBUG
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
#endif
};

}  // namespace interop
