#pragma once

#include <windows.h>
#include <vulkan/vulkan.h>
#include <string>
#include <cstdint>
#include "sl.h"
#include "sl_core_api.h"
#include "sl_dlss.h"
#include "sl_dlss_d.h"
#include "sl_helpers_vk.h"

namespace interop {

class StreamlineContext {
public:
    StreamlineContext() = default;
    ~StreamlineContext();
    StreamlineContext(const StreamlineContext&) = delete;
    StreamlineContext& operator=(const StreamlineContext&) = delete;

    bool initialize(const std::wstring& pluginsDir, const std::wstring& logDir);
    void shutdown();
    bool enabled() const { return module_ != nullptr && initialized_; }

    PFN_vkGetInstanceProcAddr instanceProcAddrProxy() const { return vkGetInstanceProcAddrProxy_; }
    PFN_vkGetDeviceProcAddr deviceProcAddrProxy() const { return vkGetDeviceProcAddrProxy_; }
    PFN_vkCreateInstance createInstanceProxy() const { return vkCreateInstanceProxy_; }
    PFN_vkCreateDevice createDeviceProxy() const { return vkCreateDeviceProxy_; }

    const sl::FeatureRequirements& featureRequirements() const { return featureRequirements_; }
    bool setVulkanInfo(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
                       uint32_t computeQueueIndex, uint32_t computeQueueFamily,
                       uint32_t graphicsQueueIndex, uint32_t graphicsQueueFamily);

    // diagnostic-only RR configuration probe
    bool probeDlssRrConfiguration(uint32_t outputWidth, uint32_t outputHeight);

    bool configureDlss(VkPhysicalDevice physicalDevice, uint32_t outputWidth, uint32_t outputHeight,
                      sl::DLSSMode mode, uint32_t* renderWidth, uint32_t* renderHeight);
    bool configureDlssRr(VkPhysicalDevice physicalDevice, uint32_t outputWidth, uint32_t outputHeight,
                        sl::DLSSMode mode, uint32_t* renderWidth, uint32_t* renderHeight);
    bool dlssConfigured() const { return dlssConfigured_; }
    bool dlssRrConfigured() const { return dlssRrConfigured_; }

    bool beginFrame(sl::FrameToken*& token, uint32_t frameIndex);
    bool setTags(const sl::FrameToken& token, const sl::ViewportHandle& viewport,
                 const sl::ResourceTag* tags, uint32_t count, VkCommandBuffer cmd);
    bool setConstants(const sl::Constants& constants, const sl::FrameToken& token,
                      const sl::ViewportHandle& viewport);
    bool evaluate(const sl::FrameToken& token, const sl::ViewportHandle& viewport, VkCommandBuffer cmd);
    bool evaluateRr(const sl::FrameToken& token, const sl::ViewportHandle& viewport, VkCommandBuffer cmd);

    const std::string& lastError() const { return lastError_; }
    void setError(const std::string& e) { lastError_ = e; }

private:
    bool resolveCoreExports();
    bool resolveDeviceFeatureFunctions();
    bool initialized_ = false;
    bool dlssConfigured_ = false;
    bool dlssRrConfigured_ = false;
    sl::FeatureRequirements featureRequirements_{};
    HMODULE module_ = nullptr;
    std::wstring pluginsDir_;
    std::wstring logDir_;
    std::string lastError_;

    PFun_slInit* slInit_ = nullptr;
    PFun_slShutdown* slShutdown_ = nullptr;
    PFun_slIsFeatureSupported* slIsFeatureSupported_ = nullptr;
    PFun_slGetFeatureRequirements* slGetFeatureRequirements_ = nullptr;
    PFun_slGetNewFrameToken* slGetNewFrameToken_ = nullptr;
    PFun_slSetTagForFrame* slSetTagForFrame_ = nullptr;
    PFun_slSetConstants* slSetConstants_ = nullptr;
    PFun_slEvaluateFeature* slEvaluateFeature_ = nullptr;
    PFun_slSetVulkanInfo* slSetVulkanInfo_ = nullptr;
    PFun_slGetFeatureFunction* slGetFeatureFunction_ = nullptr;
    PFun_slFreeResources* slFreeResources_ = nullptr;
    PFun_slDLSSGetOptimalSettings* slDLSSGetOptimalSettings_ = nullptr;
    PFun_slDLSSSetOptions* slDLSSSetOptions_ = nullptr;
    PFun_slDLSSDGetOptimalSettings* slDLSSDGetOptimalSettings_ = nullptr;
    PFun_slDLSSDSetOptions* slDLSSDSetOptions_ = nullptr;

    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddrProxy_ = nullptr;
    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddrProxy_ = nullptr;
    PFN_vkCreateInstance vkCreateInstanceProxy_ = nullptr;
    PFN_vkCreateDevice vkCreateDeviceProxy_ = nullptr;
};

}  // namespace interop
