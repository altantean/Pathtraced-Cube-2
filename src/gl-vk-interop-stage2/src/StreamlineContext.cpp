#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "StreamlineContext.h"
#include <sstream>
#include <cstdio>
#include <cstdio>

namespace interop {

StreamlineContext::~StreamlineContext() { shutdown(); }

bool StreamlineContext::resolveCoreExports() {
    slInit_ = reinterpret_cast<PFun_slInit*>(GetProcAddress(module_, "slInit"));
    slShutdown_ = reinterpret_cast<PFun_slShutdown*>(GetProcAddress(module_, "slShutdown"));
    slIsFeatureSupported_ = reinterpret_cast<PFun_slIsFeatureSupported*>(GetProcAddress(module_, "slIsFeatureSupported"));
    slGetFeatureRequirements_ = reinterpret_cast<PFun_slGetFeatureRequirements*>(GetProcAddress(module_, "slGetFeatureRequirements"));
    slGetNewFrameToken_ = reinterpret_cast<PFun_slGetNewFrameToken*>(GetProcAddress(module_, "slGetNewFrameToken"));
    slSetTagForFrame_ = reinterpret_cast<PFun_slSetTagForFrame*>(GetProcAddress(module_, "slSetTagForFrame"));
    slSetConstants_ = reinterpret_cast<PFun_slSetConstants*>(GetProcAddress(module_, "slSetConstants"));
    slEvaluateFeature_ = reinterpret_cast<PFun_slEvaluateFeature*>(GetProcAddress(module_, "slEvaluateFeature"));
    slSetVulkanInfo_ = reinterpret_cast<PFun_slSetVulkanInfo*>(GetProcAddress(module_, "slSetVulkanInfo"));
    slGetFeatureFunction_ = reinterpret_cast<PFun_slGetFeatureFunction*>(GetProcAddress(module_, "slGetFeatureFunction"));
    slFreeResources_ = reinterpret_cast<PFun_slFreeResources*>(GetProcAddress(module_, "slFreeResources"));
    vkGetInstanceProcAddrProxy_ = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(module_, "vkGetInstanceProcAddr"));
    vkGetDeviceProcAddrProxy_ = reinterpret_cast<PFN_vkGetDeviceProcAddr>(GetProcAddress(module_, "vkGetDeviceProcAddr"));
    if (!slInit_ || !slShutdown_ || !slIsFeatureSupported_ || !slGetFeatureRequirements_ ||
        !slGetNewFrameToken_ || !slSetTagForFrame_ || !slSetConstants_ || !slEvaluateFeature_ ||
        !slGetFeatureFunction_ || !slFreeResources_ || !slSetVulkanInfo_ || !vkGetInstanceProcAddrProxy_ || !vkGetDeviceProcAddrProxy_) {
        lastError_ = "StreamlineContext: required Streamline 2.12.0 exports are missing.";
        return false;
    }
    vkCreateInstanceProxy_ = reinterpret_cast<PFN_vkCreateInstance>(vkGetInstanceProcAddrProxy_(VK_NULL_HANDLE, "vkCreateInstance"));
    return vkCreateInstanceProxy_ != nullptr;
}

bool StreamlineContext::initialize(const std::wstring& pluginsDir, const std::wstring& logDir) {
    if (initialized_) return true;
    if (pluginsDir.empty()) { lastError_ = "StreamlineContext: no Streamline plugin directory given."; return false; }
    pluginsDir_ = pluginsDir;
    logDir_ = logDir;
    std::wstring path = pluginsDir_ + L"\\sl.interposer.dll";
    module_ = LoadLibraryW(path.c_str());
    if (!module_) {
        lastError_ = "StreamlineContext: LoadLibraryW failed for " + std::string(path.begin(), path.end());
        return false;
    }
    if (!resolveCoreExports()) { shutdown(); return false; }

    sl::Feature features[] = { sl::kFeatureDLSS, sl::kFeatureDLSS_RR };
    const wchar_t* pluginPaths[] = { pluginsDir_.c_str() };
    sl::Preferences pref{};
    pref.showConsole = true;
    pref.logLevel = sl::LogLevel::eVerbose;
    pref.flags = sl::PreferenceFlags::eUseManualHooking | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    pref.renderAPI = sl::RenderAPI::eVulkan;
    pref.pathsToPlugins = pluginPaths;
    pref.numPathsToPlugins = 1;
    pref.pathToLogsAndData = logDir_.empty() ? nullptr : logDir_.c_str();
    // Sauerbraten is a custom engine and has no NVIDIA-issued application ID
    pref.applicationId = 0;
    pref.engine = sl::EngineType::eCustom;
    pref.engineVersion = "1.0";
    pref.projectId = "7b8f6f5e-2f7b-4a6d-9c31-5e4d1b2a8f70";
    pref.featuresToLoad = features;
    pref.numFeaturesToLoad = static_cast<uint32_t>(sizeof(features) / sizeof(features[0]));

    sl::Result result = slInit_ (pref, sl::kSDKVersion);
    if (result != sl::Result::eOk) {
        lastError_ = "StreamlineContext: slInit failed with Result " + std::to_string(static_cast<int>(result));
        shutdown();
        return false;
    }
    featureRequirements_ = {};
    result = slGetFeatureRequirements_(sl::kFeatureDLSS, featureRequirements_);
    if (result != sl::Result::eOk) {
        lastError_ = "StreamlineContext: slGetFeatureRequirements during initialization failed with Result " +
                     std::to_string(static_cast<int>(result));
        shutdown();
        return false;
    }
    initialized_ = true;
    return true;
}

bool StreamlineContext::setVulkanInfo(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
                                       uint32_t computeQueueIndex, uint32_t computeQueueFamily,
                                       uint32_t graphicsQueueIndex, uint32_t graphicsQueueFamily) {
    if (!initialized_ || !slSetVulkanInfo_) {
        lastError_ = "StreamlineContext: slSetVulkanInfo export is unavailable or Streamline is not initialized.";
        return false;
    }
    sl::VulkanInfo info{};
    info.device = device;
    info.instance = instance;
    info.physicalDevice = physicalDevice;
    info.computeQueueIndex = computeQueueIndex;
    info.computeQueueFamily = computeQueueFamily;
    info.graphicsQueueIndex = graphicsQueueIndex;
    info.graphicsQueueFamily = graphicsQueueFamily;
    sl::Result r = slSetVulkanInfo_(info);
    if (r != sl::Result::eOk) {
        lastError_ = "StreamlineContext: slSetVulkanInfo failed, Result " + std::to_string(static_cast<int>(r));
        return false;
    }

    return true;
}

bool StreamlineContext::probeDlssRrConfiguration(uint32_t outputWidth, uint32_t outputHeight) {
    if (!initialized_ || !slGetFeatureFunction_) return false;

    void* fn = nullptr;
    sl::Result r = slGetFeatureFunction_(sl::kFeatureDLSS_RR, "slDLSSDGetOptimalSettings", fn);
    std::printf("[StreamlineContext] RR slDLSSDGetOptimalSettings resolve -> %d\n", static_cast<int>(r));
    if (r != sl::Result::eOk || !fn) return false;
    slDLSSDGetOptimalSettings_ = reinterpret_cast<PFun_slDLSSDGetOptimalSettings*>(fn);

    fn = nullptr;
    r = slGetFeatureFunction_(sl::kFeatureDLSS_RR, "slDLSSDSetOptions", fn);
    std::printf("[StreamlineContext] RR slDLSSDSetOptions resolve -> %d\n", static_cast<int>(r));
    if (r != sl::Result::eOk || !fn) return false;
    slDLSSDSetOptions_ = reinterpret_cast<PFun_slDLSSDSetOptions*>(fn);

    sl::DLSSDOptions options{};
    options.mode = sl::DLSSMode::eMaxQuality;
    options.outputWidth = outputWidth;
    options.outputHeight = outputHeight;
    options.colorBuffersHDR = sl::Boolean::eTrue;

    sl::DLSSDOptimalSettings optimal{};
    r = slDLSSDGetOptimalSettings_(options, optimal);
    std::printf("[StreamlineContext] RR optimal settings -> %d render=%ux%u\n",
                static_cast<int>(r), optimal.optimalRenderWidth, optimal.optimalRenderHeight);
    if (r != sl::Result::eOk) return false;

    sl::ViewportHandle viewport(0);
    r = slDLSSDSetOptions_(viewport, options);
    std::printf("[StreamlineContext] RR slDLSSDSetOptions -> %d\n", static_cast<int>(r));

    // leave RR switched off after the probe
    if (r == sl::Result::eOk) {
        options.mode = sl::DLSSMode::eOff;
        sl::Result off = slDLSSDSetOptions_(viewport, options);
        std::printf("[StreamlineContext] RR probe restore-off -> %d\n", static_cast<int>(off));
    }
    return r == sl::Result::eOk;
}

bool StreamlineContext::resolveDeviceFeatureFunctions() {
    if (slDLSSGetOptimalSettings_ && slDLSSSetOptions_) return true;
    void* fn = nullptr;
    if (slGetFeatureFunction_(sl::kFeatureDLSS, "slDLSSGetOptimalSettings", fn) != sl::Result::eOk || !fn) {
        lastError_ = "StreamlineContext: failed to resolve slDLSSGetOptimalSettings.";
        return false;
    }
    slDLSSGetOptimalSettings_ = reinterpret_cast<PFun_slDLSSGetOptimalSettings*>(fn);
    fn = nullptr;
    if (slGetFeatureFunction_(sl::kFeatureDLSS, "slDLSSSetOptions", fn) != sl::Result::eOk || !fn) {
        lastError_ = "StreamlineContext: failed to resolve slDLSSSetOptions.";
        return false;
    }
    slDLSSSetOptions_ = reinterpret_cast<PFun_slDLSSSetOptions*>(fn);
    return true;
}

bool StreamlineContext::configureDlss(VkPhysicalDevice physicalDevice, uint32_t outputWidth, uint32_t outputHeight,
                                      sl::DLSSMode mode, uint32_t* renderWidth, uint32_t* renderHeight) {
    if (!initialized_) { lastError_ = "StreamlineContext: not initialized."; return false; }
    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = physicalDevice;
    sl::Result r = slIsFeatureSupported_(sl::kFeatureDLSS, adapter);
    if (r != sl::Result::eOk) {
        lastError_ = "StreamlineContext: DLSS unsupported, Result " + std::to_string(static_cast<int>(r));
        return false;
    }
    sl::FeatureRequirements req{};
    r = slGetFeatureRequirements_(sl::kFeatureDLSS, req);
    if (r != sl::Result::eOk) {
        lastError_ = "StreamlineContext: slGetFeatureRequirements failed, Result " + std::to_string(static_cast<int>(r));
        return false;
    }
    if (!resolveDeviceFeatureFunctions()) return false;

    if (dlssRrConfigured_) {
        sl::ViewportHandle viewport(0);
        r = slFreeResources_(sl::kFeatureDLSS_RR, viewport);
        dlssRrConfigured_ = false;
        if (r != sl::Result::eOk) {
            lastError_ = "StreamlineContext: slFreeResources(DLSS_RR) failed while switching to DLSS, Result " +
                         std::to_string(static_cast<int>(r));
            return false;
        }
    }

    // DLSS keeps feature-specific GPU resources associated with the viewport
    if (dlssConfigured_) {
        sl::ViewportHandle viewport(0);
        r = slFreeResources_(sl::kFeatureDLSS, viewport);
        dlssConfigured_ = false;
        dlssRrConfigured_ = false;
        if (r != sl::Result::eOk) {
            lastError_ = "StreamlineContext: slFreeResources(DLSS) failed during reconfiguration, Result " +
                         std::to_string(static_cast<int>(r));
            return false;
        }
    }

    sl::DLSSOptions options{};
    options.mode = mode;
    options.outputWidth = outputWidth;
    options.outputHeight = outputHeight;
    options.colorBuffersHDR = sl::Boolean::eFalse;
    options.useAutoExposure = sl::Boolean::eFalse;

    // Streamline 2.12.0 is the current SDK used by this integration
    switch (mode) {
        case sl::DLSSMode::eDLAA:
            options.dlaaPreset = sl::DLSSPreset::ePresetK;
            break;
        case sl::DLSSMode::eMaxQuality:
            options.qualityPreset = sl::DLSSPreset::ePresetK;
            break;
        case sl::DLSSMode::eBalanced:
            options.balancedPreset = sl::DLSSPreset::ePresetK;
            break;
        case sl::DLSSMode::eMaxPerformance:
            options.performancePreset = sl::DLSSPreset::ePresetM;
            break;
        case sl::DLSSMode::eUltraPerformance:
            options.ultraPerformancePreset = sl::DLSSPreset::ePresetL;
            break;
        case sl::DLSSMode::eUltraQuality:
            break;
        default:
            break;
    }
    sl::DLSSOptimalSettings optimal{};
    r = slDLSSGetOptimalSettings_(options, optimal);
    if (r != sl::Result::eOk) {
        lastError_ = "StreamlineContext: slDLSSGetOptimalSettings failed, Result " + std::to_string(static_cast<int>(r));
        return false;
    }
    sl::ViewportHandle viewport(0);
    r = slDLSSSetOptions_(viewport, options);
    if (r != sl::Result::eOk) {
        lastError_ = "StreamlineContext: slDLSSSetOptions failed, Result " + std::to_string(static_cast<int>(r));
        return false;
    }
    *renderWidth = optimal.optimalRenderWidth;
    *renderHeight = optimal.optimalRenderHeight;
    dlssConfigured_ = true;
    return true;
}

bool StreamlineContext::configureDlssRr(VkPhysicalDevice physicalDevice, uint32_t outputWidth, uint32_t outputHeight,
                                          sl::DLSSMode mode, uint32_t* renderWidth, uint32_t* renderHeight) {
    if (!initialized_) {
        lastError_ = "StreamlineContext: not initialized.";
        return false;
    }
    if (!renderWidth || !renderHeight) {
        lastError_ = "StreamlineContext: configureDlssRr received null render-size outputs.";
        return false;
    }

    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = physicalDevice;
    sl::Result r = slIsFeatureSupported_(sl::kFeatureDLSS_RR, adapter);
    if (r != sl::Result::eOk) {
        lastError_ = "StreamlineContext: DLSS_RR unsupported on production physical device, Result " +
                     std::to_string(static_cast<int>(r));
        return false;
    }

    if (!slDLSSDGetOptimalSettings_ || !slDLSSDSetOptions_) {
        void* fn = nullptr;
        r = slGetFeatureFunction_(sl::kFeatureDLSS_RR, "slDLSSDGetOptimalSettings", fn);
        if (r != sl::Result::eOk || !fn) {
            lastError_ = "StreamlineContext: failed to resolve slDLSSDGetOptimalSettings, Result " +
                         std::to_string(static_cast<int>(r));
            return false;
        }
        slDLSSDGetOptimalSettings_ = reinterpret_cast<PFun_slDLSSDGetOptimalSettings*>(fn);
        fn = nullptr;
        r = slGetFeatureFunction_(sl::kFeatureDLSS_RR, "slDLSSDSetOptions", fn);
        if (r != sl::Result::eOk || !fn) {
            lastError_ = "StreamlineContext: failed to resolve slDLSSDSetOptions, Result " +
                         std::to_string(static_cast<int>(r));
            return false;
        }
        slDLSSDSetOptions_ = reinterpret_cast<PFun_slDLSSDSetOptions*>(fn);
    }

    // switching from normal DLSS SR to RR is a feature-state transition
    sl::ViewportHandle viewport(0);
    if (dlssConfigured_) {
        r = slFreeResources_(sl::kFeatureDLSS, viewport);
        dlssConfigured_ = false;
        if (r != sl::Result::eOk) {
            lastError_ = "StreamlineContext: slFreeResources(DLSS) failed while switching to DLSS_RR, Result " +
                         std::to_string(static_cast<int>(r));
            return false;
        }
    }
    if (dlssRrConfigured_) {
        r = slFreeResources_(sl::kFeatureDLSS_RR, viewport);
        dlssRrConfigured_ = false;
        if (r != sl::Result::eOk) {
            lastError_ = "StreamlineContext: slFreeResources(DLSS_RR) failed during reconfiguration, Result " +
                         std::to_string(static_cast<int>(r));
            return false;
        }
    }

    sl::DLSSDOptions options{};
    options.mode = mode;
    options.outputWidth = outputWidth;
    options.outputHeight = outputHeight;
    options.colorBuffersHDR = sl::Boolean::eTrue;
    options.normalRoughnessMode = sl::DLSSDNormalRoughnessMode::ePacked;
    options.alphaUpscalingEnabled = sl::Boolean::eFalse;

    options.dlaaPreset = sl::DLSSDPreset::ePresetD;
    options.qualityPreset = sl::DLSSDPreset::ePresetD;
    options.balancedPreset = sl::DLSSDPreset::ePresetD;
    options.performancePreset = sl::DLSSDPreset::ePresetD;
    options.ultraPerformancePreset = sl::DLSSDPreset::ePresetD;
    options.ultraQualityPreset = sl::DLSSDPreset::ePresetD;

    sl::DLSSDOptimalSettings optimal{};
    r = slDLSSDGetOptimalSettings_(options, optimal);
    if (r != sl::Result::eOk) {
        lastError_ = "StreamlineContext: slDLSSDGetOptimalSettings failed, Result " +
                     std::to_string(static_cast<int>(r));
        return false;
    }

    r = slDLSSDSetOptions_(viewport, options);
    if (r != sl::Result::eOk) {
        lastError_ = "StreamlineContext: slDLSSDSetOptions failed, Result " +
                     std::to_string(static_cast<int>(r));
        return false;
    }

    *renderWidth = optimal.optimalRenderWidth;
    *renderHeight = optimal.optimalRenderHeight;
    dlssRrConfigured_ = true;
    std::printf("[StreamlineContext] DLSS-RR configured: mode=%d output=%ux%u render=%ux%u packedNormalRoughness=1\n",
                static_cast<int>(mode), outputWidth, outputHeight, *renderWidth, *renderHeight);
    return true;
}

bool StreamlineContext::beginFrame(sl::FrameToken*& token, uint32_t frameIndex) {
    if (slGetNewFrameToken_(token, &frameIndex) != sl::Result::eOk || !token) {
        lastError_ = "StreamlineContext: slGetNewFrameToken failed.";
        return false;
    }
    return true;
}

bool StreamlineContext::setTags(const sl::FrameToken& token, const sl::ViewportHandle& viewport,
                                const sl::ResourceTag* tags, uint32_t count, VkCommandBuffer cmd) {
    sl::Result r = slSetTagForFrame_(token, viewport, tags, count, reinterpret_cast<sl::CommandBuffer*>(cmd));
    if (r != sl::Result::eOk) {
        lastError_ = "StreamlineContext: slSetTagForFrame failed, Result " + std::to_string(static_cast<int>(r));
        return false;
    }
    return true;
}

bool StreamlineContext::setConstants(const sl::Constants& constants, const sl::FrameToken& token,
                                     const sl::ViewportHandle& viewport) {
    sl::Result r = slSetConstants_(constants, token, viewport);
    if (r != sl::Result::eOk) {
        lastError_ = "StreamlineContext: slSetConstants failed, Result " + std::to_string(static_cast<int>(r));
        return false;
    }
    return true;
}

bool StreamlineContext::evaluate(const sl::FrameToken& token, const sl::ViewportHandle& viewport, VkCommandBuffer cmd) {
    const sl::BaseStructure* inputs[] = { &viewport };
    sl::Result r = slEvaluateFeature_(sl::kFeatureDLSS, token, inputs, 1, reinterpret_cast<sl::CommandBuffer*>(cmd));
    if (r != sl::Result::eOk) {
        lastError_ = "StreamlineContext: slEvaluateFeature failed, Result " + std::to_string(static_cast<int>(r));
        return false;
    }
    return true;
}

bool StreamlineContext::evaluateRr(const sl::FrameToken& token, const sl::ViewportHandle& viewport, VkCommandBuffer cmd) {
    const sl::BaseStructure* inputs[] = { &viewport };

	sl::Result r = slEvaluateFeature_(
		sl::kFeatureDLSS_RR,
		token,
		inputs,
		1,
    reinterpret_cast<sl::CommandBuffer*>(cmd));
    if (r != sl::Result::eOk) {
        lastError_ = "StreamlineContext: slEvaluateFeature(DLSS_RR) failed, Result " + std::to_string(static_cast<int>(r));
        return false;
    }
    return true;
}

void StreamlineContext::shutdown() {
    if (initialized_ && slShutdown_) {
        slShutdown_();
        initialized_ = false;
        dlssConfigured_ = false;
        dlssRrConfigured_ = false;
    }
    if (module_) {
        FreeLibrary(module_);
        module_ = nullptr;
    }
    slInit_ = nullptr; slShutdown_ = nullptr; slIsFeatureSupported_ = nullptr;
    slGetFeatureRequirements_ = nullptr; slGetNewFrameToken_ = nullptr;
    slSetTagForFrame_ = nullptr; slSetConstants_ = nullptr; slEvaluateFeature_ = nullptr;
    slGetFeatureFunction_ = nullptr; slFreeResources_ = nullptr; slSetVulkanInfo_ = nullptr; slDLSSGetOptimalSettings_ = nullptr; slDLSSSetOptions_ = nullptr; slDLSSDGetOptimalSettings_ = nullptr; slDLSSDSetOptions_ = nullptr;
    vkGetInstanceProcAddrProxy_ = nullptr; vkGetDeviceProcAddrProxy_ = nullptr; vkCreateInstanceProxy_ = nullptr; vkCreateDeviceProxy_ = nullptr;
}

}  // namespace interop
