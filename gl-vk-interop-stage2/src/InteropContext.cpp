#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "InteropContext.h"
#include "gl_ext_loader.h"
#include <cstring>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <utility>
#include "sl_helpers_vk.h"

namespace interop {

namespace {

#ifndef NDEBUG
VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* userData) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[VK] " << data->pMessage << std::endl;
        if (userData != nullptr) {
            auto* ctx = static_cast<InteropContext*>(userData);
            ctx->recordValidationMessage(data->pMessage);
        }
    }
    return VK_FALSE;
}
#endif

bool hasExtension(const std::vector<VkExtensionProperties>& available, const char* name) {
    for (const auto& ext : available) {
        if (std::strcmp(ext.extensionName, name) == 0) return true;
    }
    return false;
}

void addUniqueExtension(std::vector<const char*>& extensions, const char* name) {
    if (name == nullptr) return;
    for (const char* existing : extensions) {
        if (std::strcmp(existing, name) == 0) return;
    }
    extensions.push_back(name);
}

}  // namespace


InteropContext::~InteropContext() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);  // shutdown-only broad stall, see ownership.md's "Destruction order"
        // before streamline_.shutdown()/vkDestroyDevice below
        streamlineDummyPresent_.shutdown();
        if (setupCommandPool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, setupCommandPool_, nullptr);
        }
        for (auto& [id, op] : operations_) {
            for (auto& [format, variant] : op.variants) {
                if (variant.pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device_, variant.pipeline, nullptr);
                if (variant.shaderModule != VK_NULL_HANDLE) vkDestroyShaderModule(device_, variant.shaderModule, nullptr);
            }
            if (op.pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, op.pipelineLayout, nullptr);
            if (op.descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, op.descriptorSetLayout, nullptr);
        }
        if (streamlineEnabled_) { streamline_.shutdown(); streamlineEnabled_ = false; }
        vkDestroyDevice(device_, nullptr);
    } else if (streamlineEnabled_) {
        streamline_.shutdown();
        streamlineEnabled_ = false;
    }
#ifndef NDEBUG
    if (debugMessenger_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
        auto destroyFn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroyFn != nullptr) destroyFn(instance_, debugMessenger_, nullptr);
    }
#endif
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

bool InteropContext::initialize(const std::string& shaderDirectory, const std::wstring& streamlinePluginsDir, const std::wstring& streamlineLogDir) {
    if (!streamlinePluginsDir.empty()) {
        if (!streamline_.initialize(streamlinePluginsDir, streamlineLogDir)) {
            setError(streamline_.lastError());
            return false;
        }
        streamlineEnabled_ = true;
    }
    if (!createInstance()) return false;
    if (!selectPhysicalDevice()) return false;
    if (!createLogicalDeviceAndQueue()) return false;
    if (!buildOperationRegistry(shaderDirectory)) return false;
    return true;
}

bool InteropContext::createInstance() {
    uint32_t apiVersion = VK_API_VERSION_1_1;
    if (vkEnumerateInstanceVersion != nullptr) {
        uint32_t reported = 0;
        vkEnumerateInstanceVersion(&reported);
        if (reported >= VK_API_VERSION_1_1) apiVersion = reported;
    }

    uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, available.data());

    std::vector<const char*> instanceExtensions;
#ifndef NDEBUG
    if (hasExtension(available, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
#endif
    if (apiVersion < VK_API_VERSION_1_1) {
        if (!hasExtension(available, VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME) ||
            !hasExtension(available, VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME)) {
            setError("Vulkan instance reports API version below 1.1 and does not expose "
                     "VK_KHR_external_memory_capabilities/VK_KHR_external_semaphore_capabilities. "
                     "This driver cannot support the interop this library needs.");
            return false;
        }
        instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
        instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
    }

    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "gl_vk_interop";
    appInfo.applicationVersion = VK_MAKE_VERSION(2, 0, 0);
    appInfo.pEngineName = "gl_vk_interop";
    appInfo.engineVersion = VK_MAKE_VERSION(2, 0, 0);
    appInfo.apiVersion = apiVersion;

    VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    createInfo.ppEnabledExtensionNames = instanceExtensions.data();

#ifndef NDEBUG
    const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> layers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, layers.data());
    bool hasValidation = false;
    for (const auto& l : layers) {
        if (std::strcmp(l.layerName, validationLayer) == 0) { hasValidation = true; break; }
    }
    // opt-in (env SAUER_VK_VALIDATION=1)
    const char* validationEnv = std::getenv("SAUER_VK_VALIDATION");
    const bool validationWanted = validationEnv && *validationEnv && std::strcmp(validationEnv, "0") != 0;
    if (hasValidation && !validationWanted) {
        std::cerr << "[interop] Vulkan validation OFF (set SAUER_VK_VALIDATION=1 to enable)." << std::endl;
    } else if (hasValidation) {
        createInfo.enabledLayerCount = 1;
        createInfo.ppEnabledLayerNames = &validationLayer;
        std::cerr << "[interop] VK_LAYER_KHRONOS_validation ENABLED for this run -- "
                     "any [VK]-prefixed lines below are real validation output." << std::endl;
    } else {
        std::cerr << "[interop] VK_LAYER_KHRONOS_validation not found -- running WITHOUT Vulkan "
                     "validation. Install the Vulkan SDK's validation layers for Test 7."
                  << std::endl;
    }
#endif

    if (streamlineEnabled_) {
        const auto& req = streamline_.featureRequirements();
        for (uint32_t i = 0; i < req.vkNumInstanceExtensions; ++i) {
            const char* ext = req.vkInstanceExtensions[i];
            if (!hasExtension(available, ext)) {
                setError(std::string("Streamline-required instance extension is unavailable: ") + ext);
                return false;
            }
            addUniqueExtension(instanceExtensions, ext);
        }
        if (hasExtension(available, VK_KHR_SURFACE_EXTENSION_NAME) &&
            hasExtension(available, VK_KHR_WIN32_SURFACE_EXTENSION_NAME)) {
            addUniqueExtension(instanceExtensions, VK_KHR_SURFACE_EXTENSION_NAME);
            addUniqueExtension(instanceExtensions, VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
        }
    }
    createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    createInfo.ppEnabledExtensionNames = instanceExtensions.data();

    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        setError("vkCreateInstance failed with VkResult " + std::to_string(result));
        return false;
    }

#ifndef NDEBUG
    auto createDebugFn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
    if (createDebugFn != nullptr) {
        VkDebugUtilsMessengerCreateInfoEXT dbgInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        dbgInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                   VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        dbgInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dbgInfo.pfnUserCallback = debugCallback;
        dbgInfo.pUserData = this;
        createDebugFn(instance_, &dbgInfo, nullptr, &debugMessenger_);
    }
#endif

    return true;
}

bool InteropContext::selectPhysicalDevice() {
    PFNGLGETUNSIGNEDBYTEVEXTPROC_ getUnsignedBytevEXT = nullptr;
    {
        void* p = reinterpret_cast<void*>(wglGetProcAddress("glGetUnsignedBytevEXT"));
        getUnsignedBytevEXT = reinterpret_cast<PFNGLGETUNSIGNEDBYTEVEXTPROC_>(p);
    }
    if (getUnsignedBytevEXT == nullptr) {
        setError("selectPhysicalDevice: glGetUnsignedBytevEXT not available -- no current "
                 "GL context on this thread, or GL_EXT_memory_object_win32 isn't exposed. "
                 "A GL context must be current before InteropContext::initialize() is called.");
        return false;
    }
    uint8_t glLuid[8] = {};
    getUnsignedBytevEXT(GL_DEVICE_LUID_EXT_, glLuid);

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0) {
        setError("vkEnumeratePhysicalDevices returned zero physical devices.");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceIDProperties idProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &idProps;
        vkGetPhysicalDeviceProperties2(candidate, &props2);

        if (idProps.deviceLUIDValid == VK_TRUE &&
            std::memcmp(idProps.deviceLUID, glLuid, VK_LUID_SIZE) == 0) {
            physicalDevice_ = candidate;
            std::cerr << "[interop] Matched Vulkan physical device '"
                      << props2.properties.deviceName << "' to current GL context by LUID."
                      << std::endl;
            break;
        }
    }

    if (physicalDevice_ == VK_NULL_HANDLE) {
        setError("No Vulkan physical device's LUID matched the current OpenGL context's "
                 "device LUID -- failing clearly rather than silently picking a possibly-wrong "
                 "GPU (see OWNERSHIP.md and the prompt's own 'fail clearly' requirement).");
        return false;
    }

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &familyCount, families.data());

    for (uint32_t i = 0; i < familyCount; ++i) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queueFamilyIndex_ = i;
            break;
        }
    }
    if (queueFamilyIndex_ == UINT32_MAX) {
        setError("No queue family on the selected physical device supports VK_QUEUE_COMPUTE_BIT.");
        return false;
    }

    return true;
}

bool InteropContext::createLogicalDeviceAndQueue() {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount, available.data());

    const char* required[] = {
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
    };
    std::vector<const char*> deviceExtensions;
    for (const char* name : required) {
        if (!hasExtension(available, name)) {
            setError(std::string("Required device extension not available: ") + name +
                      " -- this driver cannot support Win32 GL/VK interop.");
            return false;
        }
        deviceExtensions.push_back(name);
    }
    for (const char* optional : {VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
                                  VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
                                  VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME}) {
        if (hasExtension(available, optional)) deviceExtensions.push_back(optional);
    }

    const bool rtExtsAvailable =
        hasExtension(available, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
        hasExtension(available, VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
        hasExtension(available, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

    if (streamlineEnabled_) {
        const auto& req = streamline_.featureRequirements();
        for (uint32_t i = 0; i < req.vkNumDeviceExtensions; ++i) {
            const char* ext = req.vkDeviceExtensions[i];
            if (!hasExtension(available, ext)) {
                setError(std::string("Streamline-required device extension is unavailable: ") + ext);
                return false;
            }
            addUniqueExtension(deviceExtensions, ext);
        }
        if (hasExtension(available, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
            addUniqueExtension(deviceExtensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        }
    }

    caps_.hasExternalMemoryWin32 = true;
    caps_.hasExternalSemaphoreWin32 = true;
    caps_.hasDedicatedAllocation = true;

    float queuePriority = 1.0f;
    uint32_t slComputeQueues = 0;
    uint32_t slGraphicsQueues = 0;
    if (streamlineEnabled_) {
        const auto& req = streamline_.featureRequirements();
        slComputeQueues = req.vkNumComputeQueuesRequired;
        slGraphicsQueues = req.vkNumGraphicsQueuesRequired;
    }
    const uint32_t slQueuesNeeded = std::max(slComputeQueues, slGraphicsQueues);
    const uint32_t totalQueues = 1 + slQueuesNeeded;
    std::vector<float> queuePriorities(totalQueues, 1.0f);
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = queueFamilyIndex_;
    queueInfo.queueCount = totalQueues;
    queueInfo.pQueuePriorities = queuePriorities.data();

    VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan12Features supported12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features supported13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR supportedAS{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR supportedRQ{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    supported.pNext = &supported12;
    supported12.pNext = &supported13;
    supported13.pNext = &supportedAS;
    // ray-tracing pipeline and invocation reorder support
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR supportedRTP{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT supportedSer{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT};
    const bool rtpExtAvailable = hasExtension(available, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    const bool serExtAvailable = rtpExtAvailable && hasExtension(available, VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
    supportedAS.pNext = &supportedRQ;
    if (rtpExtAvailable) { supportedAS.pNext = &supportedRTP; supportedRTP.pNext = &supportedRQ; }
    if (serExtAvailable) { supportedRTP.pNext = &supportedSer; supportedSer.pNext = &supportedRQ; }
    VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV supportedSerNV{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV};
    const bool serNvAvailable = rtpExtAvailable && hasExtension(available, VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
    if (serNvAvailable) { supportedSerNV.pNext = supportedRTP.pNext; supportedRTP.pNext = &supportedSerNV; }
    VkPhysicalDeviceShaderClockFeaturesKHR supportedClock{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
    const bool clockExtAvailable = hasExtension(available, VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
    if (clockExtAvailable) supportedRQ.pNext = &supportedClock;
    // registers / spills per shader
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR supportedPipeStats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    const char* pipeStatsEnv = getenv("SAUER_PIPELINE_STATS");
    const bool pipeStatsWanted = pipeStatsEnv && *pipeStatsEnv && hasExtension(available, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    if (pipeStatsWanted) supportedClock.pNext = &supportedPipeStats;
    vkGetPhysicalDeviceFeatures2(physicalDevice_, &supported);

    const bool rtSupported = rtExtsAvailable &&
        supported12.bufferDeviceAddress == VK_TRUE &&
        supported12.descriptorIndexing == VK_TRUE &&
        supported12.shaderSampledImageArrayNonUniformIndexing == VK_TRUE &&
        supportedAS.accelerationStructure == VK_TRUE &&
        supportedRQ.rayQuery == VK_TRUE;
    caps_.hasRayTracing = rtSupported;
    caps_.hasRayQuery = rtSupported;
    if (rtSupported) {
        addUniqueExtension(deviceExtensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        addUniqueExtension(deviceExtensions, VK_KHR_RAY_QUERY_EXTENSION_NAME);
        addUniqueExtension(deviceExtensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        if (rtpExtAvailable && supportedRTP.rayTracingPipeline == VK_TRUE) {
            addUniqueExtension(deviceExtensions, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
            caps_.hasRtPipeline = true;
            if (serExtAvailable && supportedSer.rayTracingInvocationReorder == VK_TRUE) {
                addUniqueExtension(deviceExtensions, VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
                caps_.hasInvocationReorder = true;
            }
            if (serNvAvailable && supportedSerNV.rayTracingInvocationReorder == VK_TRUE) {
                addUniqueExtension(deviceExtensions, VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
                caps_.hasInvocationReorderNV = true;
            }
        }
    }

    // Streamline's Vulkan shaders can use 64-bit values and physical-storage buffer layouts
    VkPhysicalDeviceFeatures requestedCore{};
    if (supported.features.shaderInt64 == VK_TRUE) {
        requestedCore.shaderInt64 = VK_TRUE;
    }
    if (supported.features.samplerAnisotropy == VK_TRUE) {
        requestedCore.samplerAnisotropy = VK_TRUE;
    }

    VkPhysicalDeviceVulkan12Features requested12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features requested13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR requestedAS{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR requestedRQ{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR requestedRTP{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT requestedSer{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT};
    VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV requestedSerNV{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV};
    if (streamlineEnabled_) {
        const auto& req = streamline_.featureRequirements();
        requested12 = sl::getVkPhysicalDeviceVulkan12Features(req.vkNumFeatures12, req.vkFeatures12);
        requested13 = sl::getVkPhysicalDeviceVulkan13Features(req.vkNumFeatures13, req.vkFeatures13);
        requested12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        requested13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        sl::getMergedSupportedVkPhysicalDeviceVulkanFeatures(
            reinterpret_cast<VkBaseOutStructure*>(&requested12), nullptr,
            reinterpret_cast<const VkBaseOutStructure*>(&supported12));
        sl::getMergedSupportedVkPhysicalDeviceVulkanFeatures(
            reinterpret_cast<VkBaseOutStructure*>(&requested13), nullptr,
            reinterpret_cast<const VkBaseOutStructure*>(&supported13));

        if (supported12.scalarBlockLayout == VK_TRUE) {
            requested12.scalarBlockLayout = VK_TRUE;
        }

        // Vulkan 1.2 promotes bufferDeviceAddress
        if (requested12.bufferDeviceAddress == VK_TRUE) {
            deviceExtensions.erase(
                std::remove_if(deviceExtensions.begin(), deviceExtensions.end(),
                    [](const char* name) {
                        return std::strcmp(name, VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0 ||
                               std::strcmp(name, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0;
                    }),
                deviceExtensions.end());
        }

        // Streamline's common layer uses Vulkan private-data slots
        if (supported13.privateData == VK_TRUE) {
            requested13.privateData = VK_TRUE;
        } else {
            setError("Streamline/DLSS requires Vulkan 1.3 privateData, but the selected physical device does not support it.");
            return false;
        }
    }

    if (rtSupported) {
        requested12.bufferDeviceAddress = VK_TRUE;
        requested12.descriptorIndexing = VK_TRUE;
        requested12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        if (supported12.scalarBlockLayout == VK_TRUE) requested12.scalarBlockLayout = VK_TRUE;
        requestedAS.accelerationStructure = VK_TRUE;
        requestedRQ.rayQuery = VK_TRUE;
    }

    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();
    deviceInfo.pEnabledFeatures = &requestedCore;
    requested13.pNext = &requested12;
    requested12.pNext = rtSupported ? reinterpret_cast<VkBaseOutStructure*>(&requestedAS) : nullptr;
    if (rtSupported) requestedAS.pNext = &requestedRQ;
    // between the as and ray-query structs
    if (rtSupported && caps_.hasRtPipeline) {
        requestedRTP.rayTracingPipeline = VK_TRUE;
        requestedAS.pNext = &requestedRTP;
        requestedRTP.pNext = &requestedRQ;
        if (caps_.hasInvocationReorder) {
            requestedSer.rayTracingInvocationReorder = VK_TRUE;
            requestedRTP.pNext = &requestedSer;
            requestedSer.pNext = &requestedRQ;
        }
        if (caps_.hasInvocationReorderNV) {
            requestedSerNV.rayTracingInvocationReorder = VK_TRUE;
            requestedSerNV.pNext = requestedRTP.pNext;
            requestedRTP.pNext = &requestedSerNV;
        }
    }
    VkPhysicalDeviceShaderClockFeaturesKHR requestedClock{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
    if (rtSupported && clockExtAvailable && supportedClock.shaderSubgroupClock == VK_TRUE) {
        requestedClock.shaderSubgroupClock = VK_TRUE;
        requestedRQ.pNext = &requestedClock;
        deviceExtensions.push_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
        deviceInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();
        caps_.hasShaderClock = true;
    }
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR requestedPipeStats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    if (rtSupported && pipeStatsWanted && supportedPipeStats.pipelineExecutableInfo == VK_TRUE) {
        requestedPipeStats.pipelineExecutableInfo = VK_TRUE;
        requestedPipeStats.pNext = requestedRQ.pNext;
        requestedRQ.pNext = &requestedPipeStats;
        deviceExtensions.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
        deviceInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();
        caps_.hasPipelineStats = true;
    }
    if (streamlineEnabled_ || rtSupported) deviceInfo.pNext = &requested13;

    VkResult result = vkCreateDevice(physicalDevice_, &deviceInfo, nullptr, &device_);
    if (result != VK_SUCCESS) {
        setError("vkCreateDevice failed with VkResult " + std::to_string(result));
        return false;
    }

    // tier B manual Streamline integration
    if (streamlineEnabled_) {
        const uint32_t slQueueStart = 1;
        if (!streamline_.setVulkanInfo(instance_, physicalDevice_, device_,
                                       slComputeQueues ? slQueueStart : 0, queueFamilyIndex_,
                                       slGraphicsQueues ? slQueueStart : 0, queueFamilyIndex_)) {
            setError(streamline_.lastError());
            return false;
        }

    }

    vkGetDeviceQueue(device_, queueFamilyIndex_, 0, &queue_);

    if (streamlineEnabled_) {
        std::string dummyPresentError;
        if (!streamlineDummyPresent_.initialize(streamline_, instance_, physicalDevice_, device_, queue_, queueFamilyIndex_, &dummyPresentError)) {
            std::cerr << "[interop] StreamlineDummyPresent::initialize failed (non-fatal -- Streamline's own "
                         "presentCommon() bookkeeping stays degraded, everything else keeps working): "
                      << dummyPresentError << std::endl;
        }
    }

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = queueFamilyIndex_;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device_, &poolInfo, nullptr, &setupCommandPool_) != VK_SUCCESS) {
        setError("createLogicalDeviceAndQueue: vkCreateCommandPool (setup) failed.");
        return false;
    }

    return true;
}

bool InteropContext::transitionImageToGeneral(VkImage image, VkImageAspectFlags aspectMask) {
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = setupCommandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device_, &allocInfo, &cmd) != VK_SUCCESS) {
        setError("transitionImageToGeneral: vkAllocateCommandBuffers failed.");
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {aspectMask, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 0, nullptr, 0, nullptr, 1, &barrier);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    VkResult result = vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        setError("transitionImageToGeneral: vkQueueSubmit failed with VkResult " + std::to_string(result));
        vkFreeCommandBuffers(device_, setupCommandPool_, 1, &cmd);
        return false;
    }
    vkQueueWaitIdle(queue_);
    vkFreeCommandBuffers(device_, setupCommandPool_, 1, &cmd);
    return true;
}

PFN_vkVoidFunction InteropContext::loadDeviceProc(const char* name) {
    PFN_vkVoidFunction fn = vkGetDeviceProcAddr(device_, name);
    if (fn == nullptr) {
        setError(std::string("vkGetDeviceProcAddr returned null for required function: ") + name);
    }
    return fn;
}

void InteropContext::recordValidationMessage(const std::string& message) {
    ++validationErrorCount_;
    lastValidationMessage_ = message;
}


const FormatSupport& InteropContext::queryFormatSupport(VkFormat format) {
    auto it = formatSupportCache_.find(static_cast<int32_t>(format));
    if (it != formatSupportCache_.end()) {
        return it->second;
    }

    FormatSupport support;
    support.queried = true;

    VkPhysicalDeviceExternalImageFormatInfo externalInfo{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO};
    externalInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkPhysicalDeviceImageFormatInfo2 formatInfo{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
    formatInfo.pNext = &externalInfo;
    formatInfo.format = format;
    formatInfo.type = VK_IMAGE_TYPE_2D;
    formatInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    // match the SharedTexture usage for each format
    if (format == VK_FORMAT_D32_SFLOAT) {
        formatInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                           VK_IMAGE_USAGE_SAMPLED_BIT |
                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    } else if (format == VK_FORMAT_R16G16B16_SFLOAT) {
        formatInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    } else if (format == VK_FORMAT_R8_UNORM) {
        formatInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    } else {
        formatInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }

    VkExternalImageFormatProperties externalProps{VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
    VkImageFormatProperties2 formatProps{VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
    formatProps.pNext = &externalProps;

    VkResult result = vkGetPhysicalDeviceImageFormatProperties2(physicalDevice_, &formatInfo, &formatProps);
    if (result != VK_SUCCESS) {
        std::cerr << "[interop] VkFormat " << format << " with usage "
                     "(STORAGE|SAMPLED|TRANSFER_SRC|TRANSFER_DST) is not usable at all on this "
                     "device -- VkResult " << result << "." << std::endl;
        formatSupportCache_[static_cast<int32_t>(format)] = support;  // both false
        return formatSupportCache_[static_cast<int32_t>(format)];
    }

    const auto features = externalProps.externalMemoryProperties.externalMemoryFeatures;
    support.exportable = (features & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0;
    support.importable = (features & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;

    std::cerr << "[interop] VkFormat " << format << " OPAQUE_WIN32 external memory features: "
              << "exportable=" << support.exportable << " importable=" << support.importable
              << std::endl;

    formatSupportCache_[static_cast<int32_t>(format)] = support;
    return formatSupportCache_[static_cast<int32_t>(format)];
}


bool InteropContext::configureDlss(uint32_t outputWidth, uint32_t outputHeight, sl::DLSSMode mode, uint32_t* renderWidth, uint32_t* renderHeight) {
    if (!streamlineEnabled_) {
        setError("DLSS requested but InteropContext was created without a Streamline plugin directory.");
        return false;
    }
    // propagate the failure reason into this object's lastError_
    if (!streamline_.configureDlss(physicalDevice_, outputWidth, outputHeight, mode, renderWidth, renderHeight)) {
        setError(streamline_.lastError());
        return false;
    }
    return true;
}

bool InteropContext::configureDlssRr(uint32_t outputWidth, uint32_t outputHeight, sl::DLSSMode mode, uint32_t* renderWidth, uint32_t* renderHeight) {
    if (!streamlineEnabled_) {
        setError("DLSS-RR requested but InteropContext was created without a Streamline plugin directory.");
        return false;
    }
    if (!streamline_.configureDlssRr(physicalDevice_, outputWidth, outputHeight, mode, renderWidth, renderHeight)) {
        setError(streamline_.lastError());
        return false;
    }
    return true;
}

const OperationPipeline* InteropContext::operationPipeline(InteropOperation op) const {
    auto it = operations_.find(static_cast<int32_t>(op));
    if (it == operations_.end()) return nullptr;
    return &it->second;
}

bool InteropContext::buildOperationRegistry(const std::string& shaderDirectory) {
    OperationPipeline op;
    op.localSizeX = 16;
    op.localSizeY = 16;

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;
    if (vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &op.descriptorSetLayout) != VK_SUCCESS) {
        setError("buildOperationRegistry: vkCreateDescriptorSetLayout failed.");
        return false;
    }

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &op.descriptorSetLayout;
    if (vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &op.pipelineLayout) != VK_SUCCESS) {
        setError("buildOperationRegistry: vkCreatePipelineLayout failed.");
        return false;
    }

    // (PixelFormat, spirv filename) pairs, see shaders/trivial_invert_*.comp
    const std::pair<PixelFormat, const char*> formatVariants[] = {
        {PixelFormat::Rgba8Unorm, "trivial_invert_rgba8.spv"},
        {PixelFormat::Rgba16Float, "trivial_invert_rgba16f.spv"},
        {PixelFormat::Rg16Float, "trivial_invert_rg16f.spv"},
    };

    for (const auto& [format, filename] : formatVariants) {
        const std::string spirvPath = shaderDirectory + "\\" + filename;
        std::ifstream file(spirvPath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            setError("buildOperationRegistry: could not open " + spirvPath +
                      " -- compile shaders/trivial_invert_*.comp with glslangValidator first "
                      "(see README.md), and make sure shaderDirectory points at where you put them.");
            return false;
        }
        const size_t fileSize = static_cast<size_t>(file.tellg());
        std::vector<uint32_t> spirv(fileSize / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(spirv.data()), static_cast<std::streamsize>(fileSize));
        file.close();

        OperationPipeline::FormatVariant variant;

        VkShaderModuleCreateInfo shaderInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shaderInfo.codeSize = fileSize;
        shaderInfo.pCode = spirv.data();
        if (vkCreateShaderModule(device_, &shaderInfo, nullptr, &variant.shaderModule) != VK_SUCCESS) {
            setError("buildOperationRegistry: vkCreateShaderModule failed for " + spirvPath);
            return false;
        }

        VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = variant.shaderModule;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = op.pipelineLayout;
        if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &variant.pipeline) != VK_SUCCESS) {
            setError("buildOperationRegistry: vkCreateComputePipelines failed for " + spirvPath);
            return false;
        }

        op.variants[static_cast<int32_t>(format)] = variant;
    }

    operations_[static_cast<int32_t>(InteropOperation::TrivialInvert)] = op;
    return true;
}

}  // namespace interop
