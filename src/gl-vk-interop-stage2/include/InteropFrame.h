#pragma once
// InteropFrame the single authoritative owner of every OpenGL<->Vulkan synchronization operation

#include "InteropOperation.h"
#include "gl_ext_loader.h"
#include <vulkan/vulkan.h>
#include <memory>
#include <string>
#include <vector>

namespace interop {

class InteropContext;
class SharedTexture;
class SharedSemaphore;
class RayTracingScene;
struct RtaoConstants;
struct ReflectionConstants;
struct RtdiDebugConstants;
struct RtdiConstants;
struct PathTraceConstants;
struct GpuVolFogConstants;

struct DlssFrameConstants {
    float cameraViewToClip[16]{};
    float clipToCameraView[16]{};
    float clipToPrevClip[16]{};
    float prevClipToClip[16]{};
    float jitterX = 0.0f, jitterY = 0.0f;
    float mvecScaleX = 1.0f, mvecScaleY = 1.0f;
    float cameraPosX = 0.0f, cameraPosY = 0.0f, cameraPosZ = 0.0f;
    float cameraUpX = 0.0f, cameraUpY = 1.0f, cameraUpZ = 0.0f;
    float cameraRightX = 1.0f, cameraRightY = 0.0f, cameraRightZ = 0.0f;
    float cameraFwdX = 0.0f, cameraFwdY = 0.0f, cameraFwdZ = -1.0f;
    float cameraNear = 0.1f, cameraFar = 500.0f, cameraFov = 1.2217305f, cameraAspect = 1.0f;
    bool reset = false;
};

class InteropFrame {
public:
    // poolSize
    explicit InteropFrame(InteropContext& ctx, uint32_t poolSize = 3);
    ~InteropFrame();

    InteropFrame(const InteropFrame&) = delete;
    InteropFrame& operator=(const InteropFrame&) = delete;

    bool initialize();

    // resources order (9 elements)
    bool executeRtao(RayTracingScene& scene, const std::vector<SharedTexture*>& resources,
                     const RtaoConstants& constants, std::string* errorOut = nullptr);

    bool executeReflections(RayTracingScene& scene, const std::vector<SharedTexture*>& resources,
                     const ReflectionConstants& constants, std::string* errorOut = nullptr);

    bool executeRtdiDebugBruteforce(RayTracingScene& scene, const std::vector<SharedTexture*>& resources,
                     const RtdiDebugConstants& constants, std::string* errorOut = nullptr);

    bool executeRtdi(RayTracingScene& scene, const std::vector<SharedTexture*>& resources,
                     const RtdiConstants& constants, std::string* errorOut = nullptr);

    bool executePathTrace(RayTracingScene& scene, const std::vector<SharedTexture*>& resources,
                     const PathTraceConstants& constants, std::string* errorOut = nullptr);

    bool executeVolumetricFog(RayTracingScene& scene, const GpuVolFogConstants& constants,
                     std::string* errorOut = nullptr);

    uint32_t nextSlotIndex() const { return static_cast<uint32_t>(frameIndex_ % poolSize_); }

    struct GpuStats {
        double ptCausticsMs = 0, ptTraceMs = 0, ptResolveMs = 0, fogInjectMs = 0, fogIntegrateMs = 0;
        uint32_t ptFrames = 0, fogFrames = 0;
        uint64_t rays[64] = {};
    };
    void setGpuStatsEnabled(bool on) { gpuStatsEnabled_ = on; }
    GpuStats takeGpuStats() { GpuStats s = gpuStats_; gpuStats_ = GpuStats{}; return s; }

    bool execute(InteropOperation op, const std::vector<SharedTexture*>& resources,
                 std::string* errorOut);

    bool executeDlss(const std::vector<SharedTexture*>& resources, const DlssFrameConstants& constants,
                     std::string* errorOut);

    // DLSS-RR diagnostic path
    bool executeDlssRr(const std::vector<SharedTexture*>& resources, const DlssFrameConstants& constants,
                       std::string* errorOut);

    // must be called before the Java side destroys/recreates SharedTextures during a resize or DLSS
    // mode change
    bool prepareForResourceRecreation(std::string* errorOut = nullptr);

private:
    struct FrameSlot {
        std::unique_ptr<SharedSemaphore> readyForVulkan;  // GL signals, VK waits
        std::unique_ptr<SharedSemaphore> readyForGl;  // VK signals, GL waits
        // created VK_FENCE_CREATE_SIGNALED_BIT
        VkFence workComplete = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        VkImageView imageView = VK_NULL_HANDLE;
        std::vector<VkImageView> dlssImageViews;
        uint64_t rtaoViewsEpoch = UINT64_MAX;
        VkImageView rtaoDepthView = VK_NULL_HANDLE;
        VkImageView rtaoNormalView = VK_NULL_HANDLE;
        VkImageView rtaoOutputView = VK_NULL_HANDLE;
        VkImageView rtaoMotionView = VK_NULL_HANDLE;
        VkImageView rtaoHistoryPrevView = VK_NULL_HANDLE;
        VkImageView rtaoHistoryNextView = VK_NULL_HANDLE;
        VkImageView rtaoShadowOutputView = VK_NULL_HANDLE;
        VkImageView rtaoShadowHistoryPrevView = VK_NULL_HANDLE;
        VkImageView rtaoShadowHistoryNextView = VK_NULL_HANDLE;
        // per-pixel temporal moment tracking
        VkImageView rtaoMomentsHistoryPrevView = VK_NULL_HANDLE;
        VkImageView rtaoMomentsHistoryNextView = VK_NULL_HANDLE;
        VkDescriptorSet rtaoDescriptorSet = VK_NULL_HANDLE;
        VkImageView reflDepthView = VK_NULL_HANDLE;
        VkImageView reflNormalRoughnessView = VK_NULL_HANDLE;
        VkImageView reflMotionView = VK_NULL_HANDLE;
        VkImageView reflPrevColorView = VK_NULL_HANDLE;
        VkImageView reflOutputView = VK_NULL_HANDLE;
        VkImageView reflHistoryPrevView = VK_NULL_HANDLE;
        VkImageView reflHistoryNextView = VK_NULL_HANDLE;
        VkImageView reflGeoHistoryPrevView = VK_NULL_HANDLE;
        VkImageView reflGeoHistoryNextView = VK_NULL_HANDLE;
        VkDescriptorSet reflDescriptorSet = VK_NULL_HANDLE;
        VkImageView rtdiDebugDepthView = VK_NULL_HANDLE;
        VkImageView rtdiDebugNormalRoughnessView = VK_NULL_HANDLE;
        VkImageView rtdiDebugOutputView = VK_NULL_HANDLE;
        VkDescriptorSet rtdiDebugDescriptorSet = VK_NULL_HANDLE;
        VkImageView rtdiDepthView = VK_NULL_HANDLE;
        VkImageView rtdiNormalRoughnessView = VK_NULL_HANDLE;
        VkImageView rtdiOutputView = VK_NULL_HANDLE;
        VkImageView rtdiReservoirAView = VK_NULL_HANDLE;
        VkImageView rtdiReservoirBView = VK_NULL_HANDLE;
        VkImageView rtdiHistoryReservoirAView = VK_NULL_HANDLE;
        VkImageView rtdiHistoryReservoirBView = VK_NULL_HANDLE;
        VkImageView rtdiHistoryDepthView = VK_NULL_HANDLE;
        VkImageView rtdiHistoryNormalRoughnessView = VK_NULL_HANDLE;
        VkDescriptorSet rtdiDescriptorSet = VK_NULL_HANDLE;

        uint64_t pathTraceViewsEpoch = UINT64_MAX;
        VkImageView pathTraceDepthView = VK_NULL_HANDLE;
        VkImageView pathTraceNormalView = VK_NULL_HANDLE;
        VkImageView pathTraceAlbedoView = VK_NULL_HANDLE;
        VkImageView pathTraceMotionView = VK_NULL_HANDLE;
        VkImageView pathTraceOutputView = VK_NULL_HANDLE;
        VkImageView pathTraceHistoryPrevView = VK_NULL_HANDLE;
        VkImageView pathTraceHistoryNextView = VK_NULL_HANDLE;
        VkImageView pathTraceHistoryMetaPrevView = VK_NULL_HANDLE;
        VkImageView pathTraceHistoryMetaNextView = VK_NULL_HANDLE;
        VkImageView pathTraceRawRadianceView = VK_NULL_HANDLE;
        VkImageView pathTraceGlowView = VK_NULL_HANDLE;
        VkImageView pathTracePrimaryNormalView = VK_NULL_HANDLE;
        VkImageView pathTraceLinearDepthView = VK_NULL_HANDLE;
        VkDescriptorSet pathTraceDescriptorSet = VK_NULL_HANDLE;

        VkDescriptorSet volFogDescriptorSet = VK_NULL_HANDLE;
        uint8_t pendingStatsKind = 0;
        const uint32_t* pendingRayStats = nullptr;
    };
    bool gpuStatsEnabled_ = false;
    GpuStats gpuStats_;
    VkQueryPool timestampPool_ = VK_NULL_HANDLE;  // 4 queries per slot
    double timestampPeriodNs_ = 1.0;
    void collectSlotStats(FrameSlot* slot, uint32_t slotIndex);

    FrameSlot* pickAndPrepareSlot(std::string* errorOut);

    // creates a fresh VkImageView for tex's image and assigns it to slot->imageView, called once per
    // execute() call
    bool refreshSlotImageView(FrameSlot* slot, SharedTexture* tex, std::string* errorOut);

    InteropContext& ctx_;
    uint32_t poolSize_;
    std::vector<std::unique_ptr<FrameSlot>> slots_;
    uint64_t frameIndex_ = 0;

    VkCommandPool commandPool_ = VK_NULL_HANDLE;  // per-slot reusable command buffers
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;

    GlExtFunctions gl_{};
};

}  // namespace interop
