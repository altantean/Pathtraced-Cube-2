#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "InteropFrame.h"
#include "RayTracingScene.h"
#include "InteropContext.h"
#include "SharedTexture.h"
#include "SharedTexture3D.h"
#include "SharedSemaphore.h"
#include <iostream>
#include <chrono>

namespace interop {

namespace {
void drainStaleGlErrors(const GlExtFunctions& gl, const char* before) {
    if (gl.glGetError == nullptr) return;
    static uint64_t staleCount = 0;
    GLenum e;
    while ((e = gl.glGetError()) != GL_NO_ERROR_) {
        if (staleCount++ % 600 == 0)
            std::cerr << "[interop][GL] stale GL error 0x" << std::hex << e << std::dec << " queued before " << before
                      << " (raised by earlier GL work, not the interop; " << staleCount << " so far)" << std::endl;
    }
}
}  // namespace

InteropFrame::InteropFrame(InteropContext& ctx, uint32_t poolSize)
    : ctx_(ctx), poolSize_(poolSize > 0 ? poolSize : 1) {}

InteropFrame::~InteropFrame() {
    VkDevice device = ctx_.device();
    if (device == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(device);  // shutdown-only broad stall, see ownership.md's "Destruction order"

    for (auto& slot : slots_) {
        if (slot->imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(device, slot->imageView, nullptr);
        }
        for (VkImageView view : slot->dlssImageViews) {
            if (view != VK_NULL_HANDLE) vkDestroyImageView(device, view, nullptr);
        }
        if (slot->rtaoDepthView) vkDestroyImageView(device, slot->rtaoDepthView, nullptr);
        if (slot->rtaoNormalView) vkDestroyImageView(device, slot->rtaoNormalView, nullptr);
        if (slot->rtaoOutputView) vkDestroyImageView(device, slot->rtaoOutputView, nullptr);
        if (slot->rtaoMotionView) vkDestroyImageView(device, slot->rtaoMotionView, nullptr);
        if (slot->rtaoHistoryPrevView) vkDestroyImageView(device, slot->rtaoHistoryPrevView, nullptr);
        if (slot->rtaoHistoryNextView) vkDestroyImageView(device, slot->rtaoHistoryNextView, nullptr);
        if (slot->rtaoShadowOutputView) vkDestroyImageView(device, slot->rtaoShadowOutputView, nullptr);
        if (slot->rtaoShadowHistoryPrevView) vkDestroyImageView(device, slot->rtaoShadowHistoryPrevView, nullptr);
        if (slot->rtaoShadowHistoryNextView) vkDestroyImageView(device, slot->rtaoShadowHistoryNextView, nullptr);
        if (slot->rtaoMomentsHistoryPrevView) vkDestroyImageView(device, slot->rtaoMomentsHistoryPrevView, nullptr);  // SVGF-style
        if (slot->rtaoMomentsHistoryNextView) vkDestroyImageView(device, slot->rtaoMomentsHistoryNextView, nullptr);  // SVGF-style
        if (slot->reflDepthView) vkDestroyImageView(device, slot->reflDepthView, nullptr);
        if (slot->reflNormalRoughnessView) vkDestroyImageView(device, slot->reflNormalRoughnessView, nullptr);
        if (slot->reflMotionView) vkDestroyImageView(device, slot->reflMotionView, nullptr);
        if (slot->reflPrevColorView) vkDestroyImageView(device, slot->reflPrevColorView, nullptr);
        if (slot->reflOutputView) vkDestroyImageView(device, slot->reflOutputView, nullptr);
        if (slot->reflHistoryPrevView) vkDestroyImageView(device, slot->reflHistoryPrevView, nullptr);
        if (slot->reflHistoryNextView) vkDestroyImageView(device, slot->reflHistoryNextView, nullptr);
        if (slot->reflGeoHistoryPrevView) vkDestroyImageView(device, slot->reflGeoHistoryPrevView, nullptr);
        if (slot->reflGeoHistoryNextView) vkDestroyImageView(device, slot->reflGeoHistoryNextView, nullptr);
        // same cleanup as RTAO/reflections above
        if (slot->pathTraceDepthView) vkDestroyImageView(device, slot->pathTraceDepthView, nullptr);
        if (slot->pathTraceNormalView) vkDestroyImageView(device, slot->pathTraceNormalView, nullptr);
        if (slot->pathTraceAlbedoView) vkDestroyImageView(device, slot->pathTraceAlbedoView, nullptr);
        if (slot->pathTraceMotionView) vkDestroyImageView(device, slot->pathTraceMotionView, nullptr);
        if (slot->pathTraceOutputView) vkDestroyImageView(device, slot->pathTraceOutputView, nullptr);
        if (slot->pathTraceHistoryPrevView) vkDestroyImageView(device, slot->pathTraceHistoryPrevView, nullptr);
        if (slot->pathTraceHistoryNextView) vkDestroyImageView(device, slot->pathTraceHistoryNextView, nullptr);
        if (slot->pathTraceHistoryMetaPrevView) vkDestroyImageView(device, slot->pathTraceHistoryMetaPrevView, nullptr);
        if (slot->pathTraceHistoryMetaNextView) vkDestroyImageView(device, slot->pathTraceHistoryMetaNextView, nullptr);
        if (slot->pathTraceRawRadianceView) vkDestroyImageView(device, slot->pathTraceRawRadianceView, nullptr);
        if (slot->pathTraceGlowView) vkDestroyImageView(device, slot->pathTraceGlowView, nullptr);  // pbr materials
        if (slot->pathTracePrimaryNormalView) vkDestroyImageView(device, slot->pathTracePrimaryNormalView, nullptr);
        if (slot->workComplete != VK_NULL_HANDLE) {
            vkDestroyFence(device, slot->workComplete, nullptr);
        }
        // slot->commandBuffer
    }
    if (descriptorPool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptorPool_, nullptr);
    if (commandPool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool_, nullptr);
    if (timestampPool_ != VK_NULL_HANDLE) vkDestroyQueryPool(device, timestampPool_, nullptr);
}

bool InteropFrame::initialize() {
    std::string error;
    if (!loadGlExtFunctions(gl_, &error)) {
        ctx_.setError("InteropFrame::initialize: " + error);
        return false;
    }

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = ctx_.queueFamilyIndex();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(ctx_.device(), &poolInfo, nullptr, &commandPool_) != VK_SUCCESS) {
        ctx_.setError("InteropFrame::initialize: vkCreateCommandPool (per-slot) failed.");
        return false;
    }

    // note, worth flagging rather than hiding
    const OperationPipeline* trivialInvert = ctx_.operationPipeline(InteropOperation::TrivialInvert);
    if (trivialInvert == nullptr) {
        ctx_.setError("InteropFrame::initialize: InteropOperation::TrivialInvert isn't registered "
                       "-- did InteropContext::initialize() succeed before this was called?");
        return false;
    }

    VkDescriptorPoolSize dpSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, poolSize_};
    VkDescriptorPoolCreateInfo dpInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpInfo.maxSets = poolSize_;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = &dpSize;
    if (vkCreateDescriptorPool(ctx_.device(), &dpInfo, nullptr, &descriptorPool_) != VK_SUCCESS) {
        ctx_.setError("InteropFrame::initialize: vkCreateDescriptorPool failed.");
        return false;
    }

    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(ctx_.physicalDevice(), &props);
        timestampPeriodNs_ = props.limits.timestampPeriod;
        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = poolSize_ * 4;
        if (vkCreateQueryPool(ctx_.device(), &qpi, nullptr, &timestampPool_) != VK_SUCCESS) timestampPool_ = VK_NULL_HANDLE;
    }

    for (uint32_t i = 0; i < poolSize_; ++i) {
        auto slot = std::make_unique<FrameSlot>();

        slot->readyForVulkan = std::make_unique<SharedSemaphore>(ctx_);
        if (!slot->readyForVulkan->create()) return false;
        slot->readyForGl = std::make_unique<SharedSemaphore>(ctx_);
        if (!slot->readyForGl->create()) return false;

        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (vkCreateFence(ctx_.device(), &fenceInfo, nullptr, &slot->workComplete) != VK_SUCCESS) {
            ctx_.setError("InteropFrame::initialize: vkCreateFence failed for slot " + std::to_string(i));
            return false;
        }

        VkCommandBufferAllocateInfo cbAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbAllocInfo.commandPool = commandPool_;
        cbAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbAllocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(ctx_.device(), &cbAllocInfo, &slot->commandBuffer) != VK_SUCCESS) {
            ctx_.setError("InteropFrame::initialize: vkAllocateCommandBuffers failed for slot " + std::to_string(i));
            return false;
        }

        VkDescriptorSetAllocateInfo dsAllocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsAllocInfo.descriptorPool = descriptorPool_;
        dsAllocInfo.descriptorSetCount = 1;
        dsAllocInfo.pSetLayouts = &trivialInvert->descriptorSetLayout;
        if (vkAllocateDescriptorSets(ctx_.device(), &dsAllocInfo, &slot->descriptorSet) != VK_SUCCESS) {
            ctx_.setError("InteropFrame::initialize: vkAllocateDescriptorSets failed for slot " + std::to_string(i));
            return false;
        }

        slots_.push_back(std::move(slot));
    }

    return true;
}

// called once the slot's fence has completed, so the timestamps and counters its last submission wrote
// are final
void InteropFrame::collectSlotStats(FrameSlot* slot, uint32_t slotIndex) {
    const uint8_t kind = slot->pendingStatsKind;
    slot->pendingStatsKind = 0;
    if (kind == 0) return;
    uint64_t ts[4] = {};
    const uint32_t count = kind == 1 ? 4u : 3u;
    const bool haveTs = timestampPool_ != VK_NULL_HANDLE &&
        vkGetQueryPoolResults(ctx_.device(), timestampPool_, slotIndex * 4, count, sizeof(ts), ts, sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT) == VK_SUCCESS;
    auto ms = [&](int a, int b) { return double(ts[b] - ts[a]) * timestampPeriodNs_ * 1e-6; };
    if (kind == 1) {
        if (haveTs) { gpuStats_.ptCausticsMs += ms(0, 1); gpuStats_.ptTraceMs += ms(1, 2); gpuStats_.ptResolveMs += ms(2, 3); }
        if (slot->pendingRayStats) for (int i = 0; i < 64; ++i) gpuStats_.rays[i] += slot->pendingRayStats[i];
        ++gpuStats_.ptFrames;
    } else {
        if (haveTs) { gpuStats_.fogInjectMs += ms(0, 1); gpuStats_.fogIntegrateMs += ms(1, 2); }
        ++gpuStats_.fogFrames;
    }
    slot->pendingRayStats = nullptr;
}

InteropFrame::FrameSlot* InteropFrame::pickAndPrepareSlot(std::string* errorOut) {
    FrameSlot* slot = slots_[frameIndex_ % poolSize_].get();
    VkResult status = vkGetFenceStatus(ctx_.device(), slot->workComplete);

    if (status == VK_NOT_READY) {
        std::cerr << "[interop] InteropFrame: pool exhausted at frameIndex=" << frameIndex_
                  << " (poolSize=" << poolSize_ << ") -- CPU waiting on GPU. If this happens "
                     "routinely (not just occasionally), increase poolSize." << std::endl;
        vkWaitForFences(ctx_.device(), 1, &slot->workComplete, VK_TRUE, UINT64_MAX);
    } else if (status != VK_SUCCESS) {
        ctx_.setError("InteropFrame::pickAndPrepareSlot: vkGetFenceStatus returned VkResult " +
                       std::to_string(status));
        if (errorOut != nullptr) *errorOut = ctx_.lastError();
        return nullptr;
    }
    collectSlotStats(slot, static_cast<uint32_t>(frameIndex_ % poolSize_));
    if (slot->imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(ctx_.device(), slot->imageView, nullptr);
        slot->imageView = VK_NULL_HANDLE;
    }
    for (VkImageView view : slot->dlssImageViews) {
        if (view != VK_NULL_HANDLE) vkDestroyImageView(ctx_.device(), view, nullptr);
    }
    slot->dlssImageViews.clear();
    if (slot->reflDepthView) vkDestroyImageView(ctx_.device(), slot->reflDepthView, nullptr);
    if (slot->reflNormalRoughnessView) vkDestroyImageView(ctx_.device(), slot->reflNormalRoughnessView, nullptr);
    if (slot->reflMotionView) vkDestroyImageView(ctx_.device(), slot->reflMotionView, nullptr);
    if (slot->reflPrevColorView) vkDestroyImageView(ctx_.device(), slot->reflPrevColorView, nullptr);
    if (slot->reflOutputView) vkDestroyImageView(ctx_.device(), slot->reflOutputView, nullptr);
    if (slot->reflHistoryPrevView) vkDestroyImageView(ctx_.device(), slot->reflHistoryPrevView, nullptr);
    if (slot->reflHistoryNextView) vkDestroyImageView(ctx_.device(), slot->reflHistoryNextView, nullptr);
    if (slot->reflGeoHistoryPrevView) vkDestroyImageView(ctx_.device(), slot->reflGeoHistoryPrevView, nullptr);
    if (slot->reflGeoHistoryNextView) vkDestroyImageView(ctx_.device(), slot->reflGeoHistoryNextView, nullptr);
    slot->reflDepthView = slot->reflNormalRoughnessView = slot->reflMotionView = VK_NULL_HANDLE;
    slot->reflPrevColorView = slot->reflOutputView = VK_NULL_HANDLE;
    slot->reflHistoryPrevView = slot->reflHistoryNextView = VK_NULL_HANDLE;
    slot->reflGeoHistoryPrevView = slot->reflGeoHistoryNextView = VK_NULL_HANDLE;
    if (slot->rtdiDebugDepthView) vkDestroyImageView(ctx_.device(), slot->rtdiDebugDepthView, nullptr);
    if (slot->rtdiDebugNormalRoughnessView) vkDestroyImageView(ctx_.device(), slot->rtdiDebugNormalRoughnessView, nullptr);
    if (slot->rtdiDebugOutputView) vkDestroyImageView(ctx_.device(), slot->rtdiDebugOutputView, nullptr);
    slot->rtdiDebugDepthView = slot->rtdiDebugNormalRoughnessView = slot->rtdiDebugOutputView = VK_NULL_HANDLE;
    if (slot->rtdiDepthView) vkDestroyImageView(ctx_.device(), slot->rtdiDepthView, nullptr);
    if (slot->rtdiNormalRoughnessView) vkDestroyImageView(ctx_.device(), slot->rtdiNormalRoughnessView, nullptr);
    if (slot->rtdiOutputView) vkDestroyImageView(ctx_.device(), slot->rtdiOutputView, nullptr);
    if (slot->rtdiReservoirAView) vkDestroyImageView(ctx_.device(), slot->rtdiReservoirAView, nullptr);
    if (slot->rtdiReservoirBView) vkDestroyImageView(ctx_.device(), slot->rtdiReservoirBView, nullptr);
    slot->rtdiDepthView = slot->rtdiNormalRoughnessView = slot->rtdiOutputView = VK_NULL_HANDLE;
    slot->rtdiReservoirAView = slot->rtdiReservoirBView = VK_NULL_HANDLE;
    // same fence-gated retirement timing as every other RTDI view above
    if (slot->rtdiHistoryReservoirAView) vkDestroyImageView(ctx_.device(), slot->rtdiHistoryReservoirAView, nullptr);
    if (slot->rtdiHistoryReservoirBView) vkDestroyImageView(ctx_.device(), slot->rtdiHistoryReservoirBView, nullptr);
    if (slot->rtdiHistoryDepthView) vkDestroyImageView(ctx_.device(), slot->rtdiHistoryDepthView, nullptr);
    if (slot->rtdiHistoryNormalRoughnessView) vkDestroyImageView(ctx_.device(), slot->rtdiHistoryNormalRoughnessView, nullptr);
    slot->rtdiHistoryReservoirAView = slot->rtdiHistoryReservoirBView = VK_NULL_HANDLE;
    slot->rtdiHistoryDepthView = slot->rtdiHistoryNormalRoughnessView = VK_NULL_HANDLE;
    // path tracing same fence-gated retirement timing as every other dispatch kind's view set above
    return slot;
}


bool InteropFrame::refreshSlotImageView(FrameSlot* slot, SharedTexture* tex, std::string* errorOut) {
    // slot->imageView is guaranteed VK_NULL_HANDLE here
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = tex->vkImage();
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = tex->vkFormat();
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    if (vkCreateImageView(ctx_.device(), &viewInfo, nullptr, &slot->imageView) != VK_SUCCESS) {
        ctx_.setError("InteropFrame::refreshSlotImageView: vkCreateImageView failed.");
        if (errorOut != nullptr) *errorOut = ctx_.lastError();
        return false;
    }
    return true;
}

bool InteropFrame::execute(InteropOperation op, const std::vector<SharedTexture*>& resources,
                            std::string* errorOut) {
    auto fail = [&](const std::string& msg) -> bool {
        ctx_.setError("InteropFrame::execute: " + msg);
        if (errorOut != nullptr) *errorOut = ctx_.lastError();
        return false;
    };

    const OperationPipeline* pipeline = ctx_.operationPipeline(op);
    if (pipeline == nullptr) {
        return fail("requested operation is not registered in InteropContext.");
    }

    if (resources.size() != 1) {
        return fail("InteropOperation::TrivialInvert requires exactly one resource (used in place), "
                     "got " + std::to_string(resources.size()));
    }
    SharedTexture* resource = resources[0];

    const OperationPipeline::FormatVariant* variant = pipeline->variantFor(resource->pixelFormat());
    if (variant == nullptr) {
        return fail("no pipeline variant registered for this resource's PixelFormat -- "
                     "did InteropContext::buildOperationRegistry compile all three shader "
                     "variants successfully?");
    }

    FrameSlot* slot = pickAndPrepareSlot(errorOut);
    if (slot == nullptr) return false;

    if (!refreshSlotImageView(slot, resource, errorOut)) return false;

    VkDescriptorImageInfo imgInfo{};
    imgInfo.imageView = slot->imageView;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = slot->descriptorSet;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &imgInfo;
    vkUpdateDescriptorSets(ctx_.device(), 1, &write, 0, nullptr);

    // step 2/3 (ownership.md)
    uint32_t glTex = resource->glTextureId();
    GLenum dstLayout = GL_LAYOUT_GENERAL_EXT_;
    gl_.glSignalSemaphoreEXT(slot->readyForVulkan->glSemaphore(), 0, nullptr, 1, &glTex, &dstLayout);
    checkGlError(gl_, "glSignalSemaphoreEXT (InteropFrame::execute)");

    // step 4 (ownership.md)
    vkResetCommandBuffer(slot->commandBuffer, 0);
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(slot->commandBuffer, &beginInfo);

    VkImageMemoryBarrier acquireBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    acquireBarrier.srcAccessMask = 0;
    acquireBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    acquireBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    acquireBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    acquireBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    acquireBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    acquireBarrier.image = resource->vkImage();
    acquireBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(slot->commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &acquireBarrier);

    vkCmdBindPipeline(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, variant->pipeline);
    vkCmdBindDescriptorSets(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipelineLayout,
                             0, 1, &slot->descriptorSet, 0, nullptr);
    const uint32_t groupsX = (resource->width() + pipeline->localSizeX - 1) / pipeline->localSizeX;
    const uint32_t groupsY = (resource->height() + pipeline->localSizeY - 1) / pipeline->localSizeY;
    vkCmdDispatch(slot->commandBuffer, groupsX, groupsY, 1);

    VkImageMemoryBarrier releaseBarrier = acquireBarrier;
    releaseBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    releaseBarrier.dstAccessMask = 0;
    vkCmdPipelineBarrier(slot->commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &releaseBarrier);

    vkEndCommandBuffer(slot->commandBuffer);

    // must reset before this submission signals it again, a fence must be unsignaled at submission
    // time
    vkResetFences(ctx_.device(), 1, &slot->workComplete);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSemaphore waitSem = slot->readyForVulkan->vkSemaphore();
    VkSemaphore signalSem = slot->readyForGl->vkSemaphore();

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &waitSem;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &slot->commandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &signalSem;

    VkResult submitResult = vkQueueSubmit(ctx_.queue(), 1, &submitInfo, slot->workComplete);
    if (submitResult != VK_SUCCESS) {
        return fail("vkQueueSubmit failed with VkResult " + std::to_string(submitResult));
    }

    // step 5 (ownership.md)
    uint32_t outTex = resource->glTextureId();
    GLenum srcLayout = GL_LAYOUT_GENERAL_EXT_;
    drainStaleGlErrors(gl_, "glWaitSemaphoreEXT (InteropFrame::execute)");
    gl_.glWaitSemaphoreEXT(slot->readyForGl->glSemaphore(), 0, nullptr, 1, &outTex, &srcLayout);
    checkGlError(gl_, "glWaitSemaphoreEXT (InteropFrame::execute)");

    ++frameIndex_;
    return true;
}


bool InteropFrame::prepareForResourceRecreation(std::string* errorOut) {
    auto fail = [&](const std::string& msg) -> bool {
        ctx_.setError("InteropFrame::prepareForResourceRecreation: " + msg);
        if (errorOut) *errorOut = ctx_.lastError();
        return false;
    };

    VkDevice device = ctx_.device();
    if (device == VK_NULL_HANDLE) return fail("Vulkan device is not initialized.");

    // A DLSS preset/resize can replace the shared images while previous DLSS submissions are still
    // using them
    VkResult idle = vkDeviceWaitIdle(device);
    if (idle != VK_SUCCESS) return fail("vkDeviceWaitIdle failed with VkResult " + std::to_string(idle));

    // these views point at the SharedTextures that Java is about to destroy
    for (auto& slot : slots_) {
        if (slot->imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(device, slot->imageView, nullptr);
            slot->imageView = VK_NULL_HANDLE;
        }
        for (VkImageView view : slot->dlssImageViews) {
            if (view != VK_NULL_HANDLE) vkDestroyImageView(device, view, nullptr);
        }
        slot->dlssImageViews.clear();
        if (slot->rtaoDepthView) vkDestroyImageView(device, slot->rtaoDepthView, nullptr);
        if (slot->rtaoNormalView) vkDestroyImageView(device, slot->rtaoNormalView, nullptr);
        if (slot->rtaoOutputView) vkDestroyImageView(device, slot->rtaoOutputView, nullptr);
        if (slot->rtaoMotionView) vkDestroyImageView(device, slot->rtaoMotionView, nullptr);
        if (slot->rtaoHistoryPrevView) vkDestroyImageView(device, slot->rtaoHistoryPrevView, nullptr);
        if (slot->rtaoHistoryNextView) vkDestroyImageView(device, slot->rtaoHistoryNextView, nullptr);
        if (slot->rtaoShadowOutputView) vkDestroyImageView(device, slot->rtaoShadowOutputView, nullptr);
        if (slot->rtaoShadowHistoryPrevView) vkDestroyImageView(device, slot->rtaoShadowHistoryPrevView, nullptr);
        if (slot->rtaoShadowHistoryNextView) vkDestroyImageView(device, slot->rtaoShadowHistoryNextView, nullptr);
        if (slot->rtaoMomentsHistoryPrevView) vkDestroyImageView(device, slot->rtaoMomentsHistoryPrevView, nullptr);  // SVGF-style
        if (slot->rtaoMomentsHistoryNextView) vkDestroyImageView(device, slot->rtaoMomentsHistoryNextView, nullptr);  // SVGF-style
        slot->rtaoDepthView = slot->rtaoNormalView = slot->rtaoOutputView = VK_NULL_HANDLE;
    slot->rtaoMotionView = slot->rtaoHistoryPrevView = slot->rtaoHistoryNextView = VK_NULL_HANDLE;
    slot->rtaoShadowOutputView = slot->rtaoShadowHistoryPrevView = slot->rtaoShadowHistoryNextView = VK_NULL_HANDLE;
    slot->rtaoMomentsHistoryPrevView = slot->rtaoMomentsHistoryNextView = VK_NULL_HANDLE;  // SVGF-style
    if (slot->reflDepthView) vkDestroyImageView(device, slot->reflDepthView, nullptr);
    if (slot->reflNormalRoughnessView) vkDestroyImageView(device, slot->reflNormalRoughnessView, nullptr);
    if (slot->reflMotionView) vkDestroyImageView(device, slot->reflMotionView, nullptr);
    if (slot->reflPrevColorView) vkDestroyImageView(device, slot->reflPrevColorView, nullptr);
    if (slot->reflOutputView) vkDestroyImageView(device, slot->reflOutputView, nullptr);
    if (slot->reflHistoryPrevView) vkDestroyImageView(device, slot->reflHistoryPrevView, nullptr);
    if (slot->reflHistoryNextView) vkDestroyImageView(device, slot->reflHistoryNextView, nullptr);
    if (slot->reflGeoHistoryPrevView) vkDestroyImageView(device, slot->reflGeoHistoryPrevView, nullptr);
    if (slot->reflGeoHistoryNextView) vkDestroyImageView(device, slot->reflGeoHistoryNextView, nullptr);
    slot->reflDepthView = slot->reflNormalRoughnessView = slot->reflMotionView = VK_NULL_HANDLE;
    slot->reflPrevColorView = slot->reflOutputView = VK_NULL_HANDLE;
    slot->reflHistoryPrevView = slot->reflHistoryNextView = VK_NULL_HANDLE;
    slot->reflGeoHistoryPrevView = slot->reflGeoHistoryNextView = VK_NULL_HANDLE;
    // same "views point at SharedTextures Java/the caller is about to destroy" reasoning as
    // RTAO/reflections above
    if (slot->pathTraceDepthView) vkDestroyImageView(device, slot->pathTraceDepthView, nullptr);
    if (slot->pathTraceNormalView) vkDestroyImageView(device, slot->pathTraceNormalView, nullptr);
    if (slot->pathTraceAlbedoView) vkDestroyImageView(device, slot->pathTraceAlbedoView, nullptr);
    if (slot->pathTraceMotionView) vkDestroyImageView(device, slot->pathTraceMotionView, nullptr);
    if (slot->pathTraceOutputView) vkDestroyImageView(device, slot->pathTraceOutputView, nullptr);
    if (slot->pathTraceHistoryPrevView) vkDestroyImageView(device, slot->pathTraceHistoryPrevView, nullptr);
    if (slot->pathTraceHistoryNextView) vkDestroyImageView(device, slot->pathTraceHistoryNextView, nullptr);
    if (slot->pathTraceHistoryMetaPrevView) vkDestroyImageView(device, slot->pathTraceHistoryMetaPrevView, nullptr);
    if (slot->pathTraceHistoryMetaNextView) vkDestroyImageView(device, slot->pathTraceHistoryMetaNextView, nullptr);
    if (slot->pathTraceRawRadianceView) vkDestroyImageView(device, slot->pathTraceRawRadianceView, nullptr);
    if (slot->pathTraceGlowView) vkDestroyImageView(device, slot->pathTraceGlowView, nullptr);  // pbr materials
    if (slot->pathTracePrimaryNormalView) vkDestroyImageView(device, slot->pathTracePrimaryNormalView, nullptr);
    slot->pathTraceDepthView = slot->pathTraceNormalView = slot->pathTraceAlbedoView = VK_NULL_HANDLE;
    slot->pathTraceMotionView = slot->pathTraceOutputView = VK_NULL_HANDLE;
    slot->pathTraceHistoryPrevView = slot->pathTraceHistoryNextView = VK_NULL_HANDLE;
    slot->pathTraceHistoryMetaPrevView = slot->pathTraceHistoryMetaNextView = VK_NULL_HANDLE;
    slot->pathTraceRawRadianceView = VK_NULL_HANDLE;
    slot->pathTraceGlowView = VK_NULL_HANDLE;  // pbr materials
    slot->pathTracePrimaryNormalView = VK_NULL_HANDLE;
    }

    return true;
}

bool InteropFrame::executeDlss(const std::vector<SharedTexture*>& resources,
                               const DlssFrameConstants& c, std::string* errorOut) {
    auto fail = [&](const std::string& msg) -> bool {
        ctx_.setError("InteropFrame::executeDlss: " + msg);
        if (errorOut) *errorOut = ctx_.lastError();
        return false;
    };
    if (resources.size() != 4) return fail("requires exactly four resources: color, depth, motion vectors, output");
    StreamlineContext* sl = ctx_.streamline();
    if (!sl || !sl->enabled() || !sl->dlssConfigured()) return fail("Streamline/DLSS is not configured");

    FrameSlot* slot = pickAndPrepareSlot(errorOut);
    if (!slot) return false;

    auto resetSemaphoresOnFailure = [&]() {
        slot->readyForVulkan = std::make_unique<SharedSemaphore>(ctx_);
        if (!slot->readyForVulkan->create()) {
            std::cerr << "[sauerinterop] executeDlss: failed to recreate readyForVulkan semaphore during failure recovery -- this slot may remain stuck." << std::endl;
        }
        slot->readyForGl = std::make_unique<SharedSemaphore>(ctx_);
        if (!slot->readyForGl->create()) {
            std::cerr << "[sauerinterop] executeDlss: failed to recreate readyForGl semaphore during failure recovery -- this slot may remain stuck." << std::endl;
        }
    };

    slot->dlssImageViews.resize(4, VK_NULL_HANDLE);
    for (size_t i = 0; i < resources.size(); ++i) {
        SharedTexture* tex = resources[i];
        if (!tex) return fail("null resource at index " + std::to_string(i));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = tex->vkImage();
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = tex->vkFormat();
        vi.subresourceRange.aspectMask = tex->isDepth() ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        if (vkCreateImageView(ctx_.device(), &vi, nullptr, &slot->dlssImageViews[i]) != VK_SUCCESS)
            return fail("vkCreateImageView failed for DLSS resource " + std::to_string(i));
    }

    // one GL signal covers every input and the output resource
    uint32_t glTextures[4] = {
        resources[0]->glTextureId(), resources[1]->glTextureId(),
        resources[2]->glTextureId(), resources[3]->glTextureId()
    };
    GLenum layouts[4] = { GL_LAYOUT_GENERAL_EXT_, GL_LAYOUT_GENERAL_EXT_, GL_LAYOUT_GENERAL_EXT_, GL_LAYOUT_GENERAL_EXT_ };
    gl_.glSignalSemaphoreEXT(slot->readyForVulkan->glSemaphore(), 0, nullptr, 4, glTextures, layouts);
    {
        std::string glErr;
        GLenum e;
        while ((e = gl_.glGetError()) != GL_NO_ERROR_) {
            if (!glErr.empty()) glErr += ", ";
            glErr += std::to_string(static_cast<unsigned>(e));
        }
        if (!glErr.empty()) { resetSemaphoresOnFailure(); return fail("glSignalSemaphoreEXT failed for DLSS resources, GL error code(s): " + glErr); }
    }

    vkResetCommandBuffer(slot->commandBuffer, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot->commandBuffer, &begin) != VK_SUCCESS) {
        resetSemaphoresOnFailure();
        return fail("vkBeginCommandBuffer failed");
    }

    VkImageMemoryBarrier barriers[4]{};
    for (int i = 0; i < 4; ++i) {
        barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[i].srcAccessMask = 0;
        barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[i].image = resources[i]->vkImage();
        barriers[i].subresourceRange.aspectMask = resources[i]->isDepth() ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[i].subresourceRange.levelCount = 1;
        barriers[i].subresourceRange.layerCount = 1;
    }
    vkCmdPipelineBarrier(slot->commandBuffer,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 0, nullptr, 0, nullptr, 4, barriers);

    sl::FrameToken* token = nullptr;
    if (!sl->beginFrame(token, static_cast<uint32_t>(frameIndex_))) {
        vkEndCommandBuffer(slot->commandBuffer);
        resetSemaphoresOnFailure();
        return fail(sl->lastError());
    }
    sl::ViewportHandle viewport(0);
    sl::Resource slResources[4] = {
        sl::Resource(sl::ResourceType::eTex2d, resources[0]->vkImage(), resources[0]->vkMemory(), slot->dlssImageViews[0], VK_IMAGE_LAYOUT_GENERAL),
        sl::Resource(sl::ResourceType::eTex2d, resources[1]->vkImage(), resources[1]->vkMemory(), slot->dlssImageViews[1], VK_IMAGE_LAYOUT_GENERAL),
        sl::Resource(sl::ResourceType::eTex2d, resources[2]->vkImage(), resources[2]->vkMemory(), slot->dlssImageViews[2], VK_IMAGE_LAYOUT_GENERAL),
        sl::Resource(sl::ResourceType::eTex2d, resources[3]->vkImage(), resources[3]->vkMemory(), slot->dlssImageViews[3], VK_IMAGE_LAYOUT_GENERAL)
    };
    for (int i = 0; i < 4; ++i) {
        slResources[i].width = resources[i]->width();
        slResources[i].height = resources[i]->height();
        slResources[i].nativeFormat = static_cast<uint32_t>(resources[i]->vkFormat());
        slResources[i].mipLevels = 1;
        slResources[i].arrayLayers = 1;
        slResources[i].flags = 0;
        slResources[i].usage = resources[i]->isDepth()
            ? (VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)
            : (VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    }
    sl::ResourceTag tags[] = {
        sl::ResourceTag(&slResources[1], sl::kBufferTypeDepth, sl::eValidUntilEvaluate),
        sl::ResourceTag(&slResources[2], sl::kBufferTypeMotionVectors, sl::eValidUntilEvaluate),
        sl::ResourceTag(&slResources[0], sl::kBufferTypeScalingInputColor, sl::eValidUntilEvaluate),
        sl::ResourceTag(&slResources[3], sl::kBufferTypeScalingOutputColor, sl::eValidUntilEvaluate)
    };
    if (!sl->setTags(*token, viewport, tags, 4, slot->commandBuffer)) {
        vkEndCommandBuffer(slot->commandBuffer);
        resetSemaphoresOnFailure();
        return fail(sl->lastError());
    }

    sl::Constants constants{};
    auto copyMatrix = [](const float* src, sl::float4x4& dst) {
        for (int r = 0; r < 4; ++r) {
            dst[r].x = src[r * 4 + 0];
            dst[r].y = src[r * 4 + 1];
            dst[r].z = src[r * 4 + 2];
            dst[r].w = src[r * 4 + 3];
        }
    };
    copyMatrix(c.cameraViewToClip, constants.cameraViewToClip);
    copyMatrix(c.clipToCameraView, constants.clipToCameraView);
    copyMatrix(c.clipToPrevClip, constants.clipToPrevClip);
    copyMatrix(c.prevClipToClip, constants.prevClipToClip);
    constants.jitterOffset = {c.jitterX, c.jitterY};
    constants.mvecScale = {c.mvecScaleX, c.mvecScaleY};
    constants.cameraPos = {c.cameraPosX, c.cameraPosY, c.cameraPosZ};
    constants.cameraUp = {c.cameraUpX, c.cameraUpY, c.cameraUpZ};
    constants.cameraRight = {c.cameraRightX, c.cameraRightY, c.cameraRightZ};
    constants.cameraFwd = {c.cameraFwdX, c.cameraFwdY, c.cameraFwdZ};
    constants.cameraNear = c.cameraNear;
    constants.cameraFar = c.cameraFar;
    constants.cameraFOV = c.cameraFov;
    constants.cameraAspectRatio = c.cameraAspect;
    constants.depthInverted = sl::Boolean::eFalse;
    constants.cameraMotionIncluded = sl::Boolean::eTrue;
    constants.motionVectors3D = sl::Boolean::eFalse;
    constants.motionVectorsDilated = sl::Boolean::eFalse;
    constants.motionVectorsJittered = sl::Boolean::eFalse;
    constants.orthographicProjection = sl::Boolean::eFalse;
    constants.minRelativeLinearDepthObjectSeparation = 4.0f;
    constants.reset = c.reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    if (!sl->setConstants(constants, *token, viewport)) {
        vkEndCommandBuffer(slot->commandBuffer);
        resetSemaphoresOnFailure();
        return fail(sl->lastError());
    }
    if (!sl->evaluate(*token, viewport, slot->commandBuffer)) {
        vkEndCommandBuffer(slot->commandBuffer);
        resetSemaphoresOnFailure();
        return fail(sl->lastError());
    }

    VkImageMemoryBarrier release[4]{};
    for (int i = 0; i < 4; ++i) {
        release[i] = barriers[i];
        release[i].srcAccessMask = (i == 3) ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
        release[i].dstAccessMask = 0;
    }
    vkCmdPipelineBarrier(slot->commandBuffer,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, nullptr, 0, nullptr, 4, release);
    if (vkEndCommandBuffer(slot->commandBuffer) != VK_SUCCESS) {
        resetSemaphoresOnFailure();
        return fail("vkEndCommandBuffer failed");
    }

    vkResetFences(ctx_.device(), 1, &slot->workComplete);
    VkSemaphore waitSem = slot->readyForVulkan->vkSemaphore();
    VkSemaphore signalSem = slot->readyForGl->vkSemaphore();
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &waitSem;
    submit.pWaitDstStageMask = &waitStage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &slot->commandBuffer;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &signalSem;
    VkResult sr = vkQueueSubmit(ctx_.queue(), 1, &submit, slot->workComplete);
    if (sr != VK_SUCCESS) {
        resetSemaphoresOnFailure();
        return fail("vkQueueSubmit failed with VkResult " + std::to_string(sr));
    }

    uint32_t outputTex = resources[3]->glTextureId();
    GLenum outputLayout = GL_LAYOUT_GENERAL_EXT_;
    drainStaleGlErrors(gl_, "glWaitSemaphoreEXT (DLSS)");
    gl_.glWaitSemaphoreEXT(slot->readyForGl->glSemaphore(), 0, nullptr, 1, &outputTex, &outputLayout);
    checkGlError(gl_, "glWaitSemaphoreEXT (DLSS)");
    ++frameIndex_;
    return true;
}


bool InteropFrame::executeDlssRr(const std::vector<SharedTexture*>& resources,
                                 const DlssFrameConstants& c, std::string* errorOut) {
    auto fail = [&](const std::string& msg) -> bool {
        ctx_.setError("InteropFrame::executeDlssRr: " + msg);
        if (errorOut) *errorOut = ctx_.lastError();
        return false;
    };
    if (resources.size() != 8) {
        return fail("requires exactly eight resources: color, depth, motion, output, albedo, specular albedo, packed normal+roughness, specular motion");
    }
    StreamlineContext* sl = ctx_.streamline();
    if (!sl || !sl->enabled() || !sl->dlssRrConfigured()) {
        return fail("Streamline/DLSS-RR is not configured");
    }

    FrameSlot* slot = pickAndPrepareSlot(errorOut);
    if (!slot) return false;

    auto resetSemaphoresOnFailure = [&]() {
        slot->readyForVulkan = std::make_unique<SharedSemaphore>(ctx_);
        if (!slot->readyForVulkan->create()) {
            std::cerr << "[sauerinterop] executeDlssRr: failed to recreate readyForVulkan semaphore during failure recovery -- this slot may remain stuck." << std::endl;
        }
        slot->readyForGl = std::make_unique<SharedSemaphore>(ctx_);
        if (!slot->readyForGl->create()) {
            std::cerr << "[sauerinterop] executeDlssRr: failed to recreate readyForGl semaphore during failure recovery -- this slot may remain stuck." << std::endl;
        }
    };

    slot->dlssImageViews.resize(8, VK_NULL_HANDLE);
    for (size_t i = 0; i < resources.size(); ++i) {
        SharedTexture* tex = resources[i];
        if (!tex) return fail("null resource at index " + std::to_string(i));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = tex->vkImage();
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = tex->vkFormat();
        vi.subresourceRange.aspectMask = tex->isDepth() ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        if (vkCreateImageView(ctx_.device(), &vi, nullptr, &slot->dlssImageViews[i]) != VK_SUCCESS)
            return fail("vkCreateImageView failed for DLSS-RR resource " + std::to_string(i));
    }

    uint32_t glTextures[8] = {
        resources[0]->glTextureId(), resources[1]->glTextureId(), resources[2]->glTextureId(), resources[3]->glTextureId(),
        resources[4]->glTextureId(), resources[5]->glTextureId(), resources[6]->glTextureId(), resources[7]->glTextureId()
    };
    GLenum layouts[8] = {
        GL_LAYOUT_GENERAL_EXT_, GL_LAYOUT_GENERAL_EXT_, GL_LAYOUT_GENERAL_EXT_, GL_LAYOUT_GENERAL_EXT_,
        GL_LAYOUT_GENERAL_EXT_, GL_LAYOUT_GENERAL_EXT_, GL_LAYOUT_GENERAL_EXT_, GL_LAYOUT_GENERAL_EXT_
    };
    gl_.glSignalSemaphoreEXT(slot->readyForVulkan->glSemaphore(), 0, nullptr, 8, glTextures, layouts);
    {
        std::string glErr;
        GLenum e;
        while ((e = gl_.glGetError()) != GL_NO_ERROR_) {
            if (!glErr.empty()) glErr += ", ";
            glErr += std::to_string(static_cast<unsigned>(e));
        }
        if (!glErr.empty()) { resetSemaphoresOnFailure(); return fail("glSignalSemaphoreEXT failed for DLSS-RR resources, GL error code(s): " + glErr); }
    }

    vkResetCommandBuffer(slot->commandBuffer, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot->commandBuffer, &begin) != VK_SUCCESS) {
        resetSemaphoresOnFailure();
        return fail("vkBeginCommandBuffer failed");
    }

    VkImageMemoryBarrier barriers[8]{};
    for (int i = 0; i < 8; ++i) {
        barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[i].srcAccessMask = 0;
        barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[i].image = resources[i]->vkImage();
        barriers[i].subresourceRange.aspectMask = resources[i]->isDepth() ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[i].subresourceRange.levelCount = 1;
        barriers[i].subresourceRange.layerCount = 1;
    }
    vkCmdPipelineBarrier(slot->commandBuffer,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 0, nullptr, 0, nullptr, 8, barriers);

    sl::FrameToken* token = nullptr;
    if (!sl->beginFrame(token, static_cast<uint32_t>(frameIndex_))) {
        vkEndCommandBuffer(slot->commandBuffer);
        resetSemaphoresOnFailure();
        return fail(sl->lastError());
    }
    sl::ViewportHandle viewport(0);
    sl::Resource slResources[8] = {
        sl::Resource(sl::ResourceType::eTex2d, resources[0]->vkImage(), resources[0]->vkMemory(), slot->dlssImageViews[0], VK_IMAGE_LAYOUT_GENERAL),
        sl::Resource(sl::ResourceType::eTex2d, resources[1]->vkImage(), resources[1]->vkMemory(), slot->dlssImageViews[1], VK_IMAGE_LAYOUT_GENERAL),
        sl::Resource(sl::ResourceType::eTex2d, resources[2]->vkImage(), resources[2]->vkMemory(), slot->dlssImageViews[2], VK_IMAGE_LAYOUT_GENERAL),
        sl::Resource(sl::ResourceType::eTex2d, resources[3]->vkImage(), resources[3]->vkMemory(), slot->dlssImageViews[3], VK_IMAGE_LAYOUT_GENERAL),
        sl::Resource(sl::ResourceType::eTex2d, resources[4]->vkImage(), resources[4]->vkMemory(), slot->dlssImageViews[4], VK_IMAGE_LAYOUT_GENERAL),
        sl::Resource(sl::ResourceType::eTex2d, resources[5]->vkImage(), resources[5]->vkMemory(), slot->dlssImageViews[5], VK_IMAGE_LAYOUT_GENERAL),
        sl::Resource(sl::ResourceType::eTex2d, resources[6]->vkImage(), resources[6]->vkMemory(), slot->dlssImageViews[6], VK_IMAGE_LAYOUT_GENERAL),
        sl::Resource(sl::ResourceType::eTex2d, resources[7]->vkImage(), resources[7]->vkMemory(), slot->dlssImageViews[7], VK_IMAGE_LAYOUT_GENERAL)
    };
    for (int i = 0; i < 8; ++i) {
        slResources[i].width = resources[i]->width();
        slResources[i].height = resources[i]->height();
        slResources[i].nativeFormat = static_cast<uint32_t>(resources[i]->vkFormat());
        slResources[i].mipLevels = 1;
        slResources[i].arrayLayers = 1;
        slResources[i].flags = 0;
        slResources[i].usage = resources[i]->isDepth()
            ? (VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)
            : (VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    }

    sl::ResourceTag tags[] = {
        sl::ResourceTag(&slResources[1], sl::kBufferTypeDepth, sl::eValidUntilEvaluate),
        sl::ResourceTag(&slResources[2], sl::kBufferTypeMotionVectors, sl::eValidUntilEvaluate),
        sl::ResourceTag(&slResources[0], sl::kBufferTypeScalingInputColor, sl::eValidUntilEvaluate),
        sl::ResourceTag(&slResources[4], sl::kBufferTypeAlbedo, sl::eValidUntilEvaluate),
        sl::ResourceTag(&slResources[5], sl::kBufferTypeSpecularAlbedo, sl::eValidUntilEvaluate),
        sl::ResourceTag(&slResources[6], sl::kBufferTypeNormalRoughness, sl::eValidUntilEvaluate),
        sl::ResourceTag(&slResources[7], sl::kBufferTypeSpecularMotionVectors, sl::eValidUntilEvaluate),
        sl::ResourceTag(&slResources[3], sl::kBufferTypeScalingOutputColor, sl::eValidUntilEvaluate)
    };
    if (!sl->setTags(*token, viewport, tags, 8, slot->commandBuffer)) {
        vkEndCommandBuffer(slot->commandBuffer);
        resetSemaphoresOnFailure();
        return fail(sl->lastError());
    }

    sl::Constants constants{};
    auto copyMatrix = [](const float* src, sl::float4x4& dst) {
        for (int r = 0; r < 4; ++r) {
            dst[r].x = src[r * 4 + 0];
            dst[r].y = src[r * 4 + 1];
            dst[r].z = src[r * 4 + 2];
            dst[r].w = src[r * 4 + 3];
        }
    };
    copyMatrix(c.cameraViewToClip, constants.cameraViewToClip);
    copyMatrix(c.clipToCameraView, constants.clipToCameraView);
    copyMatrix(c.clipToPrevClip, constants.clipToPrevClip);
    copyMatrix(c.prevClipToClip, constants.prevClipToClip);
    constants.jitterOffset = {c.jitterX, c.jitterY};
    constants.mvecScale = {c.mvecScaleX, c.mvecScaleY};
    constants.cameraPos = {c.cameraPosX, c.cameraPosY, c.cameraPosZ};
    constants.cameraUp = {c.cameraUpX, c.cameraUpY, c.cameraUpZ};
    constants.cameraRight = {c.cameraRightX, c.cameraRightY, c.cameraRightZ};
    constants.cameraFwd = {c.cameraFwdX, c.cameraFwdY, c.cameraFwdZ};
    constants.cameraNear = c.cameraNear;
    constants.cameraFar = c.cameraFar;
    constants.cameraFOV = c.cameraFov;
    constants.cameraAspectRatio = c.cameraAspect;
    constants.depthInverted = sl::Boolean::eFalse;
    constants.cameraMotionIncluded = sl::Boolean::eTrue;
    constants.motionVectors3D = sl::Boolean::eFalse;
    constants.motionVectorsDilated = sl::Boolean::eFalse;
    constants.motionVectorsJittered = sl::Boolean::eFalse;
    constants.orthographicProjection = sl::Boolean::eFalse;
    constants.reset = c.reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    if (!sl->setConstants(constants, *token, viewport)) {
        vkEndCommandBuffer(slot->commandBuffer);
        resetSemaphoresOnFailure();
        return fail(sl->lastError());
    }
    if (!sl->evaluateRr(*token, viewport, slot->commandBuffer)) {
        vkEndCommandBuffer(slot->commandBuffer);
        resetSemaphoresOnFailure();
        return fail(sl->lastError());
    }

    VkImageMemoryBarrier release[8]{};
    for (int i = 0; i < 8; ++i) {
        release[i] = barriers[i];
        release[i].srcAccessMask = (i == 3) ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
        release[i].dstAccessMask = 0;
    }
    vkCmdPipelineBarrier(slot->commandBuffer,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, nullptr, 0, nullptr, 8, release);
    if (vkEndCommandBuffer(slot->commandBuffer) != VK_SUCCESS) {
        resetSemaphoresOnFailure();
        return fail("vkEndCommandBuffer failed");
    }

    vkResetFences(ctx_.device(), 1, &slot->workComplete);
    VkSemaphore waitSem = slot->readyForVulkan->vkSemaphore();
    VkSemaphore signalSem = slot->readyForGl->vkSemaphore();
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &waitSem;
    submit.pWaitDstStageMask = &waitStage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &slot->commandBuffer;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &signalSem;
    VkResult sr = vkQueueSubmit(ctx_.queue(), 1, &submit, slot->workComplete);
    if (sr != VK_SUCCESS) {
        resetSemaphoresOnFailure();
        return fail("vkQueueSubmit failed with VkResult " + std::to_string(sr));
    }

    GLenum outputLayouts[8] = {
        GL_LAYOUT_GENERAL_EXT_, GL_LAYOUT_GENERAL_EXT_, GL_LAYOUT_GENERAL_EXT_, GL_LAYOUT_GENERAL_EXT_,
        GL_LAYOUT_GENERAL_EXT_, GL_LAYOUT_GENERAL_EXT_, GL_LAYOUT_GENERAL_EXT_, GL_LAYOUT_GENERAL_EXT_
    };
    drainStaleGlErrors(gl_, "glWaitSemaphoreEXT (DLSS-RR)");
    gl_.glWaitSemaphoreEXT(slot->readyForGl->glSemaphore(), 0, nullptr, 8, glTextures, outputLayouts);
    checkGlError(gl_, "glWaitSemaphoreEXT (DLSS-RR)");
    // in place and this function returning success every time
    if (gl_.glFlush) gl_.glFlush();
    ++frameIndex_;
    return true;
}

bool InteropFrame::executeRtao(RayTracingScene& scene, const std::vector<SharedTexture*>& resources,
                               const RtaoConstants& c, std::string* errorOut) {
    auto fail = [&](const std::string& msg) -> bool {
        ctx_.setError("InteropFrame::executeRtao: " + msg);
        if (errorOut) *errorOut = ctx_.lastError();
        return false;
    };
    if (!ctx_.capabilities().hasRayQuery) return fail("hardware ray query is unavailable");
    if (!scene.isBuilt()) return fail("ray tracing scene is not built");
    if (resources.size() != 11) return fail("requires eleven resources: depth, normal, motion, AO output, AO historyPrev, AO historyNext, shadow output, shadow historyPrev, shadow historyNext, moments historyPrev, moments historyNext");
    for (auto* r : resources) if (!r) return fail("null RTAO resource");

    FrameSlot* slot = pickAndPrepareSlot(errorOut);
    if (!slot) return false;
    const uint32_t slotIndex = static_cast<uint32_t>((frameIndex_) % poolSize_);

    auto makeView = [&](SharedTexture* tex, VkImageAspectFlags aspect, VkImageView* dst) -> bool {
        if (*dst != VK_NULL_HANDLE) { vkDestroyImageView(ctx_.device(), *dst, nullptr); *dst = VK_NULL_HANDLE; }
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = tex->vkImage(); vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = tex->vkFormat();
        vi.subresourceRange.aspectMask = aspect; vi.subresourceRange.levelCount = 1; vi.subresourceRange.layerCount = 1;
        return vkCreateImageView(ctx_.device(), &vi, nullptr, dst) == VK_SUCCESS;
    };
    const bool rtaoPersistentViewsStillValid = c.resourceEpoch != UINT64_MAX && slot->rtaoViewsEpoch == c.resourceEpoch &&
        slot->rtaoDepthView && slot->rtaoNormalView && slot->rtaoMotionView && slot->rtaoOutputView &&
        slot->rtaoShadowOutputView;
    if (!rtaoPersistentViewsStillValid) {
        if (!makeView(resources[0], VK_IMAGE_ASPECT_DEPTH_BIT, &slot->rtaoDepthView) ||
            !makeView(resources[1], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtaoNormalView) ||
            !makeView(resources[2], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtaoMotionView) ||
            !makeView(resources[3], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtaoOutputView) ||
            !makeView(resources[6], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtaoShadowOutputView))
            return fail("vkCreateImageView failed for RTAO resources");
        slot->rtaoViewsEpoch = c.resourceEpoch;
    }
    if (!makeView(resources[4], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtaoHistoryPrevView) ||
        !makeView(resources[5], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtaoHistoryNextView) ||
        !makeView(resources[7], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtaoShadowHistoryPrevView) ||
        !makeView(resources[8], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtaoShadowHistoryNextView) ||
        !makeView(resources[9], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtaoMomentsHistoryPrevView) ||
        !makeView(resources[10], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtaoMomentsHistoryNextView))
        return fail("vkCreateImageView failed for RTAO history resources");

    if (slot->rtaoDescriptorSet == VK_NULL_HANDLE) {
        std::vector<VkDescriptorSet> sets(1);
        std::string e;
        if (!scene.createDescriptorSets(1, sets, &e)) return fail(e);
        slot->rtaoDescriptorSet = sets[0];
    }
    std::string e;
    if (!scene.updateDescriptorSet(slot->rtaoDescriptorSet,
                                   *resources[0], *resources[1], *resources[2], *resources[3], *resources[4], *resources[5],
                                   *resources[6], *resources[7], *resources[8], *resources[9], *resources[10],
                                   slot->rtaoDepthView, slot->rtaoNormalView, slot->rtaoMotionView,
                                   slot->rtaoOutputView, slot->rtaoHistoryPrevView, slot->rtaoHistoryNextView,
                                   slot->rtaoShadowOutputView, slot->rtaoShadowHistoryPrevView, slot->rtaoShadowHistoryNextView,
                                   slot->rtaoMomentsHistoryPrevView, slot->rtaoMomentsHistoryNextView,
                                   slotIndex, c, &e)) return fail(e);

    uint32_t glTextures[11];
    for (int i = 0; i < 11; ++i) glTextures[i] = resources[i]->glTextureId();
    GLenum layouts[11];
    for (int i = 0; i < 11; ++i) layouts[i] = GL_LAYOUT_GENERAL_EXT_;
    gl_.glSignalSemaphoreEXT(slot->readyForVulkan->glSemaphore(), 0, nullptr, 11, glTextures, layouts);
    if (checkGlError(gl_, "glSignalSemaphoreEXT (RTAO)")) return fail("OpenGL signal failed for RTAO resources");

    vkResetCommandBuffer(slot->commandBuffer, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot->commandBuffer, &begin) != VK_SUCCESS) return fail("vkBeginCommandBuffer failed");

    // write targets
    VkImageMemoryBarrier bars[11]{};
    for (int i=0;i<11;++i) {
        bars[i].sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        bars[i].srcAccessMask=0;
        const bool write = (i == 3 || i == 5 || i == 6 || i == 8 || i == 10);
        bars[i].dstAccessMask=write ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
        bars[i].oldLayout=VK_IMAGE_LAYOUT_GENERAL; bars[i].newLayout=VK_IMAGE_LAYOUT_GENERAL;
        bars[i].srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; bars[i].dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        bars[i].image=resources[i]->vkImage();
        bars[i].subresourceRange.aspectMask=resources[i]->isDepth()?VK_IMAGE_ASPECT_DEPTH_BIT:VK_IMAGE_ASPECT_COLOR_BIT;
        bars[i].subresourceRange.levelCount=1; bars[i].subresourceRange.layerCount=1;
    }
    vkCmdPipelineBarrier(slot->commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 11, bars);

    vkCmdBindPipeline(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipeline());
    vkCmdBindDescriptorSets(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelineLayout(), 0, 1, &slot->rtaoDescriptorSet, 0, nullptr);
    vkCmdDispatch(slot->commandBuffer, (resources[3]->width()+7)/8, (resources[3]->height()+7)/8, 1);

    VkImageMemoryBarrier release[11]{};
    for (int i=0;i<11;++i) {
        release[i]=bars[i];
        const bool write = (i == 3 || i == 5 || i == 6 || i == 8 || i == 10);
        release[i].srcAccessMask=write ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
        release[i].dstAccessMask=0;
    }
    vkCmdPipelineBarrier(slot->commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, nullptr, 0, nullptr, 11, release);
    if (vkEndCommandBuffer(slot->commandBuffer) != VK_SUCCESS) return fail("vkEndCommandBuffer failed");

    vkResetFences(ctx_.device(), 1, &slot->workComplete);
    VkSemaphore waitSems[2];
    VkPipelineStageFlags waitStages[2];
    uint32_t waitCount = 0;
    waitSems[waitCount] = slot->readyForVulkan->vkSemaphore();
    waitStages[waitCount] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    waitCount++;
    VkSemaphore geometryReady = scene.currentGenerationAoReadySemaphore();
    if (geometryReady != VK_NULL_HANDLE) {
        waitSems[waitCount] = geometryReady;
        waitStages[waitCount] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        waitCount++;
    }
    VkSemaphore signalSem=slot->readyForGl->vkSemaphore();
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.waitSemaphoreCount=waitCount; submit.pWaitSemaphores=waitSems; submit.pWaitDstStageMask=waitStages; submit.commandBufferCount=1; submit.pCommandBuffers=&slot->commandBuffer; submit.signalSemaphoreCount=1; submit.pSignalSemaphores=&signalSem;
    VkResult sr=vkQueueSubmit(ctx_.queue(),1,&submit,slot->workComplete); if(sr!=VK_SUCCESS) return fail("vkQueueSubmit failed with VkResult "+std::to_string(sr));

    drainStaleGlErrors(gl_, "glWaitSemaphoreEXT (RTAO)");
    gl_.glWaitSemaphoreEXT(slot->readyForGl->glSemaphore(),0,nullptr,9,glTextures,layouts);
    const bool glWaitFailed = checkGlError(gl_, "glWaitSemaphoreEXT (RTAO)");
    ++frameIndex_;
    if (glWaitFailed) return fail("OpenGL wait failed for RTAO resources");
    return true;
}

bool InteropFrame::executeReflections(RayTracingScene& scene, const std::vector<SharedTexture*>& resources,
                               const ReflectionConstants& c, std::string* errorOut) {
    auto fail = [&](const std::string& msg) -> bool {
        ctx_.setError("InteropFrame::executeReflections: " + msg);
        if (errorOut) *errorOut = ctx_.lastError();
        return false;
    };
    if (!ctx_.capabilities().hasRayQuery) return fail("hardware ray query is unavailable");
    if (!scene.isBuilt()) return fail("ray tracing scene is not built");
    if (resources.size() != 9) return fail("requires nine resources: depth, normalRoughness, motion, prevSceneColor, reflOutput, reflHistoryPrev, reflHistoryNext, reflGeoHistoryPrev, reflGeoHistoryNext");
    for (auto* r : resources) if (!r) return fail("null reflection resource");

    FrameSlot* slot = pickAndPrepareSlot(errorOut);
    if (!slot) return false;
    const uint32_t slotIndex = static_cast<uint32_t>((frameIndex_) % poolSize_);

    // this lambda runs unconditionally every frame this dispatch kind executes, and *dst
    auto makeView = [&](SharedTexture* tex, VkImageAspectFlags aspect, VkImageView* dst) -> bool {
        if (*dst != VK_NULL_HANDLE) { vkDestroyImageView(ctx_.device(), *dst, nullptr); *dst = VK_NULL_HANDLE; }
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = tex->vkImage(); vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = tex->vkFormat();
        vi.subresourceRange.aspectMask = aspect; vi.subresourceRange.levelCount = 1; vi.subresourceRange.layerCount = 1;
        return vkCreateImageView(ctx_.device(), &vi, nullptr, dst) == VK_SUCCESS;
    };
    if (!makeView(resources[0], VK_IMAGE_ASPECT_DEPTH_BIT, &slot->reflDepthView) ||
        !makeView(resources[1], VK_IMAGE_ASPECT_COLOR_BIT, &slot->reflNormalRoughnessView) ||
        !makeView(resources[2], VK_IMAGE_ASPECT_COLOR_BIT, &slot->reflMotionView) ||
        !makeView(resources[3], VK_IMAGE_ASPECT_COLOR_BIT, &slot->reflPrevColorView) ||
        !makeView(resources[4], VK_IMAGE_ASPECT_COLOR_BIT, &slot->reflOutputView) ||
        !makeView(resources[5], VK_IMAGE_ASPECT_COLOR_BIT, &slot->reflHistoryPrevView) ||
        !makeView(resources[6], VK_IMAGE_ASPECT_COLOR_BIT, &slot->reflHistoryNextView) ||
        !makeView(resources[7], VK_IMAGE_ASPECT_COLOR_BIT, &slot->reflGeoHistoryPrevView) ||
        !makeView(resources[8], VK_IMAGE_ASPECT_COLOR_BIT, &slot->reflGeoHistoryNextView))
        return fail("vkCreateImageView failed for reflection resources");

    if (slot->reflDescriptorSet == VK_NULL_HANDLE) {
        std::vector<VkDescriptorSet> sets(1);
        std::string e;
        if (!scene.createReflectionDescriptorSets(1, sets, &e)) return fail(e);
        slot->reflDescriptorSet = sets[0];
    }
    std::string e;
    if (!scene.updateReflectionDescriptorSet(slot->reflDescriptorSet,
                                   *resources[0], *resources[1], *resources[2], *resources[3], *resources[4], *resources[5], *resources[6],
                                   *resources[7], *resources[8],
                                   slot->reflDepthView, slot->reflNormalRoughnessView, slot->reflMotionView,
                                   slot->reflPrevColorView, slot->reflOutputView, slot->reflHistoryPrevView, slot->reflHistoryNextView,
                                   slot->reflGeoHistoryPrevView, slot->reflGeoHistoryNextView,
                                   slotIndex, c, &e)) return fail(e);

    uint32_t glTextures[9];
    for (int i = 0; i < 9; ++i) glTextures[i] = resources[i]->glTextureId();
    GLenum layouts[9];
    for (int i = 0; i < 9; ++i) layouts[i] = GL_LAYOUT_GENERAL_EXT_;
    gl_.glSignalSemaphoreEXT(slot->readyForVulkan->glSemaphore(), 0, nullptr, 9, glTextures, layouts);
    if (checkGlError(gl_, "glSignalSemaphoreEXT (reflections)")) return fail("OpenGL signal failed for reflection resources");

    vkResetCommandBuffer(slot->commandBuffer, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot->commandBuffer, &begin) != VK_SUCCESS) return fail("vkBeginCommandBuffer failed");

    // write targets
    VkImageMemoryBarrier bars[9]{};
    for (int i=0;i<9;++i) {
        bars[i].sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        bars[i].srcAccessMask=0;
        const bool write = (i == 4 || i == 6 || i == 8);
        bars[i].dstAccessMask=write ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
        bars[i].oldLayout=VK_IMAGE_LAYOUT_GENERAL; bars[i].newLayout=VK_IMAGE_LAYOUT_GENERAL;
        bars[i].srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; bars[i].dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        bars[i].image=resources[i]->vkImage();
        bars[i].subresourceRange.aspectMask=resources[i]->isDepth()?VK_IMAGE_ASPECT_DEPTH_BIT:VK_IMAGE_ASPECT_COLOR_BIT;
        bars[i].subresourceRange.levelCount=1; bars[i].subresourceRange.layerCount=1;
    }
    vkCmdPipelineBarrier(slot->commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 9, bars);

    vkCmdBindPipeline(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelineReflection());
    vkCmdBindDescriptorSets(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelineLayoutReflection(), 0, 1, &slot->reflDescriptorSet, 0, nullptr);
    vkCmdDispatch(slot->commandBuffer, (resources[4]->width()+7)/8, (resources[4]->height()+7)/8, 1);

    VkImageMemoryBarrier release[9]{};
    for (int i=0;i<9;++i) {
        release[i]=bars[i];
        const bool write = (i == 4 || i == 6 || i == 8);
        release[i].srcAccessMask=write ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
        release[i].dstAccessMask=0;
    }
    vkCmdPipelineBarrier(slot->commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, nullptr, 0, nullptr, 9, release);
    if (vkEndCommandBuffer(slot->commandBuffer) != VK_SUCCESS) return fail("vkEndCommandBuffer failed");

    vkResetFences(ctx_.device(), 1, &slot->workComplete);
    VkSemaphore waitSems[2];
    VkPipelineStageFlags waitStages[2];
    uint32_t waitCount = 0;
    waitSems[waitCount] = slot->readyForVulkan->vkSemaphore();
    waitStages[waitCount] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    waitCount++;
    VkSemaphore geometryReady = scene.currentGenerationReflectionsReadySemaphore();
    if (geometryReady != VK_NULL_HANDLE) {
        waitSems[waitCount] = geometryReady;
        waitStages[waitCount] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        waitCount++;
    }
    VkSemaphore signalSem=slot->readyForGl->vkSemaphore();
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.waitSemaphoreCount=waitCount; submit.pWaitSemaphores=waitSems; submit.pWaitDstStageMask=waitStages; submit.commandBufferCount=1; submit.pCommandBuffers=&slot->commandBuffer; submit.signalSemaphoreCount=1; submit.pSignalSemaphores=&signalSem;
    VkResult sr=vkQueueSubmit(ctx_.queue(),1,&submit,slot->workComplete); if(sr!=VK_SUCCESS) return fail("vkQueueSubmit failed with VkResult "+std::to_string(sr));

    drainStaleGlErrors(gl_, "glWaitSemaphoreEXT (reflections)");
    gl_.glWaitSemaphoreEXT(slot->readyForGl->glSemaphore(),0,nullptr,9,glTextures,layouts);
    const bool glWaitFailed = checkGlError(gl_, "glWaitSemaphoreEXT (reflections)");
    ++frameIndex_;
    if (glWaitFailed) return fail("OpenGL wait failed for reflection resources");
    return true;
}

bool InteropFrame::executeRtdiDebugBruteforce(RayTracingScene& scene, const std::vector<SharedTexture*>& resources,
                               const RtdiDebugConstants& c, std::string* errorOut) {
    auto fail = [&](const std::string& msg) -> bool {
        ctx_.setError("InteropFrame::executeRtdiDebugBruteforce: " + msg);
        if (errorOut) *errorOut = ctx_.lastError();
        return false;
    };
    if (resources.size() != 3) return fail("requires three resources: depth, normalRoughness, output");
    for (auto* r : resources) if (!r) return fail("null RTDI debug resource");
    if (!scene.ensureRtdiDebugPipeline()) return fail("pipeline initialization failed: " + ctx_.lastError());

    FrameSlot* slot = pickAndPrepareSlot(errorOut);
    if (!slot) return false;
    const uint32_t slotIndex = static_cast<uint32_t>((frameIndex_) % poolSize_);

    // this lambda runs unconditionally every frame this dispatch kind executes, and *dst
    auto makeView = [&](SharedTexture* tex, VkImageAspectFlags aspect, VkImageView* dst) -> bool {
        if (*dst != VK_NULL_HANDLE) { vkDestroyImageView(ctx_.device(), *dst, nullptr); *dst = VK_NULL_HANDLE; }
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = tex->vkImage(); vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = tex->vkFormat();
        vi.subresourceRange.aspectMask = aspect; vi.subresourceRange.levelCount = 1; vi.subresourceRange.layerCount = 1;
        return vkCreateImageView(ctx_.device(), &vi, nullptr, dst) == VK_SUCCESS;
    };
    if (!makeView(resources[0], VK_IMAGE_ASPECT_DEPTH_BIT, &slot->rtdiDebugDepthView) ||
        !makeView(resources[1], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtdiDebugNormalRoughnessView) ||
        !makeView(resources[2], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtdiDebugOutputView))
        return fail("vkCreateImageView failed for RTDI debug resources");

    if (slot->rtdiDebugDescriptorSet == VK_NULL_HANDLE) {
        std::vector<VkDescriptorSet> sets(1);
        std::string e;
        if (!scene.createRtdiDebugDescriptorSets(1, sets, &e)) return fail(e);
        slot->rtdiDebugDescriptorSet = sets[0];
    }
    std::string e;
    if (!scene.updateRtdiDebugDescriptorSet(slot->rtdiDebugDescriptorSet,
                                   *resources[0], *resources[1], *resources[2],
                                   slot->rtdiDebugDepthView, slot->rtdiDebugNormalRoughnessView, slot->rtdiDebugOutputView,
                                   slotIndex, c, &e)) return fail(e);

    uint32_t glTextures[3];
    for (int i = 0; i < 3; ++i) glTextures[i] = resources[i]->glTextureId();
    GLenum layouts[3];
    for (int i = 0; i < 3; ++i) layouts[i] = GL_LAYOUT_GENERAL_EXT_;
    gl_.glSignalSemaphoreEXT(slot->readyForVulkan->glSemaphore(), 0, nullptr, 3, glTextures, layouts);
    if (checkGlError(gl_, "glSignalSemaphoreEXT (RTDI debug)")) return fail("OpenGL signal failed for RTDI debug resources");

    vkResetCommandBuffer(slot->commandBuffer, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot->commandBuffer, &begin) != VK_SUCCESS) return fail("vkBeginCommandBuffer failed");

    // write target
    VkImageMemoryBarrier bars[3]{};
    for (int i=0;i<3;++i) {
        bars[i].sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        bars[i].srcAccessMask=0;
        const bool write = (i == 2);
        bars[i].dstAccessMask=write ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
        bars[i].oldLayout=VK_IMAGE_LAYOUT_GENERAL; bars[i].newLayout=VK_IMAGE_LAYOUT_GENERAL;
        bars[i].srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; bars[i].dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        bars[i].image=resources[i]->vkImage();
        bars[i].subresourceRange.aspectMask=resources[i]->isDepth()?VK_IMAGE_ASPECT_DEPTH_BIT:VK_IMAGE_ASPECT_COLOR_BIT;
        bars[i].subresourceRange.levelCount=1; bars[i].subresourceRange.layerCount=1;
    }
    vkCmdPipelineBarrier(slot->commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 3, bars);

    vkCmdBindPipeline(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelineRtdiDebug());
    vkCmdBindDescriptorSets(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelineLayoutRtdiDebug(), 0, 1, &slot->rtdiDebugDescriptorSet, 0, nullptr);
    vkCmdDispatch(slot->commandBuffer, (resources[2]->width()+7)/8, (resources[2]->height()+7)/8, 1);

    VkImageMemoryBarrier release[3]{};
    for (int i=0;i<3;++i) {
        release[i]=bars[i];
        const bool write = (i == 2);
        release[i].srcAccessMask=write ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
        release[i].dstAccessMask=0;
    }
    vkCmdPipelineBarrier(slot->commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, nullptr, 0, nullptr, 3, release);
    if (vkEndCommandBuffer(slot->commandBuffer) != VK_SUCCESS) return fail("vkEndCommandBuffer failed");

    vkResetFences(ctx_.device(), 1, &slot->workComplete);
    VkSemaphore waitSem = slot->readyForVulkan->vkSemaphore();
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSemaphore signalSem=slot->readyForGl->vkSemaphore();
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.waitSemaphoreCount=1; submit.pWaitSemaphores=&waitSem; submit.pWaitDstStageMask=&waitStage; submit.commandBufferCount=1; submit.pCommandBuffers=&slot->commandBuffer; submit.signalSemaphoreCount=1; submit.pSignalSemaphores=&signalSem;
    VkResult sr=vkQueueSubmit(ctx_.queue(),1,&submit,slot->workComplete); if(sr!=VK_SUCCESS) return fail("vkQueueSubmit failed with VkResult "+std::to_string(sr));

    drainStaleGlErrors(gl_, "glWaitSemaphoreEXT (RTDI debug)");
    gl_.glWaitSemaphoreEXT(slot->readyForGl->glSemaphore(),0,nullptr,3,glTextures,layouts);
    const bool glWaitFailed = checkGlError(gl_, "glWaitSemaphoreEXT (RTDI debug)");
    ++frameIndex_;
    if (glWaitFailed) return fail("OpenGL wait failed for RTDI debug resources");
    return true;
}

bool InteropFrame::executeRtdi(RayTracingScene& scene, const std::vector<SharedTexture*>& resources,
                               const RtdiConstants& c, std::string* errorOut) {
    auto fail = [&](const std::string& msg) -> bool {
        ctx_.setError("InteropFrame::executeRtdi: " + msg);
        if (errorOut) *errorOut = ctx_.lastError();
        return false;
    };
    if (!ctx_.capabilities().hasRayQuery) return fail("hardware ray query is unavailable");
    if (!scene.isBuilt()) return fail("ray tracing scene is not built");
    if (resources.size() != 9) return fail("requires nine resources: depth, normalRoughness, output, reservoirA, reservoirB, historyReservoirA, historyReservoirB, historyDepth, historyNormalRoughness");
    for (auto* r : resources) if (!r) return fail("null RTDI resource");

    FrameSlot* slot = pickAndPrepareSlot(errorOut);
    if (!slot) return false;
    const uint32_t slotIndex = static_cast<uint32_t>((frameIndex_) % poolSize_);

    // this lambda runs unconditionally every frame this dispatch kind executes, and *dst
    auto makeView = [&](SharedTexture* tex, VkImageAspectFlags aspect, VkImageView* dst) -> bool {
        if (*dst != VK_NULL_HANDLE) { vkDestroyImageView(ctx_.device(), *dst, nullptr); *dst = VK_NULL_HANDLE; }
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = tex->vkImage(); vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = tex->vkFormat();
        vi.subresourceRange.aspectMask = aspect; vi.subresourceRange.levelCount = 1; vi.subresourceRange.layerCount = 1;
        return vkCreateImageView(ctx_.device(), &vi, nullptr, dst) == VK_SUCCESS;
    };
    if (!makeView(resources[0], VK_IMAGE_ASPECT_DEPTH_BIT, &slot->rtdiDepthView) ||
        !makeView(resources[1], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtdiNormalRoughnessView) ||
        !makeView(resources[2], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtdiOutputView) ||
        !makeView(resources[3], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtdiReservoirAView) ||
        !makeView(resources[4], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtdiReservoirBView) ||
        !makeView(resources[5], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtdiHistoryReservoirAView) ||
        !makeView(resources[6], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtdiHistoryReservoirBView) ||
        !makeView(resources[7], VK_IMAGE_ASPECT_DEPTH_BIT, &slot->rtdiHistoryDepthView) ||
        !makeView(resources[8], VK_IMAGE_ASPECT_COLOR_BIT, &slot->rtdiHistoryNormalRoughnessView))
        return fail("vkCreateImageView failed for RTDI resources");

    if (slot->rtdiDescriptorSet == VK_NULL_HANDLE) {
        std::vector<VkDescriptorSet> sets(1);
        std::string e;
        if (!scene.createRtdiDescriptorSets(1, sets, &e)) return fail(e);
        slot->rtdiDescriptorSet = sets[0];
    }
    std::string e;
    if (!scene.updateRtdiDescriptorSet(slot->rtdiDescriptorSet,
                                   *resources[0], *resources[1], *resources[2], *resources[3], *resources[4],
                                   *resources[5], *resources[6], *resources[7], *resources[8],
                                   slot->rtdiDepthView, slot->rtdiNormalRoughnessView, slot->rtdiOutputView,
                                   slot->rtdiReservoirAView, slot->rtdiReservoirBView,
                                   slot->rtdiHistoryReservoirAView, slot->rtdiHistoryReservoirBView,
                                   slot->rtdiHistoryDepthView, slot->rtdiHistoryNormalRoughnessView,
                                   slotIndex, c, &e)) return fail(e);

    uint32_t glTextures[9];
    for (int i = 0; i < 9; ++i) glTextures[i] = resources[i]->glTextureId();
    GLenum layouts[9];
    for (int i = 0; i < 9; ++i) layouts[i] = GL_LAYOUT_GENERAL_EXT_;
    gl_.glSignalSemaphoreEXT(slot->readyForVulkan->glSemaphore(), 0, nullptr, 9, glTextures, layouts);
    if (checkGlError(gl_, "glSignalSemaphoreEXT (RTDI)")) return fail("OpenGL signal failed for RTDI resources");

    vkResetCommandBuffer(slot->commandBuffer, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot->commandBuffer, &begin) != VK_SUCCESS) return fail("vkBeginCommandBuffer failed");

    // write targets
    VkImageMemoryBarrier bars[9]{};
    for (int i=0;i<9;++i) {
        bars[i].sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        bars[i].srcAccessMask=0;
        const bool write = (i == 2 || i == 3 || i == 4);
        bars[i].dstAccessMask=write ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
        bars[i].oldLayout=VK_IMAGE_LAYOUT_GENERAL; bars[i].newLayout=VK_IMAGE_LAYOUT_GENERAL;
        bars[i].srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; bars[i].dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        bars[i].image=resources[i]->vkImage();
        bars[i].subresourceRange.aspectMask=resources[i]->isDepth()?VK_IMAGE_ASPECT_DEPTH_BIT:VK_IMAGE_ASPECT_COLOR_BIT;
        bars[i].subresourceRange.levelCount=1; bars[i].subresourceRange.layerCount=1;
    }
    vkCmdPipelineBarrier(slot->commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 9, bars);

    vkCmdBindPipeline(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelineRtdi());
    vkCmdBindDescriptorSets(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelineLayoutRtdi(), 0, 1, &slot->rtdiDescriptorSet, 0, nullptr);
    vkCmdDispatch(slot->commandBuffer, (resources[2]->width()+7)/8, (resources[2]->height()+7)/8, 1);

    VkImageMemoryBarrier release[9]{};
    for (int i=0;i<9;++i) {
        release[i]=bars[i];
        const bool write = (i == 2 || i == 3 || i == 4);
        release[i].srcAccessMask=write ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
        release[i].dstAccessMask=0;
    }
    vkCmdPipelineBarrier(slot->commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, nullptr, 0, nullptr, 9, release);
    if (vkEndCommandBuffer(slot->commandBuffer) != VK_SUCCESS) return fail("vkEndCommandBuffer failed");

    vkResetFences(ctx_.device(), 1, &slot->workComplete);
    VkSemaphore waitSems[2];
    VkPipelineStageFlags waitStages[2];
    uint32_t waitCount = 0;
    waitSems[waitCount] = slot->readyForVulkan->vkSemaphore();
    waitStages[waitCount] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    waitCount++;
    VkSemaphore geometryReady = scene.currentGenerationRtdiReadySemaphore();
    if (geometryReady != VK_NULL_HANDLE) {
        waitSems[waitCount] = geometryReady;
        waitStages[waitCount] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        waitCount++;
    }
    VkSemaphore signalSem=slot->readyForGl->vkSemaphore();
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.waitSemaphoreCount=waitCount; submit.pWaitSemaphores=waitSems; submit.pWaitDstStageMask=waitStages; submit.commandBufferCount=1; submit.pCommandBuffers=&slot->commandBuffer; submit.signalSemaphoreCount=1; submit.pSignalSemaphores=&signalSem;
    VkResult sr=vkQueueSubmit(ctx_.queue(),1,&submit,slot->workComplete); if(sr!=VK_SUCCESS) return fail("vkQueueSubmit failed with VkResult "+std::to_string(sr));

    drainStaleGlErrors(gl_, "glWaitSemaphoreEXT (RTDI)");
    gl_.glWaitSemaphoreEXT(slot->readyForGl->glSemaphore(),0,nullptr,9,glTextures,layouts);
    const bool glWaitFailed = checkGlError(gl_, "glWaitSemaphoreEXT (RTDI)");
    ++frameIndex_;
    if (glWaitFailed) return fail("OpenGL wait failed for RTDI resources");
    return true;
}

// path tracing (simple pipeline)
bool InteropFrame::executePathTrace(RayTracingScene& scene, const std::vector<SharedTexture*>& resources,
                               const PathTraceConstants& c, std::string* errorOut) {
    const VkPipelineStageFlags ptStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        (ctx_.capabilities().hasRtPipeline ? VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR : VkPipelineStageFlags(0));
    auto fail = [&](const std::string& msg) -> bool {
        ctx_.setError("InteropFrame::executePathTrace: " + msg);
        if (errorOut) *errorOut = ctx_.lastError();
        return false;
    };
    if (!ctx_.capabilities().hasRayQuery) return fail("hardware ray query is unavailable");
    if (!scene.isBuilt()) return fail("ray tracing scene is not built");
    if (resources.size() != 13) return fail("requires thirteen resources: depth, normal, albedo, motion, output, historyPrev, historyNext, historyMetaPrev, historyMetaNext, rawRadiance, glow, primaryNormal, linearDepth");
    for (auto* r : resources) if (!r) return fail("null path-trace resource");

    FrameSlot* slot = pickAndPrepareSlot(errorOut);
    if (!slot) return false;
    const uint32_t slotIndex = static_cast<uint32_t>((frameIndex_) % poolSize_);

    auto resetSemaphoresOnFailure = [&]() {
        slot->readyForVulkan = std::make_unique<SharedSemaphore>(ctx_);
        if (!slot->readyForVulkan->create()) {
            std::cerr << "[sauerinterop] executePathTrace: failed to recreate readyForVulkan semaphore during failure recovery -- this slot may remain stuck." << std::endl;
        }
        slot->readyForGl = std::make_unique<SharedSemaphore>(ctx_);
        if (!slot->readyForGl->create()) {
            std::cerr << "[sauerinterop] executePathTrace: failed to recreate readyForGl semaphore during failure recovery -- this slot may remain stuck." << std::endl;
        }
    };

    auto makeView = [&](SharedTexture* tex, VkImageAspectFlags aspect, VkImageView* dst) -> bool {
        if (*dst != VK_NULL_HANDLE) { vkDestroyImageView(ctx_.device(), *dst, nullptr); *dst = VK_NULL_HANDLE; }
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = tex->vkImage(); vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = tex->vkFormat();
        vi.subresourceRange.aspectMask = aspect; vi.subresourceRange.levelCount = 1; vi.subresourceRange.layerCount = 1;
        return vkCreateImageView(ctx_.device(), &vi, nullptr, dst) == VK_SUCCESS;
    };
    const bool pathTracePersistentViewsStillValid = c.resourceEpoch != UINT64_MAX && slot->pathTraceViewsEpoch == c.resourceEpoch &&
        slot->pathTraceDepthView && slot->pathTraceNormalView && slot->pathTraceAlbedoView &&
        slot->pathTraceMotionView && slot->pathTraceOutputView &&
        slot->pathTraceRawRadianceView && slot->pathTraceGlowView && slot->pathTracePrimaryNormalView &&
        slot->pathTraceLinearDepthView;
    if (!pathTracePersistentViewsStillValid) {
        if (!makeView(resources[0], VK_IMAGE_ASPECT_DEPTH_BIT, &slot->pathTraceDepthView) ||
            !makeView(resources[1], VK_IMAGE_ASPECT_COLOR_BIT, &slot->pathTraceNormalView) ||
            !makeView(resources[2], VK_IMAGE_ASPECT_COLOR_BIT, &slot->pathTraceAlbedoView) ||
            !makeView(resources[3], VK_IMAGE_ASPECT_COLOR_BIT, &slot->pathTraceMotionView) ||
            !makeView(resources[4], VK_IMAGE_ASPECT_COLOR_BIT, &slot->pathTraceOutputView) ||
            !makeView(resources[9], VK_IMAGE_ASPECT_COLOR_BIT, &slot->pathTraceRawRadianceView) ||
            !makeView(resources[10], VK_IMAGE_ASPECT_COLOR_BIT, &slot->pathTraceGlowView) ||
            !makeView(resources[11], VK_IMAGE_ASPECT_COLOR_BIT, &slot->pathTracePrimaryNormalView) ||
            !makeView(resources[12], VK_IMAGE_ASPECT_COLOR_BIT, &slot->pathTraceLinearDepthView))
            return fail("vkCreateImageView failed for path-trace resources");
        slot->pathTraceViewsEpoch = c.resourceEpoch;
    }
    if (!makeView(resources[5], VK_IMAGE_ASPECT_COLOR_BIT, &slot->pathTraceHistoryPrevView) ||
        !makeView(resources[6], VK_IMAGE_ASPECT_COLOR_BIT, &slot->pathTraceHistoryNextView) ||
        !makeView(resources[7], VK_IMAGE_ASPECT_COLOR_BIT, &slot->pathTraceHistoryMetaPrevView) ||
        !makeView(resources[8], VK_IMAGE_ASPECT_COLOR_BIT, &slot->pathTraceHistoryMetaNextView))
        return fail("vkCreateImageView failed for path-trace history resources");

    if (slot->pathTraceDescriptorSet == VK_NULL_HANDLE) {
        std::vector<VkDescriptorSet> sets(1);
        std::string e;
        if (!scene.createPathTraceDescriptorSets(1, sets, &e)) return fail(e);
        slot->pathTraceDescriptorSet = sets[0];
    }
    std::string e;
    if (!scene.updatePathTraceDescriptorSet(slot->pathTraceDescriptorSet,
                                   *resources[0], *resources[1], *resources[2], *resources[3], *resources[4],
                                   *resources[5], *resources[6], *resources[7], *resources[8], *resources[9],
                                   *resources[10], *resources[11], *resources[12],
                                   slot->pathTraceDepthView, slot->pathTraceNormalView, slot->pathTraceAlbedoView,
                                   slot->pathTraceMotionView, slot->pathTraceOutputView,
                                   slot->pathTraceHistoryPrevView, slot->pathTraceHistoryNextView,
                                   slot->pathTraceHistoryMetaPrevView, slot->pathTraceHistoryMetaNextView,
                                   slot->pathTraceRawRadianceView, slot->pathTraceGlowView, slot->pathTracePrimaryNormalView,
                                   slot->pathTraceLinearDepthView,
                                   slotIndex, c, &e)) return fail(e);

    uint32_t glTextures[13];
    for (int i = 0; i < 13; ++i) glTextures[i] = resources[i]->glTextureId();
    GLenum layouts[13];
    for (int i = 0; i < 13; ++i) layouts[i] = GL_LAYOUT_GENERAL_EXT_;
    if (gl_.glGetError) {
        GLenum pending;
        int pendingCount = 0;
        GLenum firstPending = 0;
        while ((pending = gl_.glGetError()) != GL_NO_ERROR_ && pendingCount < 64) {
            if (pendingCount++ == 0) firstPending = pending;
        }
        static auto lastPendingLog = std::chrono::steady_clock::time_point{};
        const auto nowPending = std::chrono::steady_clock::now();
        if (pendingCount > 0 && nowPending - lastPendingLog > std::chrono::seconds(1)) {
            lastPendingLog = nowPending;
            std::cerr << "[interop][GL] " << pendingCount << " GL error(s) already pending before the path-trace signal "
                         "(from earlier rendering, NOT the interop), first 0x" << std::hex << firstPending << std::dec << std::endl;
        }
    }
    gl_.glSignalSemaphoreEXT(slot->readyForVulkan->glSemaphore(), 0, nullptr, 13, glTextures, layouts);
    if (checkGlError(gl_, "glSignalSemaphoreEXT (path trace)")) return fail("OpenGL signal failed for path-trace resources");

    vkResetCommandBuffer(slot->commandBuffer, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot->commandBuffer, &begin) != VK_SUCCESS) { resetSemaphoresOnFailure(); return fail("vkBeginCommandBuffer failed"); }

    // acquire barrier
    VkImageMemoryBarrier acquire[13]{};
    for (int i=0;i<13;++i) {
        acquire[i].sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        acquire[i].srcAccessMask=0;
        acquire[i].dstAccessMask=VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        acquire[i].oldLayout=VK_IMAGE_LAYOUT_GENERAL; acquire[i].newLayout=VK_IMAGE_LAYOUT_GENERAL;
        acquire[i].srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; acquire[i].dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        acquire[i].image=resources[i]->vkImage();
        acquire[i].subresourceRange.aspectMask=resources[i]->isDepth()?VK_IMAGE_ASPECT_DEPTH_BIT:VK_IMAGE_ASPECT_COLOR_BIT;
        acquire[i].subresourceRange.levelCount=1; acquire[i].subresourceRange.layerCount=1;
    }
    vkCmdPipelineBarrier(slot->commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ptStage,
                         0, 0, nullptr, 0, nullptr, 13, acquire);

    // zero this slot's ray counters, timestamp 0
    const bool stats = gpuStatsEnabled_;
    const uint32_t qBase = slotIndex * 4;
    if (stats) {
        VkBuffer rs = scene.rayStatsBuffer(slotIndex);
        if (rs != VK_NULL_HANDLE) {
            vkCmdFillBuffer(slot->commandBuffer, rs, 0, VK_WHOLE_SIZE, 0u);
            VkMemoryBarrier fillToCompute{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            fillToCompute.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            fillToCompute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(slot->commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, ptStage,
                                 0, 1, &fillToCompute, 0, nullptr, 0, nullptr);
        }
        if (timestampPool_ != VK_NULL_HANDLE) {
            vkCmdResetQueryPool(slot->commandBuffer, timestampPool_, qBase, 4);
            vkCmdWriteTimestamp(slot->commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampPool_, qBase + 0);
        }
    }

    const uint32_t causticPhotonCount = scene.causticPhotonDispatchCount(c);
    if (causticPhotonCount > 0 && scene.causticPhotonBuffer(slotIndex) && scene.causticHashBuffer(slotIndex)) {
        VkMemoryBarrier previousGathers{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        previousGathers.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        previousGathers.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(slot->commandBuffer, ptStage,
                             VK_PIPELINE_STAGE_TRANSFER_BIT | ptStage,
                             0, 1, &previousGathers, 0, nullptr, 0, nullptr);
        // every batch is re-emitted every frame
        vkCmdFillBuffer(slot->commandBuffer, scene.causticHashBuffer(slotIndex), 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);
        vkCmdFillBuffer(slot->commandBuffer, scene.causticPhotonBuffer(slotIndex), 0, 16 * VkDeviceSize(RayTracingScene::kCausticBatches), 0u);
        VkMemoryBarrier clearToCompute{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        clearToCompute.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        clearToCompute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(slot->commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, ptStage,
                             0, 1, &clearToCompute, 0, nullptr, 0, nullptr);
        vkCmdBindPipeline(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelinePathTraceCaustics());
        vkCmdBindDescriptorSets(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelineLayoutPathTrace(), 0, 1, &slot->pathTraceDescriptorSet, 0, nullptr);
        vkCmdDispatch(slot->commandBuffer, (causticPhotonCount + 63) / 64, RayTracingScene::kCausticBatches, 1);  // y = batch
        VkMemoryBarrier photonsToGather{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        photonsToGather.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        photonsToGather.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(slot->commandBuffer, ptStage, ptStage,
                             0, 1, &photonsToGather, 0, nullptr, 0, nullptr);
    }

    if (stats && timestampPool_ != VK_NULL_HANDLE) vkCmdWriteTimestamp(slot->commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampPool_, qBase + 1);

    const bool radianceCache = c.cacheEnabled && scene.pipelinePathTraceCache() != VK_NULL_HANDLE;
    if (radianceCache) {
        VkMemoryBarrier prevCache{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        prevCache.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        prevCache.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(slot->commandBuffer, ptStage, ptStage,
                             0, 1, &prevCache, 0, nullptr, 0, nullptr);
    }

    if (c.skyboxLight && scene.pipelinePathTraceSkyCdf() != VK_NULL_HANDLE) {
        VkMemoryBarrier prevReads{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        prevReads.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        prevReads.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(slot->commandBuffer, ptStage, ptStage,
                             0, 1, &prevReads, 0, nullptr, 0, nullptr);
        vkCmdBindPipeline(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelinePathTraceSkyCdf());
        vkCmdBindDescriptorSets(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelineLayoutPathTrace(), 0, 1, &slot->pathTraceDescriptorSet, 0, nullptr);
        vkCmdDispatch(slot->commandBuffer, 1, 1, 1);
        VkMemoryBarrier cdfToTrace{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        cdfToTrace.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        cdfToTrace.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(slot->commandBuffer, ptStage, ptStage,
                             0, 1, &cdfToTrace, 0, nullptr, 0, nullptr);
    }

    // pass 1
    const bool tracedRays = c.traceMode != 0 &&
        scene.recordPathTraceRays(slot->commandBuffer, slot->pathTraceDescriptorSet, resources[9]->width(), resources[9]->height(), c.traceMode == 2);
    if (!tracedRays) {
        vkCmdBindPipeline(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelinePathTrace());
        vkCmdBindDescriptorSets(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelineLayoutPathTrace(), 0, 1, &slot->pathTraceDescriptorSet, 0, nullptr);
        vkCmdDispatch(slot->commandBuffer, (resources[9]->width()+7)/8, (resources[9]->height()+7)/8, 1);
    }
    if (stats && timestampPool_ != VK_NULL_HANDLE) vkCmdWriteTimestamp(slot->commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampPool_, qBase + 2);

    VkImageMemoryBarrier rawRadianceBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    rawRadianceBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    rawRadianceBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    rawRadianceBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    rawRadianceBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    rawRadianceBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    rawRadianceBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    rawRadianceBarrier.image = resources[9]->vkImage();
    rawRadianceBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    rawRadianceBarrier.subresourceRange.levelCount = 1; rawRadianceBarrier.subresourceRange.layerCount = 1;
    VkMemoryBarrier traceToResolve{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    traceToResolve.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    traceToResolve.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(slot->commandBuffer, ptStage, ptStage,
                         0, 1, &traceToResolve, 0, nullptr, 1, &rawRadianceBarrier);

    // pass 2
    vkCmdBindPipeline(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelinePathTraceResolve());
    // bound here, not inherited
    vkCmdBindDescriptorSets(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelineLayoutPathTrace(), 0, 1, &slot->pathTraceDescriptorSet, 0, nullptr);
    vkCmdDispatch(slot->commandBuffer, (resources[4]->width()+7)/8, (resources[4]->height()+7)/8, 1);

    // radiance cache maintenance
    if (radianceCache) {
        VkMemoryBarrier traceToCache{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        traceToCache.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        traceToCache.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(slot->commandBuffer, ptStage, ptStage,
                             0, 1, &traceToCache, 0, nullptr, 0, nullptr);
        vkCmdBindPipeline(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelinePathTraceCache());
        // bound here, not inherited
        vkCmdBindDescriptorSets(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelineLayoutPathTrace(), 0, 1, &slot->pathTraceDescriptorSet, 0, nullptr);
        vkCmdDispatch(slot->commandBuffer, RayTracingScene::kRadianceCacheSize / 256, 1, 1);
    }
    if (stats && timestampPool_ != VK_NULL_HANDLE) vkCmdWriteTimestamp(slot->commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampPool_, qBase + 3);

    // release barrier
    VkImageMemoryBarrier release[13]{};
    for (int i=0;i<13;++i) {
        release[i]=acquire[i];
        release[i].srcAccessMask=VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        release[i].dstAccessMask=0;
    }
    vkCmdPipelineBarrier(slot->commandBuffer, ptStage, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, nullptr, 0, nullptr, 13, release);
    if (vkEndCommandBuffer(slot->commandBuffer) != VK_SUCCESS) { resetSemaphoresOnFailure(); return fail("vkEndCommandBuffer failed"); }

    vkResetFences(ctx_.device(), 1, &slot->workComplete);
    VkSemaphore waitSems[2];
    VkPipelineStageFlags waitStages[2];
    uint32_t waitCount = 0;
    const VkPipelineStageFlags waitStage = stats ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : ptStage;
    waitSems[waitCount] = slot->readyForVulkan->vkSemaphore();
    waitStages[waitCount] = waitStage;
    waitCount++;
    VkSemaphore geometryReady = scene.currentGenerationPathTraceReadySemaphore();
    if (geometryReady != VK_NULL_HANDLE) {
        waitSems[waitCount] = geometryReady;
        waitStages[waitCount] = waitStage;
        waitCount++;
    }
    VkSemaphore signalSem=slot->readyForGl->vkSemaphore();
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.waitSemaphoreCount=waitCount; submit.pWaitSemaphores=waitSems; submit.pWaitDstStageMask=waitStages; submit.commandBufferCount=1; submit.pCommandBuffers=&slot->commandBuffer; submit.signalSemaphoreCount=1; submit.pSignalSemaphores=&signalSem;
    VkResult sr=vkQueueSubmit(ctx_.queue(),1,&submit,slot->workComplete); if(sr!=VK_SUCCESS) { resetSemaphoresOnFailure(); return fail("vkQueueSubmit failed with VkResult "+std::to_string(sr)); }
    if (stats) { slot->pendingStatsKind = 1; slot->pendingRayStats = scene.rayStatsMapped(slotIndex); }

    drainStaleGlErrors(gl_, "glWaitSemaphoreEXT (path trace)");
    gl_.glWaitSemaphoreEXT(slot->readyForGl->glSemaphore(),0,nullptr,13,glTextures,layouts);
    const bool glWaitFailed = checkGlError(gl_, "glWaitSemaphoreEXT (path trace)");
    ++frameIndex_;
    if (glWaitFailed) return fail("OpenGL wait failed for path-trace resources");
    return true;
}

bool InteropFrame::executeVolumetricFog(RayTracingScene& scene, const GpuVolFogConstants& constants,
                                         std::string* errorOut) {
    auto fail = [&](const std::string& msg) -> bool {
        ctx_.setError("InteropFrame::executeVolumetricFog: " + msg);
        if (errorOut) *errorOut = ctx_.lastError();
        return false;
    };
    if (!ctx_.capabilities().hasRayQuery) return fail("hardware ray query is unavailable");
    if (!scene.isBuilt()) return fail("ray tracing scene is not built");

    if (!scene.ensureVolumetricFogPipeline()) return fail("ensureVolumetricFogPipeline failed: " + ctx_.lastError());
    if (!scene.ensureVolFogTextures(constants.gridDimX, constants.gridDimY, constants.gridDimZ, errorOut)) return false;

    FrameSlot* slot = pickAndPrepareSlot(errorOut);
    if (!slot) return false;
    const uint32_t slotIndex = static_cast<uint32_t>((frameIndex_) % poolSize_);

    // same recreate-on-failure recovery as executePathTrace()'s identical lambda
    auto resetSemaphoresOnFailure = [&]() {
        slot->readyForVulkan = std::make_unique<SharedSemaphore>(ctx_);
        if (!slot->readyForVulkan->create()) {
            std::cerr << "[sauerinterop] executeVolumetricFog: failed to recreate readyForVulkan semaphore during failure recovery -- this slot may remain stuck." << std::endl;
        }
        slot->readyForGl = std::make_unique<SharedSemaphore>(ctx_);
        if (!slot->readyForGl->create()) {
            std::cerr << "[sauerinterop] executeVolumetricFog: failed to recreate readyForGl semaphore during failure recovery -- this slot may remain stuck." << std::endl;
        }
    };

    if (slot->volFogDescriptorSet == VK_NULL_HANDLE) {
        std::vector<VkDescriptorSet> sets(1);
        std::string e;
        if (!scene.createVolFogDescriptorSets(1, sets, &e)) return fail(e);
        slot->volFogDescriptorSet = sets[0];
    }
    std::string e;
    if (!scene.updateVolFogDescriptorSet(slot->volFogDescriptorSet, slotIndex, constants, &e)) return fail(e);

    // no incoming glSignalSemaphoreEXT/acquire wait
    vkResetCommandBuffer(slot->commandBuffer, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot->commandBuffer, &begin) != VK_SUCCESS) { resetSemaphoresOnFailure(); return fail("vkBeginCommandBuffer failed"); }

    const bool fogStats = gpuStatsEnabled_ && timestampPool_ != VK_NULL_HANDLE;
    const uint32_t fogQBase = slotIndex * 4;
    if (fogStats) {
        vkCmdResetQueryPool(slot->commandBuffer, timestampPool_, fogQBase, 4);
        vkCmdWriteTimestamp(slot->commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampPool_, fogQBase + 0);
    }
    vkCmdBindDescriptorSets(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelineLayoutVolFog(), 0, 1, &slot->volFogDescriptorSet, 0, nullptr);
    if (constants.emissiveFog[2] > 0.5f) {
        // volfogcull
        vkCmdBindPipeline(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelineVolFogDepth());
        vkCmdDispatch(slot->commandBuffer, (scene.volFogGridX()+7)/8, (scene.volFogGridY()+7)/8, 1);
        VkMemoryBarrier depthDone{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        depthDone.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        depthDone.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(slot->commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &depthDone, 0, nullptr, 0, nullptr);
    }
    vkCmdBindPipeline(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelineVolFogInject());
    vkCmdDispatch(slot->commandBuffer, (scene.volFogGridX()+7)/8, (scene.volFogGridY()+7)/8, (scene.volFogGridZ()+3)/4);
    if (fogStats) vkCmdWriteTimestamp(slot->commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampPool_, fogQBase + 1);

    // inter-pass barrier
    VkImageMemoryBarrier scatterExtinctionBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    scatterExtinctionBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    scatterExtinctionBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    scatterExtinctionBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    scatterExtinctionBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    scatterExtinctionBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    scatterExtinctionBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    scatterExtinctionBarrier.image = scene.volFogScatterExtinction().vkImage();
    scatterExtinctionBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    scatterExtinctionBarrier.subresourceRange.levelCount = 1; scatterExtinctionBarrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(slot->commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &scatterExtinctionBarrier);

    // pass 2
    vkCmdBindPipeline(slot->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scene.pipelineVolFogIntegrate());
    vkCmdDispatch(slot->commandBuffer, (scene.volFogGridX()+7)/8, (scene.volFogGridY()+7)/8, 1);
    if (fogStats) vkCmdWriteTimestamp(slot->commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampPool_, fogQBase + 2);

    // release barrier
    VkImageMemoryBarrier integratedRelease{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    integratedRelease.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    integratedRelease.dstAccessMask = 0;
    integratedRelease.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    integratedRelease.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    integratedRelease.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    integratedRelease.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    integratedRelease.image = scene.volFogIntegrated().vkImage();
    integratedRelease.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    integratedRelease.subresourceRange.levelCount = 1; integratedRelease.subresourceRange.layerCount = 1;
    // this frame's light volume goes to GL too
    VkImageMemoryBarrier fogReleases[2] = { integratedRelease, integratedRelease };
    SharedTexture3D* partLight = scene.volFogPartLightCurrent();
    uint32_t releaseCount = 1;
    if (partLight) { fogReleases[1].image = partLight->vkImage(); releaseCount = 2; }
    vkCmdPipelineBarrier(slot->commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, nullptr, 0, nullptr, releaseCount, fogReleases);
    if (vkEndCommandBuffer(slot->commandBuffer) != VK_SUCCESS) { resetSemaphoresOnFailure(); return fail("vkEndCommandBuffer failed"); }

    vkResetFences(ctx_.device(), 1, &slot->workComplete);
    // same GPU-side TLAS-build-ordering wait as executePathTrace's identical block
    VkSemaphore waitSems[1];
    VkPipelineStageFlags waitStages[1];
    uint32_t waitCount = 0;
    VkSemaphore geometryReady = scene.currentGenerationPathTraceReadySemaphore();
    if (geometryReady != VK_NULL_HANDLE) {
        waitSems[waitCount] = geometryReady;
        waitStages[waitCount] = fogStats ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;  // see executePathTrace's path-trace stats wait note
        waitCount++;
    }
    VkSemaphore signalSem = slot->readyForGl->vkSemaphore();
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.waitSemaphoreCount=waitCount; submit.pWaitSemaphores=waitCount?waitSems:nullptr; submit.pWaitDstStageMask=waitCount?waitStages:nullptr; submit.commandBufferCount=1; submit.pCommandBuffers=&slot->commandBuffer; submit.signalSemaphoreCount=1; submit.pSignalSemaphores=&signalSem;
    VkResult sr=vkQueueSubmit(ctx_.queue(),1,&submit,slot->workComplete); if(sr!=VK_SUCCESS) { resetSemaphoresOnFailure(); return fail("vkQueueSubmit failed with VkResult "+std::to_string(sr)); }
    if (fogStats) slot->pendingStatsKind = 2;

    // register the GL-side wait right here
    uint32_t glTextures[2] = { scene.volFogIntegrated().glTextureId(), partLight ? partLight->glTextureId() : 0u };
    GLenum layouts[2] = { GL_LAYOUT_GENERAL_EXT_, GL_LAYOUT_GENERAL_EXT_ };
    drainStaleGlErrors(gl_, "glWaitSemaphoreEXT (volumetric fog)");
    gl_.glWaitSemaphoreEXT(slot->readyForGl->glSemaphore(), 0, nullptr, releaseCount, glTextures, layouts);
    const bool glWaitFailed = checkGlError(gl_, "glWaitSemaphoreEXT (volumetric fog)");
    ++frameIndex_;
    if (glWaitFailed) return fail("OpenGL wait failed for volumetric fog integrated texture");
    return true;
}

}  // namespace interop
