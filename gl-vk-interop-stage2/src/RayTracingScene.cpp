#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "RayTracingScene.h"
#include "InteropContext.h"
#include "InteropFrame.h"
#include "SharedTexture.h"
#include "SharedTexture3D.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

namespace interop {
namespace {
struct GpuParams {
    float invViewProj[16];
    float invView[16];
    float currentViewProjUnjittered[16];
    float prevViewProjJittered[16];
    float prevViewProjUnjittered[16];
    float settings[4];
    uint32_t frameData[4];
    float shadowLightAndRadius[4];  // lightDirection.xyz, lightRadius (soft-shadow penumbra size)
    uint32_t shadowData[4];  // shadowRaysPerPixel, shadowEnabled, pad, pad
};

struct GpuReflectionParams {
    float invViewProj[16];
    float invView[16];
    float currentViewProjUnjittered[16];
    float prevViewProjJittered[16];
    float prevViewProjUnjittered[16];
    float settings[4];  // bias, maxRayDistance, maxConeAngleRadians, historyValid
    uint32_t frameData[4];  // frameIndex, width, height, raysPerPixel
    float extra[4];  // prevColorValid, pad, pad, pad
    float skyHorizonColor[4];  // rgb, pad
    float skyZenithColor[4];  // rgb, pad
    float skyGroundColor[4];  // rgb, pad
};

struct GpuRtdiDebugParams {
    float invViewProj[16];
    float invView[16];
    uint32_t frameData[4];  // frameIndex, width, height, lightCount
};

struct GpuRtdiParams {
    float invViewProj[16];
    float invView[16];
    uint32_t frameData[4];  // frameIndex, width, height, lightCount
    // candidateCount, bias, maxRayDistance, pad
    uint32_t candidateCount;
    float bias;
    float maxRayDistance;
    float pad0;
    float previousViewProj[16];
    float maxHistoryM;
    float depthRejectThreshold;
    float pad1;
    float pad2;
};

struct GpuPathTraceParams {
    float invViewProj[16];
    float invView[16];
    float prevViewProj[16];
    uint32_t frameData[4];  // frameIndex, bounceCount, historyValid(0/1), lightCount
    float sunDirectionAndRadius[4];  // xyz = sun travel direction, w = angular radius
    float sunColor[4];  // rgb, [3] = maxHistorySamples, , repurposed pad
    float skyAmbient[4];
    float ambientFloor[4];  // rgb, pad
    float emissiveImportance[4];
    // the forward (non-inverted) view-projection matrix
    float viewProj[16];
    float prevViewProjJittered[16];
    // skybox:. Same "strictly additive, appended at the very end" discipline as
    // viewProj/prevViewProjJittered above
    uint32_t skyFaceIndexLo[4];  // lf(-X), rt(+X), ft(-Y), bk(+Y)
    uint32_t skyFaceIndexHi[4];  // dn(-Z), up(+Z), skyValid(0/1), pad
    float skyRotationPad[4];  // skyRotationRadians, pad, pad, pad
    float pomParams[4];  // enabled, max iterations, depth multiplier, mode
    float glassParams[4];
    float causticParams[4];  // on, gather radius, photons per batch, glass emitter triangle count
    float causticParams2[4];  // batch re-emitted this frame, valid-batch bitmask, pad, pad
    float waterParams[4];  // camera in water (0/1), clarity, wave strength, time (s)
    float waterCamera[4];  // camera's water colour rgb, its getwaterfog depth
    float waterParams2[4];
    float cacheParams[4];  // on, base cell size, max frames, 1-in-N update
    float cacheParams2[4];  // ambient floor scale, min frames to use, eviction frames, pad
    float skyParams[4];  // on, pad, pad, pad
    float matParams[4];  // physically-based spec translation (ptmat), pad, pad, pad
    float particleParams[4];  // x = sprite brightness in reflections (ptpartreflect)
    float waterPatch[4];  // origin xy, size (0 = none), flat water chunk's customIndex
    float waterWave[4];
    float waterFft[4];  // rt_water.glsl fft
    float waterMap[4];  // rt_water.glsl map
    float lava0[4];  // plate size, flow speed, relief, crack width
    float lava1[4];  // molten K, crust K, time, crust coverage
    float waterVolume[4];  // scattering coefficient, HG g, wave focusing
    float rrParams[4];
    float lavaPatch[4];  // flat lava chunk's customIndex (-1 none), its top triangles, pad, pad
};

struct GpuPathTraceMaterialEntry {
    uint64_t vertexAddr;
    uint64_t indexAddr;
    // this slot's previous-frame vertex buffer address
    uint64_t prevVertexAddr;
    // keeps sizeof(GpuPathTraceMaterialEntry) a multiple of 16 bytes
    uint64_t pad0;
    float albedo[3];
    float opaqueIndexCount;
};

struct GpuPathTraceMaterialInfo {
    uint32_t diffuseTexIndex;
    uint32_t normalTexIndex;
    uint32_t glowTexIndex;
    uint32_t flags;
    float colorscale[4];  // rgb, metalness
    float glowcolor[4];
    float parallax[4];
    float scroll[4];  // xy = UV units per second
};

// pbr light transport one emissive world-space triangle
struct GpuEmissiveTriangle {
    float v0[4];
    float v1[4];
    float v2[4];
    float radiance[4];
};

static bool readSpirv(const std::string& path, std::vector<uint32_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const auto size = f.tellg();
    if (size <= 0 || (size % 4) != 0) return false;
    out.resize(static_cast<size_t>(size) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(out.data()), size);
    return static_cast<bool>(f);
}
}

RayTracingScene::RayTracingScene(InteropContext& ctx, const std::string& shaderDirectory, uint32_t frameSlots, uint32_t dispatchSlots)
    : ctx_(ctx), frameSlots_(std::max(1u, frameSlots)), dispatchSlots_(std::max(1u, dispatchSlots)), shaderDirectory_(shaderDirectory) {}

RayTracingScene::~RayTracingScene() {
    if (ctx_.device() == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(ctx_.device());
    for (auto& m : meshes_) {
        if (m.blas != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_)
            vkDestroyAccelerationStructureKHR_(ctx_.device(), m.blas, nullptr);
        destroyBuffer(m.blasBuffer);
        destroyBuffer(m.vertex);
        destroyBuffer(m.index);
    }
    for (auto& generation : dynamicGenerations_) {
        for (auto& b : generation.retiredStagingBuffers_) destroyBuffer(b);
        generation.retiredStagingBuffers_.clear();
        for (auto& slot : generation.slots) {
            if (slot.blas != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_)
                vkDestroyAccelerationStructureKHR_(ctx_.device(), slot.blas, nullptr);
            destroyBuffer(slot.blasBuffer);
            destroyBuffer(slot.vertex);
            destroyBuffer(slot.index);
            destroyBuffer(slot.stagingVertex);
            destroyBuffer(slot.stagingIndex);
            destroyBuffer(slot.scratch);
            destroyBuffer(slot.boneBuffer);
            destroyBuffer(slot.boneStaging);
            destroyBuffer(slot.skinParams);
            destroyBuffer(slot.skinParamsStaging);
        }
        if (generation.tlas != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_)
            vkDestroyAccelerationStructureKHR_(ctx_.device(), generation.tlas, nullptr);
        destroyBuffer(generation.tlasBuffer);
        destroyBuffer(generation.tlasInstanceBuffer);
        destroyBuffer(generation.tlasScratch);
        destroyBuffer(generation.materials);
        if (generation.buildFence != VK_NULL_HANDLE) vkDestroyFence(ctx_.device(), generation.buildFence, nullptr);
        if (generation.readyForAo != VK_NULL_HANDLE) vkDestroySemaphore(ctx_.device(), generation.readyForAo, nullptr);
        if (generation.readyForReflections != VK_NULL_HANDLE) vkDestroySemaphore(ctx_.device(), generation.readyForReflections, nullptr);
        if (generation.readyForRtdi != VK_NULL_HANDLE) vkDestroySemaphore(ctx_.device(), generation.readyForRtdi, nullptr);
        if (generation.readyForPathTrace != VK_NULL_HANDLE) vkDestroySemaphore(ctx_.device(), generation.readyForPathTrace, nullptr);
        if (generation.cmd != VK_NULL_HANDLE) vkFreeCommandBuffers(ctx_.device(), ctx_.setupCommandPool(), 1, &generation.cmd);
    }
    if (timestampPool_ != VK_NULL_HANDLE) vkDestroyQueryPool(ctx_.device(), timestampPool_, nullptr);
    for (auto& s : paramSlots_) {
        if (s.mapped && s.memory) vkUnmapMemory(ctx_.device(), s.memory);
        if (s.buffer) vkDestroyBuffer(ctx_.device(), s.buffer, nullptr);
        if (s.memory) vkFreeMemory(ctx_.device(), s.memory, nullptr);
    }
    if (descriptorPool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), descriptorPool_, nullptr);
    if (sampler_ != VK_NULL_HANDLE) vkDestroySampler(ctx_.device(), sampler_, nullptr);
    if (historySampler_ != VK_NULL_HANDLE) vkDestroySampler(ctx_.device(), historySampler_, nullptr);
    if (rtaoPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), rtaoPipeline_, nullptr);
    if (pipelineLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), pipelineLayout_, nullptr);
    if (descriptorSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), descriptorSetLayout_, nullptr);
    if (rtaoShader_ != VK_NULL_HANDLE) vkDestroyShaderModule(ctx_.device(), rtaoShader_, nullptr);
    for (auto& kv : skinSources_) { destroyBuffer(kv.second.vertexBuffer); destroyBuffer(kv.second.indexBuffer); }
    skinSources_.clear();
    for (auto& up : pendingSkinUploads_) { destroyBuffer(up.stagingVertex); destroyBuffer(up.stagingIndex); }
    pendingSkinUploads_.clear();
    for (auto& kv : rigidSources_) {
        if (kv.second.blas != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_)
            vkDestroyAccelerationStructureKHR_(ctx_.device(), kv.second.blas, nullptr);
        destroyBuffer(kv.second.blasBuffer);
        destroyBuffer(kv.second.vertexBuffer);
        destroyBuffer(kv.second.indexBuffer);
    }
    rigidSources_.clear();
    for (auto& up : pendingRigidUploads_) { destroyBuffer(up.stagingVertex); destroyBuffer(up.stagingIndex); }
    pendingRigidUploads_.clear();
    if (descriptorPoolSkin_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), descriptorPoolSkin_, nullptr);
    if (skinPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), skinPipeline_, nullptr);
    if (pipelineLayoutSkin_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), pipelineLayoutSkin_, nullptr);
    if (descriptorSetLayoutSkin_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), descriptorSetLayoutSkin_, nullptr);
    if (skinShader_ != VK_NULL_HANDLE) vkDestroyShaderModule(ctx_.device(), skinShader_, nullptr);
    // displaced water patch
    for (auto& g : waterPatchGens_) {
        if (g.blas != VK_NULL_HANDLE) vkDestroyAccelerationStructureKHR_(ctx_.device(), g.blas, nullptr);
        destroyBuffer(g.vertex); destroyBuffer(g.blasBuffer); destroyBuffer(g.scratch); destroyBuffer(g.params);
        destroyBuffer(g.fftTemp); destroyBuffer(g.fftParams);
        if (g.fftView != VK_NULL_HANDLE) vkDestroyImageView(ctx_.device(), g.fftView, nullptr);
        if (g.fftImage != VK_NULL_HANDLE) vkDestroyImage(ctx_.device(), g.fftImage, nullptr);
        if (g.fftMemory != VK_NULL_HANDLE) vkFreeMemory(ctx_.device(), g.fftMemory, nullptr);
    }
    // FFT sea / wave map
    destroyBuffer(waveSpectrum_); destroyBuffer(waveSpectrumParams_); destroyBuffer(waveMapParams_);
    if (waveMapView_ != VK_NULL_HANDLE) vkDestroyImageView(ctx_.device(), waveMapView_, nullptr);
    if (waveMapImage_ != VK_NULL_HANDLE) vkDestroyImage(ctx_.device(), waveMapImage_, nullptr);
    if (waveMapMemory_ != VK_NULL_HANDLE) vkFreeMemory(ctx_.device(), waveMapMemory_, nullptr);
    if (waveFftSampler_ != VK_NULL_HANDLE) vkDestroySampler(ctx_.device(), waveFftSampler_, nullptr);
    if (waveMapSampler_ != VK_NULL_HANDLE) vkDestroySampler(ctx_.device(), waveMapSampler_, nullptr);
    if (descriptorPoolWave_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), descriptorPoolWave_, nullptr);
    for (VkPipeline pl : {waveSpectrumPipeline_, waveFftPipeline_, waveMapPipeline_}) if (pl != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pl, nullptr);
    for (VkPipelineLayout pl : {waveSpectrumLayout_, waveFftLayout_, waveMapLayout_}) if (pl != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), pl, nullptr);
    for (VkDescriptorSetLayout dl : {waveSpectrumSetLayout_, waveFftSetLayout_, waveMapSetLayout_}) if (dl != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), dl, nullptr);
    for (VkShaderModule sm : {waveSpectrumShader_, waveFftShader_, waveMapShader_}) if (sm != VK_NULL_HANDLE) vkDestroyShaderModule(ctx_.device(), sm, nullptr);
    waterPatchGens_.clear();
    destroyBuffer(waterRectBuffer_); destroyBuffer(waterBucketBuffer_); destroyBuffer(waterItemBuffer_); destroyBuffer(waterPatchIndexBuffer_);
    if (descriptorPoolWaterPatch_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), descriptorPoolWaterPatch_, nullptr);
    if (waterPatchPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), waterPatchPipeline_, nullptr);
    if (pipelineLayoutWaterPatch_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), pipelineLayoutWaterPatch_, nullptr);
    if (descriptorSetLayoutWaterPatch_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), descriptorSetLayoutWaterPatch_, nullptr);
    if (waterPatchShader_ != VK_NULL_HANDLE) vkDestroyShaderModule(ctx_.device(), waterPatchShader_, nullptr);
    for (auto& s : reflectionParamSlots_) {
        if (s.mapped && s.memory) vkUnmapMemory(ctx_.device(), s.memory);
        if (s.buffer) vkDestroyBuffer(ctx_.device(), s.buffer, nullptr);
        if (s.memory) vkFreeMemory(ctx_.device(), s.memory, nullptr);
    }
    if (descriptorPoolReflection_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), descriptorPoolReflection_, nullptr);
    if (reflectionPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), reflectionPipeline_, nullptr);
    if (pipelineLayoutReflection_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), pipelineLayoutReflection_, nullptr);
    if (descriptorSetLayoutReflection_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), descriptorSetLayoutReflection_, nullptr);
    if (reflectionShader_ != VK_NULL_HANDLE) vkDestroyShaderModule(ctx_.device(), reflectionShader_, nullptr);
    for (auto& s : rtdiDebugParamSlots_) {
        if (s.mapped && s.memory) vkUnmapMemory(ctx_.device(), s.memory);
        if (s.buffer) vkDestroyBuffer(ctx_.device(), s.buffer, nullptr);
        if (s.memory) vkFreeMemory(ctx_.device(), s.memory, nullptr);
    }
    for (auto& b : rtdiLightBuffers_) {
        destroyBuffer(b);
    }
    destroyBuffer(emissiveMaterialsBuffer_);
    if (descriptorPoolRtdiDebug_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), descriptorPoolRtdiDebug_, nullptr);
    if (rtdiDebugPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), rtdiDebugPipeline_, nullptr);
    if (pipelineLayoutRtdiDebug_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), pipelineLayoutRtdiDebug_, nullptr);
    if (descriptorSetLayoutRtdiDebug_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), descriptorSetLayoutRtdiDebug_, nullptr);
    if (rtdiDebugShader_ != VK_NULL_HANDLE) vkDestroyShaderModule(ctx_.device(), rtdiDebugShader_, nullptr);
    for (auto& s : rtdiParamSlots_) {
        if (s.mapped && s.memory) vkUnmapMemory(ctx_.device(), s.memory);
        if (s.buffer) vkDestroyBuffer(ctx_.device(), s.buffer, nullptr);
        if (s.memory) vkFreeMemory(ctx_.device(), s.memory, nullptr);
    }
    if (descriptorPoolRtdi_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), descriptorPoolRtdi_, nullptr);
    if (rtdiPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), rtdiPipeline_, nullptr);
    if (pipelineLayoutRtdi_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), pipelineLayoutRtdi_, nullptr);
    if (descriptorSetLayoutRtdi_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), descriptorSetLayoutRtdi_, nullptr);
    if (rtdiShader_ != VK_NULL_HANDLE) vkDestroyShaderModule(ctx_.device(), rtdiShader_, nullptr);
    for (auto& s : pathTraceParamSlots_) {
        if (s.mapped && s.memory) vkUnmapMemory(ctx_.device(), s.memory);
        if (s.buffer) vkDestroyBuffer(ctx_.device(), s.buffer, nullptr);
        if (s.memory) vkFreeMemory(ctx_.device(), s.memory, nullptr);
    }
    for (auto& s : rayStatsSlots_) {
        if (s.mapped && s.memory) vkUnmapMemory(ctx_.device(), s.memory);
        if (s.buffer) vkDestroyBuffer(ctx_.device(), s.buffer, nullptr);
        if (s.memory) vkFreeMemory(ctx_.device(), s.memory, nullptr);
    }
    for (auto& b : pathTraceLightBuffers_) {
        destroyBuffer(b);
    }
    for (auto& b : volFogLightBuffers_) {
        destroyBuffer(b);
    }
    for (auto& b : volFogDepthBuffers_) {
        destroyBuffer(b);
    }
    destroyBuffer(lightGridBuffer_);
    destroyBuffer(pathTraceMaterialsBuffer_);
    destroyBuffer(pathTraceMaterialInfoBuffer_);
    destroyBuffer(pathTraceEmissiveBuffer_);
    for (auto& tex : pathTraceTextures_) {
        if (tex.sampler) vkDestroySampler(ctx_.device(), tex.sampler, nullptr);
        if (tex.view) vkDestroyImageView(ctx_.device(), tex.view, nullptr);
        if (tex.image) vkDestroyImage(ctx_.device(), tex.image, nullptr);
        if (tex.memory) vkFreeMemory(ctx_.device(), tex.memory, nullptr);
    }
    pathTraceTextures_.clear();
    pathTraceTextureIndices_.clear();
    for (auto& up : pendingTextureUploads_) { destroyBuffer(up.staging); }
    pendingTextureUploads_.clear();
    if (descriptorPoolPathTrace_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), descriptorPoolPathTrace_, nullptr);
    if (pathTracePipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pathTracePipeline_, nullptr);
    if (pathTraceResolvePipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pathTraceResolvePipeline_, nullptr);
    if (causticPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), causticPipeline_, nullptr);
    for (int i = 0; i < 2; ++i) {  // raygen trace
        if (pathTraceRgenPipeline_[i] != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pathTraceRgenPipeline_[i], nullptr);
        if (pathTraceRgenShader_[i] != VK_NULL_HANDLE) vkDestroyShaderModule(ctx_.device(), pathTraceRgenShader_[i], nullptr);
        destroyBuffer(pathTraceRgenSbt_[i]);
        pathTraceRgenPipeline_[i] = VK_NULL_HANDLE; pathTraceRgenShader_[i] = VK_NULL_HANDLE;
    }
    if (cachePipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), cachePipeline_, nullptr);
    if (cacheShader_ != VK_NULL_HANDLE) vkDestroyShaderModule(ctx_.device(), cacheShader_, nullptr);
    destroyBuffer(rcKeyBuffer_);
    destroyBuffer(rcAccumBuffer_);
    destroyBuffer(rcRadianceBuffer_);
    destroyBuffer(restirBuffer_);
    if (skyCdfPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), skyCdfPipeline_, nullptr);
    if (skyCdfShader_ != VK_NULL_HANDLE) vkDestroyShaderModule(ctx_.device(), skyCdfShader_, nullptr);
    destroyBuffer(skyCdfBuffer_);
    if (causticShader_ != VK_NULL_HANDLE) vkDestroyShaderModule(ctx_.device(), causticShader_, nullptr);
    for (auto& b : causticPhotonBuffers_) destroyBuffer(b);
    for (auto& b : causticHashBuffers_) destroyBuffer(b);
    destroyBuffer(causticEmitterBuffer_);
    if (pipelineLayoutPathTrace_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), pipelineLayoutPathTrace_, nullptr);
    if (descriptorSetLayoutPathTrace_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), descriptorSetLayoutPathTrace_, nullptr);
    if (pathTraceShader_ != VK_NULL_HANDLE) vkDestroyShaderModule(ctx_.device(), pathTraceShader_, nullptr);
    if (pathTraceResolveShader_ != VK_NULL_HANDLE) vkDestroyShaderModule(ctx_.device(), pathTraceResolveShader_, nullptr);
}

bool RayTracingScene::loadRtFunctions() {
    auto load = [&](const char* n) { return ctx_.loadDeviceProc(n); };
    vkCreateAccelerationStructureKHR_ = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(load("vkCreateAccelerationStructureKHR"));
    vkDestroyAccelerationStructureKHR_ = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(load("vkDestroyAccelerationStructureKHR"));
    vkGetAccelerationStructureBuildSizesKHR_ = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(load("vkGetAccelerationStructureBuildSizesKHR"));
    vkGetAccelerationStructureDeviceAddressKHR_ = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(load("vkGetAccelerationStructureDeviceAddressKHR"));
    vkCmdBuildAccelerationStructuresKHR_ = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(load("vkCmdBuildAccelerationStructuresKHR"));
    vkGetBufferDeviceAddress_ = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(load("vkGetBufferDeviceAddress"));
    vkCmdWriteAccelerationStructuresPropertiesKHR_ = reinterpret_cast<PFN_vkCmdWriteAccelerationStructuresPropertiesKHR>(load("vkCmdWriteAccelerationStructuresPropertiesKHR"));
    vkCmdCopyAccelerationStructureKHR_ = reinterpret_cast<PFN_vkCmdCopyAccelerationStructureKHR>(load("vkCmdCopyAccelerationStructureKHR"));
    return vkCreateAccelerationStructureKHR_ && vkDestroyAccelerationStructureKHR_ &&
           vkGetAccelerationStructureBuildSizesKHR_ && vkGetAccelerationStructureDeviceAddressKHR_ &&
           vkCmdBuildAccelerationStructuresKHR_ && vkGetBufferDeviceAddress_;
}

bool RayTracingScene::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                                    Buffer& out, VkMemoryAllocateFlags allocFlags) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx_.device(), &bi, nullptr, &out.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx_.device(), out.buffer, &req);
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(ctx_.physicalDevice(), &mp);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) == 0) continue;
        if ((mp.memoryTypes[i].propertyFlags & properties) == properties) { type = i; break; }
    }
    if (type == UINT32_MAX) { vkDestroyBuffer(ctx_.device(), out.buffer, nullptr); out.buffer = VK_NULL_HANDLE; return false; }
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = allocFlags;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    ai.pNext = allocFlags ? &flags : nullptr;
    if (vkAllocateMemory(ctx_.device(), &ai, nullptr, &out.memory) != VK_SUCCESS) {
        vkDestroyBuffer(ctx_.device(), out.buffer, nullptr); out.buffer = VK_NULL_HANDLE; return false;
    }
    if (vkBindBufferMemory(ctx_.device(), out.buffer, out.memory, 0) != VK_SUCCESS) return false;
    out.size = size;
    out.usage = usage;
    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo addr{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        addr.buffer = out.buffer;
        out.address = vkGetBufferDeviceAddress_(ctx_.device(), &addr);
        if (!out.address) return false;
    }
    return true;
}

void RayTracingScene::destroyBuffer(Buffer& b) {
    if (!b.owned) { b = {}; return; }  // a view into a scene-owned arena (chunkArenas_)
    if (b.buffer) vkDestroyBuffer(ctx_.device(), b.buffer, nullptr);
    if (b.memory) vkFreeMemory(ctx_.device(), b.memory, nullptr);
    b = {};
}

bool RayTracingScene::ensureBufferCapacity(Buffer& buf, VkDeviceSize requiredSize, VkBufferUsageFlags usage,
                                           VkMemoryPropertyFlags properties, VkMemoryAllocateFlags allocFlags) {
    const bool usageCovered = !buf.owned || (buf.usage & usage) == usage;
    if (buf.buffer != VK_NULL_HANDLE && buf.size >= requiredSize && usageCovered) {
        return true;
    }
    if (buf.buffer != VK_NULL_HANDLE && buf.owned && !usageCovered) {
        usage |= buf.usage;
        requiredSize = std::max(requiredSize, buf.size);
        if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) allocFlags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    }
    if (buf.buffer != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(ctx_.device());
    }
    destroyBuffer(buf);
    return createBuffer(requiredSize, usage, properties, buf, allocFlags);
}

bool RayTracingScene::addMesh(const float* positions, uint32_t vertexCount, const uint32_t* indices, uint32_t indexCount, uint32_t materialSlot) {
    // guards on staticBuilt_, not tlas_
    if (staticBuilt_ || !positions || !indices || vertexCount == 0 || indexCount == 0 || (indexCount % 3) != 0) {
        ctx_.setError("RayTracingScene::addMesh: invalid input or static geometry already built.");
        return false;
    }
    for (uint32_t i = 0; i < indexCount; ++i) {
        if (indices[i] >= vertexCount) {
            ctx_.setError("RayTracingScene::addMesh: index out of range.");
            return false;
        }
    }
    Mesh m;
    m.positions.assign(positions, positions + static_cast<size_t>(vertexCount) * 3);
    m.indices.assign(indices, indices + indexCount);
    m.materialSlot = materialSlot;
    meshes_.push_back(std::move(m));
    return true;
}

bool RayTracingScene::addDynamicMesh(const float* positions, uint32_t vertexCount, const uint32_t* indices, uint32_t indexCount, uint32_t materialSlot) {
    if (!positions || !indices || vertexCount == 0 || indexCount == 0 || (indexCount % 3) != 0) {
        ctx_.setError("RayTracingScene::addDynamicMesh: invalid input.");
        return false;
    }
    for (uint32_t i = 0; i < indexCount; ++i) {
        if (indices[i] >= vertexCount) {
            ctx_.setError("RayTracingScene::addDynamicMesh: index out of range.");
            return false;
        }
    }
    if (dynamicGenerations_.empty()) {
        ctx_.setError("RayTracingScene::addDynamicMesh: call resetDynamicGeometry() first.");
        return false;
    }
    DynamicGeneration& generation = dynamicGenerations_[dynamicGenerationIndex_];
    if (generation.slotsUsed >= generation.slots.size()) {
        generation.slots.emplace_back();
    }
    DynamicMeshSlot& slot = generation.slots[generation.slotsUsed];
    slot.positions.assign(positions, positions + static_cast<size_t>(vertexCount) * 3);
    slot.indices.assign(indices, indices + indexCount);
    slot.materialSlot = materialSlot;
    slot.skinned = false;  // explicit reset
    slot.water = false;  // same reset, same reason
    slot.particles = false;
    slot.alphaTested = false;
    slot.textured = false;  // same reset, same reason
    generation.slotsUsed++;
    return true;
}

bool RayTracingScene::hasSkinSource(uint64_t modelKey, uint32_t boneCount) const {
    auto it = skinSources_.find(modelKey);
    return it != skinSources_.end() && it->second.boneCount == boneCount;
}

bool RayTracingScene::registerSkinSource(uint64_t modelKey, const SkinSourceVertex* vertices, uint32_t vertexCount,
                                          const uint32_t* indices, uint32_t indexCount, uint32_t boneCount,
                                          std::string* errorOut) {
    auto existing = skinSources_.find(modelKey);
    if (existing != skinSources_.end() && existing->second.boneCount == boneCount) return true;
    if (!vertices || !indices || vertexCount == 0 || indexCount == 0 || (indexCount % 3) != 0 || boneCount == 0) {
        if (errorOut) *errorOut = "RayTracingScene::registerSkinSource: invalid input.";
        return false;
    }
    for (uint32_t i = 0; i < indexCount; ++i) {
        if (indices[i] >= vertexCount) { if (errorOut) *errorOut = "RayTracingScene::registerSkinSource: index out of range."; return false; }
    }
    SkinSource src;
    src.vertexCount = vertexCount; src.indexCount = indexCount; src.boneCount = boneCount;
    const VkDeviceSize vbSize = static_cast<VkDeviceSize>(vertexCount) * sizeof(SkinSourceVertex);
    const VkDeviceSize ibSize = static_cast<VkDeviceSize>(indexCount) * sizeof(uint32_t);
    Buffer stagingV, stagingI;
    if (!createBuffer(vbSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingV) ||
        !createBuffer(ibSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingI)) {
        if (errorOut) *errorOut = "RayTracingScene::registerSkinSource: staging buffer allocation failed.";
        destroyBuffer(stagingV); destroyBuffer(stagingI);
        return false;
    }
    void* p = nullptr;
    vkMapMemory(ctx_.device(), stagingV.memory, 0, vbSize, 0, &p); std::memcpy(p, vertices, vbSize); vkUnmapMemory(ctx_.device(), stagingV.memory);
    vkMapMemory(ctx_.device(), stagingI.memory, 0, ibSize, 0, &p); std::memcpy(p, indices, ibSize); vkUnmapMemory(ctx_.device(), stagingI.memory);
    const VkBufferUsageFlags geoUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!createBuffer(vbSize, geoUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, src.vertexBuffer, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) ||
        !createBuffer(ibSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, src.indexBuffer, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
        if (errorOut) *errorOut = "RayTracingScene::registerSkinSource: device-local buffer allocation failed.";
        destroyBuffer(stagingV); destroyBuffer(stagingI); destroyBuffer(src.vertexBuffer); destroyBuffer(src.indexBuffer);
        return false;
    }
    PendingSkinUpload up;
    up.modelKey = modelKey;
    up.stagingVertex = stagingV;
    up.stagingIndex = stagingI;
    up.vertexBytes = vbSize;
    up.indexBytes = ibSize;
    pendingSkinUploads_.push_back(up);
    skinSources_[modelKey] = std::move(src);
    return true;
}

bool RayTracingScene::registerRigidSource(uint64_t modelKey, const SkinSourceVertex* vertices, uint32_t vertexCount,
                                           const uint32_t* indices, uint32_t indexCount, std::string* errorOut) {
    if (rigidSources_.find(modelKey) != rigidSources_.end()) return true;
    if (!vertices || !indices || vertexCount == 0 || indexCount == 0 || (indexCount % 3) != 0) {
        if (errorOut) *errorOut = "RayTracingScene::registerRigidSource: invalid input.";
        return false;
    }
    for (uint32_t i = 0; i < indexCount; ++i) {
        if (indices[i] >= vertexCount) { if (errorOut) *errorOut = "RayTracingScene::registerRigidSource: index out of range."; return false; }
    }
    if (nextRigidMaterialOrdinal_ >= kMaxRigidMaterialSlots) {
        if (errorOut) *errorOut = "RayTracingScene::registerRigidSource: kMaxRigidMaterialSlots exhausted.";
        return false;
    }
    RigidMeshSource src;
    src.vertexCount = vertexCount; src.indexCount = indexCount;
    // the vertex buffer must be in ChunkRtVertex layout
    std::vector<ChunkRtVertex> converted(vertexCount);
    for (uint32_t i = 0; i < vertexCount; ++i) {
        const SkinSourceVertex& sv = vertices[i];
        ChunkRtVertex& v = converted[i];
        v.px = sv.px; v.py = sv.py; v.pz = sv.pz;
        v.nx = sv.nx; v.ny = sv.ny; v.nz = sv.nz;
        v.tx = sv.tx; v.ty = sv.ty; v.tz = sv.tz; v.tw = sv.tw;
        v.u = sv.u; v.v = sv.v;
        v.textureIndex = sv.textureIndex;
        // no VSlot texture comes from textureIndex alone
        v.materialIndex = (sv.materialIndex & 0x80000000u) != 0u ? sv.materialIndex : 0xFFFFFFFFu;
        v.blendMaterialIndex = 0xFFFFFFFFu;
        v.blendAlpha = 0.0f;
        v.blendU = 0.0f; v.blendV = 0.0f;
        v.alphaTestCutoff = 0.0f;
    }
    const VkDeviceSize vbSize = static_cast<VkDeviceSize>(vertexCount) * sizeof(ChunkRtVertex);
    const VkDeviceSize ibSize = static_cast<VkDeviceSize>(indexCount) * sizeof(uint32_t);
    Buffer stagingV, stagingI;
    if (!createBuffer(vbSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingV) ||
        !createBuffer(ibSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingI)) {
        if (errorOut) *errorOut = "RayTracingScene::registerRigidSource: staging buffer allocation failed.";
        destroyBuffer(stagingV); destroyBuffer(stagingI);
        return false;
    }
    void* p = nullptr;
    vkMapMemory(ctx_.device(), stagingV.memory, 0, vbSize, 0, &p); std::memcpy(p, converted.data(), vbSize); vkUnmapMemory(ctx_.device(), stagingV.memory);
    vkMapMemory(ctx_.device(), stagingI.memory, 0, ibSize, 0, &p); std::memcpy(p, indices, ibSize); vkUnmapMemory(ctx_.device(), stagingI.memory);
    const VkBufferUsageFlags geoUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!createBuffer(vbSize, geoUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, src.vertexBuffer, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) ||
        !createBuffer(ibSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, src.indexBuffer, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
        if (errorOut) *errorOut = "RayTracingScene::registerRigidSource: device-local buffer allocation failed.";
        destroyBuffer(stagingV); destroyBuffer(stagingI); destroyBuffer(src.vertexBuffer); destroyBuffer(src.indexBuffer);
        return false;
    }
    src.materialSlot = dynamicMaterialBase_ + (kMaxDynamicMaterialSlots - kMaxRigidMaterialSlots) + nextRigidMaterialOrdinal_;
    ++nextRigidMaterialOrdinal_;
    PendingRigidUpload up;
    up.modelKey = modelKey;
    up.stagingVertex = stagingV;
    up.stagingIndex = stagingI;
    up.vertexBytes = vbSize;
    up.indexBytes = ibSize;
    pendingRigidUploads_.push_back(up);
    rigidSources_[modelKey] = std::move(src);
    return true;
}

bool RayTracingScene::addRigidInstance(uint64_t modelKey, const float worldMatrix[12]) {
    auto it = rigidSources_.find(modelKey);
    if (it == rigidSources_.end()) {
        ctx_.setError("RayTracingScene::addRigidInstance: modelKey not registered -- call registerRigidSource first.");
        return false;
    }
    RigidInstanceDraw draw;
    draw.modelKey = modelKey;
    std::memcpy(draw.worldMatrix, worldMatrix, sizeof(float) * 12);
    pendingRigidInstances_.push_back(draw);
    return true;
}

void RayTracingScene::recordPendingUploads(VkCommandBuffer cmd, DynamicGeneration& generation) {
    if (pendingSkinUploads_.empty() && pendingTextureUploads_.empty() && pendingRigidUploads_.empty()) return;

    // skin sources first plain buffer-to-buffer copies, no layout to manage
    for (auto& up : pendingSkinUploads_) {
        auto it = skinSources_.find(up.modelKey);
        if (it == skinSources_.end()) continue;
        VkBufferCopy vc{0, 0, up.vertexBytes};
        vkCmdCopyBuffer(cmd, up.stagingVertex.buffer, it->second.vertexBuffer.buffer, 1, &vc);
        VkBufferCopy ic{0, 0, up.indexBytes};
        vkCmdCopyBuffer(cmd, up.stagingIndex.buffer, it->second.indexBuffer.buffer, 1, &ic);
        generation.retiredStagingBuffers_.push_back(up.stagingVertex);
        generation.retiredStagingBuffers_.push_back(up.stagingIndex);
    }

    std::vector<DynamicMaterialUpdate> rigidMaterialUpdates;
    for (auto& up : pendingRigidUploads_) {
        auto it = rigidSources_.find(up.modelKey);
        if (it == rigidSources_.end()) continue;
        RigidMeshSource& src = it->second;
        VkBufferCopy vc{0, 0, up.vertexBytes};
        vkCmdCopyBuffer(cmd, up.stagingVertex.buffer, src.vertexBuffer.buffer, 1, &vc);
        VkBufferCopy ic{0, 0, up.indexBytes};
        vkCmdCopyBuffer(cmd, up.stagingIndex.buffer, src.indexBuffer.buffer, 1, &ic);
        generation.retiredStagingBuffers_.push_back(up.stagingVertex);
        generation.retiredStagingBuffers_.push_back(up.stagingIndex);
        {
            VkMemoryBarrier copyToBuild{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            copyToBuild.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            copyToBuild.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                 0, 1, &copyToBuild, 0, nullptr, 0, nullptr);
        }

        VkAccelerationStructureGeometryTrianglesDataKHR tri{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
        tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT; tri.vertexData.deviceAddress = src.vertexBuffer.address; tri.vertexStride = sizeof(ChunkRtVertex);  // matches registerRigidSource()'s converted layout
        tri.maxVertex = src.vertexCount - 1; tri.indexType = VK_INDEX_TYPE_UINT32; tri.indexData.deviceAddress = src.indexBuffer.address;
        VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR; geom.geometry.triangles = tri; geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        info.geometryCount = 1; info.pGeometries = &geom;
        uint32_t primCount = src.indexCount / 3;
        VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        vkGetAccelerationStructureBuildSizesKHR_(ctx_.device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &primCount, &sizes);
        if (!createBuffer(sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, src.blasBuffer, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
            lastRigidDiag_ = "rigid BLAS buffer alloc FAILED modelKey=" + std::to_string(up.modelKey);
            continue;
        }
        VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        aci.buffer = src.blasBuffer.buffer; aci.size = sizes.accelerationStructureSize; aci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        if (vkCreateAccelerationStructureKHR_(ctx_.device(), &aci, nullptr, &src.blas) != VK_SUCCESS) {
            lastRigidDiag_ = "vkCreateAccelerationStructureKHR FAILED modelKey=" + std::to_string(up.modelKey);
            continue;
        }
        Buffer scratch;
        if (!createBuffer(sizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratch, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
            lastRigidDiag_ = "rigid BLAS scratch alloc FAILED modelKey=" + std::to_string(up.modelKey);
            continue;
        }
        info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        info.srcAccelerationStructure = VK_NULL_HANDLE;
        info.dstAccelerationStructure = src.blas;
        info.scratchData.deviceAddress = scratch.address;
        VkAccelerationStructureBuildRangeInfoKHR range{}; range.primitiveCount = primCount;
        const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;
        vkCmdBuildAccelerationStructuresKHR_(cmd, 1, &info, &ranges);
        src.blasBuilt = true;
        lastRigidDiag_ = "rigid BLAS built OK modelKey=" + std::to_string(up.modelKey) +
            " materialSlot=" + std::to_string(src.materialSlot) +
            " vertexAddr=" + std::to_string(src.vertexBuffer.address) +
            " indexAddr=" + std::to_string(src.indexBuffer.address);
        // one-time materials[] entry on materialSlot's permanent assignment
        rigidMaterialUpdates.push_back({src.materialSlot - dynamicMaterialBase_, static_cast<uint64_t>(src.vertexBuffer.address), static_cast<uint64_t>(src.indexBuffer.address)});
        generation.retiredStagingBuffers_.push_back(scratch);
    }
    if (!rigidMaterialUpdates.empty()) {
        std::string rigidErr;
        updateDynamicMaterialEntries(rigidMaterialUpdates, &rigidErr);
    }
    pendingRigidUploads_.clear();

    // textures
    if (!pendingTextureUploads_.empty()) {
        std::vector<VkImageMemoryBarrier> toTransferDst;
        toTransferDst.reserve(pendingTextureUploads_.size());
        for (auto& up : pendingTextureUploads_) {
            if (up.textureIndex >= pathTraceTextures_.size()) continue;
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.image = pathTraceTextures_[up.textureIndex].image;
            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            b.subresourceRange.levelCount = pathTraceTextures_[up.textureIndex].preMipped ? pathTraceTextures_[up.textureIndex].mipLevels : 1;
            b.subresourceRange.layerCount = 1;
            toTransferDst.push_back(b);
        }
        if (!toTransferDst.empty()) {
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                  0, nullptr, 0, nullptr,
                                  static_cast<uint32_t>(toTransferDst.size()), toTransferDst.data());
        }
        for (auto& up : pendingTextureUploads_) {
            if (up.textureIndex >= pathTraceTextures_.size()) continue;
            const PathTraceTexture& ct = pathTraceTextures_[up.textureIndex];
            const uint32_t copyLevels = ct.preMipped ? ct.mipLevels : 1u;
            std::vector<VkBufferImageCopy> regions(copyLevels);
            VkDeviceSize offset = 0;
            for (uint32_t level = 0; level < copyLevels; ++level) {
                const uint32_t lw = std::max(1u, up.width >> level), lh = std::max(1u, up.height >> level);
                VkBufferImageCopy& c = regions[level];
                c = VkBufferImageCopy{};
                c.bufferOffset = offset;
                c.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                c.imageSubresource.mipLevel = level;
                c.imageSubresource.layerCount = 1;
                c.imageExtent = {lw, lh, 1};
                offset += VkDeviceSize(lw) * lh * 4u;
            }
            vkCmdCopyBufferToImage(cmd, up.staging.buffer, ct.image,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copyLevels, regions.data());
            generation.retiredStagingBuffers_.push_back(up.staging);
        }

        // mip-chain generation
        for (auto& up : pendingTextureUploads_) {
            if (up.textureIndex >= pathTraceTextures_.size()) continue;
            PathTraceTexture& t = pathTraceTextures_[up.textureIndex];
            if (t.preMipped) continue;  // every level already copied above
            int32_t mipW = static_cast<int32_t>(up.width), mipH = static_cast<int32_t>(up.height);
            for (uint32_t level = 1; level < t.mipLevels; ++level) {
                const int32_t nextW = std::max(1, mipW / 2), nextH = std::max(1, mipH / 2);
                VkImageMemoryBarrier toSrc{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                toSrc.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                toSrc.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                toSrc.image = t.image;
                toSrc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 1, 0, 1};
                VkImageMemoryBarrier dstUndefToDst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                dstUndefToDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                dstUndefToDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; dstUndefToDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                dstUndefToDst.image = t.image;
                dstUndefToDst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1};
                VkImageMemoryBarrier pre[2] = {toSrc, dstUndefToDst};
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                      0, nullptr, 0, nullptr, 2, pre);

                VkImageBlit blit{};
                blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1};
                blit.srcOffsets[0] = {0, 0, 0}; blit.srcOffsets[1] = {mipW, mipH, 1};
                blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
                blit.dstOffsets[0] = {0, 0, 0}; blit.dstOffsets[1] = {nextW, nextH, 1};
                vkCmdBlitImage(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

                VkImageMemoryBarrier srcToShaderRead{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                srcToShaderRead.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; srcToShaderRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                srcToShaderRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; srcToShaderRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                srcToShaderRead.image = t.image;
                srcToShaderRead.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 1, 0, 1};
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                      0, nullptr, 0, nullptr, 1, &srcToShaderRead);

                mipW = nextW; mipH = nextH;
            }
        }
    }

    VkMemoryBarrier bufBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    bufBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bufBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    std::vector<VkImageMemoryBarrier> toShaderRead;
    toShaderRead.reserve(pendingTextureUploads_.size());
    for (auto& up : pendingTextureUploads_) {
        if (up.textureIndex >= pathTraceTextures_.size()) continue;
        const PathTraceTexture& t = pathTraceTextures_[up.textureIndex];
        // only the level still in TRANSFER_DST_OPTIMAL needs this barrier
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.image = t.image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.baseMipLevel = t.preMipped ? 0 : t.mipLevels - 1;
        b.subresourceRange.levelCount = t.preMipped ? t.mipLevels : 1;
        b.subresourceRange.layerCount = 1;
        toShaderRead.push_back(b);
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                          1, &bufBarrier, 0, nullptr,
                          static_cast<uint32_t>(toShaderRead.size()), toShaderRead.data());

    pendingSkinUploads_.clear();
    pendingTextureUploads_.clear();
}

bool RayTracingScene::addSkinnedDynamicMesh(uint64_t modelKey, const float* boneDualQuats, uint32_t boneCount,
                                             const float worldMatrix[12], uint32_t materialSlot, uint32_t instanceMask) {
    auto srcIt = skinSources_.find(modelKey);
    if (srcIt == skinSources_.end()) {
        ctx_.setError("RayTracingScene::addSkinnedDynamicMesh: modelKey not registered -- call registerSkinSource first.");
        return false;
    }
    if (!boneDualQuats || boneCount != srcIt->second.boneCount) {
        // see's live "player model doesn't render, log mentions bones" report
        ctx_.setError("RayTracingScene::addSkinnedDynamicMesh: boneCount does not match this model's registered skin source (modelKey=" +
            std::to_string(modelKey) + ", registered boneCount=" + std::to_string(srcIt->second.boneCount) +
            ", this call's boneCount=" + std::to_string(boneCount) + ", boneDualQuats=" + (boneDualQuats ? "non-null" : "NULL") + ").");
        return false;
    }
    if (dynamicGenerations_.empty()) {
        ctx_.setError("RayTracingScene::addSkinnedDynamicMesh: call resetDynamicGeometry() first.");
        return false;
    }
    DynamicGeneration& generation = dynamicGenerations_[dynamicGenerationIndex_];
    if (generation.slotsUsed >= generation.slots.size()) generation.slots.emplace_back();  // same growth rule as addDynamicMesh's identical block
    DynamicMeshSlot& slot = generation.slots[generation.slotsUsed];
    slot.skinned = true;
    slot.instanceMask = instanceMask & 0xFFu;
    slot.water = false;
    slot.particles = false;
    slot.alphaTested = false;
    slot.textured = false;  // same reset, same reason
    slot.skinSourceKey = modelKey;
    slot.boneMatrixData.assign(boneDualQuats, boneDualQuats + static_cast<size_t>(boneCount) * 8);
    std::memcpy(slot.worldMatrix, worldMatrix, sizeof(float) * 12);
    slot.materialSlot = materialSlot;
    generation.slotsUsed++;
    return true;
}

bool RayTracingScene::addWaterMesh(const float* positions, const float* normals, const float* uvs,
                                    const uint32_t* materialIndices, uint32_t vertexCount,
                                    const uint32_t* indices, uint32_t indexCount) {
    if (!positions || !normals || !uvs || !materialIndices || !indices || vertexCount == 0 || indexCount == 0 || (indexCount % 3) != 0) {
        ctx_.setError("RayTracingScene::addWaterMesh: invalid input.");
        return false;
    }
    for (uint32_t i = 0; i < indexCount; ++i) {
        if (indices[i] >= vertexCount) {
            ctx_.setError("RayTracingScene::addWaterMesh: index out of range.");
            return false;
        }
    }
    if (dynamicGenerations_.empty()) {
        ctx_.setError("RayTracingScene::addWaterMesh: call resetDynamicGeometry() first.");
        return false;
    }
    DynamicGeneration& generation = dynamicGenerations_[dynamicGenerationIndex_];
    if (generation.slotsUsed >= generation.slots.size()) generation.slots.emplace_back();  // same growth rule as addDynamicMesh's identical block
    DynamicMeshSlot& slot = generation.slots[generation.slotsUsed];
    slot.water = true;
    slot.particles = false;
    slot.alphaTested = false;
    slot.skinned = false;
    slot.textured = false;  // same reset, same reason
    slot.waterVertices.resize(vertexCount);
    for (uint32_t i = 0; i < vertexCount; ++i) {
        ChunkRtVertex& v = slot.waterVertices[i];
        v.px = positions[i*3+0]; v.py = positions[i*3+1]; v.pz = positions[i*3+2];
        v.nx = normals[i*3+0]; v.ny = normals[i*3+1]; v.nz = normals[i*3+2];
        v.tx = 1.0f; v.ty = 0.0f; v.tz = 0.0f; v.tw = 1.0f;
        v.u = uvs[i*2+0]; v.v = uvs[i*2+1];
        v.textureIndex = 0;
        v.materialIndex = materialIndices[i];
        v.blendMaterialIndex = 0xFFFFFFFFu;
        v.blendAlpha = 0.0f;
        v.blendU = 0.0f; v.blendV = 0.0f;
        v.alphaTestCutoff = 0.0f;
    }
    slot.indices.assign(indices, indices + indexCount);
    generation.slotsUsed++;
    return true;
}

bool RayTracingScene::addParticleSpriteMesh(const SkinSourceVertex* vertices, uint32_t vertexCount,
                                            const uint32_t* indices, uint32_t indexCount) {
    // the textured path keeps materialIndex 0xFFFFFFFE (bit31 set) as given
    if (!addTexturedDynamicMesh(vertices, vertexCount, indices, indexCount)) return false;
    DynamicMeshSlot& slot = dynamicGenerations_[dynamicGenerationIndex_].slots[dynamicGenerations_[dynamicGenerationIndex_].slotsUsed - 1];
    slot.particles = true;
    slot.instanceMask = 0x80u;
    return true;
}

bool RayTracingScene::addGrassMesh(const SkinSourceVertex* vertices, const float* alphaCutoffs, uint32_t vertexCount,
                                   const uint32_t* indices, uint32_t indexCount) {
    if (!alphaCutoffs || !addTexturedDynamicMesh(vertices, vertexCount, indices, indexCount)) return false;
    DynamicMeshSlot& slot = dynamicGenerations_[dynamicGenerationIndex_].slots[dynamicGenerations_[dynamicGenerationIndex_].slotsUsed - 1];
    slot.alphaTested = true;  // non-opaque geometry, full TLAS mask (every ray kind sees it)
    for (uint32_t i = 0; i < vertexCount; ++i) slot.texturedVertices[i].alphaTestCutoff = alphaCutoffs[i];
    return true;
}

bool RayTracingScene::addTexturedDynamicMesh(const SkinSourceVertex* vertices, uint32_t vertexCount,
                                              const uint32_t* indices, uint32_t indexCount) {
    if (!vertices || !indices || vertexCount == 0 || indexCount == 0 || (indexCount % 3) != 0) {
        ctx_.setError("RayTracingScene::addTexturedDynamicMesh: invalid input.");
        return false;
    }
    for (uint32_t i = 0; i < indexCount; ++i) {
        if (indices[i] >= vertexCount) {
            ctx_.setError("RayTracingScene::addTexturedDynamicMesh: index out of range.");
            return false;
        }
    }
    if (dynamicGenerations_.empty()) {
        ctx_.setError("RayTracingScene::addTexturedDynamicMesh: call resetDynamicGeometry() first.");
        return false;
    }
    DynamicGeneration& generation = dynamicGenerations_[dynamicGenerationIndex_];
    if (generation.slotsUsed >= generation.slots.size()) generation.slots.emplace_back();  // same growth rule as addDynamicMesh's identical block
    DynamicMeshSlot& slot = generation.slots[generation.slotsUsed];
    slot.textured = true;
    slot.skinned = false;
    slot.water = false;
    slot.particles = false;
    slot.alphaTested = false;
    slot.texturedVertices.resize(vertexCount);
    for (uint32_t i = 0; i < vertexCount; ++i) {
        const SkinSourceVertex& sv = vertices[i];
        ChunkRtVertex& v = slot.texturedVertices[i];
        v.px = sv.px; v.py = sv.py; v.pz = sv.pz;
        v.nx = sv.nx; v.ny = sv.ny; v.nz = sv.nz;
        v.tx = sv.tx; v.ty = sv.ty; v.tz = sv.tz; v.tw = sv.tw;
        v.u = sv.u; v.v = sv.v;
        v.textureIndex = sv.textureIndex;
        v.materialIndex = (sv.materialIndex & 0x80000000u) != 0u ? sv.materialIndex : 0xFFFFFFFFu;
        v.blendMaterialIndex = 0xFFFFFFFFu;
        v.blendAlpha = 0.0f;
        v.blendU = 0.0f; v.blendV = 0.0f;
        v.alphaTestCutoff = 0.0f;
    }
    slot.indices.assign(indices, indices + indexCount);
    generation.slotsUsed++;
    return true;
}

void RayTracingScene::resetGeometry() {
    if (ctx_.device() == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(ctx_.device());
    for (auto& m : meshes_) {
        if (m.blas != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_)
            vkDestroyAccelerationStructureKHR_(ctx_.device(), m.blas, nullptr);
        destroyBuffer(m.blasBuffer);
        destroyBuffer(m.vertex);
        destroyBuffer(m.index);
    }
    meshes_.clear();
    staticBuilt_ = false;
    for (auto& kv : chunkMeshes_) {
        destroyChunkBlas(kv.second);
    }
    chunkMeshes_.clear();
    for (Buffer& a : chunkArenas_) destroyBuffer(a);
    chunkArenas_.clear();
    for (auto& tex : pathTraceTextures_) {
        if (tex.sampler) vkDestroySampler(ctx_.device(), tex.sampler, nullptr);
        if (tex.view) vkDestroyImageView(ctx_.device(), tex.view, nullptr);
        if (tex.image) vkDestroyImage(ctx_.device(), tex.image, nullptr);
        if (tex.memory) vkFreeMemory(ctx_.device(), tex.memory, nullptr);
    }
    pathTraceTextures_.clear();
    pathTraceTextureIndices_.clear();
    for (auto& kv : skinSources_) { destroyBuffer(kv.second.vertexBuffer); destroyBuffer(kv.second.indexBuffer); }
    skinSources_.clear();
    for (auto& up : pendingSkinUploads_) { destroyBuffer(up.stagingVertex); destroyBuffer(up.stagingIndex); }
    pendingSkinUploads_.clear();
    for (auto& up : pendingTextureUploads_) { destroyBuffer(up.staging); }
    pendingTextureUploads_.clear();
    for (auto& kv : rigidSources_) {
        if (kv.second.blas != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_)
            vkDestroyAccelerationStructureKHR_(ctx_.device(), kv.second.blas, nullptr);
        destroyBuffer(kv.second.blasBuffer);
        destroyBuffer(kv.second.vertexBuffer);
        destroyBuffer(kv.second.indexBuffer);
    }
    rigidSources_.clear();
    nextRigidMaterialOrdinal_ = 0;
    for (auto& up : pendingRigidUploads_) { destroyBuffer(up.stagingVertex); destroyBuffer(up.stagingIndex); }
    pendingRigidUploads_.clear();
    pendingRigidInstances_.clear();
    ++geometryEpoch_;
    for (auto& generation : dynamicGenerations_) {
        for (auto& b : generation.retiredStagingBuffers_) destroyBuffer(b);
        generation.retiredStagingBuffers_.clear();
        for (auto& slot : generation.slots) {
            if (slot.blas != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_)
                vkDestroyAccelerationStructureKHR_(ctx_.device(), slot.blas, nullptr);
            destroyBuffer(slot.blasBuffer);
            destroyBuffer(slot.vertex);
            destroyBuffer(slot.index);
            destroyBuffer(slot.stagingVertex);
            destroyBuffer(slot.stagingIndex);
            destroyBuffer(slot.scratch);
            destroyBuffer(slot.boneBuffer);
            destroyBuffer(slot.boneStaging);
            destroyBuffer(slot.skinParams);
            destroyBuffer(slot.skinParamsStaging);
        }
        if (generation.tlas != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_)
            vkDestroyAccelerationStructureKHR_(ctx_.device(), generation.tlas, nullptr);
        destroyBuffer(generation.tlasBuffer);
        destroyBuffer(generation.tlasInstanceBuffer);
        destroyBuffer(generation.tlasScratch);
        destroyBuffer(generation.materials);
        if (generation.buildFence != VK_NULL_HANDLE) vkDestroyFence(ctx_.device(), generation.buildFence, nullptr);
        if (generation.readyForAo != VK_NULL_HANDLE) vkDestroySemaphore(ctx_.device(), generation.readyForAo, nullptr);
        if (generation.readyForReflections != VK_NULL_HANDLE) vkDestroySemaphore(ctx_.device(), generation.readyForReflections, nullptr);
        if (generation.readyForRtdi != VK_NULL_HANDLE) vkDestroySemaphore(ctx_.device(), generation.readyForRtdi, nullptr);
        if (generation.readyForPathTrace != VK_NULL_HANDLE) vkDestroySemaphore(ctx_.device(), generation.readyForPathTrace, nullptr);
        if (generation.cmd != VK_NULL_HANDLE) vkFreeCommandBuffers(ctx_.device(), ctx_.setupCommandPool(), 1, &generation.cmd);
    }
    dynamicGenerations_.clear();
    dynamicGenerationIndex_ = 0;
}

bool RayTracingScene::ensureDynamicGenerations() {
    if (!dynamicGenerations_.empty()) return true;
    dynamicGenerations_.resize(frameSlots_);
    for (auto& generation : dynamicGenerations_) {
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (vkCreateFence(ctx_.device(), &fi, nullptr, &generation.buildFence) != VK_SUCCESS) {
            ctx_.setError("RayTracingScene: vkCreateFence (dynamic generation) failed.");
            return false;
        }
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (vkCreateSemaphore(ctx_.device(), &si, nullptr, &generation.readyForAo) != VK_SUCCESS ||
            vkCreateSemaphore(ctx_.device(), &si, nullptr, &generation.readyForReflections) != VK_SUCCESS ||
            vkCreateSemaphore(ctx_.device(), &si, nullptr, &generation.readyForRtdi) != VK_SUCCESS ||
            vkCreateSemaphore(ctx_.device(), &si, nullptr, &generation.readyForPathTrace) != VK_SUCCESS) {
            ctx_.setError("RayTracingScene: vkCreateSemaphore (dynamic generation) failed.");
            return false;
        }
    }
    return true;
}

void RayTracingScene::resetDynamicGeometry() {
    if (!ensureDynamicGenerations()) return;
    dynamicGenerationIndex_ = (dynamicGenerationIndex_ + 1) % frameSlots_;
    DynamicGeneration& generation = dynamicGenerations_[dynamicGenerationIndex_];
    VkResult status = vkGetFenceStatus(ctx_.device(), generation.buildFence);
    if (status == VK_NOT_READY) {
        std::cerr << "[RayTracingScene] dynamic generation pool exhausted (frameSlots="
                  << frameSlots_ << ") -- CPU waiting on GPU. If this happens routinely "
                     "(not just occasionally), the dynamic geometry rebuild is taking "
                     "longer than a frame; investigate before increasing pool size." << std::endl;
        vkWaitForFences(ctx_.device(), 1, &generation.buildFence, VK_TRUE, UINT64_MAX);
    }
    auto recreateIfStale = [&](bool& pending, VkSemaphore& sem, const char* name) {
        if (!pending) return;
        vkDestroySemaphore(ctx_.device(), sem, nullptr);
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (vkCreateSemaphore(ctx_.device(), &si, nullptr, &sem) != VK_SUCCESS) {
            std::cerr << "[RayTracingScene] resetDynamicGeometry: failed to recreate stale "
                      << name << " semaphore -- this generation may remain stuck." << std::endl;
        }
        pending = false;
    };
    recreateIfStale(generation.aoReadyPending, generation.readyForAo, "readyForAo");
    recreateIfStale(generation.reflectionsReadyPending, generation.readyForReflections, "readyForReflections");
    recreateIfStale(generation.rtdiReadyPending, generation.readyForRtdi, "readyForRtdi");
    recreateIfStale(generation.pathTraceReadyPending, generation.readyForPathTrace, "readyForPathTrace");
    for (auto& b : generation.retiredStagingBuffers_) destroyBuffer(b);
    generation.retiredStagingBuffers_.clear();
    generation.slotsUsed = 0;
    pendingRigidInstances_.clear();
}

bool RayTracingScene::build(bool reflectionsActive, bool rtdiActive, bool pathTraceActive) {
    return buildWithRtao(true, reflectionsActive, rtdiActive, pathTraceActive);
}

bool RayTracingScene::buildWithRtao(bool rtaoActive, bool reflectionsActive, bool rtdiActive, bool pathTraceActive) {
    if (!ctx_.capabilities().hasRayQuery) { ctx_.setError("Hardware ray tracing/ray query is unavailable on the selected Vulkan device."); return false; }
    flushChunkBuilds();
    if (!staticBuilt_) {
        if (!loadRtFunctions()) { ctx_.setError("Required Vulkan acceleration-structure entry points are unavailable."); return false; }
        for (auto& m : meshes_) if (!buildBlas(m)) return false;
        staticBuilt_ = true;
    }
    if (!ensureDynamicGenerations()) return false;
    ensureTimestampPool();
    DynamicGeneration& generation = dynamicGenerations_[dynamicGenerationIndex_];
    // this guard was never updated to know about the chunk tier
    if (meshes_.empty() && generation.slotsUsed == 0 && chunkMeshes_.empty()) { ctx_.setError("RayTracingScene::build: no meshes were added."); return false; }

    // one command buffer for the whole frame's dynamic work
    if (generation.cmd == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = ctx_.setupCommandPool(); cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(ctx_.device(), &cai, &generation.cmd) != VK_SUCCESS) {
            ctx_.setError("RayTracingScene: vkAllocateCommandBuffers (dynamic generation) failed."); return false;
        }
    } else {
        // reused from a previous use of this same generation
        vkResetCommandBuffer(generation.cmd, 0);
    }
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(generation.cmd, &cbi) != VK_SUCCESS) {
        ctx_.setError("RayTracingScene: vkBeginCommandBuffer (dynamic generation) failed."); return false;
    }

    recordPendingUploads(generation.cmd, generation);

    if (waterPatchGens_.size() != frameSlots_) waterPatchGens_.resize(frameSlots_);
    bool bakeWaveMap = false;
    {
        WaterPatchGen& wgen = waterPatchGens_[dynamicGenerationIndex_];
        wgen.active = false;
        wgen.fftValid = false;
        if ((waterPatchSettings_.waves || waterPatchSettings_.enabled || lavaTopTris_ > 0) && waterRectCount_ > 0 && !pathTraceMaterialMirror_.empty() && ensureWaveResources()) {
            if (waveMapNeedsImage_ && !recreateWaveMapImage()) {
                static bool loggedMap = false;
                if (!loggedMap) { loggedMap = true; std::cerr << "[RayTracingScene] water wave map image failed" << std::endl; }
            }
            if (waterPatchSettings_.waves && waterTopTris_ > 0) {  // the FFT sea, only with water (lava-only maps skip it)
                wgen.fftValid = recordWaveFft(generation.cmd, wgen);
                bakeWaveMap = true;
            }
            if (lavaTopTris_ > 0) bakeWaveMap = true;  // the bake also measures each lava spot's distance to its bank
            if (waterPatchSettings_.enabled && !recordWaterPatch(generation.cmd, wgen)) {
                static bool logged = false;
                if (!logged) { logged = true; std::cerr << "[RayTracingScene] water patch disabled: " << ctx_.lastError() << std::endl; }
                wgen.active = false;
            }
        }
    }

    auto dynamicStart = std::chrono::steady_clock::now();
    std::vector<DynamicMaterialUpdate> dynamicMaterialUpdates;
    for (size_t i = 0; i < generation.slotsUsed; ++i) {
        DynamicMeshSlot& slot = generation.slots[i];
        if (!recordDynamicSlotBuild(generation.cmd, slot, static_cast<uint32_t>(i))) return false;
        if (slot.skinned) {
            auto srcIt = skinSources_.find(slot.skinSourceKey);
            if (srcIt != skinSources_.end()) {
                // this slot's previous-frame vertex buffer
                uint64_t prevVertexAddr = 0;
                if (frameSlots_ > 0) {
                    const uint32_t prevGenIdx = static_cast<uint32_t>((dynamicGenerationIndex_ + frameSlots_ - 1) % frameSlots_);
                    if (prevGenIdx < dynamicGenerations_.size()) {
                        DynamicGeneration& prevGeneration = dynamicGenerations_[prevGenIdx];
                        if (i < prevGeneration.slotsUsed) {
                            DynamicMeshSlot& prevSlot = prevGeneration.slots[i];
                            if (prevSlot.skinned && prevSlot.skinSourceKey == slot.skinSourceKey && prevSlot.vertex.address != 0) {
                                prevVertexAddr = static_cast<uint64_t>(prevSlot.vertex.address);
                            }
                        }
                    }
                }
                dynamicMaterialUpdates.push_back({static_cast<uint32_t>(i),
                    static_cast<uint64_t>(slot.vertex.address), static_cast<uint64_t>(srcIt->second.indexBuffer.address),
                    prevVertexAddr});
            }
        } else if (slot.water) {
            dynamicMaterialUpdates.push_back({static_cast<uint32_t>(i),
                static_cast<uint64_t>(slot.vertex.address), static_cast<uint64_t>(slot.index.address)});
        } else if (slot.textured) {
            dynamicMaterialUpdates.push_back({static_cast<uint32_t>(i),
                static_cast<uint64_t>(slot.vertex.address), static_cast<uint64_t>(slot.index.address)});
        }
    }
    if (!dynamicMaterialUpdates.empty()) {
        std::string err;
        if (!updateDynamicMaterialEntries(dynamicMaterialUpdates, &err)) {
            ctx_.setError("RayTracingScene::updateDynamicMaterialEntries failed: " + err);
            return false;
        }
    }
    if (!pathTraceMaterialMirror_.empty()) {
        const VkDeviceSize matBytes = static_cast<VkDeviceSize>(pathTraceMaterialMirror_.size());
        if (!ensureBufferCapacity(generation.materials, matBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            ctx_.setError("RayTracingScene: per-generation material table allocation failed.");
            return false;
        }
        void* matMapped = nullptr;
        if (vkMapMemory(ctx_.device(), generation.materials.memory, 0, matBytes, 0, &matMapped) == VK_SUCCESS) {
            std::memcpy(matMapped, pathTraceMaterialMirror_.data(), static_cast<size_t>(matBytes));
            // the water patch's entry, per generation, since its vertex buffer is (the mirror holds
            // zeros at this slot)
            const WaterPatchGen& wgen = waterPatchGens_[dynamicGenerationIndex_];
            if (wgen.active && (size_t(waterPatchMaterialSlot_) + 1) * sizeof(GpuPathTraceMaterialEntry) <= size_t(matBytes)) {
                GpuPathTraceMaterialEntry e{};
                e.vertexAddr = static_cast<uint64_t>(wgen.vertex.address);
                e.indexAddr = static_cast<uint64_t>(waterPatchIndexBuffer_.address);
                e.albedo[0] = e.albedo[1] = e.albedo[2] = 0.6f;
                e.opaqueIndexCount = 0.0f;  // the whole index buffer is the "glass" range, like the flat water chunk
                std::memcpy(static_cast<uint8_t*>(matMapped) + size_t(waterPatchMaterialSlot_) * sizeof(GpuPathTraceMaterialEntry), &e, sizeof(e));
            }
            vkUnmapMemory(ctx_.device(), generation.materials.memory);
        }
    }
    lastDynamicRebuildMicros_ = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - dynamicStart).count());
    if (generation.slotsUsed > 0) {
        static auto lastDiagLog = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastDiagLog).count() > 1000) {
            lastDiagLog = now;
            std::cerr << "[RayTracingScene] buildWithRtao: generation.slotsUsed=" << generation.slotsUsed
                      << " (about to recordTlasBuild)" << std::endl;
        }
    }
    // BLAS builds -> TLAS build
    {
        VkMemoryBarrier blasToTlas{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        blasToTlas.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        blasToTlas.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(generation.cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &blasToTlas, 0, nullptr, 0, nullptr);
    }
    auto tlasStart = std::chrono::steady_clock::now();
    bool tlasOk = recordTlasBuild(generation.cmd, generation);
    lastTlasRebuildMicros_ = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tlasStart).count());
    if (!tlasOk) return false;
    if (bakeWaveMap) recordWaveMapBake(generation.cmd, generation.tlas);  // no-op unless the water or a setting changed
    if (vkEndCommandBuffer(generation.cmd) != VK_SUCCESS) {
        ctx_.setError("RayTracingScene: vkEndCommandBuffer (dynamic generation) failed."); return false;
    }

    vkResetFences(ctx_.device(), 1, &generation.buildFence);
    VkSemaphore signalSems[4];
    uint32_t signalCount = 0;
    if (rtaoActive) {
        signalSems[signalCount++] = generation.readyForAo;
        generation.aoReadyPending = true;
    }
    if (reflectionsActive) {
        signalSems[signalCount++] = generation.readyForReflections;
        generation.reflectionsReadyPending = true;
    }
    if (rtdiActive) {
        signalSems[signalCount++] = generation.readyForRtdi;
        generation.rtdiReadyPending = true;
    }
    if (pathTraceActive) {
        signalSems[signalCount++] = generation.readyForPathTrace;
        generation.pathTraceReadyPending = true;
    }
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &generation.cmd;
    si.signalSemaphoreCount = signalCount; si.pSignalSemaphores = signalSems;
    VkResult sr = vkQueueSubmit(ctx_.queue(), 1, &si, generation.buildFence);
    if (sr != VK_SUCCESS) {
        ctx_.setError("RayTracingScene: vkQueueSubmit (dynamic generation) failed with VkResult " + std::to_string(sr));
        return false;
    }
    const bool firstCompile = pathTracePipeline_ == VK_NULL_HANDLE;
    const auto compileStart = std::chrono::steady_clock::now();
    const bool pipelinesOk = ensureRtaoPipeline() && ensureReflectionPipeline() && ensureRtdiPipeline() && ensurePathTracePipeline() &&
        ensureVolumetricFogPipeline();
    if (firstCompile && pathTracePipeline_ != VK_NULL_HANDLE)
        std::cerr << "[RayTracingScene] pipelines created in " << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - compileStart).count() << " ms" << std::endl;
    return pipelinesOk;
}
bool RayTracingScene::ensureTimestampPool() {
    if (timestampPool_ != VK_NULL_HANDLE) return true;
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx_.physicalDevice(), &props);
    timestampPeriodNs_ = props.limits.timestampPeriod;
    if (timestampPeriodNs_ <= 0.0f) {
        std::cerr << "[RayTracingScene] device reports timestampPeriod <= 0 -- GPU timing unavailable, CPU timing only." << std::endl;
        return false;
    }
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 4;
    if (vkCreateQueryPool(ctx_.device(), &qi, nullptr, &timestampPool_) != VK_SUCCESS) {
        std::cerr << "[RayTracingScene] vkCreateQueryPool failed -- GPU timing unavailable, CPU timing only." << std::endl;
        return false;
    }
    return true;
}

bool RayTracingScene::buildBlas(Mesh& m) {
    const VkDeviceSize vbSize = m.positions.size() * sizeof(float);
    const VkDeviceSize ibSize = m.indices.size() * sizeof(uint32_t);
    const VkBufferUsageFlags geoUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!createBuffer(vbSize, geoUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m.vertex, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
        ctx_.setError("RayTracingScene: vertex buffer allocation failed."); return false;
    }
    if (!createBuffer(ibSize, geoUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m.index, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
        ctx_.setError("RayTracingScene: index buffer allocation failed."); return false;
    }
    Buffer stagingV, stagingI;
    if (!createBuffer(vbSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingV) ||
        !createBuffer(ibSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingI)) {
        ctx_.setError("RayTracingScene: staging buffer allocation failed."); return false;
    }
    void* p = nullptr;
    vkMapMemory(ctx_.device(), stagingV.memory, 0, vbSize, 0, &p); std::memcpy(p, m.positions.data(), vbSize); vkUnmapMemory(ctx_.device(), stagingV.memory);
    vkMapMemory(ctx_.device(), stagingI.memory, 0, ibSize, 0, &p); std::memcpy(p, m.indices.data(), ibSize); vkUnmapMemory(ctx_.device(), stagingI.memory);

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = ctx_.setupCommandPool(); cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(ctx_.device(), &cai, &cmd);
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbi);
    VkBufferCopy vc{0,0,vbSize}; vkCmdCopyBuffer(cmd, stagingV.buffer, m.vertex.buffer, 1, &vc);
    VkBufferCopy ic{0,0,ibSize}; vkCmdCopyBuffer(cmd, stagingI.buffer, m.index.buffer, 1, &ic);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
    vkQueueSubmit(ctx_.queue(),1,&si,VK_NULL_HANDLE); vkQueueWaitIdle(ctx_.queue());
    vkFreeCommandBuffers(ctx_.device(), ctx_.setupCommandPool(), 1, &cmd);
    destroyBuffer(stagingV); destroyBuffer(stagingI);

    VkAccelerationStructureGeometryTrianglesDataKHR tri{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT; tri.vertexData.deviceAddress = m.vertex.address; tri.vertexStride = sizeof(float)*3;
    tri.maxVertex = static_cast<uint32_t>(m.positions.size()/3 - 1); tri.indexType = VK_INDEX_TYPE_UINT32; tri.indexData.deviceAddress = m.index.address;
    VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR}; geom.geometryType=VK_GEOMETRY_TYPE_TRIANGLES_KHR; geom.geometry.triangles=tri; geom.flags=VK_GEOMETRY_OPAQUE_BIT_KHR;
    VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    info.type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR; info.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR|VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR; info.geometryCount=1; info.pGeometries=&geom;
    uint32_t primCount = static_cast<uint32_t>(m.indices.size()/3); VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR_(ctx_.device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &primCount, &sizes);
    if (!createBuffer(sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m.blasBuffer, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) { ctx_.setError("RayTracingScene: BLAS buffer allocation failed."); return false; }
    VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR}; aci.buffer=m.blasBuffer.buffer; aci.size=sizes.accelerationStructureSize; aci.type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (vkCreateAccelerationStructureKHR_(ctx_.device(), &aci, nullptr, &m.blas)!=VK_SUCCESS) { ctx_.setError("RayTracingScene: vkCreateAccelerationStructureKHR(BLAS) failed."); return false; }
    Buffer scratch; if (!createBuffer(sizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratch, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) { ctx_.setError("RayTracingScene: BLAS scratch allocation failed."); return false; }
    info.dstAccelerationStructure=m.blas; info.scratchData.deviceAddress=scratch.address; VkAccelerationStructureBuildRangeInfoKHR range{}; range.primitiveCount=primCount; const VkAccelerationStructureBuildRangeInfoKHR* ranges=&range;
    cai.commandPool=ctx_.setupCommandPool(); vkAllocateCommandBuffers(ctx_.device(), &cai, &cmd); vkBeginCommandBuffer(cmd,&cbi); vkCmdBuildAccelerationStructuresKHR_(cmd,1,&info,&ranges); vkEndCommandBuffer(cmd); si.pCommandBuffers=&cmd; vkQueueSubmit(ctx_.queue(),1,&si,VK_NULL_HANDLE); vkQueueWaitIdle(ctx_.queue()); vkFreeCommandBuffers(ctx_.device(),ctx_.setupCommandPool(),1,&cmd); destroyBuffer(scratch);
    compactBlas(m.blas, m.blasBuffer);
    return true;
}

bool RayTracingScene::compactBlas(VkAccelerationStructureKHR& blas, Buffer& blasBuffer) {
    if (!vkCmdWriteAccelerationStructuresPropertiesKHR_ || !vkCmdCopyAccelerationStructureKHR_)
        return false;

    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    qi.queryCount = 1;
    VkQueryPool pool = VK_NULL_HANDLE;
    if (vkCreateQueryPool(ctx_.device(), &qi, nullptr, &pool) != VK_SUCCESS)
        return false;

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = ctx_.setupCommandPool(); cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(ctx_.device(), &cai, &cmd);
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbi);
    vkCmdResetQueryPool(cmd, pool, 0, 1);
    vkCmdWriteAccelerationStructuresPropertiesKHR_(cmd, 1, &blas, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, pool, 0);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx_.queue(), 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx_.queue());
    vkFreeCommandBuffers(ctx_.device(), ctx_.setupCommandPool(), 1, &cmd);

    VkDeviceSize compactedSize = 0;
    const VkResult qr = vkGetQueryPoolResults(ctx_.device(), pool, 0, 1, sizeof(VkDeviceSize), &compactedSize, sizeof(VkDeviceSize),
                                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    vkDestroyQueryPool(ctx_.device(), pool, nullptr);
    // never regress
    if (qr != VK_SUCCESS || compactedSize == 0 || compactedSize >= blasBuffer.size)
        return false;

    Buffer newBuffer{};
    if (!createBuffer(compactedSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, newBuffer, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT))
        return false;

    VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    aci.buffer = newBuffer.buffer; aci.size = compactedSize; aci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    VkAccelerationStructureKHR newBlas = VK_NULL_HANDLE;
    if (vkCreateAccelerationStructureKHR_(ctx_.device(), &aci, nullptr, &newBlas) != VK_SUCCESS) {
        destroyBuffer(newBuffer);
        return false;
    }

    cai.commandPool = ctx_.setupCommandPool();
    vkAllocateCommandBuffers(ctx_.device(), &cai, &cmd);
    vkBeginCommandBuffer(cmd, &cbi);
    VkCopyAccelerationStructureInfoKHR copy{VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
    copy.src = blas; copy.dst = newBlas; copy.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
    vkCmdCopyAccelerationStructureKHR_(cmd, &copy);
    vkEndCommandBuffer(cmd);
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx_.queue(), 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx_.queue());
    vkFreeCommandBuffers(ctx_.device(), ctx_.setupCommandPool(), 1, &cmd);

    vkDestroyAccelerationStructureKHR_(ctx_.device(), blas, nullptr);
    destroyBuffer(blasBuffer);
    blas = newBlas;
    blasBuffer = newBuffer;
    return true;
}

bool RayTracingScene::buildChunkBlas(ChunkMesh& m) {
    if (!vkGetBufferDeviceAddress_ && !loadRtFunctions()) {
        ctx_.setError("RayTracingScene::buildChunkBlas: required Vulkan acceleration-structure entry points are unavailable.");
        return false;
    }
    const VkDeviceSize vbSize = m.vertices.size() * sizeof(ChunkRtVertex);
    const VkDeviceSize ibSize = m.indices.size() * sizeof(uint32_t);
    const VkBufferUsageFlags geoUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!createBuffer(vbSize, geoUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m.vertex, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
        ctx_.setError("RayTracingScene: chunk vertex buffer allocation failed."); return false;
    }
    if (!createBuffer(ibSize, geoUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m.index, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
        ctx_.setError("RayTracingScene: chunk index buffer allocation failed."); return false;
    }
    Buffer stagingV, stagingI;
    if (!createBuffer(vbSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingV) ||
        !createBuffer(ibSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingI)) {
        ctx_.setError("RayTracingScene: chunk staging buffer allocation failed."); return false;
    }
    void* p = nullptr;
    vkMapMemory(ctx_.device(), stagingV.memory, 0, vbSize, 0, &p); std::memcpy(p, m.vertices.data(), vbSize); vkUnmapMemory(ctx_.device(), stagingV.memory);
    vkMapMemory(ctx_.device(), stagingI.memory, 0, ibSize, 0, &p); std::memcpy(p, m.indices.data(), ibSize); vkUnmapMemory(ctx_.device(), stagingI.memory);

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = ctx_.setupCommandPool(); cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(ctx_.device(), &cai, &cmd);
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbi);
    VkBufferCopy vc{0,0,vbSize}; vkCmdCopyBuffer(cmd, stagingV.buffer, m.vertex.buffer, 1, &vc);
    VkBufferCopy ic{0,0,ibSize}; vkCmdCopyBuffer(cmd, stagingI.buffer, m.index.buffer, 1, &ic);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
    vkQueueSubmit(ctx_.queue(),1,&si,VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx_.queue());
    vkFreeCommandBuffers(ctx_.device(), ctx_.setupCommandPool(), 1, &cmd);
    destroyBuffer(stagingV); destroyBuffer(stagingI);

    const bool hasGlassSplit = m.opaqueIndexCount < static_cast<uint32_t>(m.indices.size());
    const uint32_t opaquePrimCount = (hasGlassSplit ? m.opaqueIndexCount : static_cast<uint32_t>(m.indices.size())) / 3;
    const uint32_t glassPrimCount = hasGlassSplit ? (static_cast<uint32_t>(m.indices.size()) - m.opaqueIndexCount) / 3 : 0;

    VkAccelerationStructureGeometryTrianglesDataKHR tri{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT; tri.vertexData.deviceAddress = m.vertex.address; tri.vertexStride = sizeof(ChunkRtVertex);
    tri.maxVertex = static_cast<uint32_t>(m.vertices.size() - 1); tri.indexType = VK_INDEX_TYPE_UINT32; tri.indexData.deviceAddress = m.index.address;
    VkAccelerationStructureGeometryKHR geoms[2]{};
    geoms[0] = VkAccelerationStructureGeometryKHR{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geoms[0].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR; geoms[0].geometry.triangles = tri;
    geoms[0].flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    if (hasGlassSplit) {
        geoms[1] = VkAccelerationStructureGeometryKHR{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geoms[1].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR; geoms[1].geometry.triangles = tri;
        geoms[1].flags = 0;
    }
    const uint32_t geometryCount = hasGlassSplit ? 2u : 1u;
    VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    info.type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR; info.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR|VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR; info.geometryCount=geometryCount; info.pGeometries=geoms;
    // pMaxPrimitiveCounts is a per-geometry array once geometryCount > 1
    uint32_t primCounts[2] = { opaquePrimCount, glassPrimCount };
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR_(ctx_.device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, primCounts, &sizes);
    if (!createBuffer(sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m.blasBuffer, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) { ctx_.setError("RayTracingScene: chunk BLAS buffer allocation failed."); return false; }
    VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR}; aci.buffer=m.blasBuffer.buffer; aci.size=sizes.accelerationStructureSize; aci.type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (vkCreateAccelerationStructureKHR_(ctx_.device(), &aci, nullptr, &m.blas)!=VK_SUCCESS) { ctx_.setError("RayTracingScene: chunk vkCreateAccelerationStructureKHR(BLAS) failed."); return false; }
    Buffer scratch; if (!createBuffer(sizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratch, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) { ctx_.setError("RayTracingScene: chunk BLAS scratch allocation failed."); return false; }
    info.dstAccelerationStructure=m.blas; info.scratchData.deviceAddress=scratch.address;
    VkAccelerationStructureBuildRangeInfoKHR ranges[2]{};
    ranges[0].primitiveCount = opaquePrimCount; ranges[0].primitiveOffset = 0;
    if (hasGlassSplit) {
        ranges[1].primitiveCount = glassPrimCount; ranges[1].primitiveOffset = m.opaqueIndexCount * static_cast<uint32_t>(sizeof(uint32_t));
    }
    const VkAccelerationStructureBuildRangeInfoKHR* rangePtrs = ranges;
    cai.commandPool=ctx_.setupCommandPool(); vkAllocateCommandBuffers(ctx_.device(), &cai, &cmd); vkBeginCommandBuffer(cmd,&cbi); vkCmdBuildAccelerationStructuresKHR_(cmd,1,&info,&rangePtrs); vkEndCommandBuffer(cmd); si.pCommandBuffers=&cmd; vkQueueSubmit(ctx_.queue(),1,&si,VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx_.queue());
    vkFreeCommandBuffers(ctx_.device(),ctx_.setupCommandPool(),1,&cmd); destroyBuffer(scratch);
    compactBlas(m.blas, m.blasBuffer);
    return true;
}

void RayTracingScene::destroyChunkBlas(ChunkMesh& m) {
    if (m.blas != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_) {
        vkDestroyAccelerationStructureKHR_(ctx_.device(), m.blas, nullptr);
        m.blas = VK_NULL_HANDLE;
    }
    destroyBuffer(m.blasBuffer);
    destroyBuffer(m.vertex);
    destroyBuffer(m.index);
}

bool RayTracingScene::addChunkMesh(uint64_t chunkId, const float* positions, uint32_t vertexCount,
                                    const uint32_t* indices, uint32_t indexCount, uint32_t materialSlot,
                                    const float* avgAlbedo) {
    return addTexturedChunkMesh(chunkId, positions, nullptr, nullptr, vertexCount, indices, indexCount, materialSlot, avgAlbedo);
}

bool RayTracingScene::addTexturedChunkMesh(uint64_t chunkId, const float* positions, const float* uvs,
                                           const uint32_t* textureGlIds, uint32_t vertexCount,
                                           const uint32_t* indices, uint32_t indexCount, uint32_t materialSlot,
                                           const float* avgAlbedo,
                                           const float* normals, const float* tangents,
                                           const uint32_t* materialIndices,
                                           uint32_t opaqueIndexCount,
                                           const uint32_t* blendMaterialIndices,
                                           const float* blendAlphas,
                                           const float* blendUvs,
                                           const float* alphaTestCutoffs,
                                           const float* blendLmUvs,
                                           const uint32_t* blendLmTexGlIds) {
    if (!positions || !indices || vertexCount == 0 || indexCount == 0 || (indexCount % 3) != 0) {
        ctx_.setError("RayTracingScene::addTexturedChunkMesh: invalid input."); return false;
    }
    if ((uvs == nullptr) != (textureGlIds == nullptr)) {
        ctx_.setError("RayTracingScene::addTexturedChunkMesh: UV and texture-id arrays must both be supplied for textured geometry."); return false;
    }
    for (uint32_t i = 0; i < indexCount; ++i) if (indices[i] >= vertexCount) {
        ctx_.setError("RayTracingScene::addTexturedChunkMesh: index out of range."); return false;
    }
    if (chunkMeshes_.find(chunkId) != chunkMeshes_.end()) {
        ctx_.setError("RayTracingScene::addTexturedChunkMesh: chunkId already exists -- call updateChunkMesh instead."); return false;
    }
    ChunkMesh m;
    m.vertices.resize(vertexCount);
    for (uint32_t i = 0; i < vertexCount; ++i) {
        m.vertices[i].px = positions[i*3+0];
        m.vertices[i].py = positions[i*3+1];
        m.vertices[i].pz = positions[i*3+2];
        // Sauerbraten smoothed normal/tangent
        m.vertices[i].nx = normals ? normals[i*3+0] : 0.0f;
        m.vertices[i].ny = normals ? normals[i*3+1] : 0.0f;
        m.vertices[i].nz = normals ? normals[i*3+2] : 1.0f;
        m.vertices[i].tx = tangents ? tangents[i*4+0] : 1.0f;
        m.vertices[i].ty = tangents ? tangents[i*4+1] : 0.0f;
        m.vertices[i].tz = tangents ? tangents[i*4+2] : 0.0f;
        m.vertices[i].tw = tangents ? tangents[i*4+3] : 1.0f;
        m.vertices[i].u = uvs ? uvs[i*2+0] : 0.0f;
        m.vertices[i].v = uvs ? uvs[i*2+1] : 0.0f;
        m.vertices[i].textureIndex = 0;
        m.vertices[i].materialIndex = materialIndices ? materialIndices[i] : 0u;
        // texture blending
        m.vertices[i].blendMaterialIndex = blendMaterialIndices ? blendMaterialIndices[i] : 0xFFFFFFFFu;
        m.vertices[i].blendAlpha = blendAlphas ? blendAlphas[i] : 0.0f;
        m.vertices[i].blendU = blendUvs ? blendUvs[i*2+0] : 0.0f;
        m.vertices[i].blendV = blendUvs ? blendUvs[i*2+1] : 0.0f;
        // mapmodel alpha-test
        m.vertices[i].alphaTestCutoff = alphaTestCutoffs ? alphaTestCutoffs[i] : 0.0f;
        if (blendLmUvs) { m.vertices[i].blendLmU = blendLmUvs[i*2+0]; m.vertices[i].blendLmV = blendLmUvs[i*2+1]; }
        if (blendLmTexGlIds && blendLmTexGlIds[i] != 0) {
            auto lit = pathTraceTextureIndices_.find(blendLmTexGlIds[i]);
            if (lit != pathTraceTextureIndices_.end()) m.vertices[i].blendLmTexIndex = lit->second;
        }
        if (textureGlIds && textureGlIds[i] != 0) {
            auto it = pathTraceTextureIndices_.find(textureGlIds[i]);
            m.vertices[i].textureIndex = it != pathTraceTextureIndices_.end() ? it->second : 0u;
        }
    }
    m.indices.assign(indices, indices + indexCount);
    m.materialSlot = materialSlot;
    m.opaqueIndexCount = (opaqueIndexCount >= indexCount) ? indexCount : opaqueIndexCount;
    if (avgAlbedo) { m.avgAlbedo[0]=avgAlbedo[0]; m.avgAlbedo[1]=avgAlbedo[1]; m.avgAlbedo[2]=avgAlbedo[2]; }
    // built by the next flushChunkBuilds()
    m.pendingBuild = true;
    chunkMeshes_.emplace(chunkId, std::move(m));
    return true;
}

bool RayTracingScene::flushChunkBuilds() {
    std::vector<std::pair<uint64_t, ChunkMesh*>> pending;
    for (auto& kv : chunkMeshes_) if (kv.second.pendingBuild) pending.emplace_back(kv.first, &kv.second);
    if (pending.empty()) return true;
    if (!vkGetBufferDeviceAddress_ && !loadRtFunctions()) {
        ctx_.setError("RayTracingScene::flushChunkBuilds: required Vulkan acceleration-structure entry points are unavailable.");
        return false;
    }
    static VkDeviceSize scratchAlign = 0;
    if (!scratchAlign) {
        VkPhysicalDeviceAccelerationStructurePropertiesKHR asp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &asp;
        vkGetPhysicalDeviceProperties2(ctx_.physicalDevice(), &p2);
        scratchAlign = std::max<VkDeviceSize>(asp.minAccelerationStructureScratchOffsetAlignment, 256);
    }
    auto alignUp = [](VkDeviceSize v, VkDeviceSize a) { return (v + a - 1) / a * a; };
    const VkDeviceSize kAsAlign = 256;  // VkAccelerationStructureCreateInfoKHR::offset
    const VkDeviceSize kBatchBudget = VkDeviceSize(256) << 20;  // per batch
    const bool canCompact = vkCmdWriteAccelerationStructuresPropertiesKHR_ && vkCmdCopyAccelerationStructureKHR_;
    const VkBufferUsageFlags geoUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkBufferUsageFlags asUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    struct Job {
        ChunkMesh* m;
        uint64_t id;
        VkDeviceSize vbSize, ibSize, geoOffset, asSize, asOffset, scratchOffset;
        uint32_t primCounts[2];
        uint32_t geometryCount;
        VkAccelerationStructureGeometryKHR geoms[2];
        VkAccelerationStructureBuildRangeInfoKHR ranges[2];
    };
    std::vector<uint64_t> failed;
    auto beginCmd = [&]() {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = ctx_.setupCommandPool(); cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(ctx_.device(), &cai, &cmd);
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &cbi);
        return cmd;
    };
    auto submitAndWait = [&](VkCommandBuffer cmd) {
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        vkQueueSubmit(ctx_.queue(), 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(ctx_.queue());
        vkFreeCommandBuffers(ctx_.device(), ctx_.setupCommandPool(), 1, &cmd);
    };
    auto view = [](const Buffer& arena, VkDeviceSize offset, VkDeviceSize size) {
        Buffer v{};
        v.buffer = arena.buffer; v.memory = VK_NULL_HANDLE; v.address = arena.address + offset; v.size = size; v.offset = offset; v.owned = false;
        return v;
    };
    auto dropJobs = [&](std::vector<Job>& jobs) {
        for (Job& j : jobs) {
            if (j.m->blas != VK_NULL_HANDLE) vkDestroyAccelerationStructureKHR_(ctx_.device(), j.m->blas, nullptr);
            j.m->blas = VK_NULL_HANDLE; j.m->vertex = Buffer{}; j.m->index = Buffer{}; j.m->blasBuffer = Buffer{};
            failed.push_back(j.id);
        }
    };

    using FlushClock = std::chrono::steady_clock;
    auto flushMs = [](FlushClock::time_point a) { return std::chrono::duration<double, std::milli>(FlushClock::now() - a).count(); };
    lastFlushStats_ = FlushStats{};
    lastFlushStats_.chunks = static_cast<uint32_t>(pending.size());
    size_t next = 0;
    while (next < pending.size()) {
        ++lastFlushStats_.batches;
        FlushClock::time_point phase = FlushClock::now();
        std::vector<Job> jobs;
        VkDeviceSize geoTotal = 0, asTotal = 0, scratchTotal = 0;
        for (; next < pending.size(); ++next) {
            ChunkMesh& m = *pending[next].second;
            Job j{};
            j.m = &m; j.id = pending[next].first;
            j.vbSize = m.vertices.size() * sizeof(ChunkRtVertex);
            j.ibSize = m.indices.size() * sizeof(uint32_t);
            const uint32_t indexCount = static_cast<uint32_t>(m.indices.size());
            const bool glassSplit = m.opaqueIndexCount < indexCount;
            j.primCounts[0] = (glassSplit ? m.opaqueIndexCount : indexCount) / 3;
            j.primCounts[1] = glassSplit ? (indexCount - m.opaqueIndexCount) / 3 : 0;
            j.geometryCount = glassSplit ? 2u : 1u;
            VkAccelerationStructureGeometryTrianglesDataKHR tri{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
            tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT; tri.vertexStride = sizeof(ChunkRtVertex);
            tri.maxVertex = static_cast<uint32_t>(m.vertices.size() - 1); tri.indexType = VK_INDEX_TYPE_UINT32;
            for (uint32_t g = 0; g < j.geometryCount; ++g) {
                j.geoms[g] = VkAccelerationStructureGeometryKHR{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
                j.geoms[g].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR; j.geoms[g].geometry.triangles = tri;
                j.geoms[g].flags = g == 0 ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;  // glass tail non-opaque, see buildChunkBlas()
            }
            VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
            info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
            info.geometryCount = j.geometryCount; info.pGeometries = j.geoms;
            VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
            vkGetAccelerationStructureBuildSizesKHR_(ctx_.device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, j.primCounts, &sizes);
            const VkDeviceSize geoNeed = alignUp(j.vbSize, kAsAlign) + alignUp(j.ibSize, kAsAlign);
            const VkDeviceSize asNeed = alignUp(sizes.accelerationStructureSize, kAsAlign);
            const VkDeviceSize scratchNeed = alignUp(sizes.buildScratchSize, scratchAlign);
            if (!jobs.empty() && (geoTotal + geoNeed > kBatchBudget || asTotal + asNeed > kBatchBudget || scratchTotal + scratchNeed > kBatchBudget)) break;
            j.asSize = sizes.accelerationStructureSize;
            j.geoOffset = geoTotal; geoTotal += geoNeed;
            j.asOffset = asTotal; asTotal += asNeed;
            j.scratchOffset = scratchTotal; scratchTotal += scratchNeed;
            jobs.push_back(j);
        }
        if (jobs.empty()) continue;

        Buffer geoArena, asArena, scratch, staging;
        if (!createBuffer(geoTotal, geoUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, geoArena, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) ||
            !createBuffer(asTotal, asUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, asArena, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) ||
            !createBuffer(scratchTotal, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratch, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) ||
            !createBuffer(geoTotal, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging)) {
            ctx_.setError("RayTracingScene::flushChunkBuilds: batch buffer allocation failed.");
            destroyBuffer(geoArena); destroyBuffer(asArena); destroyBuffer(scratch); destroyBuffer(staging);
            dropJobs(jobs);
            continue;
        }
        unsigned char* mapped = nullptr;
        vkMapMemory(ctx_.device(), staging.memory, 0, geoTotal, 0, reinterpret_cast<void**>(&mapped));
        std::vector<VkAccelerationStructureBuildGeometryInfoKHR> infos;
        std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rangePtrs;
        std::vector<VkAccelerationStructureKHR> blases;
        std::vector<Job> built;
        for (Job& j : jobs) {
            ChunkMesh& m = *j.m;
            m.vertex = view(geoArena, j.geoOffset, j.vbSize);
            m.index = view(geoArena, j.geoOffset + alignUp(j.vbSize, kAsAlign), j.ibSize);
            m.blasBuffer = view(asArena, j.asOffset, j.asSize);
            std::memcpy(mapped + m.vertex.offset, m.vertices.data(), static_cast<size_t>(j.vbSize));
            std::memcpy(mapped + m.index.offset, m.indices.data(), static_cast<size_t>(j.ibSize));
            VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
            aci.buffer = asArena.buffer; aci.offset = j.asOffset; aci.size = j.asSize; aci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            if (vkCreateAccelerationStructureKHR_(ctx_.device(), &aci, nullptr, &m.blas) != VK_SUCCESS) {
                ctx_.setError("RayTracingScene::flushChunkBuilds: vkCreateAccelerationStructureKHR(BLAS) failed.");
                m.blas = VK_NULL_HANDLE; m.vertex = Buffer{}; m.index = Buffer{}; m.blasBuffer = Buffer{};
                failed.push_back(j.id);
                continue;
            }
            for (uint32_t g = 0; g < j.geometryCount; ++g) {
                j.geoms[g].geometry.triangles.vertexData.deviceAddress = m.vertex.address;
                j.geoms[g].geometry.triangles.indexData.deviceAddress = m.index.address;
            }
            j.ranges[0] = VkAccelerationStructureBuildRangeInfoKHR{};
            j.ranges[0].primitiveCount = j.primCounts[0];
            j.ranges[1] = VkAccelerationStructureBuildRangeInfoKHR{};
            j.ranges[1].primitiveCount = j.primCounts[1];
            j.ranges[1].primitiveOffset = m.opaqueIndexCount * static_cast<uint32_t>(sizeof(uint32_t));  // bytes into the same index buffer
            built.push_back(j);
        }
        vkUnmapMemory(ctx_.device(), staging.memory);
        // pointers into `built` only once it has stopped growing
        for (Job& j : built) {
            VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
            info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
            info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            info.geometryCount = j.geometryCount; info.pGeometries = j.geoms;
            info.dstAccelerationStructure = j.m->blas;
            info.scratchData.deviceAddress = scratch.address + j.scratchOffset;
            infos.push_back(info);
            rangePtrs.push_back(j.ranges);
            blases.push_back(j.m->blas);
        }
        lastFlushStats_.createMs += flushMs(phase);
        phase = FlushClock::now();

        VkQueryPool pool = VK_NULL_HANDLE;
        if (canCompact && !built.empty()) {
            VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qi.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
            qi.queryCount = static_cast<uint32_t>(built.size());
            if (vkCreateQueryPool(ctx_.device(), &qi, nullptr, &pool) != VK_SUCCESS) pool = VK_NULL_HANDLE;
        }
        VkCommandBuffer cmd = beginCmd();
        VkBufferCopy whole{0, 0, geoTotal};
        vkCmdCopyBuffer(cmd, staging.buffer, geoArena.buffer, 1, &whole);  // same layout on both sides
        VkMemoryBarrier toBuild{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        toBuild.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toBuild.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &toBuild, 0, nullptr, 0, nullptr);
        if (!infos.empty()) vkCmdBuildAccelerationStructuresKHR_(cmd, static_cast<uint32_t>(infos.size()), infos.data(), rangePtrs.data());
        if (pool) {
            VkMemoryBarrier builtBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            builtBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            builtBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &builtBarrier, 0, nullptr, 0, nullptr);
            vkCmdResetQueryPool(cmd, pool, 0, static_cast<uint32_t>(blases.size()));
            vkCmdWriteAccelerationStructuresPropertiesKHR_(cmd, static_cast<uint32_t>(blases.size()), blases.data(),
                                                           VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, pool, 0);
        }
        submitAndWait(cmd);
        destroyBuffer(staging);
        destroyBuffer(scratch);
        chunkArenas_.push_back(geoArena);
        lastFlushStats_.buildMs += flushMs(phase);
        phase = FlushClock::now();

        bool asArenaKept = true;
        if (pool) {
            std::vector<VkDeviceSize> compacted(built.size(), 0);
            const VkResult qr = vkGetQueryPoolResults(ctx_.device(), pool, 0, static_cast<uint32_t>(built.size()),
                                                      compacted.size() * sizeof(VkDeviceSize), compacted.data(), sizeof(VkDeviceSize),
                                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            vkDestroyQueryPool(ctx_.device(), pool, nullptr);
            bool valid = qr == VK_SUCCESS;
            VkDeviceSize packedTotal = 0;
            std::vector<VkDeviceSize> packedOffset(built.size(), 0);
            for (size_t k = 0; k < built.size() && valid; ++k) {
                if (compacted[k] == 0) { valid = false; break; }
                packedOffset[k] = packedTotal;
                packedTotal += alignUp(compacted[k], kAsAlign);
            }
            Buffer packed;
            if (valid && packedTotal < asTotal &&
                createBuffer(packedTotal, asUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, packed, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
                std::vector<VkAccelerationStructureKHR> newBlas(built.size(), VK_NULL_HANDLE);
                bool ok = true;
                for (size_t k = 0; k < built.size() && ok; ++k) {
                    VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
                    aci.buffer = packed.buffer; aci.offset = packedOffset[k]; aci.size = compacted[k]; aci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                    ok = vkCreateAccelerationStructureKHR_(ctx_.device(), &aci, nullptr, &newBlas[k]) == VK_SUCCESS;
                }
                if (ok) {
                    VkCommandBuffer ccmd = beginCmd();
                    for (size_t k = 0; k < built.size(); ++k) {
                        VkCopyAccelerationStructureInfoKHR copy{VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
                        copy.src = built[k].m->blas; copy.dst = newBlas[k]; copy.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
                        vkCmdCopyAccelerationStructureKHR_(ccmd, &copy);
                    }
                    submitAndWait(ccmd);
                    for (size_t k = 0; k < built.size(); ++k) {
                        vkDestroyAccelerationStructureKHR_(ctx_.device(), built[k].m->blas, nullptr);
                        built[k].m->blas = newBlas[k];
                        built[k].m->blasBuffer = view(packed, packedOffset[k], compacted[k]);
                    }
                    chunkArenas_.push_back(packed);
                    destroyBuffer(asArena);
                    asArenaKept = false;
                } else {
                    for (VkAccelerationStructureKHR b : newBlas) if (b != VK_NULL_HANDLE) vkDestroyAccelerationStructureKHR_(ctx_.device(), b, nullptr);
                    destroyBuffer(packed);
                }
            }
        }
        if (asArenaKept) chunkArenas_.push_back(asArena);
        for (Job& j : built) j.m->pendingBuild = false;
        lastFlushStats_.compactMs += flushMs(phase);
    }
    for (uint64_t id : failed) chunkMeshes_.erase(id);
    return failed.empty();
}

bool RayTracingScene::updateChunkMesh(uint64_t chunkId, const float* positions, uint32_t vertexCount,
                                       const uint32_t* indices, uint32_t indexCount, uint32_t materialSlot,
                                       const float* avgAlbedo) {
    if (!positions || !indices || vertexCount == 0 || indexCount == 0 || (indexCount % 3) != 0) {
        ctx_.setError("RayTracingScene::updateChunkMesh: invalid input.");
        return false;
    }
    for (uint32_t i = 0; i < indexCount; ++i) {
        if (indices[i] >= vertexCount) {
            ctx_.setError("RayTracingScene::updateChunkMesh: index out of range.");
            return false;
        }
    }
    auto it = chunkMeshes_.find(chunkId);
    if (it == chunkMeshes_.end()) {
        ctx_.setError("RayTracingScene::updateChunkMesh: chunkId does not exist -- call addChunkMesh instead.");
        return false;
    }
    destroyChunkBlas(it->second);
    ChunkMesh fresh;
    fresh.vertices.resize(vertexCount);
    for (uint32_t i=0; i<vertexCount; ++i) {
        fresh.vertices[i].px=positions[i*3+0]; fresh.vertices[i].py=positions[i*3+1]; fresh.vertices[i].pz=positions[i*3+2];
        fresh.vertices[i].u=0.0f; fresh.vertices[i].v=0.0f; fresh.vertices[i].textureIndex=0;
        fresh.vertices[i].nx=0.0f; fresh.vertices[i].ny=0.0f; fresh.vertices[i].nz=1.0f;
        fresh.vertices[i].tx=1.0f; fresh.vertices[i].ty=0.0f; fresh.vertices[i].tz=0.0f; fresh.vertices[i].tw=1.0f;
        fresh.vertices[i].materialIndex=0;
    }
    fresh.indices.assign(indices, indices + indexCount);
    fresh.materialSlot = materialSlot;
    if (avgAlbedo) { fresh.avgAlbedo[0] = avgAlbedo[0]; fresh.avgAlbedo[1] = avgAlbedo[1]; fresh.avgAlbedo[2] = avgAlbedo[2]; }
    if (!buildChunkBlas(fresh)) {
        chunkMeshes_.erase(it);
        return false;
    }
    it->second = std::move(fresh);
    return true;
}

bool RayTracingScene::removeChunkMesh(uint64_t chunkId) {
    auto it = chunkMeshes_.find(chunkId);
    if (it == chunkMeshes_.end()) return false;
    destroyChunkBlas(it->second);
    chunkMeshes_.erase(it);
    return true;
}

bool RayTracingScene::recordSkinDispatch(VkCommandBuffer cmd, DynamicMeshSlot& slot) {
    if (!createSkinPipeline()) return false;
    auto srcIt = skinSources_.find(slot.skinSourceKey);
    if (srcIt == skinSources_.end()) { ctx_.setError("RayTracingScene::recordSkinDispatch: skinSourceKey not registered."); return false; }
    const SkinSource& src = srcIt->second;
    if (slot.boneMatrixData.size() != static_cast<size_t>(src.boneCount) * 8) {
        ctx_.setError("RayTracingScene::recordSkinDispatch: bone data size does not match this model's registered bone count.");
        return false;
    }

    const VkDeviceSize boneSize = static_cast<VkDeviceSize>(slot.boneMatrixData.size()) * sizeof(float);
    if (!ensureBufferCapacity(slot.boneBuffer, boneSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
        !ensureBufferCapacity(slot.boneStaging, boneSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        ctx_.setError("RayTracingScene: skin bone buffer allocation failed."); return false;
    }
    void* p = nullptr;
    vkMapMemory(ctx_.device(), slot.boneStaging.memory, 0, boneSize, 0, &p); std::memcpy(p, slot.boneMatrixData.data(), boneSize); vkUnmapMemory(ctx_.device(), slot.boneStaging.memory);

    struct SkinParamsCpu { float worldRow0[4], worldRow1[4], worldRow2[4]; uint32_t vertexCount; float pad[3]; };
    SkinParamsCpu params{};
    std::memcpy(params.worldRow0, &slot.worldMatrix[0], sizeof(float) * 4);
    std::memcpy(params.worldRow1, &slot.worldMatrix[4], sizeof(float) * 4);
    std::memcpy(params.worldRow2, &slot.worldMatrix[8], sizeof(float) * 4);
    params.vertexCount = src.vertexCount;
    const VkDeviceSize paramsSize = sizeof(SkinParamsCpu);
    if (!ensureBufferCapacity(slot.skinParams, paramsSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
        !ensureBufferCapacity(slot.skinParamsStaging, paramsSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        ctx_.setError("RayTracingScene: skin params buffer allocation failed."); return false;
    }
    vkMapMemory(ctx_.device(), slot.skinParamsStaging.memory, 0, paramsSize, 0, &p); std::memcpy(p, &params, paramsSize); vkUnmapMemory(ctx_.device(), slot.skinParamsStaging.memory);

    VkBufferCopy bc{0,0,boneSize}; vkCmdCopyBuffer(cmd, slot.boneStaging.buffer, slot.boneBuffer.buffer, 1, &bc);
    VkBufferCopy pc{0,0,paramsSize}; vkCmdCopyBuffer(cmd, slot.skinParamsStaging.buffer, slot.skinParams.buffer, 1, &pc);
    VkMemoryBarrier toCompute{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    toCompute.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toCompute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &toCompute, 0, nullptr, 0, nullptr);

    if (slot.skinDescriptorSet == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPoolSkin_; ai.descriptorSetCount = 1; ai.pSetLayouts = &descriptorSetLayoutSkin_;
        if (vkAllocateDescriptorSets(ctx_.device(), &ai, &slot.skinDescriptorSet) != VK_SUCCESS) {
            ctx_.setError("RayTracingScene: skin descriptor set allocation failed (too many animated instances?)");
            return false;
        }
    }
    // always rewrite the four bindings (cheap)
    const bool needsUpdate = true;
    if (needsUpdate) {
        VkDescriptorBufferInfo bi0{src.vertexBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo bi1{slot.boneBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo bi2{slot.vertex.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo bi3{slot.skinParams.buffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w[4]{};
        w[0] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[0].dstSet=slot.skinDescriptorSet; w[0].dstBinding=0; w[0].descriptorCount=1; w[0].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo=&bi0;
        w[1] = w[0]; w[1].dstBinding=1; w[1].pBufferInfo=&bi1;
        w[2] = w[0]; w[2].dstBinding=2; w[2].pBufferInfo=&bi2;
        w[3] = w[0]; w[3].dstBinding=3; w[3].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[3].pBufferInfo=&bi3;
        vkUpdateDescriptorSets(ctx_.device(), 4, w, 0, nullptr);
        slot.skinDescriptorBoundSource = src.vertexBuffer.buffer;
        slot.skinDescriptorBoundBones = slot.boneBuffer.buffer;
        slot.skinDescriptorBoundOutput = slot.vertex.buffer;
        slot.skinDescriptorBoundParams = slot.skinParams.buffer;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, skinPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayoutSkin_, 0, 1, &slot.skinDescriptorSet, 0, nullptr);
    const uint32_t groups = (src.vertexCount + 63) / 64;  // matches skin.comp's own local_size_x=64
    vkCmdDispatch(cmd, groups, 1, 1);

    VkMemoryBarrier toBuild{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    toBuild.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toBuild.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &toBuild, 0, nullptr, 0, nullptr);
    return true;
}

bool RayTracingScene::recordDynamicSlotBuild(VkCommandBuffer cmd, DynamicMeshSlot& slot, uint32_t ordinal) {
    uint32_t vertexCount = 0, indexCount = 0;
    VkDeviceSize vertexStride = 0;
    VkDeviceAddress indexDeviceAddress = 0;

    if (slot.skinned) {
        auto srcIt = skinSources_.find(slot.skinSourceKey);
        if (srcIt == skinSources_.end()) {
            ctx_.setError("RayTracingScene: dynamic slot's skinSourceKey is no longer registered.");
            return false;
        }
        const SkinSource& src = srcIt->second;
        vertexCount = src.vertexCount;
        indexCount = src.indexCount;
        vertexStride = sizeof(ChunkRtVertex);
        indexDeviceAddress = src.indexBuffer.address;
        slot.materialSlot = dynamicMaterialBase_ + ordinal;
        const VkDeviceSize vbSize = static_cast<VkDeviceSize>(vertexCount) * vertexStride;
        const VkBufferUsageFlags skinnedVertexUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (!ensureBufferCapacity(slot.vertex, vbSize, skinnedVertexUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
            ctx_.setError("RayTracingScene: skinned dynamic vertex buffer allocation failed."); return false;
        }
        if (!recordSkinDispatch(cmd, slot)) return false;
    } else if (slot.water) {
        slot.materialSlot = dynamicMaterialBase_ + ordinal;
        const VkDeviceSize vbSize = slot.waterVertices.size() * sizeof(ChunkRtVertex);
        const VkDeviceSize ibSize = slot.indices.size() * sizeof(uint32_t);
        const VkBufferUsageFlags geoUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (!ensureBufferCapacity(slot.vertex, vbSize, geoUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
            ctx_.setError("RayTracingScene: water dynamic vertex buffer allocation failed."); return false;
        }
        if (!ensureBufferCapacity(slot.index, ibSize, geoUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
            ctx_.setError("RayTracingScene: water dynamic index buffer allocation failed."); return false;
        }
        if (!ensureBufferCapacity(slot.stagingVertex, vbSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ||
            !ensureBufferCapacity(slot.stagingIndex, ibSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            ctx_.setError("RayTracingScene: water dynamic staging buffer allocation failed."); return false;
        }
        void* p = nullptr;
        vkMapMemory(ctx_.device(), slot.stagingVertex.memory, 0, vbSize, 0, &p); std::memcpy(p, slot.waterVertices.data(), vbSize); vkUnmapMemory(ctx_.device(), slot.stagingVertex.memory);
        vkMapMemory(ctx_.device(), slot.stagingIndex.memory, 0, ibSize, 0, &p); std::memcpy(p, slot.indices.data(), ibSize); vkUnmapMemory(ctx_.device(), slot.stagingIndex.memory);

        vertexCount = static_cast<uint32_t>(slot.waterVertices.size());
        indexCount = static_cast<uint32_t>(slot.indices.size());
        vertexStride = sizeof(ChunkRtVertex);
        indexDeviceAddress = slot.index.address;

        VkBufferCopy vc{0,0,vbSize}; vkCmdCopyBuffer(cmd, slot.stagingVertex.buffer, slot.vertex.buffer, 1, &vc);
        VkBufferCopy ic{0,0,ibSize}; vkCmdCopyBuffer(cmd, slot.stagingIndex.buffer, slot.index.buffer, 1, &ic);
        VkMemoryBarrier waterBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        waterBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        waterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             0, 1, &waterBarrier, 0, nullptr, 0, nullptr);
    } else if (slot.textured) {
        slot.materialSlot = dynamicMaterialBase_ + ordinal;
        const VkDeviceSize vbSize = slot.texturedVertices.size() * sizeof(ChunkRtVertex);
        const VkDeviceSize ibSize = slot.indices.size() * sizeof(uint32_t);
        const VkBufferUsageFlags geoUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (!ensureBufferCapacity(slot.vertex, vbSize, geoUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
            ctx_.setError("RayTracingScene: textured dynamic vertex buffer allocation failed."); return false;
        }
        if (!ensureBufferCapacity(slot.index, ibSize, geoUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
            ctx_.setError("RayTracingScene: textured dynamic index buffer allocation failed."); return false;
        }
        if (!ensureBufferCapacity(slot.stagingVertex, vbSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ||
            !ensureBufferCapacity(slot.stagingIndex, ibSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            ctx_.setError("RayTracingScene: textured dynamic staging buffer allocation failed."); return false;
        }
        void* p = nullptr;
        vkMapMemory(ctx_.device(), slot.stagingVertex.memory, 0, vbSize, 0, &p); std::memcpy(p, slot.texturedVertices.data(), vbSize); vkUnmapMemory(ctx_.device(), slot.stagingVertex.memory);
        vkMapMemory(ctx_.device(), slot.stagingIndex.memory, 0, ibSize, 0, &p); std::memcpy(p, slot.indices.data(), ibSize); vkUnmapMemory(ctx_.device(), slot.stagingIndex.memory);

        vertexCount = static_cast<uint32_t>(slot.texturedVertices.size());
        indexCount = static_cast<uint32_t>(slot.indices.size());
        vertexStride = sizeof(ChunkRtVertex);
        indexDeviceAddress = slot.index.address;

        VkBufferCopy vc{0,0,vbSize}; vkCmdCopyBuffer(cmd, slot.stagingVertex.buffer, slot.vertex.buffer, 1, &vc);
        VkBufferCopy ic{0,0,ibSize}; vkCmdCopyBuffer(cmd, slot.stagingIndex.buffer, slot.index.buffer, 1, &ic);
        VkMemoryBarrier texturedBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        texturedBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        texturedBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             0, 1, &texturedBarrier, 0, nullptr, 0, nullptr);
    } else {
        // CPU-supplied positions (addDynamicMesh), unchanged from before GPU skinning existed
        const VkDeviceSize vbSize = slot.positions.size() * sizeof(float);
        const VkDeviceSize ibSize = slot.indices.size() * sizeof(uint32_t);
        const VkBufferUsageFlags geoUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (!ensureBufferCapacity(slot.vertex, vbSize, geoUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
            ctx_.setError("RayTracingScene: dynamic vertex buffer allocation failed."); return false;
        }
        if (!ensureBufferCapacity(slot.index, ibSize, geoUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
            ctx_.setError("RayTracingScene: dynamic index buffer allocation failed."); return false;
        }
        if (!ensureBufferCapacity(slot.stagingVertex, vbSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ||
            !ensureBufferCapacity(slot.stagingIndex, ibSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            ctx_.setError("RayTracingScene: dynamic staging buffer allocation failed."); return false;
        }
        void* p = nullptr;
        vkMapMemory(ctx_.device(), slot.stagingVertex.memory, 0, vbSize, 0, &p); std::memcpy(p, slot.positions.data(), vbSize); vkUnmapMemory(ctx_.device(), slot.stagingVertex.memory);
        vkMapMemory(ctx_.device(), slot.stagingIndex.memory, 0, ibSize, 0, &p); std::memcpy(p, slot.indices.data(), ibSize); vkUnmapMemory(ctx_.device(), slot.stagingIndex.memory);

        vertexCount = static_cast<uint32_t>(slot.positions.size() / 3);
        indexCount = static_cast<uint32_t>(slot.indices.size());
        vertexStride = sizeof(float) * 3;
        indexDeviceAddress = slot.index.address;

        VkBufferCopy vc{0,0,vbSize}; vkCmdCopyBuffer(cmd, slot.stagingVertex.buffer, slot.vertex.buffer, 1, &vc);
        VkBufferCopy ic{0,0,ibSize}; vkCmdCopyBuffer(cmd, slot.stagingIndex.buffer, slot.index.buffer, 1, &ic);
        // same TRANSFER_WRITE -> ACCELERATION_STRUCTURE_READ barrier as always
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    VkAccelerationStructureGeometryTrianglesDataKHR tri{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT; tri.vertexData.deviceAddress = slot.vertex.address; tri.vertexStride = vertexStride;
    tri.maxVertex = vertexCount - 1; tri.indexType = VK_INDEX_TYPE_UINT32; tri.indexData.deviceAddress = indexDeviceAddress;
    VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR}; geom.geometryType=VK_GEOMETRY_TYPE_TRIANGLES_KHR; geom.geometry.triangles=tri;
    geom.flags = (slot.water || slot.particles || slot.alphaTested) ? 0 : VK_GEOMETRY_OPAQUE_BIT_KHR;
    VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    info.type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    // ALLOW_UPDATE_BIT unconditionally
    info.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    info.geometryCount=1; info.pGeometries=&geom;
    uint32_t primCount = indexCount / 3;
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR_(ctx_.device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &primCount, &sizes);

    bool blasBufferNeedsGrow = slot.blasBuffer.buffer == VK_NULL_HANDLE || slot.blasBuffer.size < sizes.accelerationStructureSize;
    if (!ensureBufferCapacity(slot.blasBuffer, sizes.accelerationStructureSize,
                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
        ctx_.setError("RayTracingScene: dynamic BLAS buffer allocation failed."); return false;
    }
    if (blasBufferNeedsGrow && slot.blas != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructureKHR_(ctx_.device(), slot.blas, nullptr);
        slot.blas = VK_NULL_HANDLE;
    }
    if (slot.blas == VK_NULL_HANDLE) {
        VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        aci.buffer = slot.blasBuffer.buffer; aci.size = sizes.accelerationStructureSize; aci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        if (vkCreateAccelerationStructureKHR_(ctx_.device(), &aci, nullptr, &slot.blas) != VK_SUCCESS) {
            ctx_.setError("RayTracingScene: vkCreateAccelerationStructureKHR(dynamic BLAS) failed."); return false;
        }
    }
    if (!ensureBufferCapacity(slot.scratch, sizes.buildScratchSize,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
        ctx_.setError("RayTracingScene: dynamic BLAS scratch allocation failed."); return false;
    }
    bool canUpdate = !forceAccelRebuild_ && slot.blasBuilt && !blasBufferNeedsGrow &&
                      slot.lastBuiltVertexCount == vertexCount && slot.lastBuiltIndexCount == indexCount;
    info.mode = canUpdate ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.srcAccelerationStructure = canUpdate ? slot.blas : VK_NULL_HANDLE;
    info.dstAccelerationStructure = slot.blas;
    info.scratchData.deviceAddress = slot.scratch.address;
    VkAccelerationStructureBuildRangeInfoKHR range{}; range.primitiveCount = primCount;
    const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;

    vkCmdBuildAccelerationStructuresKHR_(cmd, 1, &info, &ranges);
    slot.blasBuilt = true;
    slot.lastBuiltVertexCount = vertexCount;
    slot.lastBuiltIndexCount = indexCount;
    return true;
}

bool RayTracingScene::recordTlasBuild(VkCommandBuffer cmd, DynamicGeneration& generation) {
    lastRigidDiag_ = "recordTlasBuild ENTERED rigidSources_.size()=" + std::to_string(rigidSources_.size()) +
        " pendingRigidInstances_.size()=" + std::to_string(pendingRigidInstances_.size());
    const size_t staticCount = meshes_.size();
    const size_t dynamicCount = generation.slotsUsed;
    const size_t chunkCount = chunkMeshes_.size();
    std::vector<const RigidInstanceDraw*> validRigidInstances;
    validRigidInstances.reserve(pendingRigidInstances_.size());
    for (const RigidInstanceDraw& d : pendingRigidInstances_) {
        auto rit = rigidSources_.find(d.modelKey);
        if (rit != rigidSources_.end() && rit->second.blasBuilt) validRigidInstances.push_back(&d);
    }
    const size_t rigidCount = validRigidInstances.size();
    lastRigidDiag_ = "recordTlasBuild pendingRigidInstances_.size()=" + std::to_string(pendingRigidInstances_.size()) +
        " validRigidInstances.size()=" + std::to_string(rigidCount) + " rigidSources_.size()=" + std::to_string(rigidSources_.size());
    // displaced water patch (recordWaterPatch), one extra instance, last
    const WaterPatchGen* waterGen = (dynamicGenerationIndex_ < waterPatchGens_.size() && waterPatchGens_[dynamicGenerationIndex_].active)
        ? &waterPatchGens_[dynamicGenerationIndex_] : nullptr;
    const size_t totalCount = staticCount + dynamicCount + chunkCount + rigidCount + (waterGen ? 1 : 0);
    std::vector<VkAccelerationStructureInstanceKHR> instances(totalCount);
    if (waterGen) {
        VkAccelerationStructureInstanceKHR& wi = instances[totalCount - 1];
        wi.transform = VkTransformMatrixKHR{{{1,0,0,0},{0,1,0,0},{0,0,1,0}}};
        wi.instanceCustomIndex = waterPatchMaterialSlot_; wi.mask = 0xff; wi.instanceShaderBindingTableRecordOffset = 0;
        wi.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR}; ai.accelerationStructure = waterGen->blas;
        wi.accelerationStructureReference = vkGetAccelerationStructureDeviceAddressKHR_(ctx_.device(), &ai);
    }
    for (uint32_t i = 0; i < instances.size(); ++i) {
        if (i >= staticCount + dynamicCount) continue;  // filled by the chunk loop below
        VkAccelerationStructureKHR blas = (i < staticCount) ? meshes_[i].blas : generation.slots[i - staticCount].blas;
        uint32_t materialSlot = (i < staticCount) ? meshes_[i].materialSlot : generation.slots[i - staticCount].materialSlot;
        const uint32_t instMask = (i < staticCount) ? 0xFFu
            : ((generation.slots[i - staticCount].skinned || generation.slots[i - staticCount].particles) ? generation.slots[i - staticCount].instanceMask : 0xFFu);
        VkTransformMatrixKHR t{{{1,0,0,0},{0,1,0,0},{0,0,1,0}}};
        instances[i].transform=t; instances[i].instanceCustomIndex=materialSlot; instances[i].mask=instMask; instances[i].instanceShaderBindingTableRecordOffset=0; instances[i].flags=VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR}; ai.accelerationStructure=blas; instances[i].accelerationStructureReference=vkGetAccelerationStructureDeviceAddressKHR_(ctx_.device(),&ai);
    }
    {
        uint32_t i = static_cast<uint32_t>(staticCount + dynamicCount);
        for (auto& kv : chunkMeshes_) {
            ChunkMesh& cm = kv.second;
            VkTransformMatrixKHR t{{{1,0,0,0},{0,1,0,0},{0,0,1,0}}};
            instances[i].transform=t; instances[i].instanceCustomIndex=cm.materialSlot; instances[i].mask=0xff; instances[i].instanceShaderBindingTableRecordOffset=0; instances[i].flags=VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR}; ai.accelerationStructure=cm.blas; instances[i].accelerationStructureReference=vkGetAccelerationStructureDeviceAddressKHR_(ctx_.device(),&ai);
            ++i;
        }
    }
    {
        uint32_t i = static_cast<uint32_t>(staticCount + dynamicCount + chunkCount);
        bool diagLoggedOne = false;
        for (const RigidInstanceDraw* d : validRigidInstances) {
            RigidMeshSource& src = rigidSources_[d->modelKey];
            if (!diagLoggedOne) {
                diagLoggedOne = true;
                char buf[512];
                snprintf(buf, sizeof(buf),
                    "instance sample modelKey=%llu blas=%p materialSlot=%u m=[%.2f %.2f %.2f %.2f | %.2f %.2f %.2f %.2f | %.2f %.2f %.2f %.2f]",
                    (unsigned long long)d->modelKey, (void*)src.blas, src.materialSlot,
                    d->worldMatrix[0], d->worldMatrix[1], d->worldMatrix[2], d->worldMatrix[3],
                    d->worldMatrix[4], d->worldMatrix[5], d->worldMatrix[6], d->worldMatrix[7],
                    d->worldMatrix[8], d->worldMatrix[9], d->worldMatrix[10], d->worldMatrix[11]);
                lastRigidDiag_ = buf;
            }
            VkTransformMatrixKHR t{{
                {d->worldMatrix[0], d->worldMatrix[1], d->worldMatrix[2], d->worldMatrix[3]},
                {d->worldMatrix[4], d->worldMatrix[5], d->worldMatrix[6], d->worldMatrix[7]},
                {d->worldMatrix[8], d->worldMatrix[9], d->worldMatrix[10], d->worldMatrix[11]}
            }};
            instances[i].transform=t; instances[i].instanceCustomIndex=src.materialSlot; instances[i].mask=0xff; instances[i].instanceShaderBindingTableRecordOffset=0; instances[i].flags=VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR}; ai.accelerationStructure=src.blas; instances[i].accelerationStructureReference=vkGetAccelerationStructureDeviceAddressKHR_(ctx_.device(),&ai);
            ++i;
        }
    }
    const VkDeviceSize bytes = instances.size() * sizeof(VkAccelerationStructureInstanceKHR);
    if (!ensureBufferCapacity(generation.tlasInstanceBuffer, bytes,
                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
        ctx_.setError("RayTracingScene: TLAS instance buffer allocation failed."); return false;
    }
    void* p=nullptr; vkMapMemory(ctx_.device(),generation.tlasInstanceBuffer.memory,0,bytes,0,&p); std::memcpy(p,instances.data(),bytes); vkUnmapMemory(ctx_.device(),generation.tlasInstanceBuffer.memory);
    VkAccelerationStructureGeometryInstancesDataKHR ids{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR}; ids.arrayOfPointers=VK_FALSE; ids.data.deviceAddress=generation.tlasInstanceBuffer.address;
    VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR}; geom.geometryType=VK_GEOMETRY_TYPE_INSTANCES_KHR; geom.geometry.instances=ids;
    VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR}; info.type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    info.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    info.geometryCount=1; info.pGeometries=&geom;
    uint32_t primCount=static_cast<uint32_t>(instances.size()); VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR}; vkGetAccelerationStructureBuildSizesKHR_(ctx_.device(),VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&info,&primCount,&sizes);
    bool tlasBufferNeedsGrow = generation.tlasBuffer.buffer == VK_NULL_HANDLE || generation.tlasBuffer.size < sizes.accelerationStructureSize;
    if(!ensureBufferCapacity(generation.tlasBuffer, sizes.accelerationStructureSize,VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)){ctx_.setError("RayTracingScene: TLAS buffer allocation failed.");return false;}
    if (tlasBufferNeedsGrow && generation.tlas != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructureKHR_(ctx_.device(), generation.tlas, nullptr);
        generation.tlas = VK_NULL_HANDLE;
    }
    if (generation.tlas == VK_NULL_HANDLE) {
        VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR}; aci.buffer=generation.tlasBuffer.buffer; aci.size=sizes.accelerationStructureSize; aci.type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        if(vkCreateAccelerationStructureKHR_(ctx_.device(),&aci,nullptr,&generation.tlas)!=VK_SUCCESS){ctx_.setError("RayTracingScene: vkCreateAccelerationStructureKHR(TLAS) failed.");return false;}
    }
    if(!ensureBufferCapacity(generation.tlasScratch, sizes.buildScratchSize,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)){ctx_.setError("RayTracingScene: TLAS scratch allocation failed.");return false;}
    bool canUpdate = !forceAccelRebuild_ && generation.tlas != VK_NULL_HANDLE && !tlasBufferNeedsGrow &&
                      generation.lastBuiltInstanceCount == totalCount;
    info.mode = canUpdate ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.srcAccelerationStructure = canUpdate ? generation.tlas : VK_NULL_HANDLE;
    info.dstAccelerationStructure=generation.tlas; info.scratchData.deviceAddress=generation.tlasScratch.address; VkAccelerationStructureBuildRangeInfoKHR range{}; range.primitiveCount=primCount; const VkAccelerationStructureBuildRangeInfoKHR* ranges=&range;
    vkCmdBuildAccelerationStructuresKHR_(cmd,1,&info,&ranges);
    generation.lastBuiltInstanceCount = static_cast<uint32_t>(totalCount);
    return true;
}

bool RayTracingScene::ensureRtaoPipeline() {
    if (rtaoPipeline_ != VK_NULL_HANDLE) return true;
    std::vector<uint32_t> spirv; if(!readSpirv(shaderDirectory_ + "\\rtao.spv",spirv) && !readSpirv(shaderDirectory_ + "/rtao.spv",spirv)){ctx_.setError("RayTracingScene: missing rtao.spv in shader directory.");return false;}
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; sm.codeSize=spirv.size()*4; sm.pCode=spirv.data(); if(vkCreateShaderModule(ctx_.device(),&sm,nullptr,&rtaoShader_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: vkCreateShaderModule(rtao) failed.");return false;}
    VkDescriptorSetLayoutBinding b[15]{};
    b[0]={0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[1]={1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[2]={2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[3]={3,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[4]={4,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[5]={5,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[6]={6,VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[7]={7,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[8]={8,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[9]={9,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[10]={10,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[11]={11,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[12]={12,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    // the path tracer's scene data, shared (same buffers/images, see rt_common.glsl)
    b[13]={13,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    {
        VkPhysicalDeviceProperties lp{}; vkGetPhysicalDeviceProperties(ctx_.physicalDevice(), &lp);
        const uint32_t need = kMaxPathTraceTextures + 16u;
        if (lp.limits.maxPerStageDescriptorSamplers < need || lp.limits.maxPerStageDescriptorSampledImages < need ||
            lp.limits.maxDescriptorSetSamplers < need || lp.limits.maxDescriptorSetSampledImages < need) {
            ctx_.setError("RayTracingScene: this GPU's descriptor limits are below the path tracer's " + std::to_string(kMaxPathTraceTextures) + "-texture table.");
            return false;
        }
    }
    b[14]={14,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,kMaxPathTraceTextures,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};dl.bindingCount=15;dl.pBindings=b;if(vkCreateDescriptorSetLayout(ctx_.device(),&dl,nullptr,&descriptorSetLayout_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: descriptor set layout failed.");return false;}
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&descriptorSetLayout_;if(vkCreatePipelineLayout(ctx_.device(),&pl,nullptr,&pipelineLayout_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: pipeline layout failed.");return false;}
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cp.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cp.stage.module=rtaoShader_;cp.stage.pName="main";cp.layout=pipelineLayout_;if(vkCreateComputePipelines(ctx_.device(),VK_NULL_HANDLE,1,&cp,nullptr,&rtaoPipeline_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: compute pipeline failed.");return false;}
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};si.magFilter=VK_FILTER_NEAREST;si.minFilter=VK_FILTER_NEAREST;si.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST;si.addressModeU=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;si.addressModeV=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;si.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;si.maxLod=0.0f;if(vkCreateSampler(ctx_.device(),&si,nullptr,&sampler_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: sampler creation failed.");return false;}
    // +1 combined-image-sampler (binding 11), +1 storage-image (binding 12) per dispatch slot
    VkDescriptorPoolSize ps[5]={{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,dispatchSlots_*(6+kMaxPathTraceTextures)},{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,dispatchSlots_*5},{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,dispatchSlots_},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,dispatchSlots_},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,dispatchSlots_}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dpi.maxSets=dispatchSlots_;dpi.poolSizeCount=5;dpi.pPoolSizes=ps;if(vkCreateDescriptorPool(ctx_.device(),&dpi,nullptr,&descriptorPool_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: descriptor pool creation failed.");return false;}
    return createParamSlots();
}

bool RayTracingScene::createSkinPipeline() {
    if (skinPipeline_ != VK_NULL_HANDLE) return true;
    std::vector<uint32_t> spirv; if(!readSpirv(shaderDirectory_ + "\\skin.spv",spirv) && !readSpirv(shaderDirectory_ + "/skin.spv",spirv)){ctx_.setError("RayTracingScene: missing skin.spv in shader directory.");return false;}
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; sm.codeSize=spirv.size()*4; sm.pCode=spirv.data(); if(vkCreateShaderModule(ctx_.device(),&sm,nullptr,&skinShader_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: vkCreateShaderModule(skin) failed.");return false;}
    VkDescriptorSetLayoutBinding b[4]{};
    b[0]={0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[1]={1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[2]={2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[3]={3,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};dl.bindingCount=4;dl.pBindings=b;if(vkCreateDescriptorSetLayout(ctx_.device(),&dl,nullptr,&descriptorSetLayoutSkin_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: skin descriptor set layout failed.");return false;}
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&descriptorSetLayoutSkin_;if(vkCreatePipelineLayout(ctx_.device(),&pl,nullptr,&pipelineLayoutSkin_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: skin pipeline layout failed.");return false;}
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cp.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cp.stage.module=skinShader_;cp.stage.pName="main";cp.layout=pipelineLayoutSkin_;if(vkCreateComputePipelines(ctx_.device(),VK_NULL_HANDLE,1,&cp,nullptr,&skinPipeline_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: skin compute pipeline failed.");return false;}
    const uint32_t kMaxSkinDescriptorSets = 4096;
    VkDescriptorPoolSize ps[2]={{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,kMaxSkinDescriptorSets*3},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,kMaxSkinDescriptorSets}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dpi.maxSets=kMaxSkinDescriptorSets;dpi.poolSizeCount=2;dpi.pPoolSizes=ps;if(vkCreateDescriptorPool(ctx_.device(),&dpi,nullptr,&descriptorPoolSkin_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: skin descriptor pool creation failed.");return false;}
    return true;
}


// once per map
bool RayTracingScene::setWaterSurfaces(const float* rects, uint32_t count, uint64_t flatWaterChunkId, uint32_t waterTopTris,
                                       uint64_t flatLavaChunkId, uint32_t lavaTopTris) {
    vkDeviceWaitIdle(ctx_.device());
    destroyBuffer(waterRectBuffer_);
    destroyBuffer(waterBucketBuffer_);
    destroyBuffer(waterItemBuffer_);
    waterRectCount_ = 0;
    waterFlatChunkId_ = flatWaterChunkId;
    lavaFlatChunkId_ = flatLavaChunkId;
    waterTopTris_ = waterTopTris;
    lavaTopTris_ = lavaTopTris;
    waveMapValid_ = false;  // rebaked for this map's water (recordWaveMapBake)
    for (auto& g : waterPatchGens_) g.built = false;
    if (!rects || count == 0) return true;

    float minX = 1e30f, minY = 1e30f, maxX = -1e30f, maxY = -1e30f, minZ = 1e30f;
    for (uint32_t i = 0; i < count; ++i) {
        const float* r = rects + size_t(i) * 8;
        minX = std::min(minX, r[0]); minY = std::min(minY, r[1]);
        maxX = std::max(maxX, r[2]); maxY = std::max(maxY, r[3]);
        minZ = std::min(minZ, r[4]);
    }
    // wave map over the water's bounds
    waveMapTexel_ = std::max(4.0f, std::ceil(std::max(maxX - minX, maxY - minY) / 2046.0f));
    waveMapOriginX_ = minX - waveMapTexel_; waveMapOriginY_ = minY - waveMapTexel_;
    waveMapW_ = static_cast<uint32_t>(std::ceil((maxX - minX) / waveMapTexel_)) + 2;
    waveMapH_ = static_cast<uint32_t>(std::ceil((maxY - minY) / waveMapTexel_)) + 2;
    waveMapNeedsImage_ = true;
    waterBucketSize_ = 64.0f;
    waterBucketOriginX_ = minX; waterBucketOriginY_ = minY; waterMinZ_ = minZ;
    waterBucketsX_ = std::max(1u, static_cast<uint32_t>(std::ceil((maxX - minX) / waterBucketSize_)));
    waterBucketsY_ = std::max(1u, static_cast<uint32_t>(std::ceil((maxY - minY) / waterBucketSize_)));
    std::vector<std::vector<uint32_t>> lists(size_t(waterBucketsX_) * waterBucketsY_);
    for (uint32_t i = 0; i < count; ++i) {
        const float* r = rects + size_t(i) * 8;
        const int bx0 = std::max(0, int((r[0] - minX) / waterBucketSize_)), by0 = std::max(0, int((r[1] - minY) / waterBucketSize_));
        const int bx1 = std::min(int(waterBucketsX_) - 1, int((r[2] - minX) / waterBucketSize_));
        const int by1 = std::min(int(waterBucketsY_) - 1, int((r[3] - minY) / waterBucketSize_));
        for (int by = by0; by <= by1; ++by) for (int bx = bx0; bx <= bx1; ++bx) lists[size_t(by) * waterBucketsX_ + bx].push_back(i);
    }
    std::vector<float> rectData(size_t(count) * 8, 0.0f);
    for (uint32_t i = 0; i < count; ++i) {
        const float* r = rects + size_t(i) * 8;
        float* o = &rectData[size_t(i) * 8];
        o[0] = r[0]; o[1] = r[1]; o[2] = r[2]; o[3] = r[3];
        o[4] = r[4];
        const uint32_t mat = static_cast<uint32_t>(r[5]);
        std::memcpy(&o[5], &mat, sizeof(mat));  // read back with floatBitsToUint
        o[6] = r[6]; o[7] = r[7];  // kind (0 water, 1 lava), lava glow
    }
    std::vector<uint32_t> heads(lists.size() * 2), items;
    for (size_t b = 0; b < lists.size(); ++b) {
        heads[b * 2] = static_cast<uint32_t>(items.size());
        heads[b * 2 + 1] = static_cast<uint32_t>(lists[b].size());
        items.insert(items.end(), lists[b].begin(), lists[b].end());
    }
    if (items.empty()) items.push_back(0);
    auto upload = [&](Buffer& buf, const void* data, VkDeviceSize bytes) -> bool {
        if (!createBuffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buf)) return false;
        void* m = nullptr;
        if (vkMapMemory(ctx_.device(), buf.memory, 0, bytes, 0, &m) != VK_SUCCESS) return false;
        std::memcpy(m, data, static_cast<size_t>(bytes));
        vkUnmapMemory(ctx_.device(), buf.memory);
        return true;
    };
    if (!upload(waterRectBuffer_, rectData.data(), rectData.size() * sizeof(float)) ||
        !upload(waterBucketBuffer_, heads.data(), heads.size() * sizeof(uint32_t)) ||
        !upload(waterItemBuffer_, items.data(), items.size() * sizeof(uint32_t))) {
        ctx_.setError("RayTracingScene::setWaterSurfaces: buffer allocation failed.");
        return false;
    }
    waterRectCount_ = count;
    return true;
}

bool RayTracingScene::createWaterPatchPipeline() {
    if (waterPatchPipeline_ != VK_NULL_HANDLE) return true;
    if (waterPatchPipelineFailed_) return false;
    waterPatchPipelineFailed_ = true;  // cleared on success
    std::vector<uint32_t> spirv;
    if (!readSpirv(shaderDirectory_ + "\\water_patch.spv", spirv) && !readSpirv(shaderDirectory_ + "/water_patch.spv", spirv)) {
        ctx_.setError("RayTracingScene: missing water_patch.spv in shader directory."); return false;
    }
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; sm.codeSize = spirv.size() * 4; sm.pCode = spirv.data();
    if (vkCreateShaderModule(ctx_.device(), &sm, nullptr, &waterPatchShader_) != VK_SUCCESS) { ctx_.setError("RayTracingScene: water_patch shader module failed."); return false; }
    VkDescriptorSetLayoutBinding b[7]{};
    for (uint32_t i = 0; i < 4; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[4] = {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[5] = {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[6] = {6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dl.bindingCount = 7; dl.pBindings = b;
    if (vkCreateDescriptorSetLayout(ctx_.device(), &dl, nullptr, &descriptorSetLayoutWaterPatch_) != VK_SUCCESS) { ctx_.setError("RayTracingScene: water_patch set layout failed."); return false; }
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pl.setLayoutCount = 1; pl.pSetLayouts = &descriptorSetLayoutWaterPatch_;
    if (vkCreatePipelineLayout(ctx_.device(), &pl, nullptr, &pipelineLayoutWaterPatch_) != VK_SUCCESS) { ctx_.setError("RayTracingScene: water_patch pipeline layout failed."); return false; }
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cp.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cp.stage.module = waterPatchShader_; cp.stage.pName = "main"; cp.layout = pipelineLayoutWaterPatch_;
    if (vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE, 1, &cp, nullptr, &waterPatchPipeline_) != VK_SUCCESS) { ctx_.setError("RayTracingScene: water_patch pipeline failed."); return false; }
    const uint32_t sets = std::max(1u, frameSlots_);
    VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sets * 4}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sets},
                                  {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sets * 2}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = sets; dpi.poolSizeCount = 3; dpi.pPoolSizes = ps;
    if (vkCreateDescriptorPool(ctx_.device(), &dpi, nullptr, &descriptorPoolWaterPatch_) != VK_SUCCESS) { ctx_.setError("RayTracingScene: water_patch descriptor pool failed."); return false; }
    waterPatchPipelineFailed_ = false;
    return true;
}

// one frame's patch for this generation
bool RayTracingScene::recordWaterPatch(VkCommandBuffer cmd, WaterPatchGen& gen) {
    gen.active = false;
    const WaterPatchSettings& ws = waterPatchSettings_;
    if (!createWaterPatchPipeline()) return false;

    float cell = 1.0f;
    while (cell * 2.0f <= std::max(1.0f, std::min(ws.cellSize, 8.0f))) cell *= 2.0f;
    const float snap = 32.0f;
    const float halfSize = std::max(16.0f, ws.range);
    uint32_t cells = static_cast<uint32_t>(std::ceil((2.0f * halfSize + snap) / cell));
    cells = std::min(cells, 256u);
    const float originX = std::floor((ws.camX - 0.5f * cells * cell) / snap) * snap;
    const float originY = std::floor((ws.camY - 0.5f * cells * cell) / snap) * snap;
    const uint32_t cellCount = cells * cells;

    if (waterPatchIndexCells_ != cells) {
        vkDeviceWaitIdle(ctx_.device());
        destroyBuffer(waterPatchIndexBuffer_);
        std::vector<uint32_t> idx(size_t(cellCount) * 6);
        for (uint32_t c = 0; c < cellCount; ++c) {
            const uint32_t v = c * 4;
            const uint32_t t[6] = { v, v + 1, v + 2, v, v + 2, v + 3 };
            std::memcpy(&idx[size_t(c) * 6], t, sizeof(t));
        }
        const VkDeviceSize bytes = idx.size() * sizeof(uint32_t);
        if (!createBuffer(bytes, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, waterPatchIndexBuffer_, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
            ctx_.setError("RayTracingScene: water patch index buffer allocation failed."); return false;
        }
        void* m = nullptr;
        if (vkMapMemory(ctx_.device(), waterPatchIndexBuffer_.memory, 0, bytes, 0, &m) != VK_SUCCESS) return false;
        std::memcpy(m, idx.data(), static_cast<size_t>(bytes));
        vkUnmapMemory(ctx_.device(), waterPatchIndexBuffer_.memory);
        waterPatchIndexCells_ = cells;
        for (auto& g : waterPatchGens_) g.built = false;
    }

    const uint32_t vertexCount = cellCount * 4;
    const VkDeviceSize vbSize = VkDeviceSize(vertexCount) * sizeof(ChunkRtVertex);
    const bool vertexGrew = gen.vertex.buffer == VK_NULL_HANDLE || gen.vertex.size < vbSize;
    if (!ensureBufferCapacity(gen.vertex, vbSize,
                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) ||
        !ensureBufferCapacity(gen.params, 128, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        ctx_.setError("RayTracingScene: water patch buffer allocation failed."); return false;
    }
    if (vertexGrew) gen.built = false;

    // parameters, matches water_patch.comp's PatchParams (6 x vec4, std140)
    float fft[4], map[4], amp[4];
    waveShaderParams(gen, fft, map, amp);
    const float params[32] = {
        originX, originY, cell, float(cells),
        waterBucketOriginX_, waterBucketOriginY_, waterBucketSize_, float(waterBucketsX_),
        float(waterBucketsY_), ws.fadeWidth, ws.height, waterMinZ_,
        fft[0], fft[1], fft[2], fft[3],
        map[0], map[1], map[2], map[3],
        amp[0], amp[1], amp[2], amp[3],
        ws.lava0[0], ws.lava0[1], ws.lava0[2], ws.lava0[3],  // displaced lava (rt_lava.glsl)
        ws.lava1[0], ws.lava1[1], ws.lava1[2], ws.lava1[3] };
    void* m = nullptr;
    if (vkMapMemory(ctx_.device(), gen.params.memory, 0, sizeof(params), 0, &m) != VK_SUCCESS) return false;
    std::memcpy(m, params, sizeof(params));
    vkUnmapMemory(ctx_.device(), gen.params.memory);

    if (gen.set == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPoolWaterPatch_; ai.descriptorSetCount = 1; ai.pSetLayouts = &descriptorSetLayoutWaterPatch_;
        if (vkAllocateDescriptorSets(ctx_.device(), &ai, &gen.set) != VK_SUCCESS) { ctx_.setError("RayTracingScene: water patch descriptor set allocation failed."); return false; }
    }
    // rewritten every frame (cheap), buffers can be replaced by a map change or a grow
    VkDescriptorBufferInfo bi[5] = {
        {waterRectBuffer_.buffer, 0, VK_WHOLE_SIZE}, {waterBucketBuffer_.buffer, 0, VK_WHOLE_SIZE},
        {waterItemBuffer_.buffer, 0, VK_WHOLE_SIZE}, {gen.vertex.buffer, 0, VK_WHOLE_SIZE}, {gen.params.buffer, 0, VK_WHOLE_SIZE} };
    VkDescriptorImageInfo ii[2] = {
        {waveFftSampler_, gen.fftView, VK_IMAGE_LAYOUT_GENERAL}, {waveMapSampler_, waveMapView_, VK_IMAGE_LAYOUT_GENERAL} };
    VkWriteDescriptorSet w[7]{};
    for (uint32_t i = 0; i < 7; ++i) {
        w[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = gen.set; w[i].dstBinding = i; w[i].descriptorCount = 1;
        if (i < 5) {
            w[i].descriptorType = i == 4 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &bi[i];
        } else {
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[i].pImageInfo = &ii[i - 5];
        }
    }
    vkUpdateDescriptorSets(ctx_.device(), 7, w, 0, nullptr);

    // vertices
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, waterPatchPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayoutWaterPatch_, 0, 1, &gen.set, 0, nullptr);
    vkCmdDispatch(cmd, (cellCount + 63) / 64, 1, 1);
    VkMemoryBarrier toBuild{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    toBuild.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toBuild.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &toBuild, 0, nullptr, 0, nullptr);

    // BLAS: refit while the patch hasn't moved, full build when it re-centred
    VkAccelerationStructureGeometryTrianglesDataKHR tri{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT; tri.vertexData.deviceAddress = gen.vertex.address; tri.vertexStride = sizeof(ChunkRtVertex);
    tri.maxVertex = vertexCount - 1; tri.indexType = VK_INDEX_TYPE_UINT32; tri.indexData.deviceAddress = waterPatchIndexBuffer_.address;
    VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR; geom.geometry.triangles = tri;
    geom.flags = 0;  // non-opaque
    VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    info.geometryCount = 1; info.pGeometries = &geom;
    const uint32_t primCount = cellCount * 2;
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR_(ctx_.device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &primCount, &sizes);
    const bool blasGrow = gen.blasBuffer.buffer == VK_NULL_HANDLE || gen.blasBuffer.size < sizes.accelerationStructureSize;
    if (!ensureBufferCapacity(gen.blasBuffer, sizes.accelerationStructureSize,
                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) ||
        !ensureBufferCapacity(gen.scratch, std::max(sizes.buildScratchSize, sizes.updateScratchSize),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
        ctx_.setError("RayTracingScene: water patch BLAS allocation failed."); return false;
    }
    if (blasGrow && gen.blas != VK_NULL_HANDLE) { vkDestroyAccelerationStructureKHR_(ctx_.device(), gen.blas, nullptr); gen.blas = VK_NULL_HANDLE; }
    if (gen.blas == VK_NULL_HANDLE) {
        VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        aci.buffer = gen.blasBuffer.buffer; aci.size = sizes.accelerationStructureSize; aci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        if (vkCreateAccelerationStructureKHR_(ctx_.device(), &aci, nullptr, &gen.blas) != VK_SUCCESS) { ctx_.setError("RayTracingScene: water patch BLAS creation failed."); return false; }
        gen.built = false;
    }
    const bool refit = gen.built && gen.cells == cells && gen.cellSize == cell && gen.originX == originX && gen.originY == originY && !forceAccelRebuild_;
    info.mode = refit ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.srcAccelerationStructure = refit ? gen.blas : VK_NULL_HANDLE;
    info.dstAccelerationStructure = gen.blas;
    info.scratchData.deviceAddress = gen.scratch.address;
    VkAccelerationStructureBuildRangeInfoKHR range{}; range.primitiveCount = primCount;
    const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;
    vkCmdBuildAccelerationStructuresKHR_(cmd, 1, &info, &ranges);
    gen.built = true;
    gen.cells = cells; gen.cellSize = cell; gen.originX = originX; gen.originY = originY;
    gen.active = true;
    return true;
}


namespace {
    // records + submits + waits
    template <typename F>
    bool submitOneTime(InteropContext& ctx, F&& record) {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = ctx.setupCommandPool(); cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(ctx.device(), &cai, &cmd) != VK_SUCCESS) return false;
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &cbi);
        record(cmd);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        const bool ok = vkQueueSubmit(ctx.queue(), 1, &si, VK_NULL_HANDLE) == VK_SUCCESS && vkQueueWaitIdle(ctx.queue()) == VK_SUCCESS;
        vkFreeCommandBuffers(ctx.device(), ctx.setupCommandPool(), 1, &cmd);
        return ok;
    }
    void toGeneral(VkCommandBuffer cmd, VkImage image, uint32_t layers) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    }
    bool makeComputePipeline(InteropContext& ctx, const std::string& dir, const char* name,
                             const VkDescriptorSetLayoutBinding* bindings, uint32_t count, uint32_t pushBytes,
                             VkShaderModule& shader, VkDescriptorSetLayout& setLayout, VkPipelineLayout& layout, VkPipeline& pipeline) {
        std::vector<uint32_t> spirv;
        if (!readSpirv(dir + "\\" + name + ".spv", spirv) && !readSpirv(dir + "/" + name + ".spv", spirv)) {
            ctx.setError(std::string("RayTracingScene: missing ") + name + ".spv in shader directory."); return false;
        }
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; sm.codeSize = spirv.size() * 4; sm.pCode = spirv.data();
        if (vkCreateShaderModule(ctx.device(), &sm, nullptr, &shader) != VK_SUCCESS) { ctx.setError(std::string("RayTracingScene: shader module failed: ") + name); return false; }
        VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dl.bindingCount = count; dl.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(ctx.device(), &dl, nullptr, &setLayout) != VK_SUCCESS) { ctx.setError(std::string("RayTracingScene: set layout failed: ") + name); return false; }
        VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pl.setLayoutCount = 1; pl.pSetLayouts = &setLayout;
        if (pushBytes) { pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &pr; }
        if (vkCreatePipelineLayout(ctx.device(), &pl, nullptr, &layout) != VK_SUCCESS) { ctx.setError(std::string("RayTracingScene: pipeline layout failed: ") + name); return false; }
        VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cp.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cp.stage.module = shader; cp.stage.pName = "main"; cp.layout = layout;
        if (vkCreateComputePipelines(ctx.device(), VK_NULL_HANDLE, 1, &cp, nullptr, &pipeline) != VK_SUCCESS) { ctx.setError(std::string("RayTracingScene: compute pipeline failed: ") + name); return false; }
        return true;
    }
    void writeUbo(VkDevice device, VkDeviceMemory memory, const float* data, size_t floats) {
        void* m = nullptr;
        if (vkMapMemory(device, memory, 0, floats * sizeof(float), 0, &m) == VK_SUCCESS) {
            std::memcpy(m, data, floats * sizeof(float));
            vkUnmapMemory(device, memory);
        }
    }
    void computeBarrier(VkCommandBuffer cmd, VkPipelineStageFlags dstStages, VkAccessFlags dstAccess) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = dstAccess;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, dstStages, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
}  // namespace

bool RayTracingScene::createWaveImage(uint32_t w, uint32_t h, uint32_t layers, VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D; ii.format = VK_FORMAT_R16G16B16A16_SFLOAT; ii.extent = {w, h, 1};
    ii.mipLevels = 1; ii.arrayLayers = layers; ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;  // TRANSFER_SRC: ptwavestats readback
    if (vkCreateImage(ctx_.device(), &ii, nullptr, &image) != VK_SUCCESS) return false;
    VkMemoryRequirements req{}; vkGetImageMemoryRequirements(ctx_.device(), image, &req);
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(ctx_.physicalDevice(), &mp);
    uint32_t mt = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { mt = i; break; }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = req.size; mai.memoryTypeIndex = mt;
    if (mt == UINT32_MAX || vkAllocateMemory(ctx_.device(), &mai, nullptr, &memory) != VK_SUCCESS) return false;
    if (vkBindImageMemory(ctx_.device(), image, memory, 0) != VK_SUCCESS) return false;
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = image; vi.viewType = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D; vi.format = ii.format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
    if (vkCreateImageView(ctx_.device(), &vi, nullptr, &view) != VK_SUCCESS) return false;
    return submitOneTime(ctx_, [&](VkCommandBuffer cmd) { toGeneral(cmd, image, layers); });
}

// everything the waves need, created once (the wave map image is resized per map by
// recreateWaveMapImage)
bool RayTracingScene::ensureWaveResources() {
    if (wavesCreated_) return true;
    if (wavesFailed_) return false;
    wavesFailed_ = true;  // cleared on success
    if (waterPatchGens_.size() != frameSlots_) waterPatchGens_.resize(frameSlots_);
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_LINEAR; si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT; si.maxLod = 0.0f;
    if (vkCreateSampler(ctx_.device(), &si, nullptr, &waveFftSampler_) != VK_SUCCESS) { ctx_.setError("RayTracingScene: wave sampler failed."); return false; }
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(ctx_.device(), &si, nullptr, &waveMapSampler_) != VK_SUCCESS) { ctx_.setError("RayTracingScene: wave map sampler failed."); return false; }

    const VkShaderStageFlags cs = VK_SHADER_STAGE_COMPUTE_BIT;
    const VkDescriptorSetLayoutBinding spectrumB[2] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr}, {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, cs, nullptr} };
    const VkDescriptorSetLayoutBinding fftB[4] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, cs, nullptr}, {3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, cs, nullptr} };
    const VkDescriptorSetLayoutBinding mapB[6] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr}, {3, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, cs, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, cs, nullptr}, {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, cs, nullptr} };
    if (!makeComputePipeline(ctx_, shaderDirectory_, "water_spectrum", spectrumB, 2, 0, waveSpectrumShader_, waveSpectrumSetLayout_, waveSpectrumLayout_, waveSpectrumPipeline_) ||
        !makeComputePipeline(ctx_, shaderDirectory_, "water_fft", fftB, 4, 4, waveFftShader_, waveFftSetLayout_, waveFftLayout_, waveFftPipeline_) ||
        !makeComputePipeline(ctx_, shaderDirectory_, "water_wavemap", mapB, 6, 0, waveMapShader_, waveMapSetLayout_, waveMapLayout_, waveMapPipeline_))
        return false;
    const uint32_t gens = std::max(1u, frameSlots_);
    VkDescriptorPoolSize ps[4] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 + gens * 2 + 3}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 + gens + 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, gens + 1}, {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1} };
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = gens + 2; dpi.poolSizeCount = 4; dpi.pPoolSizes = ps;
    if (vkCreateDescriptorPool(ctx_.device(), &dpi, nullptr, &descriptorPoolWave_) != VK_SUCCESS) { ctx_.setError("RayTracingScene: wave descriptor pool failed."); return false; }
    auto allocSet = [&](VkDescriptorSetLayout layout, VkDescriptorSet& set) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPoolWave_; ai.descriptorSetCount = 1; ai.pSetLayouts = &layout;
        return vkAllocateDescriptorSets(ctx_.device(), &ai, &set) == VK_SUCCESS;
    };
    const VkMemoryPropertyFlags hostVis = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkDeviceSize spectrumBytes = VkDeviceSize(kWaveN) * kWaveN * 3 * 16;
    if (!createBuffer(spectrumBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, waveSpectrum_) ||
        !createBuffer(64, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hostVis, waveSpectrumParams_) ||
        !createBuffer(96, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hostVis, waveMapParams_) ||
        !allocSet(waveSpectrumSetLayout_, waveSpectrumSet_) || !allocSet(waveMapSetLayout_, waveMapSet_)) {
        ctx_.setError("RayTracingScene: wave buffers failed."); return false;
    }
    {
        VkDescriptorBufferInfo b0{waveSpectrum_.buffer, 0, VK_WHOLE_SIZE}, b1{waveSpectrumParams_.buffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w[2]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
        w[0].dstSet = waveSpectrumSet_; w[0].dstBinding = 0; w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &b0;
        w[1].dstSet = waveSpectrumSet_; w[1].dstBinding = 1; w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[1].pBufferInfo = &b1;
        vkUpdateDescriptorSets(ctx_.device(), 2, w, 0, nullptr);
    }
    const VkDeviceSize tempBytes = VkDeviceSize(kWaveN) * kWaveN * 3 * kWaveFftChannels * 8;
    for (WaterPatchGen& g : waterPatchGens_) {
        if (!createWaveImage(kWaveN, kWaveN, kWaveFftLayers, g.fftImage, g.fftMemory, g.fftView) ||
            !createBuffer(tempBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, g.fftTemp) ||
            !createBuffer(32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hostVis, g.fftParams) ||
            !allocSet(waveFftSetLayout_, g.fftSet)) {
            ctx_.setError("RayTracingScene: wave FFT resources failed."); return false;
        }
        VkDescriptorBufferInfo b0{waveSpectrum_.buffer, 0, VK_WHOLE_SIZE}, b1{g.fftTemp.buffer, 0, VK_WHOLE_SIZE}, b3{g.fftParams.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorImageInfo i2{VK_NULL_HANDLE, g.fftView, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w[4]{};
        for (uint32_t i = 0; i < 4; ++i) { w[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[i].dstSet = g.fftSet; w[i].dstBinding = i; w[i].descriptorCount = 1; }
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &b0;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &b1;
        w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[2].pImageInfo = &i2;
        w[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[3].pBufferInfo = &b3;
        vkUpdateDescriptorSets(ctx_.device(), 4, w, 0, nullptr);
    }
    if (!createWaveImage(1, 1, 1, waveMapImage_, waveMapMemory_, waveMapView_)) { ctx_.setError("RayTracingScene: wave map image failed."); return false; }
    wavesCreated_ = true;
    wavesFailed_ = false;
    return true;
}

bool RayTracingScene::recreateWaveMapImage() {
    vkDeviceWaitIdle(ctx_.device());
    if (waveMapView_ != VK_NULL_HANDLE) vkDestroyImageView(ctx_.device(), waveMapView_, nullptr);
    if (waveMapImage_ != VK_NULL_HANDLE) vkDestroyImage(ctx_.device(), waveMapImage_, nullptr);
    if (waveMapMemory_ != VK_NULL_HANDLE) vkFreeMemory(ctx_.device(), waveMapMemory_, nullptr);
    waveMapView_ = VK_NULL_HANDLE; waveMapImage_ = VK_NULL_HANDLE; waveMapMemory_ = VK_NULL_HANDLE;
    waveMapNeedsImage_ = false;
    return createWaveImage(waveMapW_, waveMapH_, 1, waveMapImage_, waveMapMemory_, waveMapView_);
}

// cascade tile sizes (world units)
void RayTracingScene::waveCascadeSizes(float out[3]) const {
    const float L0 = std::max(4.0f, waterPatchSettings_.sizeMeters) * kUnitsPerMetre;
    out[0] = L0; out[1] = L0 / 3.71f; out[2] = L0 / 13.93f;
}
void RayTracingScene::waveBands(float out[3]) const {
    float L[3]; waveCascadeSizes(L);
    const float twoPi = 6.2831853f;
    out[0] = twoPi * 6.0f / L[1];
    out[1] = twoPi * 6.0f / L[2];
    out[2] = twoPi / std::max(1.0f, 4.0f * L[2] / float(kWaveN));
}

void RayTracingScene::waveShaderParams(const WaterPatchGen& gen, float fft[4], float map[4], float amp[4]) const {
    const WaterPatchSettings& ws = waterPatchSettings_;
    waveCascadeSizes(fft);
    fft[3] = std::max(0.0f, ws.chop);
    const bool useMap = ws.depthAware && waveMapValid_ && !waveMapNeedsImage_;
    map[0] = waveMapOriginX_; map[1] = waveMapOriginY_;
    map[2] = useMap ? 1.0f / (float(waveMapW_) * waveMapTexel_) : 0.0f;
    map[3] = useMap ? 1.0f / (float(waveMapH_) * waveMapTexel_) : 0.0f;
    amp[0] = ws.strength; amp[1] = ws.detail; amp[2] = 0.0f; amp[3] = gen.fftValid ? 1.0f : 0.0f;
}

// this frame's sea into gen's image
bool RayTracingScene::recordWaveFft(VkCommandBuffer cmd, WaterPatchGen& gen) {
    const WaterPatchSettings& ws = waterPatchSettings_;
    const float g = 9.81f * kUnitsPerMetre;
    float L[3]; waveCascadeSizes(L);
    const float key[6] = { ws.windSpeed, ws.windDir, ws.fetch, ws.spread, ws.sizeMeters, 0.0f };
    if (std::memcmp(key, waveSpectrumKey_, sizeof(key)) != 0) {
        vkDeviceWaitIdle(ctx_.device());
        std::memcpy(waveSpectrumKey_, key, sizeof(key));
        float bands[3]; waveBands(bands);
        const float s = std::max(0.5f, ws.spread);
        const float norm = std::exp(std::lgamma(s + 1.0f) - std::lgamma(s + 0.5f)) / (2.0f * std::sqrt(3.14159265f));
        const float dir = ws.windDir * 3.14159265f / 180.0f;
        const float params[16] = {
            float(kWaveN), g, std::max(0.5f, ws.windSpeed) * kUnitsPerMetre, std::max(10.0f, ws.fetch) * kUnitsPerMetre,
            std::cos(dir), std::sin(dir), norm, s,
            L[0], L[1], L[2], 1337.0f,
            bands[0], bands[1], bands[2], 0.0f };
        writeUbo(ctx_.device(), waveSpectrumParams_.memory, params, 16);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, waveSpectrumPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, waveSpectrumLayout_, 0, 1, &waveSpectrumSet_, 0, nullptr);
        vkCmdDispatch(cmd, kWaveN / 8, kWaveN / 8, 3);
        computeBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    const double period = 225.0;  // divides the scroll clock's 1 h wrap
    const float tWrapped = static_cast<float>(std::fmod(double(ws.time) * double(std::max(0.0f, ws.speed)), period));
    const float fftParams[8] = { g, tWrapped, static_cast<float>(6.283185307 / period), 0.0f, L[0], L[1], L[2], 0.0f };
    writeUbo(ctx_.device(), gen.fftParams.memory, fftParams, 8);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, waveFftPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, waveFftLayout_, 0, 1, &gen.fftSet, 0, nullptr);
    uint32_t pass = 0;
    vkCmdPushConstants(cmd, waveFftLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &pass);
    vkCmdDispatch(cmd, kWaveN, 3, 1);
    computeBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    pass = 1;
    vkCmdPushConstants(cmd, waveFftLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &pass);
    vkCmdDispatch(cmd, kWaveN, 3, 1);
    computeBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    return true;
}

// depth + fetch wave map (water_wavemap.comp), only when the map's water or a wave setting changed
bool RayTracingScene::recordWaveMapBake(VkCommandBuffer cmd, VkAccelerationStructureKHR tlas) {
    const WaterPatchSettings& ws = waterPatchSettings_;
    if (waveMapStatsPending_) { waveMapStatsPending_ = false; logWaveMapStats(); }  // the last bake has been submitted by now
    if (!ws.depthAware) { waveMapValid_ = false; waveMapKey_[0] = -1.0f; return true; }
    const float key[5] = { ws.windSpeed, ws.windDir, ws.fetch, ws.sizeMeters, 0.0f };
    if (waveMapValid_ && std::memcmp(key, waveMapKey_, sizeof(key)) == 0) return true;
    if (tlas == VK_NULL_HANDLE || waterRectCount_ == 0 || waveMapW_ <= 1) return true;
    vkDeviceWaitIdle(ctx_.device());
    std::memcpy(waveMapKey_, key, sizeof(key));
    float bands[3]; waveBands(bands);
    float L[3]; waveCascadeSizes(L);
    auto flat = chunkMeshes_.find(waterFlatChunkId_);
    auto flatLava = chunkMeshes_.find(lavaFlatChunkId_);
    const float dir = ws.windDir * 3.14159265f / 180.0f;
    const float params[24] = {
        waveMapOriginX_, waveMapOriginY_, waveMapTexel_, 0.0f,
        waterBucketOriginX_, waterBucketOriginY_, waterBucketSize_, float(waterBucketsX_),
        float(waterBucketsY_), flat != chunkMeshes_.end() ? float(flat->second.materialSlot) : -1.0f, float(waterPatchMaterialSlot_),
        flatLava != chunkMeshes_.end() ? float(flatLava->second.materialSlot) : -1.0f,
        std::cos(dir), std::sin(dir), std::max(0.5f, ws.windSpeed) * kUnitsPerMetre, std::max(10.0f, ws.fetch) * kUnitsPerMetre,
        bands[0], bands[1], bands[2], 9.81f * kUnitsPerMetre,
        L[0], std::max(0.0f, ws.worldSize), 0.0f, 0.0f };
    writeUbo(ctx_.device(), waveMapParams_.memory, params, 24);
    VkDescriptorBufferInfo b0{waterRectBuffer_.buffer, 0, VK_WHOLE_SIZE}, b1{waterBucketBuffer_.buffer, 0, VK_WHOLE_SIZE},
                           b2{waterItemBuffer_.buffer, 0, VK_WHOLE_SIZE}, b5{waveMapParams_.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo i4{VK_NULL_HANDLE, waveMapView_, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSetAccelerationStructureKHR as{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    as.accelerationStructureCount = 1; as.pAccelerationStructures = &tlas;
    VkWriteDescriptorSet w[6]{};
    for (uint32_t i = 0; i < 6; ++i) { w[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[i].dstSet = waveMapSet_; w[i].dstBinding = i; w[i].descriptorCount = 1; }
    w[0].descriptorType = w[1].descriptorType = w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[0].pBufferInfo = &b0; w[1].pBufferInfo = &b1; w[2].pBufferInfo = &b2;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; w[3].pNext = &as;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[4].pImageInfo = &i4;
    w[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[5].pBufferInfo = &b5;
    vkUpdateDescriptorSets(ctx_.device(), 6, w, 0, nullptr);
    VkMemoryBarrier asRead{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    asRead.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR; asRead.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &asRead, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, waveMapPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, waveMapLayout_, 0, 1, &waveMapSet_, 0, nullptr);
    vkCmdDispatch(cmd, (waveMapW_ + 7) / 8, (waveMapH_ + 7) / 8, 1);
    waveMapValid_ = true;
    waveMapStatsPending_ = ws.mapStats;
    std::cerr << "[RayTracingScene] water wave map baked: " << waveMapW_ << "x" << waveMapH_ << " at " << waveMapTexel_ << " units/texel" << std::endl;
    return true;
}

void RayTracingScene::logWaveMapStats() {
    if (waveMapImage_ == VK_NULL_HANDLE || waveMapW_ <= 1 || waveMapH_ <= 1) return;
    vkDeviceWaitIdle(ctx_.device());
    const uint32_t W = waveMapW_, H = waveMapH_;
    const VkDeviceSize bytes = VkDeviceSize(W) * H * 8;  // rgba16f
    Buffer rb;
    if (!createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, rb)) return;
    VkImage img = waveMapImage_;
    const bool copied = submitOneTime(ctx_, [&](VkCommandBuffer c) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {W, H, 1};
        vkCmdCopyImageToBuffer(c, img, VK_IMAGE_LAYOUT_GENERAL, rb.buffer, 1, &r);
    });
    void* m = nullptr;
    if (copied && vkMapMemory(ctx_.device(), rb.memory, 0, bytes, 0, &m) == VK_SUCCESS) {
        auto halfToFloat = [](uint16_t h) {
            const uint32_t e = (h >> 10) & 31, f = h & 1023;
            const float v = e == 0 ? std::ldexp(float(f), -24) : (e == 31 ? 65504.0f : std::ldexp(float(f | 1024), int(e) - 25));
            return (h >> 15) ? -v : v;
        };
        const uint16_t* px = static_cast<const uint16_t*>(m);
        std::vector<float> c[4];
        for (size_t i = 0; i < size_t(W) * H; ++i) {
            float v[4];
            for (int k = 0; k < 4; ++k) v[k] = halfToFloat(px[i * 4 + k]);
            if ((v[0] == 0.0f && v[1] == 0.0f && v[2] == 0.0f && v[3] == 0.0f) || v[0] < 0.0f) continue;
            for (int k = 0; k < 4; ++k) c[k].push_back(v[k]);
        }
        auto pc = [](std::vector<float>& v, float q) {
            if (v.empty()) return 0.0f;
            std::sort(v.begin(), v.end());
            return v[std::min(v.size() - 1, size_t(q * float(v.size())))];
        };
        char line[512];
        snprintf(line, sizeof(line), "[RayTracingScene] ptwavestats: %zu water texels | depth (m) p10 %.1f p50 %.1f p90 %.1f | wave weights p10/p50/p90 -- large %.2f/%.2f/%.2f  medium %.2f/%.2f/%.2f  small %.2f/%.2f/%.2f",
            c[3].size(), pc(c[3], 0.1f) / kUnitsPerMetre, pc(c[3], 0.5f) / kUnitsPerMetre, pc(c[3], 0.9f) / kUnitsPerMetre,
            pc(c[0], 0.1f), pc(c[0], 0.5f), pc(c[0], 0.9f), pc(c[1], 0.1f), pc(c[1], 0.5f), pc(c[1], 0.9f), pc(c[2], 0.1f), pc(c[2], 0.5f), pc(c[2], 0.9f));
        std::cerr << line << std::endl;
        vkUnmapMemory(ctx_.device(), rb.memory);
    }
    destroyBuffer(rb);
}

bool RayTracingScene::ensureCausticCapacity(uint32_t photonsPerBatch) {
    if (photonsPerBatch <= causticCapacity_ || causticPhotonBuffers_.empty()) return true;
    vkDeviceWaitIdle(ctx_.device());
    Buffer grown;
    if (!createBuffer(16 * VkDeviceSize(kCausticBatches) + VkDeviceSize(kCausticBatches) * photonsPerBatch * 48,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, grown)) {
        std::cerr << "[RayTracingScene] caustics: couldn't allocate " << photonsPerBatch / 1024 << "k photons a batch ("
                  << (VkDeviceSize(kCausticBatches) * photonsPerBatch * 48) / (1024 * 1024) << " MB) -- staying at " << causticCapacity_ / 1024 << "k" << std::endl;
        return false;
    }
    Buffer grownHash;  // the hash heads grow with it (causticHashSizeFor)
    if (!createBuffer(VkDeviceSize(kCausticBatches) * causticHashSizeFor(photonsPerBatch) * 4,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, grownHash)) {
        destroyBuffer(grown);
        std::cerr << "[RayTracingScene] caustics: couldn't allocate the photon hash for " << photonsPerBatch / 1024 << "k photons a batch" << std::endl;
        return false;
    }
    destroyBuffer(causticPhotonBuffers_[0]);
    causticPhotonBuffers_[0] = grown;
    destroyBuffer(causticHashBuffers_[0]);
    causticHashBuffers_[0] = grownHash;
    causticCapacity_ = photonsPerBatch;
    std::cerr << "[RayTracingScene] caustics: photon buffer grown to " << photonsPerBatch / 1024 << "k a batch ("
              << (VkDeviceSize(kCausticBatches) * photonsPerBatch * 48) / (1024 * 1024) << " MB)" << std::endl;
    return true;
}

bool RayTracingScene::createParamSlots(){
    paramSlots_.resize(dispatchSlots_); for(auto& s:paramSlots_){ Buffer b; if(!createBuffer(sizeof(GpuParams),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,b)){ctx_.setError("RayTracingScene: parameter buffer allocation failed.");return false;} s.buffer=b.buffer;s.memory=b.memory;vkMapMemory(ctx_.device(),s.memory,0,sizeof(GpuParams),0,&s.mapped);b.buffer=VK_NULL_HANDLE;b.memory=VK_NULL_HANDLE; }
    return true;
}

bool RayTracingScene::createDescriptorSets(uint32_t count,std::vector<VkDescriptorSet>& sets,std::string* errorOut){
    if(count>dispatchSlots_){if(errorOut)*errorOut="requested descriptor-set count exceeds scene dispatch slot count";return false;} std::vector<VkDescriptorSetLayout> layouts(count,descriptorSetLayout_);VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};ai.descriptorPool=descriptorPool_;ai.descriptorSetCount=count;ai.pSetLayouts=layouts.data();if(vkAllocateDescriptorSets(ctx_.device(),&ai,sets.data())!=VK_SUCCESS){if(errorOut)*errorOut="vkAllocateDescriptorSets for RTAO failed";return false;}return true;
}

bool RayTracingScene::updateDescriptorSet(VkDescriptorSet set, SharedTexture& depth, SharedTexture& normal,
                                           SharedTexture& motion, SharedTexture& output,
                                           SharedTexture& historyPrev, SharedTexture& historyNext,
                                           SharedTexture& shadowOutput, SharedTexture& shadowHistoryPrev, SharedTexture& shadowHistoryNext,
                                           SharedTexture& momentsHistoryPrev, SharedTexture& momentsHistoryNext,
                                           VkImageView depthView, VkImageView normalView, VkImageView motionView,
                                           VkImageView outputView, VkImageView historyPrevView, VkImageView historyNextView,
                                           VkImageView shadowOutputView, VkImageView shadowHistoryPrevView, VkImageView shadowHistoryNextView,
                                           VkImageView momentsHistoryPrevView, VkImageView momentsHistoryNextView,
                                           uint32_t slotIndex, const RtaoConstants& c, std::string* errorOut) {
    if (slotIndex >= paramSlots_.size()) {
        if (errorOut) *errorOut = "invalid RTAO frame slot";
        return false;
    }
    GpuParams p{};
    std::memcpy(p.invViewProj, c.invViewProj, sizeof(p.invViewProj));
    std::memcpy(p.invView, c.invView, sizeof(p.invView));
    std::memcpy(p.currentViewProjUnjittered, c.currentViewProjUnjittered, sizeof(p.currentViewProjUnjittered));
    std::memcpy(p.prevViewProjJittered, c.prevViewProjJittered, sizeof(p.prevViewProjJittered));
    std::memcpy(p.prevViewProjUnjittered, c.prevViewProjUnjittered, sizeof(p.prevViewProjUnjittered));
    p.settings[0] = c.radius;
    p.settings[1] = c.intensity;
    p.settings[2] = c.bias;
    p.settings[3] = c.historyValid ? 1.0f : 0.0f;
    p.frameData[0] = c.frameIndex;
    p.frameData[1] = output.width();
    p.frameData[2] = output.height();
    p.frameData[3] = c.raysPerPixel;
    p.shadowLightAndRadius[0] = c.lightDirection[0];
    p.shadowLightAndRadius[1] = c.lightDirection[1];
    p.shadowLightAndRadius[2] = c.lightDirection[2];
    p.shadowLightAndRadius[3] = c.lightRadius;
    p.shadowData[0] = c.shadowRaysPerPixel;
    p.shadowData[1] = c.shadowEnabled ? 1u : 0u;
    // shared scene data (bindings 13/14)
    const Buffer* sharedMaterials = nullptr;
    if (!dynamicGenerations_.empty() && dynamicGenerations_[dynamicGenerationIndex_].materials.buffer != VK_NULL_HANDLE)
        sharedMaterials = &dynamicGenerations_[dynamicGenerationIndex_].materials;
    else if (pathTraceMaterialsBuffer_.buffer != VK_NULL_HANDLE)
        sharedMaterials = &pathTraceMaterialsBuffer_;
    const bool sharedSceneData = sharedMaterials != nullptr && !pathTraceTextures_.empty();
    p.shadowData[2] = sharedSceneData ? 1u : 0u;
    p.shadowData[3] = 0u;
    std::memcpy(paramSlots_[slotIndex].mapped, &p, sizeof(GpuParams));

    VkDescriptorImageInfo di0{sampler_, depthView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo di1{sampler_, normalView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo di2{sampler_, motionView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dout{VK_NULL_HANDLE, outputView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dhprev{sampler_, historyPrevView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dhnext{VK_NULL_HANDLE, historyNextView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dshadowOut{VK_NULL_HANDLE, shadowOutputView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dshadowHprev{sampler_, shadowHistoryPrevView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dshadowHnext{VK_NULL_HANDLE, shadowHistoryNextView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dmomentsHprev{sampler_, momentsHistoryPrevView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dmomentsHnext{VK_NULL_HANDLE, momentsHistoryNextView, VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet w[13]{};
    for (auto& x : w) x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet=set; w[0].dstBinding=0; w[0].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].descriptorCount=1; w[0].pImageInfo=&di0;
    w[1]=w[0]; w[1].dstBinding=1; w[1].pImageInfo=&di1;
    w[2]=w[0]; w[2].dstBinding=2; w[2].pImageInfo=&di2;
    w[3]=w[0]; w[3].dstBinding=3; w[3].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[3].pImageInfo=&dout;
    w[4]=w[0]; w[4].dstBinding=4; w[4].pImageInfo=&dhprev;
    w[5]=w[3]; w[5].dstBinding=5; w[5].pImageInfo=&dhnext;

    // local copy, not a pointer into a member field
    VkAccelerationStructureKHR currentTlas = tlas();
    VkWriteDescriptorSetAccelerationStructureKHR as{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    as.accelerationStructureCount=1;
    as.pAccelerationStructures=&currentTlas;
    w[6]=w[0]; w[6].dstBinding=6; w[6].descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; w[6].pNext=&as; w[6].pImageInfo=nullptr;

    VkDescriptorBufferInfo db{paramSlots_[slotIndex].buffer,0,sizeof(GpuParams)};
    w[7]=w[0]; w[7].dstBinding=7; w[7].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[7].pBufferInfo=&db; w[7].pImageInfo=nullptr;

    w[8]=w[3]; w[8].dstBinding=8; w[8].pImageInfo=&dshadowOut;
    w[9]=w[0]; w[9].dstBinding=9; w[9].pImageInfo=&dshadowHprev;
    w[10]=w[3]; w[10].dstBinding=10; w[10].pImageInfo=&dshadowHnext;
    w[11]=w[0]; w[11].dstBinding=11; w[11].pImageInfo=&dmomentsHprev;
    w[12]=w[3]; w[12].dstBinding=12; w[12].pImageInfo=&dmomentsHnext;

    vkUpdateDescriptorSets(ctx_.device(),13,w,0,nullptr);

    if (sharedSceneData) {
        VkDescriptorBufferInfo dbMaterials{sharedMaterials->buffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet wm{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wm.dstSet=set; wm.dstBinding=13; wm.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wm.descriptorCount=1; wm.pBufferInfo=&dbMaterials;
        // same fill as updatePathTraceDescriptorSet()
        std::vector<VkDescriptorImageInfo> textureInfos(kMaxPathTraceTextures,
            VkDescriptorImageInfo{pathTraceTextures_[0].sampler, pathTraceTextures_[0].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
        for (size_t i = 0; i < pathTraceTextures_.size() && i < textureInfos.size(); ++i)
            textureInfos[i] = {pathTraceTextures_[i].sampler, pathTraceTextures_[i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet wt{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wt.dstSet=set; wt.dstBinding=14; wt.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wt.descriptorCount=static_cast<uint32_t>(textureInfos.size()); wt.pImageInfo=textureInfos.data();
        VkWriteDescriptorSet shared[2] = {wm, wt};
        vkUpdateDescriptorSets(ctx_.device(), 2, shared, 0, nullptr);
    }
    return true;
}

bool RayTracingScene::ensureReflectionPipeline() {
    if (reflectionPipeline_ != VK_NULL_HANDLE) return true;
    std::vector<uint32_t> spirv; if(!readSpirv(shaderDirectory_ + "\\reflections.spv",spirv) && !readSpirv(shaderDirectory_ + "/reflections.spv",spirv)){ctx_.setError("RayTracingScene: missing reflections.spv in shader directory.");return false;}
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; sm.codeSize=spirv.size()*4; sm.pCode=spirv.data(); if(vkCreateShaderModule(ctx_.device(),&sm,nullptr,&reflectionShader_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: vkCreateShaderModule(reflections) failed.");return false;}
    VkDescriptorSetLayoutBinding b[11]{};
    b[0]={0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[1]={1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[2]={2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[3]={3,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[4]={4,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[5]={5,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[6]={6,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[7]={7,VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[8]={8,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[9]={9,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[10]={10,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};dl.bindingCount=11;dl.pBindings=b;if(vkCreateDescriptorSetLayout(ctx_.device(),&dl,nullptr,&descriptorSetLayoutReflection_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: reflection descriptor set layout failed.");return false;}
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&descriptorSetLayoutReflection_;if(vkCreatePipelineLayout(ctx_.device(),&pl,nullptr,&pipelineLayoutReflection_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: reflection pipeline layout failed.");return false;}
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cp.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cp.stage.module=reflectionShader_;cp.stage.pName="main";cp.layout=pipelineLayoutReflection_;if(vkCreateComputePipelines(ctx_.device(),VK_NULL_HANDLE,1,&cp,nullptr,&reflectionPipeline_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: reflection compute pipeline failed.");return false;}
    // own descriptor pool rather than resizing/sharing descriptorPool_
    VkDescriptorPoolSize ps[4]={{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,dispatchSlots_*6},{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,dispatchSlots_*3},{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,dispatchSlots_},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,dispatchSlots_}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dpi.maxSets=dispatchSlots_;dpi.poolSizeCount=4;dpi.pPoolSizes=ps;if(vkCreateDescriptorPool(ctx_.device(),&dpi,nullptr,&descriptorPoolReflection_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: reflection descriptor pool creation failed.");return false;}
    return createReflectionParamSlots();
}

bool RayTracingScene::createReflectionParamSlots(){
    reflectionParamSlots_.resize(dispatchSlots_); for(auto& s:reflectionParamSlots_){ Buffer b; if(!createBuffer(sizeof(GpuReflectionParams),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,b)){ctx_.setError("RayTracingScene: reflection parameter buffer allocation failed.");return false;} s.buffer=b.buffer;s.memory=b.memory;vkMapMemory(ctx_.device(),s.memory,0,sizeof(GpuReflectionParams),0,&s.mapped);b.buffer=VK_NULL_HANDLE;b.memory=VK_NULL_HANDLE; }
    return true;
}

bool RayTracingScene::createReflectionDescriptorSets(uint32_t count,std::vector<VkDescriptorSet>& sets,std::string* errorOut){
    if(count>dispatchSlots_){if(errorOut)*errorOut="requested descriptor-set count exceeds scene dispatch slot count";return false;} std::vector<VkDescriptorSetLayout> layouts(count,descriptorSetLayoutReflection_);VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};ai.descriptorPool=descriptorPoolReflection_;ai.descriptorSetCount=count;ai.pSetLayouts=layouts.data();if(vkAllocateDescriptorSets(ctx_.device(),&ai,sets.data())!=VK_SUCCESS){if(errorOut)*errorOut="vkAllocateDescriptorSets for reflections failed";return false;}return true;
}

bool RayTracingScene::updateReflectionDescriptorSet(VkDescriptorSet set, SharedTexture& depth, SharedTexture& normalRoughness,
                                           SharedTexture& motion, SharedTexture& prevSceneColor,
                                           SharedTexture& reflOutput, SharedTexture& reflHistoryPrev, SharedTexture& reflHistoryNext,
                                           SharedTexture& reflGeoHistoryPrev, SharedTexture& reflGeoHistoryNext,
                                           VkImageView depthView, VkImageView normalRoughnessView, VkImageView motionView,
                                           VkImageView prevSceneColorView, VkImageView reflOutputView,
                                           VkImageView reflHistoryPrevView, VkImageView reflHistoryNextView,
                                           VkImageView reflGeoHistoryPrevView, VkImageView reflGeoHistoryNextView,
                                           uint32_t slotIndex, const ReflectionConstants& c, std::string* errorOut) {
    if (slotIndex >= reflectionParamSlots_.size()) {
        if (errorOut) *errorOut = "invalid reflection frame slot";
        return false;
    }
    GpuReflectionParams p{};
    std::memcpy(p.invViewProj, c.invViewProj, sizeof(p.invViewProj));
    std::memcpy(p.invView, c.invView, sizeof(p.invView));
    std::memcpy(p.currentViewProjUnjittered, c.currentViewProjUnjittered, sizeof(p.currentViewProjUnjittered));
    std::memcpy(p.prevViewProjJittered, c.prevViewProjJittered, sizeof(p.prevViewProjJittered));
    std::memcpy(p.prevViewProjUnjittered, c.prevViewProjUnjittered, sizeof(p.prevViewProjUnjittered));
    p.settings[0] = c.bias;
    p.settings[1] = c.maxRayDistance;
    p.settings[2] = c.maxConeAngleRadians;
    p.settings[3] = c.historyValid ? 1.0f : 0.0f;
    p.frameData[0] = c.frameIndex;
    p.frameData[1] = reflOutput.width();
    p.frameData[2] = reflOutput.height();
    p.frameData[3] = c.raysPerPixel;
    p.extra[0] = c.prevColorValid ? 1.0f : 0.0f;
    p.extra[1] = 0.0f;
    p.extra[2] = 0.0f;
    p.extra[3] = 0.0f;
    p.skyHorizonColor[0] = c.skyHorizonColor[0];
    p.skyHorizonColor[1] = c.skyHorizonColor[1];
    p.skyHorizonColor[2] = c.skyHorizonColor[2];
    p.skyHorizonColor[3] = 0.0f;
    p.skyZenithColor[0] = c.skyZenithColor[0];
    p.skyZenithColor[1] = c.skyZenithColor[1];
    p.skyZenithColor[2] = c.skyZenithColor[2];
    p.skyZenithColor[3] = 0.0f;
    p.skyGroundColor[0] = c.skyGroundColor[0];
    p.skyGroundColor[1] = c.skyGroundColor[1];
    p.skyGroundColor[2] = c.skyGroundColor[2];
    p.skyGroundColor[3] = 0.0f;
    std::memcpy(reflectionParamSlots_[slotIndex].mapped, &p, sizeof(GpuReflectionParams));

    VkDescriptorImageInfo di0{sampler_, depthView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo di1{sampler_, normalRoughnessView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo di2{sampler_, motionView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo di3{sampler_, prevSceneColorView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dout{VK_NULL_HANDLE, reflOutputView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dhprev{sampler_, reflHistoryPrevView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dhnext{VK_NULL_HANDLE, reflHistoryNextView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dgeohprev{sampler_, reflGeoHistoryPrevView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dgeohnext{VK_NULL_HANDLE, reflGeoHistoryNextView, VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet w[11]{};
    for (auto& x : w) x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet=set; w[0].dstBinding=0; w[0].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].descriptorCount=1; w[0].pImageInfo=&di0;
    w[1]=w[0]; w[1].dstBinding=1; w[1].pImageInfo=&di1;
    w[2]=w[0]; w[2].dstBinding=2; w[2].pImageInfo=&di2;
    w[3]=w[0]; w[3].dstBinding=3; w[3].pImageInfo=&di3;
    w[4]=w[0]; w[4].dstBinding=4; w[4].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[4].pImageInfo=&dout;
    w[5]=w[0]; w[5].dstBinding=5; w[5].pImageInfo=&dhprev;
    w[6]=w[4]; w[6].dstBinding=6; w[6].pImageInfo=&dhnext;

    // local copy, not a pointer into a member field
    VkAccelerationStructureKHR currentTlas = tlas();
    VkWriteDescriptorSetAccelerationStructureKHR as{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    as.accelerationStructureCount=1;
    as.pAccelerationStructures=&currentTlas;
    w[7]=w[0]; w[7].dstBinding=7; w[7].descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; w[7].pNext=&as; w[7].pImageInfo=nullptr;

    VkDescriptorBufferInfo db{reflectionParamSlots_[slotIndex].buffer,0,sizeof(GpuReflectionParams)};
    w[8]=w[0]; w[8].dstBinding=8; w[8].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[8].pBufferInfo=&db; w[8].pImageInfo=nullptr;

    w[9]=w[0]; w[9].dstBinding=9; w[9].pImageInfo=&dgeohprev;
    w[10]=w[4]; w[10].dstBinding=10; w[10].pImageInfo=&dgeohnext;

    vkUpdateDescriptorSets(ctx_.device(),11,w,0,nullptr);
    return true;
}

VkSemaphore RayTracingScene::currentGenerationAoReadySemaphore() {
    if (dynamicGenerations_.empty()) return VK_NULL_HANDLE;
    auto& g = dynamicGenerations_[dynamicGenerationIndex_];
    if (!g.aoReadyPending) return VK_NULL_HANDLE;
    g.aoReadyPending = false;
    return g.readyForAo;
}
VkSemaphore RayTracingScene::currentGenerationReflectionsReadySemaphore() {
    if (dynamicGenerations_.empty()) return VK_NULL_HANDLE;
    auto& g = dynamicGenerations_[dynamicGenerationIndex_];
    if (!g.reflectionsReadyPending) return VK_NULL_HANDLE;
    g.reflectionsReadyPending = false;
    return g.readyForReflections;
}
VkSemaphore RayTracingScene::currentGenerationRtdiReadySemaphore() {
    if (dynamicGenerations_.empty()) return VK_NULL_HANDLE;
    auto& g = dynamicGenerations_[dynamicGenerationIndex_];
    if (!g.rtdiReadyPending) return VK_NULL_HANDLE;
    g.rtdiReadyPending = false;
    return g.readyForRtdi;
}

bool RayTracingScene::ensureRtdiDebugPipeline() {
    if (rtdiDebugPipeline_ != VK_NULL_HANDLE) return true;
    std::vector<uint32_t> spirv; if(!readSpirv(shaderDirectory_ + "\\restirDiDebugBruteforce.spv",spirv) && !readSpirv(shaderDirectory_ + "/restirDiDebugBruteforce.spv",spirv)){ctx_.setError("RayTracingScene: missing restirDiDebugBruteforce.spv in shader directory.");return false;}
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; sm.codeSize=spirv.size()*4; sm.pCode=spirv.data(); if(vkCreateShaderModule(ctx_.device(),&sm,nullptr,&rtdiDebugShader_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: vkCreateShaderModule(restirDiDebugBruteforce) failed.");return false;}
    VkDescriptorSetLayoutBinding b[5]{};
    b[0]={0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[1]={1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[2]={2,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[3]={3,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[4]={4,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};dl.bindingCount=5;dl.pBindings=b;if(vkCreateDescriptorSetLayout(ctx_.device(),&dl,nullptr,&descriptorSetLayoutRtdiDebug_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: RTDI debug descriptor set layout failed.");return false;}
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&descriptorSetLayoutRtdiDebug_;if(vkCreatePipelineLayout(ctx_.device(),&pl,nullptr,&pipelineLayoutRtdiDebug_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: RTDI debug pipeline layout failed.");return false;}
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cp.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cp.stage.module=rtdiDebugShader_;cp.stage.pName="main";cp.layout=pipelineLayoutRtdiDebug_;if(vkCreateComputePipelines(ctx_.device(),VK_NULL_HANDLE,1,&cp,nullptr,&rtdiDebugPipeline_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: RTDI debug compute pipeline failed.");return false;}
    VkDescriptorPoolSize ps[4]={{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,dispatchSlots_*2},{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,dispatchSlots_},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,dispatchSlots_},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,dispatchSlots_}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dpi.maxSets=dispatchSlots_;dpi.poolSizeCount=4;dpi.pPoolSizes=ps;if(vkCreateDescriptorPool(ctx_.device(),&dpi,nullptr,&descriptorPoolRtdiDebug_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: RTDI debug descriptor pool creation failed.");return false;}
    rtdiDebugParamSlots_.resize(dispatchSlots_);
    for (auto& s : rtdiDebugParamSlots_) {
        Buffer b; if(!createBuffer(sizeof(GpuRtdiDebugParams),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,b)){ctx_.setError("RayTracingScene: RTDI debug parameter buffer allocation failed.");return false;}
        s.buffer=b.buffer;s.memory=b.memory;vkMapMemory(ctx_.device(),s.memory,0,sizeof(GpuRtdiDebugParams),0,&s.mapped);b.buffer=VK_NULL_HANDLE;b.memory=VK_NULL_HANDLE;
    }
    if (!ensureRtdiLightBuffers()) return false;
    return true;
}

bool RayTracingScene::ensureRtdiLightBuffers() {
    if (!rtdiLightBuffers_.empty()) return true;
    static constexpr VkDeviceSize RTDI_LIGHT_INITIAL_CAPACITY = 128;
    const VkDeviceSize initialBytes = RTDI_LIGHT_INITIAL_CAPACITY * 16 * sizeof(float);
    rtdiLightBuffers_.resize(dispatchSlots_);
    for (auto& buf : rtdiLightBuffers_) {
        if (!createBuffer(initialBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buf)) {
            ctx_.setError("RayTracingScene: RTDI light buffer initial allocation failed.");
            rtdiLightBuffers_.clear();
            return false;
        }
    }
    rtdiLightCounts_.assign(dispatchSlots_, 0);
    return true;
}

bool RayTracingScene::ensureRtdiPipeline() {
    if (rtdiPipeline_ != VK_NULL_HANDLE) return true;
    if (!ensureRtdiLightBuffers()) return false;
    std::vector<uint32_t> spirv; if(!readSpirv(shaderDirectory_ + "\\restirDi.spv",spirv) && !readSpirv(shaderDirectory_ + "/restirDi.spv",spirv)){ctx_.setError("RayTracingScene: missing restirDi.spv in shader directory.");return false;}
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; sm.codeSize=spirv.size()*4; sm.pCode=spirv.data(); if(vkCreateShaderModule(ctx_.device(),&sm,nullptr,&rtdiShader_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: vkCreateShaderModule(restirDi) failed.");return false;}
    VkDescriptorSetLayoutBinding b[12]{};
    b[0]={0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[1]={1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[2]={2,VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[3]={3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[4]={4,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[5]={5,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[6]={6,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[7]={7,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[8]={8,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[9]={9,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[10]={10,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[11]={11,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};dl.bindingCount=12;dl.pBindings=b;if(vkCreateDescriptorSetLayout(ctx_.device(),&dl,nullptr,&descriptorSetLayoutRtdi_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: RTDI descriptor set layout failed.");return false;}
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&descriptorSetLayoutRtdi_;if(vkCreatePipelineLayout(ctx_.device(),&pl,nullptr,&pipelineLayoutRtdi_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: RTDI pipeline layout failed.");return false;}
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cp.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cp.stage.module=rtdiShader_;cp.stage.pName="main";cp.layout=pipelineLayoutRtdi_;if(vkCreateComputePipelines(ctx_.device(),VK_NULL_HANDLE,1,&cp,nullptr,&rtdiPipeline_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: RTDI compute pipeline failed.");return false;}
    VkDescriptorPoolSize ps[5]={{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,dispatchSlots_*6},{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,dispatchSlots_},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,dispatchSlots_},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,dispatchSlots_},{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,dispatchSlots_*3}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dpi.maxSets=dispatchSlots_;dpi.poolSizeCount=5;dpi.pPoolSizes=ps;if(vkCreateDescriptorPool(ctx_.device(),&dpi,nullptr,&descriptorPoolRtdi_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: RTDI descriptor pool creation failed.");return false;}
    rtdiParamSlots_.resize(dispatchSlots_);
    for (auto& s : rtdiParamSlots_) {
        Buffer b; if(!createBuffer(sizeof(GpuRtdiParams),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,b)){ctx_.setError("RayTracingScene: RTDI parameter buffer allocation failed.");return false;}
        s.buffer=b.buffer;s.memory=b.memory;vkMapMemory(ctx_.device(),s.memory,0,sizeof(GpuRtdiParams),0,&s.mapped);b.buffer=VK_NULL_HANDLE;b.memory=VK_NULL_HANDLE;
    }
    return true;
}

bool RayTracingScene::createRtdiDescriptorSets(uint32_t count, std::vector<VkDescriptorSet>& sets, std::string* errorOut) {
    if (count > dispatchSlots_) { if (errorOut) *errorOut = "requested descriptor-set count exceeds scene dispatch slot count"; return false; }
    std::vector<VkDescriptorSetLayout> layouts(count, descriptorSetLayoutRtdi_);
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; ai.descriptorPool=descriptorPoolRtdi_; ai.descriptorSetCount=count; ai.pSetLayouts=layouts.data();
    if (vkAllocateDescriptorSets(ctx_.device(), &ai, sets.data()) != VK_SUCCESS) { if (errorOut) *errorOut = "vkAllocateDescriptorSets for RTDI failed"; return false; }
    return true;
}

bool RayTracingScene::updateRtdiDescriptorSet(VkDescriptorSet set, SharedTexture& depth, SharedTexture& normalRoughness,
                                              SharedTexture& output, SharedTexture& reservoirA, SharedTexture& reservoirB,
                                              SharedTexture& historyReservoirA, SharedTexture& historyReservoirB,
                                              SharedTexture& historyDepth, SharedTexture& historyNormalRoughness,
                                              VkImageView depthView, VkImageView normalRoughnessView, VkImageView outputView,
                                              VkImageView reservoirAView, VkImageView reservoirBView,
                                              VkImageView historyReservoirAView, VkImageView historyReservoirBView,
                                              VkImageView historyDepthView, VkImageView historyNormalRoughnessView,
                                              uint32_t slotIndex, const RtdiConstants& c, std::string* errorOut) {
    if (slotIndex >= rtdiParamSlots_.size() || slotIndex >= rtdiLightBuffers_.size()) {
        if (errorOut) *errorOut = "invalid RTDI frame slot";
        return false;
    }
    GpuRtdiParams p{};
    std::memcpy(p.invViewProj, c.invViewProj, sizeof(p.invViewProj));
    std::memcpy(p.invView, c.invView, sizeof(p.invView));
    p.frameData[0] = c.frameIndex;
    p.frameData[1] = output.width();
    p.frameData[2] = output.height();
    p.frameData[3] = rtdiLightCounts_[slotIndex];
    p.candidateCount = std::clamp(c.candidateCount, 1u, 64u);
    p.bias = c.bias;
    p.maxRayDistance = c.maxRayDistance;
    std::memcpy(p.previousViewProj, c.previousViewProj, sizeof(p.previousViewProj));
    p.maxHistoryM = c.maxHistoryM;
    p.depthRejectThreshold = c.depthRejectThreshold;
    std::memcpy(rtdiParamSlots_[slotIndex].mapped, &p, sizeof(GpuRtdiParams));

    VkDescriptorImageInfo di0{sampler_, depthView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo di1{sampler_, normalRoughnessView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dout{VK_NULL_HANDLE, outputView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dresA{VK_NULL_HANDLE, reservoirAView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dresB{VK_NULL_HANDLE, reservoirBView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dhresA{sampler_, historyReservoirAView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dhresB{sampler_, historyReservoirBView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dhdepth{sampler_, historyDepthView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dhnormal{sampler_, historyNormalRoughnessView, VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet w[12]{};
    for (auto& x : w) x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet=set; w[0].dstBinding=0; w[0].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].descriptorCount=1; w[0].pImageInfo=&di0;
    w[1]=w[0]; w[1].dstBinding=1; w[1].pImageInfo=&di1;

    VkAccelerationStructureKHR currentTlas = tlas();
    VkWriteDescriptorSetAccelerationStructureKHR as{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    as.accelerationStructureCount=1;
    as.pAccelerationStructures=&currentTlas;
    w[2]=w[0]; w[2].dstBinding=2; w[2].descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; w[2].pNext=&as; w[2].pImageInfo=nullptr;

    VkDescriptorBufferInfo dbLights{rtdiLightBuffers_[slotIndex].buffer, 0, VK_WHOLE_SIZE};
    w[3]=w[0]; w[3].dstBinding=3; w[3].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[3].pBufferInfo=&dbLights; w[3].pImageInfo=nullptr;

    VkDescriptorBufferInfo dbParams{rtdiParamSlots_[slotIndex].buffer, 0, sizeof(GpuRtdiParams)};
    w[4]=w[0]; w[4].dstBinding=4; w[4].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[4].pBufferInfo=&dbParams; w[4].pImageInfo=nullptr;

    w[5]=w[0]; w[5].dstBinding=5; w[5].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[5].pImageInfo=&dout;
    w[6]=w[0]; w[6].dstBinding=6; w[6].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[6].pImageInfo=&dresA;
    w[7]=w[0]; w[7].dstBinding=7; w[7].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[7].pImageInfo=&dresB;
    w[8]=w[0]; w[8].dstBinding=8; w[8].pImageInfo=&dhresA;
    w[9]=w[0]; w[9].dstBinding=9; w[9].pImageInfo=&dhresB;
    w[10]=w[0]; w[10].dstBinding=10; w[10].pImageInfo=&dhdepth;
    w[11]=w[0]; w[11].dstBinding=11; w[11].pImageInfo=&dhnormal;

    vkUpdateDescriptorSets(ctx_.device(), 12, w, 0, nullptr);
    return true;
}

bool RayTracingScene::setRtdiLights(uint32_t slotIndex, const float* packed, uint32_t count) {
    if (slotIndex >= rtdiLightBuffers_.size()) {
        ctx_.setError("RayTracingScene::setRtdiLights: invalid RTDI frame slot -- call ensureRtdiDebugPipeline (via a real dispatch) first.");
        return false;
    }
    if (count == 0) {
        rtdiLightCounts_[slotIndex] = 0;
        return true;
    }
    if (!packed) {
        ctx_.setError("RayTracingScene::setRtdiLights: null light data with nonzero count.");
        return false;
    }
    const uint32_t strideFloats = 16;
    const VkDeviceSize requiredSize = static_cast<VkDeviceSize>(count) * strideFloats * sizeof(float);
    if (rtdiLightBuffers_[slotIndex].size < requiredSize) {
        VkResult idle = vkDeviceWaitIdle(ctx_.device());
        if (idle != VK_SUCCESS) {
            ctx_.setError("RayTracingScene::setRtdiLights: vkDeviceWaitIdle failed with VkResult " + std::to_string(idle));
            return false;
        }
        std::cerr << "[RayTracingScene] RTDI light buffer growing past its initial capacity (slot "
                  << slotIndex << ", " << count << " lights) -- stalled the GPU to do it safely. "
                     "If this happens routinely, raise RTDI_LIGHT_INITIAL_CAPACITY." << std::endl;
    }
    if (!ensureBufferCapacity(rtdiLightBuffers_[slotIndex], requiredSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        ctx_.setError("RayTracingScene::setRtdiLights: light buffer allocation failed.");
        return false;
    }
    void* mapped = nullptr;
    if (vkMapMemory(ctx_.device(), rtdiLightBuffers_[slotIndex].memory, 0, requiredSize, 0, &mapped) != VK_SUCCESS) {
        ctx_.setError("RayTracingScene::setRtdiLights: vkMapMemory failed.");
        return false;
    }
    std::memcpy(mapped, packed, static_cast<size_t>(count) * strideFloats * sizeof(float));
    vkUnmapMemory(ctx_.device(), rtdiLightBuffers_[slotIndex].memory);
    rtdiLightCounts_[slotIndex] = count;
    return true;
}

bool RayTracingScene::setEmissiveMaterials(const float* packed, uint32_t count) {
    if (count == 0) {
        emissiveMaterialCount_ = 0;
        return true;
    }
    if (!packed) {
        ctx_.setError("RayTracingScene::setEmissiveMaterials: null material data with nonzero count.");
        return false;
    }
    const uint32_t strideFloats = 4;
    const VkDeviceSize requiredSize = static_cast<VkDeviceSize>(count) * strideFloats * sizeof(float);
    if (emissiveMaterialsBuffer_.size < requiredSize) {
        VkResult idle = vkDeviceWaitIdle(ctx_.device());
        if (idle != VK_SUCCESS) {
            ctx_.setError("RayTracingScene::setEmissiveMaterials: vkDeviceWaitIdle failed with VkResult " + std::to_string(idle));
            return false;
        }
    }
    if (!ensureBufferCapacity(emissiveMaterialsBuffer_, requiredSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        ctx_.setError("RayTracingScene::setEmissiveMaterials: buffer allocation failed.");
        return false;
    }
    void* mapped = nullptr;
    if (vkMapMemory(ctx_.device(), emissiveMaterialsBuffer_.memory, 0, requiredSize, 0, &mapped) != VK_SUCCESS) {
        ctx_.setError("RayTracingScene::setEmissiveMaterials: vkMapMemory failed.");
        return false;
    }
    std::memcpy(mapped, packed, static_cast<size_t>(count) * strideFloats * sizeof(float));
    vkUnmapMemory(ctx_.device(), emissiveMaterialsBuffer_.memory);
    emissiveMaterialCount_ = count;
    return true;
}

bool RayTracingScene::createRtdiDebugDescriptorSets(uint32_t count, std::vector<VkDescriptorSet>& sets, std::string* errorOut) {
    if (count > dispatchSlots_) { if (errorOut) *errorOut = "requested descriptor-set count exceeds scene dispatch slot count"; return false; }
    std::vector<VkDescriptorSetLayout> layouts(count, descriptorSetLayoutRtdiDebug_);
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; ai.descriptorPool=descriptorPoolRtdiDebug_; ai.descriptorSetCount=count; ai.pSetLayouts=layouts.data();
    if (vkAllocateDescriptorSets(ctx_.device(), &ai, sets.data()) != VK_SUCCESS) { if (errorOut) *errorOut = "vkAllocateDescriptorSets for RTDI debug failed"; return false; }
    return true;
}

bool RayTracingScene::updateRtdiDebugDescriptorSet(VkDescriptorSet set, SharedTexture& depth, SharedTexture& normalRoughness,
                                                   SharedTexture& output, VkImageView depthView, VkImageView normalRoughnessView,
                                                   VkImageView outputView, uint32_t slotIndex, const RtdiDebugConstants& c,
                                                   std::string* errorOut) {
    if (slotIndex >= rtdiDebugParamSlots_.size() || slotIndex >= rtdiLightBuffers_.size()) {
        if (errorOut) *errorOut = "invalid RTDI debug frame slot";
        return false;
    }
    GpuRtdiDebugParams p{};
    std::memcpy(p.invViewProj, c.invViewProj, sizeof(p.invViewProj));
    std::memcpy(p.invView, c.invView, sizeof(p.invView));
    p.frameData[0] = c.frameIndex;
    p.frameData[1] = output.width();
    p.frameData[2] = output.height();
    // the light count for this slot's buffer
    p.frameData[3] = rtdiLightCounts_[slotIndex];
    std::memcpy(rtdiDebugParamSlots_[slotIndex].mapped, &p, sizeof(GpuRtdiDebugParams));

    VkDescriptorImageInfo di0{sampler_, depthView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo di1{sampler_, normalRoughnessView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dout{VK_NULL_HANDLE, outputView, VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet w[5]{};
    for (auto& x : w) x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet=set; w[0].dstBinding=0; w[0].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].descriptorCount=1; w[0].pImageInfo=&di0;
    w[1]=w[0]; w[1].dstBinding=1; w[1].pImageInfo=&di1;
    w[2]=w[0]; w[2].dstBinding=2; w[2].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[2].pImageInfo=&dout;

    VkDescriptorBufferInfo dbParams{rtdiDebugParamSlots_[slotIndex].buffer, 0, sizeof(GpuRtdiDebugParams)};
    w[3]=w[0]; w[3].dstBinding=3; w[3].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[3].pBufferInfo=&dbParams; w[3].pImageInfo=nullptr;

    VkDescriptorBufferInfo dbLights{rtdiLightBuffers_[slotIndex].buffer, 0, VK_WHOLE_SIZE};
    w[4]=w[0]; w[4].dstBinding=4; w[4].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[4].pBufferInfo=&dbLights; w[4].pImageInfo=nullptr;

    vkUpdateDescriptorSets(ctx_.device(), 5, w, 0, nullptr);
    return true;
}


VkSemaphore RayTracingScene::currentGenerationPathTraceReadySemaphore() {
    if (dynamicGenerations_.empty()) return VK_NULL_HANDLE;
    auto& g = dynamicGenerations_[dynamicGenerationIndex_];
    if (!g.pathTraceReadyPending) return VK_NULL_HANDLE;
    g.pathTraceReadyPending = false;
    return g.readyForPathTrace;
}

bool RayTracingScene::ensurePathTraceLightBuffers() {
    if (!pathTraceLightBuffers_.empty()) return true;
    // 256 lights is a generous starting point for a Sauerbraten map's ET_LIGHT entity count
    static constexpr VkDeviceSize PATHTRACE_LIGHT_INITIAL_CAPACITY = 256;
    // 12 floats per light
    const VkDeviceSize initialBytes = PATHTRACE_LIGHT_INITIAL_CAPACITY * 12 * sizeof(float);
    pathTraceLightBuffers_.resize(dispatchSlots_);
    for (auto& buf : pathTraceLightBuffers_) {
        if (!createBuffer(initialBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buf)) {
            ctx_.setError("RayTracingScene: path-trace light buffer initial allocation failed.");
            pathTraceLightBuffers_.clear();
            return false;
        }
    }
    pathTraceLightCounts_.assign(dispatchSlots_, 0);
    return true;
}

void RayTracingScene::reportPipelineStats(const char* spvName, VkPipelineLayout layout) {
    if (!ctx_.capabilities().hasPipelineStats || layout == VK_NULL_HANDLE) return;
    auto getProps = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(ctx_.loadDeviceProc("vkGetPipelineExecutablePropertiesKHR"));
    auto getStats = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(ctx_.loadDeviceProc("vkGetPipelineExecutableStatisticsKHR"));
    if (!getProps || !getStats) return;
    std::vector<uint32_t> spirv;
    if (!readSpirv(shaderDirectory_ + "\\" + spvName, spirv) && !readSpirv(shaderDirectory_ + "/" + spvName, spirv)) return;
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; sm.codeSize = spirv.size() * 4; sm.pCode = spirv.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx_.device(), &sm, nullptr, &module) != VK_SUCCESS) return;
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    cp.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cp.stage.module = module; cp.stage.pName = "main"; cp.layout = layout;
    VkPipeline pipe = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE, 1, &cp, nullptr, &pipe) == VK_SUCCESS) {
        VkPipelineInfoKHR pi{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR}; pi.pipeline = pipe;
        uint32_t execCount = 0;
        getProps(ctx_.device(), &pi, &execCount, nullptr);
        std::vector<VkPipelineExecutablePropertiesKHR> execs(execCount, VkPipelineExecutablePropertiesKHR{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
        getProps(ctx_.device(), &pi, &execCount, execs.data());
        for (uint32_t e = 0; e < execCount; ++e) {
            VkPipelineExecutableInfoKHR ei{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR}; ei.pipeline = pipe; ei.executableIndex = e;
            uint32_t statCount = 0;
            getStats(ctx_.device(), &ei, &statCount, nullptr);
            std::vector<VkPipelineExecutableStatisticKHR> stats(statCount, VkPipelineExecutableStatisticKHR{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
            getStats(ctx_.device(), &ei, &statCount, stats.data());
            std::cerr << "[pipestats] " << spvName << " [" << execs[e].name << ", subgroup " << execs[e].subgroupSize << "]:";
            for (const auto& st : stats) {
                std::cerr << " " << st.name << "=";
                switch (st.format) {
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: std::cerr << (st.value.b32 ? "true" : "false"); break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: std::cerr << st.value.i64; break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: std::cerr << st.value.u64; break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: std::cerr << st.value.f64; break;
                    default: std::cerr << "?"; break;
                }
                std::cerr << ";";
            }
            std::cerr << std::endl;
        }
        vkDestroyPipeline(ctx_.device(), pipe, nullptr);
    }
    vkDestroyShaderModule(ctx_.device(), module, nullptr);
}

void RayTracingScene::createPathTraceRaygenPipelines() {
    if (!ctx_.capabilities().hasRtPipeline || pipelineLayoutPathTrace_ == VK_NULL_HANDLE) return;
    auto createRt = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(ctx_.loadDeviceProc("vkCreateRayTracingPipelinesKHR"));
    auto getHandles = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(ctx_.loadDeviceProc("vkGetRayTracingShaderGroupHandlesKHR"));
    cmdTraceRays_ = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(ctx_.loadDeviceProc("vkCmdTraceRaysKHR"));
    cmdSetRtStack_ = reinterpret_cast<PFN_vkCmdSetRayTracingPipelineStackSizeKHR>(ctx_.loadDeviceProc("vkCmdSetRayTracingPipelineStackSizeKHR"));
    if (!createRt || !getHandles || !cmdTraceRays_) return;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props.pNext = &rtp;
    vkGetPhysicalDeviceProperties2(ctx_.physicalDevice(), &props);
    const VkDeviceSize handleSize = rtp.shaderGroupHandleSize;
    const VkDeviceSize handleAlign = std::max<VkDeviceSize>(rtp.shaderGroupHandleAlignment, 1);
    const VkDeviceSize baseAlign = std::max<VkDeviceSize>(rtp.shaderGroupBaseAlignment, 1);
    const VkDeviceSize stride = (handleSize + handleAlign - 1) / handleAlign * handleAlign;
    const char* names[2] = {"pathtrace_trace_rgen.spv", "pathtrace_trace_ser.spv"};
    for (int v = 0; v < 2; ++v) {
        if (v == 1 && !ctx_.capabilities().hasInvocationReorder && !ctx_.capabilities().hasInvocationReorderNV) break;
        std::vector<uint32_t> spirv;
        if (!readSpirv(shaderDirectory_ + "\\" + names[v], spirv) && !readSpirv(shaderDirectory_ + "/" + names[v], spirv)) continue;
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sm.codeSize = spirv.size() * 4; sm.pCode = spirv.data();
        if (vkCreateShaderModule(ctx_.device(), &sm, nullptr, &pathTraceRgenShader_[v]) != VK_SUCCESS) continue;
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR; stage.module = pathTraceRgenShader_[v]; stage.pName = "main";
        VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
        group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        group.generalShader = 0;
        group.closestHitShader = group.anyHitShader = group.intersectionShader = VK_SHADER_UNUSED_KHR;
        const VkDynamicState rtDynamic = VK_DYNAMIC_STATE_RAY_TRACING_PIPELINE_STACK_SIZE_KHR;
        VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dyn.dynamicStateCount = 1; dyn.pDynamicStates = &rtDynamic;
        VkRayTracingPipelineCreateInfoKHR ci{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
        ci.stageCount = 1; ci.pStages = &stage;
        ci.groupCount = 1; ci.pGroups = &group;
        ci.maxPipelineRayRecursionDepth = 1;
        ci.layout = pipelineLayoutPathTrace_;
        ci.pDynamicState = &dyn;
        const auto rgenStart = std::chrono::steady_clock::now();
        if (createRt(ctx_.device(), VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &ci, nullptr, &pathTraceRgenPipeline_[v]) != VK_SUCCESS) {
            pathTraceRgenPipeline_[v] = VK_NULL_HANDLE;
            std::cerr << "[interop] RAYGEN TRACE: " << names[v] << " pipeline creation failed" << std::endl;
            continue;
        }
        std::vector<uint8_t> handle(handleSize);
        if (getHandles(ctx_.device(), pathTraceRgenPipeline_[v], 0, 1, handle.size(), handle.data()) != VK_SUCCESS ||
            !createBuffer(stride + baseAlign, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, pathTraceRgenSbt_[v],
                          VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)) {
            vkDestroyPipeline(ctx_.device(), pathTraceRgenPipeline_[v], nullptr);
            pathTraceRgenPipeline_[v] = VK_NULL_HANDLE;
            continue;
        }
        const VkDeviceAddress aligned = (pathTraceRgenSbt_[v].address + baseAlign - 1) / baseAlign * baseAlign;
        void* mapped = nullptr;
        vkMapMemory(ctx_.device(), pathTraceRgenSbt_[v].memory, 0, VK_WHOLE_SIZE, 0, &mapped);
        std::memcpy(static_cast<uint8_t*>(mapped) + (aligned - pathTraceRgenSbt_[v].address), handle.data(), handle.size());
        vkUnmapMemory(ctx_.device(), pathTraceRgenSbt_[v].memory);
        pathTraceRgenRegion_[v] = {aligned, stride, stride};  // raygen
        auto groupStack = reinterpret_cast<PFN_vkGetRayTracingShaderGroupStackSizeKHR>(ctx_.loadDeviceProc("vkGetRayTracingShaderGroupStackSizeKHR"));
        const VkDeviceSize reported = groupStack ? groupStack(ctx_.device(), pathTraceRgenPipeline_[v], 0, VK_SHADER_GROUP_SHADER_GENERAL_KHR) : 0;
        pathTraceRgenStack_[v] = static_cast<uint32_t>(std::max<VkDeviceSize>(reported, 0));
        std::cerr << "[interop] RAYGEN TRACE: " << names[v] << " ready in " << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - rgenStart).count() << " ms (driver-reported raygen stack " << reported << " bytes)" << std::endl;
    }
}

bool RayTracingScene::recordPathTraceRays(VkCommandBuffer cmd, VkDescriptorSet set, uint32_t width, uint32_t height, bool ser) {
    const int v = ser ? 1 : 0;
    if (pathTraceRgenPipeline_[v] == VK_NULL_HANDLE || !cmdTraceRays_) return false;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pathTraceRgenPipeline_[v]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayoutPathTrace_, 0, 1, &set, 0, nullptr);
    if (cmdSetRtStack_) cmdSetRtStack_(cmd, std::max<uint32_t>(pathTraceRgenStack_[v], pathTraceRgenStackMin_));
    const VkStridedDeviceAddressRegionKHR empty{};
    cmdTraceRays_(cmd, &pathTraceRgenRegion_[v], &empty, &empty, &empty, width, height, 1);
    return true;
}

bool RayTracingScene::ensurePathTracePipeline() {
    if (pathTracePipeline_ != VK_NULL_HANDLE) return true;
    if (!ensurePathTraceLightBuffers()) return false;
    if (historySampler_ == VK_NULL_HANDLE) {
        VkSamplerCreateInfo hsi{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        hsi.magFilter = VK_FILTER_LINEAR;
        hsi.minFilter = VK_FILTER_LINEAR;
        hsi.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        hsi.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        hsi.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        hsi.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        hsi.maxLod = 0.0f;
        if (vkCreateSampler(ctx_.device(), &hsi, nullptr, &historySampler_) != VK_SUCCESS) {
            ctx_.setError("RayTracingScene: history sampler creation failed.");
            return false;
        }
    }
    std::vector<uint32_t> spirv;
    const bool profBuild = ctx_.capabilities().hasShaderClock &&
        (readSpirv(shaderDirectory_ + "\\pathtrace_trace_prof.spv", spirv) || readSpirv(shaderDirectory_ + "/pathtrace_trace_prof.spv", spirv));
    if(!profBuild && !readSpirv(shaderDirectory_ + "\\pathtrace_trace.spv",spirv) && !readSpirv(shaderDirectory_ + "/pathtrace_trace.spv",spirv)){ctx_.setError("RayTracingScene: missing pathtrace_trace.spv in shader directory.");return false;}
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; sm.codeSize=spirv.size()*4; sm.pCode=spirv.data(); if(vkCreateShaderModule(ctx_.device(),&sm,nullptr,&pathTraceShader_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: vkCreateShaderModule(pathtrace_trace) failed.");return false;}
    // second shader module, pass 2 (resolve)
    std::vector<uint32_t> spirvResolve; if(!readSpirv(shaderDirectory_ + "\\pathtrace_resolve.spv",spirvResolve) && !readSpirv(shaderDirectory_ + "/pathtrace_resolve.spv",spirvResolve)){ctx_.setError("RayTracingScene: missing pathtrace_resolve.spv in shader directory.");return false;}
    VkShaderModuleCreateInfo smr{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smr.codeSize=spirvResolve.size()*4; smr.pCode=spirvResolve.data(); if(vkCreateShaderModule(ctx_.device(),&smr,nullptr,&pathTraceResolveShader_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: vkCreateShaderModule(pathtrace_resolve) failed.");return false;}
    // the raygen variant binds this same set
    const VkShaderStageFlags ptStages = VK_SHADER_STAGE_COMPUTE_BIT | (ctx_.capabilities().hasRtPipeline ? VK_SHADER_STAGE_RAYGEN_BIT_KHR : 0);
    VkDescriptorSetLayoutBinding b[32]{};
    b[0]={0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,ptStages,nullptr};
    b[1]={1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,ptStages,nullptr};
    b[2]={2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,ptStages,nullptr};
    b[3]={3,VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1,ptStages,nullptr};
    b[4]={4,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,ptStages,nullptr};
    b[5]={5,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,ptStages,nullptr};
    b[6]={6,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,ptStages,nullptr};
    b[7]={7,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,ptStages,nullptr};
    b[8]={8,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,ptStages,nullptr};
    b[9]={9,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,ptStages,nullptr};
    b[10]={10,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,ptStages,nullptr};
    b[11]={11,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,ptStages,nullptr};
    b[12]={12,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,ptStages,nullptr};
    {
        VkPhysicalDeviceProperties lp{}; vkGetPhysicalDeviceProperties(ctx_.physicalDevice(), &lp);
        const uint32_t need = kMaxPathTraceTextures + 16u;
        if (lp.limits.maxPerStageDescriptorSamplers < need || lp.limits.maxPerStageDescriptorSampledImages < need ||
            lp.limits.maxDescriptorSetSamplers < need || lp.limits.maxDescriptorSetSampledImages < need) {
            ctx_.setError("RayTracingScene: this GPU's descriptor limits are below the path tracer's " + std::to_string(kMaxPathTraceTextures) + "-texture table.");
            return false;
        }
    }
    b[13]={13,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,kMaxPathTraceTextures,ptStages,nullptr};
    // raw-radiance intermediate, trace/resolve split
    b[14]={14,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,ptStages,nullptr};
    b[15]={15,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,ptStages,nullptr};
    b[16]={16,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,ptStages,nullptr};
    b[17]={17,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,ptStages,nullptr};
    b[18]={18,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,ptStages,nullptr};
    // uncompressed linear distance (r16f), written by the trace pass
    b[19]={19,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,ptStages,nullptr};
    b[20]={20,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,ptStages,nullptr};
    b[21]={21,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,ptStages,nullptr};
    b[22]={22,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,ptStages,nullptr};
    // ray counters, see rayStatsBuffer()
    b[23]={25,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,ptStages,nullptr};
    // see setLightGrid()
    b[24]={26,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,ptStages,nullptr};
    // keys, this frame's sums, stored radiance
    b[25]={27,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,ptStages,nullptr};
    b[26]={28,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,ptStages,nullptr};
    b[27]={29,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,ptStages,nullptr};
    // the sky's sampling CDF
    b[28]={30,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,ptStages,nullptr};
    // the FFT sea + depth/fetch wave map (rt_water.glsl)
    b[29]={31,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,ptStages,nullptr};
    b[30]={32,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,ptStages,nullptr};
    b[31]={33,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,ptStages,nullptr};  // RESTIR DI reservoirs
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};dl.bindingCount=32;dl.pBindings=b;if(vkCreateDescriptorSetLayout(ctx_.device(),&dl,nullptr,&descriptorSetLayoutPathTrace_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: path-trace descriptor set layout failed.");return false;}
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&descriptorSetLayoutPathTrace_;if(vkCreatePipelineLayout(ctx_.device(),&pl,nullptr,&pipelineLayoutPathTrace_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: path-trace pipeline layout failed.");return false;}
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cp.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cp.stage.module=pathTraceShader_;cp.stage.pName="main";cp.layout=pipelineLayoutPathTrace_;if(vkCreateComputePipelines(ctx_.device(),VK_NULL_HANDLE,1,&cp,nullptr,&pathTracePipeline_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: path-trace compute pipeline (trace) failed.");return false;}
    VkComputePipelineCreateInfo cpr{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cpr.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};cpr.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cpr.stage.module=pathTraceResolveShader_;cpr.stage.pName="main";cpr.layout=pipelineLayoutPathTrace_;if(vkCreateComputePipelines(ctx_.device(),VK_NULL_HANDLE,1,&cpr,nullptr,&pathTraceResolvePipeline_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: path-trace compute pipeline (resolve) failed.");return false;}
    createPathTraceRaygenPipelines();  // non-fatal, compute stays the fallback
    {
        std::vector<uint32_t> spirvCaustics;
        if (readSpirv(shaderDirectory_ + "\\pathtrace_caustics.spv", spirvCaustics) || readSpirv(shaderDirectory_ + "/pathtrace_caustics.spv", spirvCaustics)) {
            VkShaderModuleCreateInfo smc{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smc.codeSize = spirvCaustics.size() * 4; smc.pCode = spirvCaustics.data();
            if (vkCreateShaderModule(ctx_.device(), &smc, nullptr, &causticShader_) == VK_SUCCESS) {
                VkComputePipelineCreateInfo cpc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpc.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
                cpc.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpc.stage.module = causticShader_; cpc.stage.pName = "main"; cpc.layout = pipelineLayoutPathTrace_;
                if (vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE, 1, &cpc, nullptr, &causticPipeline_) != VK_SUCCESS) causticPipeline_ = VK_NULL_HANDLE;
            }
        }
    }
    // fourth pipeline, same layout, non-fatal if missing (the cache then never settles, so lookups
    // never hit)
    {
        std::vector<uint32_t> spirvCache;
        if (readSpirv(shaderDirectory_ + "\\pathtrace_cache.spv", spirvCache) || readSpirv(shaderDirectory_ + "/pathtrace_cache.spv", spirvCache)) {
            VkShaderModuleCreateInfo smc{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smc.codeSize = spirvCache.size() * 4; smc.pCode = spirvCache.data();
            if (vkCreateShaderModule(ctx_.device(), &smc, nullptr, &cacheShader_) == VK_SUCCESS) {
                VkComputePipelineCreateInfo cpc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpc.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
                cpc.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpc.stage.module = cacheShader_; cpc.stage.pName = "main"; cpc.layout = pipelineLayoutPathTrace_;
                if (vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE, 1, &cpc, nullptr, &cachePipeline_) != VK_SUCCESS) cachePipeline_ = VK_NULL_HANDLE;
            }
        }
    }
    {
        std::vector<uint32_t> spirvSky;
        if (readSpirv(shaderDirectory_ + "\\pathtrace_skycdf.spv", spirvSky) || readSpirv(shaderDirectory_ + "/pathtrace_skycdf.spv", spirvSky)) {
            VkShaderModuleCreateInfo smc{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smc.codeSize = spirvSky.size() * 4; smc.pCode = spirvSky.data();
            if (vkCreateShaderModule(ctx_.device(), &smc, nullptr, &skyCdfShader_) == VK_SUCCESS) {
                VkComputePipelineCreateInfo cpc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpc.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
                cpc.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpc.stage.module = skyCdfShader_; cpc.stage.pName = "main"; cpc.layout = pipelineLayoutPathTrace_;
                if (vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE, 1, &cpc, nullptr, &skyCdfPipeline_) != VK_SUCCESS) skyCdfPipeline_ = VK_NULL_HANDLE;
            }
        }
    }
    // pipeline stats diagnostic (env SAUER_PIPELINE_STATS only)
    for (const char* n : {"pathtrace_trace.spv", "pathtrace_trace_prof.spv", "pathtrace_resolve.spv", "pathtrace_caustics.spv",
                          "pathtrace_cache.spv", "pathtrace_skycdf.spv"})
        reportPipelineStats(n, pipelineLayoutPathTrace_);
    VkDescriptorPoolSize ps[5]={{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,dispatchSlots_*(9u+kMaxPathTraceTextures)},{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,dispatchSlots_},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,dispatchSlots_*14},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,dispatchSlots_},{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,dispatchSlots_*6}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dpi.maxSets=dispatchSlots_;dpi.poolSizeCount=5;dpi.pPoolSizes=ps;if(vkCreateDescriptorPool(ctx_.device(),&dpi,nullptr,&descriptorPoolPathTrace_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: path-trace descriptor pool creation failed.");return false;}
    // device-local (read on every sky sample), zeroed with the cache below (header invalid until the
    // first CDF pass)
    if (skyCdfBuffer_.buffer == VK_NULL_HANDLE &&
        !createBuffer(16 + VkDeviceSize(kSkyCdfCells) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, skyCdfBuffer_)) {
        ctx_.setError("RayTracingScene: sky CDF buffer allocation failed."); return false;
    }
    // radiance cache buffers (bindings 27-29), device-local, zeroed
    if (rcKeyBuffer_.buffer == VK_NULL_HANDLE) {
        const VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (!createBuffer(VkDeviceSize(kRadianceCacheSize) * 8, u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, rcKeyBuffer_) ||
            !createBuffer(VkDeviceSize(kRadianceCacheSize) * 16, u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, rcAccumBuffer_) ||
            !createBuffer(VkDeviceSize(kRadianceCacheSize) * 16, u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, rcRadianceBuffer_)) {
            ctx_.setError("RayTracingScene: radiance cache allocation failed."); return false;
        }
        clearRadianceCache();
    }
    // caustics buffers, always allocated (bindings 20-22 are statically used by the trace shader)
    causticPhotonBuffers_.resize(1);
    causticHashBuffers_.resize(1);
    for (uint32_t i = 0; i < 1; ++i) {
        // sized for the default (128k a batch) to start
        causticCapacity_ = 128u * 1024u;
        if (!createBuffer(16 * VkDeviceSize(kCausticBatches) + VkDeviceSize(kCausticBatches) * causticCapacity_ * 48, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, causticPhotonBuffers_[i]) ||
            !createBuffer(VkDeviceSize(kCausticBatches) * causticHashSizeFor(causticCapacity_) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, causticHashBuffers_[i])) {
            ctx_.setError("RayTracingScene: caustic photon buffer allocation failed."); return false;
        }
    }
    if (causticEmitterBuffer_.buffer == VK_NULL_HANDLE) {
        // placeholder until the first setCausticEmitters()
        if (!createBuffer(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, causticEmitterBuffer_)) {
            ctx_.setError("RayTracingScene: caustic emitter buffer allocation failed."); return false;
        }
        void* m = nullptr;
        if (vkMapMemory(ctx_.device(), causticEmitterBuffer_.memory, 0, 64, 0, &m) == VK_SUCCESS) { std::memset(m, 0, 64); vkUnmapMemory(ctx_.device(), causticEmitterBuffer_.memory); }
    }
    if (!ensureLightGridPlaceholder()) return false;
    rayStatsSlots_.resize(dispatchSlots_);
    for (auto& s : rayStatsSlots_) {
        const VkDeviceSize sz = kRayStatCount * sizeof(uint32_t);
        Buffer buf; if(!createBuffer(sz,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,buf)){ctx_.setError("RayTracingScene: ray-stats buffer allocation failed.");return false;}
        s.buffer=buf.buffer;s.memory=buf.memory;vkMapMemory(ctx_.device(),s.memory,0,sz,0,&s.mapped);std::memset(s.mapped,0,static_cast<size_t>(sz));buf.buffer=VK_NULL_HANDLE;buf.memory=VK_NULL_HANDLE;
    }
    pathTraceParamSlots_.resize(dispatchSlots_);
    for (auto& s : pathTraceParamSlots_) {
        Buffer buf; if(!createBuffer(sizeof(GpuPathTraceParams),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,buf)){ctx_.setError("RayTracingScene: path-trace parameter buffer allocation failed.");return false;}
        s.buffer=buf.buffer;s.memory=buf.memory;vkMapMemory(ctx_.device(),s.memory,0,sizeof(GpuPathTraceParams),0,&s.mapped);buf.buffer=VK_NULL_HANDLE;buf.memory=VK_NULL_HANDLE;
    }
    pathTraceStaticDescriptorWrittenGeneration_.assign(dispatchSlots_, UINT64_MAX);
    // defensive
    if (pathTraceMaterialsBuffer_.buffer == VK_NULL_HANDLE) {
        GpuPathTraceMaterialEntry placeholder{};
        if (!ensureBufferCapacity(pathTraceMaterialsBuffer_, sizeof(placeholder), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            ctx_.setError("RayTracingScene: path-trace material table placeholder allocation failed.");
            return false;
        }
        void* mapped = nullptr;
        if (vkMapMemory(ctx_.device(), pathTraceMaterialsBuffer_.memory, 0, sizeof(placeholder), 0, &mapped) == VK_SUCCESS) {
            std::memcpy(mapped, &placeholder, sizeof(placeholder));
            vkUnmapMemory(ctx_.device(), pathTraceMaterialsBuffer_.memory);
        }
    }
    if (pathTraceMaterialInfoBuffer_.buffer == VK_NULL_HANDLE) {
        GpuPathTraceMaterialInfo placeholder{};
        placeholder.colorscale[0] = placeholder.colorscale[1] = placeholder.colorscale[2] = 1.0f;
        placeholder.glowcolor[0] = placeholder.glowcolor[1] = placeholder.glowcolor[2] = 1.0f;
        if (!ensureBufferCapacity(pathTraceMaterialInfoBuffer_, sizeof(placeholder), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            ctx_.setError("RayTracingScene: path-trace material-info placeholder allocation failed.");
            return false;
        }
        void* mapped = nullptr;
        if (vkMapMemory(ctx_.device(), pathTraceMaterialInfoBuffer_.memory, 0, sizeof(placeholder), 0, &mapped) == VK_SUCCESS) {
            std::memcpy(mapped, &placeholder, sizeof(placeholder));
            vkUnmapMemory(ctx_.device(), pathTraceMaterialInfoBuffer_.memory);
        }
    }
    if (pathTraceEmissiveBuffer_.buffer == VK_NULL_HANDLE) {
        GpuEmissiveTriangle placeholder{};
        if (!ensureBufferCapacity(pathTraceEmissiveBuffer_, sizeof(placeholder), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            ctx_.setError("RayTracingScene: path-trace emissive-triangle placeholder allocation failed.");
            return false;
        }
        void* mapped = nullptr;
        if (vkMapMemory(ctx_.device(), pathTraceEmissiveBuffer_.memory, 0, sizeof(placeholder), 0, &mapped) == VK_SUCCESS) {
            std::memcpy(mapped, &placeholder, sizeof(placeholder));
            vkUnmapMemory(ctx_.device(), pathTraceEmissiveBuffer_.memory);
        }
        pathTraceEmissiveCount_ = 0;
        pathTraceEmissiveTotalPower_ = 0.0f;
    }
    return true;
}

bool RayTracingScene::ensureVolumetricFogPipeline() {
    if (volFogInjectPipeline_ != VK_NULL_HANDLE) return true;
    if (historySampler_ == VK_NULL_HANDLE) {
        VkSamplerCreateInfo hsi{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        hsi.magFilter = VK_FILTER_LINEAR;
        hsi.minFilter = VK_FILTER_LINEAR;
        hsi.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        hsi.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        hsi.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        hsi.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        hsi.maxLod = 0.0f;
        if (vkCreateSampler(ctx_.device(), &hsi, nullptr, &historySampler_) != VK_SUCCESS) {
            ctx_.setError("RayTracingScene: history/volfog sampler creation failed.");
            return false;
        }
    }
    std::vector<uint32_t> spirvInject; if(!readSpirv(shaderDirectory_ + "\\volfog_inject.spv",spirvInject) && !readSpirv(shaderDirectory_ + "/volfog_inject.spv",spirvInject)){ctx_.setError("RayTracingScene: missing volfog_inject.spv in shader directory.");return false;}
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize=spirvInject.size()*4; smi.pCode=spirvInject.data(); if(vkCreateShaderModule(ctx_.device(),&smi,nullptr,&volFogInjectShader_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: vkCreateShaderModule(volfog_inject) failed.");return false;}
    std::vector<uint32_t> spirvIntegrate; if(!readSpirv(shaderDirectory_ + "\\volfog_integrate.spv",spirvIntegrate) && !readSpirv(shaderDirectory_ + "/volfog_integrate.spv",spirvIntegrate)){ctx_.setError("RayTracingScene: missing volfog_integrate.spv in shader directory.");return false;}
    VkShaderModuleCreateInfo smg{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smg.codeSize=spirvIntegrate.size()*4; smg.pCode=spirvIntegrate.data(); if(vkCreateShaderModule(ctx_.device(),&smg,nullptr,&volFogIntegrateShader_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: vkCreateShaderModule(volfog_integrate) failed.");return false;}
    std::vector<uint32_t> spirvDepth; if(!readSpirv(shaderDirectory_ + "\volfog_depth.spv",spirvDepth) && !readSpirv(shaderDirectory_ + "/volfog_depth.spv",spirvDepth)){ctx_.setError("RayTracingScene: missing volfog_depth.spv in shader directory.");return false;}
    VkShaderModuleCreateInfo smd{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smd.codeSize=spirvDepth.size()*4; smd.pCode=spirvDepth.data(); if(vkCreateShaderModule(ctx_.device(),&smd,nullptr,&volFogDepthShader_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: vkCreateShaderModule(volfog_depth) failed.");return false;}

    VkDescriptorSetLayoutBinding b[17]{};
    b[0]={0,VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[1]={1,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[2]={2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[3]={3,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[4]={4,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[5]={5,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[6]={6,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    // the path tracer's scene data (rt_common.glsl), so a direction sample can evaluate the glow it
    // hits
    b[13]={13,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[14]={14,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,kMaxPathTraceTextures,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[15]={15,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[7]={7,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[8]={8,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[9]={9,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[10]={10,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};  // light grid
    // this frame's light volume (11) and last frame's (12)
    b[11]={11,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[12]={12,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    b[16]={16,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};  // volfogcull column depths
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};dl.bindingCount=17;dl.pBindings=b;if(vkCreateDescriptorSetLayout(ctx_.device(),&dl,nullptr,&descriptorSetLayoutVolFog_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: volfog descriptor set layout failed.");return false;}
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&descriptorSetLayoutVolFog_;if(vkCreatePipelineLayout(ctx_.device(),&pl,nullptr,&pipelineLayoutVolFog_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: volfog pipeline layout failed.");return false;}
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cpi.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};cpi.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cpi.stage.module=volFogInjectShader_;cpi.stage.pName="main";cpi.layout=pipelineLayoutVolFog_;if(vkCreateComputePipelines(ctx_.device(),VK_NULL_HANDLE,1,&cpi,nullptr,&volFogInjectPipeline_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: volfog compute pipeline (inject) failed.");return false;}
    VkComputePipelineCreateInfo cpg{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cpg.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};cpg.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cpg.stage.module=volFogIntegrateShader_;cpg.stage.pName="main";cpg.layout=pipelineLayoutVolFog_;if(vkCreateComputePipelines(ctx_.device(),VK_NULL_HANDLE,1,&cpg,nullptr,&volFogIntegratePipeline_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: volfog compute pipeline (integrate) failed.");return false;}

    VkComputePipelineCreateInfo cpd{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cpd.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};cpd.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cpd.stage.module=volFogDepthShader_;cpd.stage.pName="main";cpd.layout=pipelineLayoutVolFog_;if(vkCreateComputePipelines(ctx_.device(),VK_NULL_HANDLE,1,&cpd,nullptr,&volFogDepthPipeline_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: volfog compute pipeline (depth) failed.");return false;}
    reportPipelineStats("volfog_inject.spv", pipelineLayoutVolFog_);  // pipeline stats (env SAUER_PIPELINE_STATS only)
    reportPipelineStats("volfog_integrate.spv", pipelineLayoutVolFog_);
    VkDescriptorPoolSize ps[5]={{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,dispatchSlots_},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,dispatchSlots_},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,dispatchSlots_*(4u+kMaxPathTraceTextures)},{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,dispatchSlots_*4u},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,dispatchSlots_*6u}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dpi.maxSets=dispatchSlots_;dpi.poolSizeCount=5;dpi.pPoolSizes=ps;if(vkCreateDescriptorPool(ctx_.device(),&dpi,nullptr,&descriptorPoolVolFog_)!=VK_SUCCESS){ctx_.setError("RayTracingScene: volfog descriptor pool creation failed.");return false;}
    if (!ensureLightGridPlaceholder()) return false;
    volFogLightBuffers_.resize(dispatchSlots_);
    for (auto& lb : volFogLightBuffers_) {
        if (!ensureBufferCapacity(lb, 256u * 12u * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            ctx_.setError("RayTracingScene: volfog light buffer allocation failed.");
            return false;
        }
    }

    // column depths
    volFogDepthBuffers_.resize(dispatchSlots_);
    for (auto& db : volFogDepthBuffers_) {
        if (!ensureBufferCapacity(db, 320u * 180u * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            ctx_.setError("RayTracingScene: volfog depth buffer allocation failed.");
            return false;
        }
    }

    volFogParamSlots_.resize(dispatchSlots_);
    for (auto& s : volFogParamSlots_) {
        Buffer buf; if(!createBuffer(sizeof(GpuVolFogConstants),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,buf)){ctx_.setError("RayTracingScene: volfog parameter buffer allocation failed.");return false;}
        s.buffer=buf.buffer;s.memory=buf.memory;vkMapMemory(ctx_.device(),s.memory,0,sizeof(GpuVolFogConstants),0,&s.mapped);buf.buffer=VK_NULL_HANDLE;buf.memory=VK_NULL_HANDLE;
    }

    volFogNoisePlaceholder_ = std::make_unique<SharedTexture3D>(ctx_);
    if (!volFogNoisePlaceholder_->create(1, 1, 1, PixelFormat::R16Float)) {
        ctx_.setError("RayTracingScene: volfog noise placeholder texture creation failed: " + ctx_.lastError());
        return false;
    }
    VkImageViewCreateInfo noiseViewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    noiseViewInfo.image = volFogNoisePlaceholder_->vkImage();
    noiseViewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
    noiseViewInfo.format = volFogNoisePlaceholder_->vkFormat();
    noiseViewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(ctx_.device(), &noiseViewInfo, nullptr, &volFogNoisePlaceholderView_) != VK_SUCCESS) {
        ctx_.setError("RayTracingScene: volfog noise placeholder image view creation failed.");
        return false;
    }
    return true;
}

// allocates (or, on a dimension change, destroys and recreates) the four persistent froxel-grid
// textures
bool RayTracingScene::ensureVolFogTextures(uint32_t gridX, uint32_t gridY, uint32_t gridZ, std::string* errorOut) {
    if (volFogVisibilityA_ && volFogGridX_ == gridX && volFogGridY_ == gridY && volFogGridZ_ == gridZ) return true;
    auto destroyView = [this](VkImageView& v) { if (v != VK_NULL_HANDLE) { vkDestroyImageView(ctx_.device(), v, nullptr); v = VK_NULL_HANDLE; } };
    destroyView(volFogVisibilityAView_);
    destroyView(volFogVisibilityBView_);
    destroyView(volFogScatterExtinctionView_);
    destroyView(volFogScatterExtinctionBView_);
    destroyView(volFogIntegratedView_);
    destroyView(volFogPartLightAView_);
    destroyView(volFogPartLightBView_);
    volFogPartLightA_.reset();
    volFogPartLightB_.reset();
    volFogVisibilityA_.reset();
    volFogVisibilityB_.reset();
    volFogScatterExtinction_.reset();
    volFogScatterExtinctionB_.reset();
    volFogIntegrated_.reset();

    volFogVisibilityA_ = std::make_unique<SharedTexture3D>(ctx_);
    volFogVisibilityB_ = std::make_unique<SharedTexture3D>(ctx_);
    volFogScatterExtinction_ = std::make_unique<SharedTexture3D>(ctx_);
    volFogScatterExtinctionB_ = std::make_unique<SharedTexture3D>(ctx_);
    volFogIntegrated_ = std::make_unique<SharedTexture3D>(ctx_);
    volFogPartLightA_ = std::make_unique<SharedTexture3D>(ctx_);
    volFogPartLightB_ = std::make_unique<SharedTexture3D>(ctx_);
    if (!volFogPartLightA_->create(gridX, gridY, gridZ, PixelFormat::Rgba16Float) ||
        !volFogPartLightB_->create(gridX, gridY, gridZ, PixelFormat::Rgba16Float) ||
        !volFogVisibilityA_->create(gridX, gridY, gridZ, PixelFormat::R16Float) ||
        !volFogVisibilityB_->create(gridX, gridY, gridZ, PixelFormat::R16Float) ||
        !volFogScatterExtinction_->create(gridX, gridY, gridZ, PixelFormat::Rgba16Float) ||
        !volFogScatterExtinctionB_->create(gridX, gridY, gridZ, PixelFormat::Rgba16Float) ||
        !volFogIntegrated_->create(gridX, gridY, gridZ, PixelFormat::Rgba16Float)) {
        if (errorOut) *errorOut = "RayTracingScene::ensureVolFogTextures: SharedTexture3D::create failed: " + ctx_.lastError();
        volFogVisibilityA_.reset(); volFogVisibilityB_.reset(); volFogScatterExtinction_.reset(); volFogScatterExtinctionB_.reset(); volFogIntegrated_.reset();
        volFogPartLightA_.reset(); volFogPartLightB_.reset();
        volFogGridX_ = volFogGridY_ = volFogGridZ_ = 0;
        return false;
    }

    auto makeView = [this](SharedTexture3D& tex, VkImageView& outView) -> bool {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = tex.vkImage(); vi.viewType = VK_IMAGE_VIEW_TYPE_3D; vi.format = tex.vkFormat();
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        return vkCreateImageView(ctx_.device(), &vi, nullptr, &outView) == VK_SUCCESS;
    };
    if (!makeView(*volFogVisibilityA_, volFogVisibilityAView_) ||
        !makeView(*volFogVisibilityB_, volFogVisibilityBView_) ||
        !makeView(*volFogScatterExtinction_, volFogScatterExtinctionView_) ||
        !makeView(*volFogScatterExtinctionB_, volFogScatterExtinctionBView_) ||
        !makeView(*volFogIntegrated_, volFogIntegratedView_) ||
        !makeView(*volFogPartLightA_, volFogPartLightAView_) ||
        !makeView(*volFogPartLightB_, volFogPartLightBView_)) {
        if (errorOut) *errorOut = "RayTracingScene::ensureVolFogTextures: image view creation failed";
        return false;
    }
    volFogGridX_ = gridX; volFogGridY_ = gridY; volFogGridZ_ = gridZ;
    return true;
}

bool RayTracingScene::createVolFogDescriptorSets(uint32_t count, std::vector<VkDescriptorSet>& sets, std::string* errorOut) {
    if (count > dispatchSlots_) { if (errorOut) *errorOut = "requested descriptor-set count exceeds scene dispatch slot count"; return false; }
    std::vector<VkDescriptorSetLayout> layouts(count, descriptorSetLayoutVolFog_);
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; ai.descriptorPool=descriptorPoolVolFog_; ai.descriptorSetCount=count; ai.pSetLayouts=layouts.data();
    if (vkAllocateDescriptorSets(ctx_.device(), &ai, sets.data()) != VK_SUCCESS) { if (errorOut) *errorOut = "vkAllocateDescriptorSets for volumetric fog failed"; return false; }
    return true;
}

bool RayTracingScene::updateVolFogDescriptorSet(VkDescriptorSet set, uint32_t slotIndex,
                         const GpuVolFogConstants& constants, std::string* errorOut) {
    if (slotIndex >= volFogParamSlots_.size()) { if (errorOut) *errorOut = "invalid volfog frame slot"; return false; }
    if (!volFogVisibilityA_ || !volFogVisibilityB_ || !volFogScatterExtinction_ || !volFogScatterExtinctionB_ || !volFogIntegrated_ || !volFogNoisePlaceholder_) {
        if (errorOut) *errorOut = "updateVolFogDescriptorSet called before ensureVolFogTextures/ensureVolumetricFogPipeline";
        return false;
    }
    GpuVolFogConstants cst = constants;
    const uint32_t fogLightCount = static_cast<uint32_t>(volFogPendingLights_.size() / 12);
    if (slotIndex >= volFogLightBuffers_.size()) { if (errorOut) *errorOut = "volfog light buffers not allocated"; return false; }
    {
        Buffer& lb = volFogLightBuffers_[slotIndex];
        const VkDeviceSize need = static_cast<VkDeviceSize>(std::max(fogLightCount, 1u)) * 12u * sizeof(float);
        if (lb.size < need) {
            if (vkDeviceWaitIdle(ctx_.device()) != VK_SUCCESS) { if (errorOut) *errorOut = "vkDeviceWaitIdle failed growing the volfog light buffer"; return false; }
            if (!ensureBufferCapacity(lb, need, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                if (errorOut) *errorOut = "volfog light buffer allocation failed";
                return false;
            }
        }
        if (fogLightCount > 0) {
            void* mapped = nullptr;
            if (vkMapMemory(ctx_.device(), lb.memory, 0, need, 0, &mapped) != VK_SUCCESS) { if (errorOut) *errorOut = "vkMapMemory(volfog lights) failed"; return false; }
            std::memcpy(mapped, volFogPendingLights_.data(), static_cast<size_t>(fogLightCount) * 12u * sizeof(float));
            vkUnmapMemory(ctx_.device(), lb.memory);
        }
    }
    const bool haveEmissiveBuf = pathTraceEmissiveBuffer_.buffer != VK_NULL_HANDLE;
    cst.reservedFrameData = fogLightCount;
    cst.localLightParams[2] = haveEmissiveBuf ? pathTraceEmissiveTotalPower_ : 0.0f;
    cst.localLightParams[3] = haveEmissiveBuf ? static_cast<float>(pathTraceEmissiveCount_) : 0.0f;
    {
        auto flatLava = chunkMeshes_.find(lavaFlatChunkId_);
        cst.lavaFog[0] = (lavaFlatChunkId_ != 0 && flatLava != chunkMeshes_.end()) ? float(flatLava->second.materialSlot) : -1.0f;
    }
    // same availability rule as RTAO's shared scene data
    const Buffer* fogSceneMaterials = (!dynamicGenerations_.empty() && dynamicGenerations_[dynamicGenerationIndex_].materials.buffer != VK_NULL_HANDLE)
        ? &dynamicGenerations_[dynamicGenerationIndex_].materials
        : (pathTraceMaterialsBuffer_.buffer != VK_NULL_HANDLE ? &pathTraceMaterialsBuffer_ : nullptr);
    const bool fogSceneData = fogSceneMaterials != nullptr && !pathTraceTextures_.empty() && pathTraceMaterialInfoBuffer_.buffer != VK_NULL_HANDLE;
    cst.emissiveFog[0] = (fogSceneData && constants.emissiveFog[1] > 0.5f) ? 1.0f : 0.0f;  // [1] = volfogglowmis
    std::memcpy(volFogParamSlots_[slotIndex].mapped, &cst, sizeof(GpuVolFogConstants));

    volFogPingPong_ = !volFogPingPong_;
    VkImageView visPrevView = volFogPingPong_ ? volFogVisibilityBView_ : volFogVisibilityAView_;
    VkImageView visNextView = volFogPingPong_ ? volFogVisibilityAView_ : volFogVisibilityBView_;
    // same A/B alternation, same toggle
    VkImageView scatterPrevView = volFogPingPong_ ? volFogScatterExtinctionBView_ : volFogScatterExtinctionView_;
    VkImageView scatterNextView = volFogPingPong_ ? volFogScatterExtinctionView_ : volFogScatterExtinctionBView_;

    // local copy, not a pointer into a member field
    VkAccelerationStructureKHR currentTlas = tlas();
    VkWriteDescriptorSetAccelerationStructureKHR asInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    asInfo.accelerationStructureCount = 1;
    asInfo.pAccelerationStructures = &currentTlas;

    VkDescriptorBufferInfo bufInfo{volFogParamSlots_[slotIndex].buffer, 0, sizeof(GpuVolFogConstants)};
    VkDescriptorImageInfo visPrevInfo{historySampler_, visPrevView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo visNextInfo{VK_NULL_HANDLE, visNextView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo scatterInfo{VK_NULL_HANDLE, scatterNextView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo integratedInfo{VK_NULL_HANDLE, volFogIntegratedView_, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo noiseInfo{historySampler_, volFogNoisePlaceholderView_, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo scatterPrevInfo{historySampler_, scatterPrevView, VK_IMAGE_LAYOUT_GENERAL};

    VkDescriptorBufferInfo lightsInfo{volFogLightBuffers_[slotIndex].buffer, 0, VK_WHOLE_SIZE};
    // no emissive list yet (path-trace pipeline not built)
    VkDescriptorBufferInfo emissiveInfo{haveEmissiveBuf ? pathTraceEmissiveBuffer_.buffer : volFogLightBuffers_[slotIndex].buffer, 0, VK_WHOLE_SIZE};

    VkDescriptorBufferInfo lightGridInfo{lightGridBuffer_.buffer, 0, VK_WHOLE_SIZE};
    // same A/B toggle
    VkDescriptorImageInfo partLightNextInfo{VK_NULL_HANDLE, volFogPingPong_ ? volFogPartLightAView_ : volFogPartLightBView_, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo partLightPrevInfo{historySampler_, volFogPingPong_ ? volFogPartLightBView_ : volFogPartLightAView_, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo colDepthInfo{volFogDepthBuffers_[slotIndex].buffer, 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet w[14]{};
    w[0]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w[0].pNext=&asInfo;w[0].dstSet=set;w[0].dstBinding=0;w[0].descriptorCount=1;w[0].descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    w[1]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w[1].dstSet=set;w[1].dstBinding=1;w[1].descriptorCount=1;w[1].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;w[1].pBufferInfo=&bufInfo;
    w[2]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w[2].dstSet=set;w[2].dstBinding=2;w[2].descriptorCount=1;w[2].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;w[2].pImageInfo=&visPrevInfo;
    w[3]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w[3].dstSet=set;w[3].dstBinding=3;w[3].descriptorCount=1;w[3].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;w[3].pImageInfo=&visNextInfo;
    w[4]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w[4].dstSet=set;w[4].dstBinding=4;w[4].descriptorCount=1;w[4].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;w[4].pImageInfo=&scatterInfo;
    w[5]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w[5].dstSet=set;w[5].dstBinding=5;w[5].descriptorCount=1;w[5].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;w[5].pImageInfo=&integratedInfo;
    w[6]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w[6].dstSet=set;w[6].dstBinding=6;w[6].descriptorCount=1;w[6].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;w[6].pImageInfo=&noiseInfo;
    w[7]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w[7].dstSet=set;w[7].dstBinding=7;w[7].descriptorCount=1;w[7].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;w[7].pImageInfo=&scatterPrevInfo;
    w[8]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w[8].dstSet=set;w[8].dstBinding=8;w[8].descriptorCount=1;w[8].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w[8].pBufferInfo=&lightsInfo;
    w[9]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w[9].dstSet=set;w[9].dstBinding=9;w[9].descriptorCount=1;w[9].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w[9].pBufferInfo=&emissiveInfo;
    w[10]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w[10].dstSet=set;w[10].dstBinding=10;w[10].descriptorCount=1;w[10].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w[10].pBufferInfo=&lightGridInfo;
    w[11]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w[11].dstSet=set;w[11].dstBinding=11;w[11].descriptorCount=1;w[11].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;w[11].pImageInfo=&partLightNextInfo;
    w[12]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w[12].dstSet=set;w[12].dstBinding=12;w[12].descriptorCount=1;w[12].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;w[12].pImageInfo=&partLightPrevInfo;
    w[13]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w[13].dstSet=set;w[13].dstBinding=16;w[13].descriptorCount=1;w[13].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w[13].pBufferInfo=&colDepthInfo;
    vkUpdateDescriptorSets(ctx_.device(), 14, w, 0, nullptr);
    if (fogSceneData) {
        VkDescriptorBufferInfo dbMaterials{fogSceneMaterials->buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo dbMaterialInfo{pathTraceMaterialInfoBuffer_.buffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet sw[3]{};
        sw[0] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        sw[0].dstSet = set; sw[0].dstBinding = 13; sw[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; sw[0].descriptorCount = 1; sw[0].pBufferInfo = &dbMaterials;
        sw[1] = sw[0]; sw[1].dstBinding = 15; sw[1].pBufferInfo = &dbMaterialInfo;
        uint32_t swCount = 2;
        std::vector<VkDescriptorImageInfo> textureInfos;
        if (volFogTexGenWritten_.size() < dispatchSlots_) volFogTexGenWritten_.resize(dispatchSlots_, ~0ull);
        if (volFogTexGenWritten_[slotIndex] != pathTraceStaticDescriptorGeneration_) {
            textureInfos.assign(kMaxPathTraceTextures, VkDescriptorImageInfo{pathTraceTextures_[0].sampler, pathTraceTextures_[0].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
            for (size_t i = 0; i < pathTraceTextures_.size() && i < textureInfos.size(); ++i)
                textureInfos[i] = {pathTraceTextures_[i].sampler, pathTraceTextures_[i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            sw[2] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            sw[2].dstSet = set; sw[2].dstBinding = 14; sw[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            sw[2].descriptorCount = static_cast<uint32_t>(textureInfos.size()); sw[2].pImageInfo = textureInfos.data();
            swCount = 3;
            volFogTexGenWritten_[slotIndex] = pathTraceStaticDescriptorGeneration_;
        }
        vkUpdateDescriptorSets(ctx_.device(), swCount, sw, 0, nullptr);
    }
    return true;
}

bool RayTracingScene::ensureLightGridPlaceholder() {
    if (lightGridBuffer_.buffer != VK_NULL_HANDLE) return true;
    uint32_t empty[16] = {};
    std::string e;
    if (!setLightGrid(empty, 16, &e)) { ctx_.setError("RayTracingScene: light grid placeholder failed: " + e); return false; }
    return true;
}

void RayTracingScene::fillBufferNow(Buffer& buf, uint32_t value) {
    if (buf.buffer == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(ctx_.device());
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = ctx_.setupCommandPool(); cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(ctx_.device(), &cai, &cmd) != VK_SUCCESS) return;
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbi);
    vkCmdFillBuffer(cmd, buf.buffer, 0, VK_WHOLE_SIZE, value);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx_.queue(), 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(ctx_.queue());
    vkFreeCommandBuffers(ctx_.device(), ctx_.setupCommandPool(), 1, &cmd);
}

void RayTracingScene::clearRadianceCache() {
    if (rcKeyBuffer_.buffer == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(ctx_.device());
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = ctx_.setupCommandPool(); cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(ctx_.device(), &cai, &cmd) != VK_SUCCESS) return;
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbi);
    vkCmdFillBuffer(cmd, rcKeyBuffer_.buffer, 0, VK_WHOLE_SIZE, 0u);
    vkCmdFillBuffer(cmd, rcAccumBuffer_.buffer, 0, VK_WHOLE_SIZE, 0u);
    vkCmdFillBuffer(cmd, rcRadianceBuffer_.buffer, 0, VK_WHOLE_SIZE, 0u);
    if (skyCdfBuffer_.buffer != VK_NULL_HANDLE) vkCmdFillBuffer(cmd, skyCdfBuffer_.buffer, 0, VK_WHOLE_SIZE, 0u);  // rebuilt by the next SKY_CDF_PASS
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx_.queue(), 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(ctx_.queue());
    vkFreeCommandBuffers(ctx_.device(), ctx_.setupCommandPool(), 1, &cmd);
}

bool RayTracingScene::setLightGrid(const uint32_t* data, size_t count, std::string* errorOut) {
    if (!data || count < 16) { if (errorOut) *errorOut = "setLightGrid: empty data"; return false; }
    const VkDeviceSize size = static_cast<VkDeviceSize>(count) * sizeof(uint32_t);
    if (lightGridBuffer_.buffer != VK_NULL_HANDLE) {
        if (vkDeviceWaitIdle(ctx_.device()) != VK_SUCCESS) { if (errorOut) *errorOut = "setLightGrid: vkDeviceWaitIdle failed"; return false; }
        destroyBuffer(lightGridBuffer_);
    }
    Buffer staging;
    if (!createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, lightGridBuffer_) ||
        !createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging)) {
        destroyBuffer(lightGridBuffer_);
        if (errorOut) *errorOut = "setLightGrid: buffer allocation failed";
        return false;
    }
    void* mapped = nullptr;
    vkMapMemory(ctx_.device(), staging.memory, 0, size, 0, &mapped);
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vkUnmapMemory(ctx_.device(), staging.memory);
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = ctx_.setupCommandPool(); cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(ctx_.device(), &cai, &cmd);
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbi);
    VkBufferCopy bc{0, 0, size}; vkCmdCopyBuffer(cmd, staging.buffer, lightGridBuffer_.buffer, 1, &bc);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx_.queue(), 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(ctx_.queue());
    vkFreeCommandBuffers(ctx_.device(), ctx_.setupCommandPool(), 1, &cmd);
    destroyBuffer(staging);
    return true;
}

void RayTracingScene::setVolFogLights(const float* packed, uint32_t count) {
    if (!packed || count == 0) { volFogPendingLights_.clear(); return; }
    volFogPendingLights_.assign(packed, packed + static_cast<size_t>(count) * 12u);
}

bool RayTracingScene::createPathTraceDescriptorSets(uint32_t count, std::vector<VkDescriptorSet>& sets, std::string* errorOut) {
    if (count > dispatchSlots_) { if (errorOut) *errorOut = "requested descriptor-set count exceeds scene dispatch slot count"; return false; }
    std::vector<VkDescriptorSetLayout> layouts(count, descriptorSetLayoutPathTrace_);
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; ai.descriptorPool=descriptorPoolPathTrace_; ai.descriptorSetCount=count; ai.pSetLayouts=layouts.data();
    if (vkAllocateDescriptorSets(ctx_.device(), &ai, sets.data()) != VK_SUCCESS) { if (errorOut) *errorOut = "vkAllocateDescriptorSets for path tracing failed"; return false; }
    return true;
}

bool RayTracingScene::updatePathTraceDescriptorSet(VkDescriptorSet set, SharedTexture& depth, SharedTexture& normal,
                         SharedTexture& albedo, SharedTexture& motion, SharedTexture& output,
                         SharedTexture& historyPrev, SharedTexture& historyNext,
                         SharedTexture& historyMetaPrev, SharedTexture& historyMetaNext,
                         SharedTexture& rawRadiance, SharedTexture& glow, SharedTexture& primaryNormal,
                         SharedTexture& linearDepth,
                         VkImageView depthView, VkImageView normalView, VkImageView albedoView,
                         VkImageView motionView, VkImageView outputView,
                         VkImageView historyPrevView, VkImageView historyNextView,
                         VkImageView historyMetaPrevView, VkImageView historyMetaNextView,
                         VkImageView rawRadianceView, VkImageView glowView, VkImageView primaryNormalView,
                         VkImageView linearDepthView,
                         uint32_t slotIndex, const PathTraceConstants& c, std::string* errorOut) {
    if (slotIndex >= pathTraceParamSlots_.size() || slotIndex >= pathTraceLightBuffers_.size()) {
        if (errorOut) *errorOut = "invalid path-trace frame slot";
        return false;
    }
    GpuPathTraceParams p{};
    std::memcpy(p.invViewProj, c.invViewProj, sizeof(p.invViewProj));
    std::memcpy(p.invView, c.invView, sizeof(p.invView));
    std::memcpy(p.prevViewProj, c.prevViewProj, sizeof(p.prevViewProj));
    p.frameData[0] = c.frameIndex;
    p.frameData[1] = c.bounceCount;
    p.frameData[2] = c.historyValid ? 1u : 0u;
    p.frameData[3] = pathTraceLightCounts_[slotIndex];
    p.sunDirectionAndRadius[0] = c.sunDirection[0];
    p.sunDirectionAndRadius[1] = c.sunDirection[1];
    p.sunDirectionAndRadius[2] = c.sunDirection[2];
    p.sunDirectionAndRadius[3] = c.sunAngularRadius;
    p.sunColor[0] = c.sunColor[0]; p.sunColor[1] = c.sunColor[1]; p.sunColor[2] = c.sunColor[2]; p.sunColor[3] = c.maxHistorySamples;
    p.skyAmbient[0] = c.skyAmbient[0]; p.skyAmbient[1] = c.skyAmbient[1]; p.skyAmbient[2] = c.skyAmbient[2];
    // cross-checked against the uploaded count (pathTraceEmissiveCount_)
    p.skyAmbient[3] = static_cast<float>(pathTraceEmissiveCount_);
    p.ambientFloor[0] = c.ambientFloor[0]; p.ambientFloor[1] = c.ambientFloor[1]; p.ambientFloor[2] = c.ambientFloor[2];
    // pbr light transport
    p.ambientFloor[3] = c.glowScale;
    p.emissiveImportance[0] = pathTraceEmissiveTotalPower_;
    p.emissiveImportance[1] = c.pixelSpreadAngle;
    p.emissiveImportance[2] = static_cast<float>(dynamicMaterialBase_);
    p.emissiveImportance[3] = c.glassThinness;
    std::memcpy(p.viewProj, c.viewProj, sizeof(p.viewProj));
    std::memcpy(p.prevViewProjJittered, c.prevViewProjJittered, sizeof(p.prevViewProjJittered));
    // skybox
    p.skyFaceIndexLo[0] = c.skyFaceTexIndex[0];
    p.skyFaceIndexLo[1] = c.skyFaceTexIndex[1];
    p.skyFaceIndexLo[2] = c.skyFaceTexIndex[2];
    p.skyFaceIndexLo[3] = c.skyFaceTexIndex[3];
    p.skyFaceIndexHi[0] = c.skyFaceTexIndex[4];
    p.skyFaceIndexHi[1] = c.skyFaceTexIndex[5];
    p.skyFaceIndexHi[2] = c.skyValid ? 1u : 0u;
    p.skyFaceIndexHi[3] = c.blueNoiseTexIndex;
    p.skyRotationPad[0] = c.skyRotationRadians;
    p.skyRotationPad[1] = c.skyLightScale;
    p.skyRotationPad[2] = c.lightmapClamp ? 1.0f : 0.0f;
    p.skyRotationPad[3] = c.plainSpecular;
    p.pomParams[0] = c.pomEnabled ? 1.0f : 0.0f;
    p.pomParams[1] = static_cast<float>(c.pomMaxIterations);
    p.pomParams[2] = c.pomDepthScale;
    p.pomParams[3] = static_cast<float>(c.pomMode);
    p.glassParams[0] = c.glassVolumeTint;
    p.glassParams[1] = c.alphaGlassTint;
    p.glassParams[2] = c.modelEnvReflect;
    p.glassParams[3] = c.rayStats ? (c.rayStatsProfile ? 2.0f : 1.0f) : 0.0f;  // path-trace stats toggle (2 = + profile)
    {
        uint32_t photons = causticPhotonDispatchCount(c);
        if (photons > causticCapacity_ && !ensureCausticCapacity(photons)) photons = causticCapacity_;  // couldn't grow
        p.causticParams[0] = photons > 0 ? 1.0f : 0.0f;
        p.causticParams[1] = c.causticRadius;
        p.causticParams[2] = static_cast<float>(photons);
        p.causticParams[3] = c.causticRange;  // photon window half-size
        // every batch is re-emitted every frame (the waves move), so all are valid whenever the pass
        // runs
        causticValidMask_ = photons > 0 ? (1u << kCausticBatches) - 1u : 0u;
        p.causticParams2[0] = c.causticRipple;
        p.causticParams2[1] = static_cast<float>(causticValidMask_);
        p.causticParams2[2] = c.textureGradScale;
        p.causticParams2[3] = static_cast<float>((c.lightSampleBounces ? 1u : 0u) |
                                                 (std::min(c.lightSplitExact, 8u) << 1) |
                                                 (std::min(c.lightSplitRandom, 15u) << 5) |
                                                 ((c.risEnabled ? 1u : 0u) << 9) |
                                                 (std::clamp(c.risCandidates, 1u, 63u) << 10) |
                                                 (std::clamp(c.risSamples, 1u, 4u) << 16) |
                                                 (std::min(c.perfCut, 7u) << 19) | ((c.perfSkipShadow ? 1u : 0u) << 22) |  // perf diagnostic (ptcut, ptcutshadow)
                                                 ((c.risImportance ? 1u : 0u) << 23));  // bit 23
    }
    p.waterParams[0] = c.cameraInWater ? 1.0f : 0.0f;
    p.waterParams[1] = c.waterClarity;
    p.waterParams[2] = c.waterWaves;
    p.waterParams[3] = c.timeSeconds;
    p.waterCamera[0] = c.cameraWaterColor[0]; p.waterCamera[1] = c.cameraWaterColor[1]; p.waterCamera[2] = c.cameraWaterColor[2];
    p.waterCamera[3] = c.cameraWaterDepth;
    p.waterParams2[0] = c.waterScatter;
    p.waterParams2[1] = c.cameraWaterSurfaceZ;
    p.waterParams2[2] = c.waterHistory;
    p.waterParams2[3] = c.waterDebug;  // ptwaterdebug
    // off when the maintenance pipeline is missing
    p.cacheParams[0] = (c.cacheEnabled && cachePipeline_ != VK_NULL_HANDLE) ? 1.0f : 0.0f;
    p.cacheParams[1] = c.cacheCell;
    p.cacheParams[2] = c.cacheFrames;
    p.cacheParams[3] = c.cacheUpdate;
    p.cacheParams2[0] = c.cacheFloor;
    p.cacheParams2[1] = c.cacheMinFrames;
    p.cacheParams2[2] = c.cacheEvict;
    p.cacheParams2[3] = static_cast<float>((c.cacheFallback ? 1u : 0u) | (c.cacheCoherent ? 2u : 0u));  // bit0 fallback, bit1 coherent update paths
    // off when the CDF pipeline is missing
    p.skyParams[0] = (c.skyboxLight && skyCdfPipeline_ != VK_NULL_HANDLE) ? 1.0f : 0.0f;
    p.skyParams[1] = p.skyParams[2] = p.skyParams[3] = 0.0f;
    p.matParams[0] = c.physicalMaterials ? 1.0f : 0.0f;
    p.particleParams[0] = c.particleReflect;
    {
        bool anyParticles = false;
        if (dynamicGenerationIndex_ < dynamicGenerations_.size()) {
            const DynamicGeneration& g = dynamicGenerations_[dynamicGenerationIndex_];
            for (size_t i = 0; i < g.slotsUsed && !anyParticles; ++i) anyParticles = g.slots[i].particles;
        }
        p.particleParams[1] = anyParticles ? 1.0f : 0.0f;
    }
    p.particleParams[2] = p.particleParams[3] = 0.0f;
    std::memset(p.waterPatch, 0, sizeof(p.waterPatch));
    if (dynamicGenerationIndex_ < waterPatchGens_.size() && waterPatchGens_[dynamicGenerationIndex_].active) {
        const WaterPatchGen& wgen = waterPatchGens_[dynamicGenerationIndex_];
        auto flat = chunkMeshes_.find(waterFlatChunkId_);
        auto flatLava = chunkMeshes_.find(lavaFlatChunkId_);
        p.waterPatch[0] = wgen.originX;
        p.waterPatch[1] = wgen.originY;
        p.waterPatch[2] = float(wgen.cells) * wgen.cellSize;
        p.waterPatch[3] = flat != chunkMeshes_.end() ? float(flat->second.materialSlot) : -1.0f;
        p.lavaPatch[0] = flatLava != chunkMeshes_.end() ? float(flatLava->second.materialSlot) : -1.0f;
        p.lavaPatch[1] = float(lavaTopTris_);
    } else {
        p.lavaPatch[0] = -1.0f; p.lavaPatch[1] = 0.0f;
    }
    p.lavaPatch[2] = c.lavaTintHue; p.lavaPatch[3] = c.lavaTintSat;  // lava colour (rt_lava.glsl lavaTint)
    p.waterWave[0] = waterPatchSettings_.height;
    p.lava0[0] = c.lavaSize; p.lava0[1] = c.lavaFlow; p.lava0[2] = c.lavaRelief; p.lava0[3] = c.lavaCrack;
    p.lava1[0] = c.lavaTemp; p.lava1[1] = c.lavaCrustTemp; p.lava1[2] = c.timeSeconds * c.lavaSpeed; p.lava1[3] = c.lavaCrustAmount;
    p.waterVolume[0] = c.waterRays; p.waterVolume[1] = c.waterRaysG; p.waterVolume[2] = c.waterRaysFocus; p.waterVolume[3] = c.waterFlowDt;
    p.rrParams[0] = c.rrSpecularMotion ? 1.0f : 0.0f;
    p.rrParams[1] = c.restirEnabled ? 1.0f + static_cast<float>(c.restirMinLights) : 0.0f;  // 1 + min lights in range
    // bits 0-2 ReSTIR spatial neighbours, 3-5 bounce RIS samples, 6-11 bounce RIS candidates
    p.rrParams[2] = static_cast<float>(std::min(c.restirSpatial, 4u) | (std::clamp(c.risBounceSamples, 1u, 4u) << 3) |
                                       (std::clamp(c.risBounceCandidates, 1u, 63u) << 6));
    p.rrParams[3] = static_cast<float>(std::clamp(c.restirCandidates, 1u, 32u));
    p.waterWave[2] = float(waterTopTris_);  // setWaterSurfaces' contract
    {
        static const WaterPatchGen kNoGen{};
        const WaterPatchGen& wgen = dynamicGenerationIndex_ < waterPatchGens_.size() ? waterPatchGens_[dynamicGenerationIndex_] : kNoGen;
        float amp[4];
        waveShaderParams(wgen, p.waterFft, p.waterMap, amp);
        p.waterWave[1] = amp[1];
        p.waterWave[3] = amp[3];
    }
    p.matParams[1] = c.transportAlbedo ? 1.0f : 0.0f;
    p.matParams[2] = c.transportAlbedoGamma;
    p.matParams[3] = 0.0f;
    std::memcpy(pathTraceParamSlots_[slotIndex].mapped, &p, sizeof(GpuPathTraceParams));

    VkDescriptorImageInfo di0{sampler_, depthView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo di1{sampler_, normalView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo di2{sampler_, albedoView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dout{VK_NULL_HANDLE, outputView, VK_IMAGE_LAYOUT_GENERAL};
    // historySampler_ (linear filtering), not sampler_ (nearest)
    VkDescriptorImageInfo dhprev{historySampler_, historyPrevView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dhnext{VK_NULL_HANDLE, historyNextView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dmotion{sampler_, motionView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dhmetaprev{historySampler_, historyMetaPrevView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dhmetanext{VK_NULL_HANDLE, historyMetaNextView, VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet w[13]{};
    for (auto& x : w) x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet=set; w[0].dstBinding=0; w[0].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].descriptorCount=1; w[0].pImageInfo=&di0;
    w[1]=w[0]; w[1].dstBinding=1; w[1].pImageInfo=&di1;
    w[2]=w[0]; w[2].dstBinding=2; w[2].pImageInfo=&di2;

    VkAccelerationStructureKHR currentTlas = tlas();
    VkWriteDescriptorSetAccelerationStructureKHR as{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    as.accelerationStructureCount=1; as.pAccelerationStructures=&currentTlas;
    w[3]=w[0]; w[3].dstBinding=3; w[3].descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; w[3].pNext=&as; w[3].pImageInfo=nullptr;

    VkDescriptorBufferInfo dbLights{pathTraceLightBuffers_[slotIndex].buffer, 0, VK_WHOLE_SIZE};
    w[4]=w[0]; w[4].dstBinding=4; w[4].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[4].pBufferInfo=&dbLights; w[4].pImageInfo=nullptr;

    VkDescriptorBufferInfo dbParams{pathTraceParamSlots_[slotIndex].buffer, 0, sizeof(GpuPathTraceParams)};
    w[6]=w[0]; w[6].dstBinding=6; w[6].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[6].pBufferInfo=&dbParams; w[6].pImageInfo=nullptr;

    w[7]=w[0]; w[7].dstBinding=7; w[7].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[7].pImageInfo=&dout;
    w[8]=w[0]; w[8].dstBinding=8; w[8].pImageInfo=&dhprev;
    w[9]=w[7]; w[9].dstBinding=9; w[9].pImageInfo=&dhnext;
    w[10]=w[0]; w[10].dstBinding=10; w[10].pImageInfo=&dmotion;
    w[11]=w[0]; w[11].dstBinding=11; w[11].pImageInfo=&dhmetaprev;
    w[12]=w[7]; w[12].dstBinding=12; w[12].pImageInfo=&dhmetanext;

    // per-instance material table single buffer, not slot-indexed
    const Buffer& currentMaterials = (!dynamicGenerations_.empty() && dynamicGenerations_[dynamicGenerationIndex_].materials.buffer != VK_NULL_HANDLE)
        ? dynamicGenerations_[dynamicGenerationIndex_].materials : pathTraceMaterialsBuffer_;
    VkDescriptorBufferInfo dbMaterials{currentMaterials.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wMaterials = w[4]; wMaterials.dstBinding = 5; wMaterials.pBufferInfo = &dbMaterials;

    // raw-radiance intermediate (trace/resolve split)
    VkDescriptorImageInfo drawradiance{VK_NULL_HANDLE, rawRadianceView, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wraw = w[7]; wraw.dstBinding = 14; wraw.pImageInfo = &drawradiance;

    std::vector<VkDescriptorImageInfo> textureInfos(kMaxPathTraceTextures);
    VkDescriptorImageInfo fallback{};
    if (!pathTraceTextures_.empty())
        fallback = {pathTraceTextures_[0].sampler, pathTraceTextures_[0].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    for (auto &x : textureInfos) x = fallback;
    for (size_t i=0; i<pathTraceTextures_.size() && i<textureInfos.size(); ++i)
        textureInfos[i] = {pathTraceTextures_[i].sampler, pathTraceTextures_[i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wt{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wt.dstSet=set; wt.dstBinding=13; wt.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wt.descriptorCount=static_cast<uint32_t>(textureInfos.size()); wt.pImageInfo=textureInfos.data();

    // Sauerbraten-VSlot-indexed material table
    VkDescriptorBufferInfo dbMaterialInfo{pathTraceMaterialInfoBuffer_.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wMaterialInfo = w[4]; wMaterialInfo.dstBinding = 15; wMaterialInfo.pBufferInfo = &dbMaterialInfo;

    // emissive-triangle list
    VkDescriptorBufferInfo dbEmissive{pathTraceEmissiveBuffer_.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wEmissive = w[4]; wEmissive.dstBinding = 16; wEmissive.pBufferInfo = &dbEmissive;

    VkDescriptorImageInfo dglow{sampler_, glowView, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wGlow = w[0]; wGlow.dstBinding = 17; wGlow.pImageInfo = &dglow;

    // primary-hit normal (octahedral-encoded)
    VkDescriptorImageInfo dprimarynormal{VK_NULL_HANDLE, primaryNormalView, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wPrimaryNormal = w[7]; wPrimaryNormal.dstBinding = 18; wPrimaryNormal.pImageInfo = &dprimarynormal;

    VkDescriptorImageInfo dlineardepth{VK_NULL_HANDLE, linearDepthView, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wLinearDepth = w[7]; wLinearDepth.dstBinding = 19; wLinearDepth.pImageInfo = &dlineardepth;

    const bool needsStaticRewrite = slotIndex >= pathTraceStaticDescriptorWrittenGeneration_.size() ||
        pathTraceStaticDescriptorWrittenGeneration_[slotIndex] != pathTraceStaticDescriptorGeneration_;

    VkWriteDescriptorSet writes[40];
    uint32_t writeCount = 0;
    writes[writeCount++] = w[0]; writes[writeCount++] = w[1]; writes[writeCount++] = w[2];
    writes[writeCount++] = w[3]; writes[writeCount++] = w[4]; writes[writeCount++] = w[6];
    writes[writeCount++] = w[7]; writes[writeCount++] = w[8]; writes[writeCount++] = w[9];
    writes[writeCount++] = w[10]; writes[writeCount++] = w[11]; writes[writeCount++] = w[12];
    writes[writeCount++] = wraw; writes[writeCount++] = wGlow; writes[writeCount++] = wPrimaryNormal;
    writes[writeCount++] = wLinearDepth;
    writes[writeCount++] = wMaterials;  // per-generation buffer, changes every frame
    // bindings 20-22 (see ensurePathTracePipeline())
    VkDescriptorBufferInfo dbCausticPhotons{causticPhotonBuffer(slotIndex), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo dbCausticHash{causticHashBuffer(slotIndex), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo dbCausticEmitters{causticEmitterBuffer_.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wCausticPhotons = w[4]; wCausticPhotons.dstBinding = 20; wCausticPhotons.pBufferInfo = &dbCausticPhotons;
    VkWriteDescriptorSet wCausticHash = w[4]; wCausticHash.dstBinding = 21; wCausticHash.pBufferInfo = &dbCausticHash;
    VkWriteDescriptorSet wCausticEmitters = w[4]; wCausticEmitters.dstBinding = 22; wCausticEmitters.pBufferInfo = &dbCausticEmitters;
    if (dbCausticPhotons.buffer != VK_NULL_HANDLE && dbCausticHash.buffer != VK_NULL_HANDLE && dbCausticEmitters.buffer != VK_NULL_HANDLE) {
        writes[writeCount++] = wCausticPhotons;
        writes[writeCount++] = wCausticHash;
        writes[writeCount++] = wCausticEmitters;
    }
    VkDescriptorBufferInfo dbRayStats{rayStatsBuffer(slotIndex), 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wRayStats = w[4]; wRayStats.dstBinding = 25; wRayStats.pBufferInfo = &dbRayStats;
    if (dbRayStats.buffer != VK_NULL_HANDLE) writes[writeCount++] = wRayStats;
    VkDescriptorBufferInfo dbLightGrid{lightGridBuffer_.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wLightGrid = w[4]; wLightGrid.dstBinding = 26; wLightGrid.pBufferInfo = &dbLightGrid;
    if (dbLightGrid.buffer != VK_NULL_HANDLE) writes[writeCount++] = wLightGrid;
    // bindings 27-29
    VkDescriptorBufferInfo dbRcKeys{rcKeyBuffer_.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo dbRcAccum{rcAccumBuffer_.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo dbRcRadiance{rcRadianceBuffer_.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wRcKeys = w[4]; wRcKeys.dstBinding = 27; wRcKeys.pBufferInfo = &dbRcKeys;
    VkWriteDescriptorSet wRcAccum = w[4]; wRcAccum.dstBinding = 28; wRcAccum.pBufferInfo = &dbRcAccum;
    VkWriteDescriptorSet wRcRadiance = w[4]; wRcRadiance.dstBinding = 29; wRcRadiance.pBufferInfo = &dbRcRadiance;
    if (dbRcKeys.buffer != VK_NULL_HANDLE) {
        writes[writeCount++] = wRcKeys;
        writes[writeCount++] = wRcAccum;
        writes[writeCount++] = wRcRadiance;
    }
    // binding 33
    {
        const VkDeviceSize need = VkDeviceSize(2) * rawRadiance.width() * rawRadiance.height() * 16;
        const VkBuffer before = restirBuffer_.buffer;
        if (!ensureBufferCapacity(restirBuffer_, need, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)) {
            if (errorOut) *errorOut = "ReSTIR reservoir buffer: " + ctx_.lastError();
            return false;
        }
        if (restirBuffer_.buffer != before) fillBufferNow(restirBuffer_, 0xFFFFFFFFu);
    }
    VkDescriptorBufferInfo dbRestir{restirBuffer_.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wRestir = w[4]; wRestir.dstBinding = 33; wRestir.pBufferInfo = &dbRestir;
    writes[writeCount++] = wRestir;
    // binding 30
    VkDescriptorBufferInfo dbSkyCdf{skyCdfBuffer_.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wSkyCdf = w[4]; wSkyCdf.dstBinding = 30; wSkyCdf.pBufferInfo = &dbSkyCdf;
    if (dbSkyCdf.buffer != VK_NULL_HANDLE) writes[writeCount++] = wSkyCdf;
    // bindings 31/32
    if (!ensureWaveResources()) {
        if (errorOut) *errorOut = "water wave resources: " + ctx_.lastError();
        return false;
    }
    const uint32_t waveGen = dynamicGenerationIndex_ < waterPatchGens_.size() ? dynamicGenerationIndex_ : 0u;
    VkDescriptorImageInfo dWaveFft{waveFftSampler_, waterPatchGens_[waveGen].fftView, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dWaveMap{waveMapSampler_, waveMapView_, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wWaveFft = w[0]; wWaveFft.dstBinding = 31; wWaveFft.pImageInfo = &dWaveFft;
    VkWriteDescriptorSet wWaveMap = w[0]; wWaveMap.dstBinding = 32; wWaveMap.pImageInfo = &dWaveMap;
    writes[writeCount++] = wWaveFft;
    writes[writeCount++] = wWaveMap;
    if (needsStaticRewrite) {
        writes[writeCount++] = wt;
        writes[writeCount++] = wMaterialInfo;
        writes[writeCount++] = wEmissive;
        if (slotIndex < pathTraceStaticDescriptorWrittenGeneration_.size())
            pathTraceStaticDescriptorWrittenGeneration_[slotIndex] = pathTraceStaticDescriptorGeneration_;
    }
    vkUpdateDescriptorSets(ctx_.device(), writeCount, writes, 0, nullptr);
    return true;
}

bool RayTracingScene::setCausticEmitters(const std::vector<float>& corners) {
    const uint32_t triCount = static_cast<uint32_t>(corners.size() / 9);
    std::vector<float> data(4 + static_cast<size_t>(triCount) * 12, 0.0f);
    double total = 0.0;
    std::vector<double> areas(triCount);
    for (uint32_t t = 0; t < triCount; ++t) {
        const float* v = &corners[static_cast<size_t>(t) * 9];
        const double e1x = v[3]-v[0], e1y = v[4]-v[1], e1z = v[5]-v[2];
        const double e2x = v[6]-v[0], e2y = v[7]-v[1], e2z = v[8]-v[2];
        const double cx = e1y*e2z - e1z*e2y, cy = e1z*e2x - e1x*e2z, cz = e1x*e2y - e1y*e2x;
        areas[t] = 0.5 * std::sqrt(cx*cx + cy*cy + cz*cz);
        total += areas[t];
    }
    data[0] = static_cast<float>(total);
    double acc = 0.0;
    for (uint32_t t = 0; t < triCount; ++t) {
        const float* v = &corners[static_cast<size_t>(t) * 9];
        acc += areas[t];
        float* o = &data[4 + static_cast<size_t>(t) * 12];
        o[0] = v[0]; o[1] = v[1]; o[2] = v[2]; o[3] = total > 0.0 ? static_cast<float>(acc / total) : 1.0f;
        o[4] = v[3]; o[5] = v[4]; o[6] = v[5]; o[7] = 0.0f;
        o[8] = v[6]; o[9] = v[7]; o[10] = v[8]; o[11] = 0.0f;
    }
    if (triCount > 0) data[4 + static_cast<size_t>(triCount - 1) * 12 + 3] = 1.0f;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(data.size() * sizeof(float));
    vkDeviceWaitIdle(ctx_.device());
    destroyBuffer(causticEmitterBuffer_);
    if (!createBuffer(bytes < 64 ? 64 : bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, causticEmitterBuffer_)) {
        causticEmitterTriCount_ = 0;
        ctx_.setError("RayTracingScene::setCausticEmitters: buffer allocation failed.");
        return false;
    }
    void* m = nullptr;
    if (vkMapMemory(ctx_.device(), causticEmitterBuffer_.memory, 0, bytes, 0, &m) != VK_SUCCESS) {
        causticEmitterTriCount_ = 0;
        return false;
    }
    std::memcpy(m, data.data(), static_cast<size_t>(bytes));
    vkUnmapMemory(ctx_.device(), causticEmitterBuffer_.memory);
    causticEmitterTriCount_ = total > 0.0 ? triCount : 0;
    causticEmitterArea_ = static_cast<float>(total);
    ++causticEmitterVersion_;  // invalidates every photon batch
    return true;
}

bool RayTracingScene::setPathTraceLights(uint32_t slotIndex, const float* packed, uint32_t count) {
    if (slotIndex >= pathTraceLightBuffers_.size()) {
        ctx_.setError("RayTracingScene::setPathTraceLights: invalid path-trace frame slot -- call ensurePathTracePipeline (via build()) first.");
        return false;
    }
    if (count == 0) {
        pathTraceLightCounts_[slotIndex] = 0;
        return true;
    }
    if (!packed) {
        ctx_.setError("RayTracingScene::setPathTraceLights: null light data with nonzero count.");
        return false;
    }
    const uint32_t strideFloats = 12;
    const VkDeviceSize requiredSize = static_cast<VkDeviceSize>(count) * strideFloats * sizeof(float);
    if (pathTraceLightBuffers_[slotIndex].size < requiredSize) {
        VkResult idle = vkDeviceWaitIdle(ctx_.device());
        if (idle != VK_SUCCESS) {
            ctx_.setError("RayTracingScene::setPathTraceLights: vkDeviceWaitIdle failed with VkResult " + std::to_string(idle));
            return false;
        }
        std::cerr << "[RayTracingScene] path-trace light buffer growing past its initial capacity (slot "
                  << slotIndex << ", " << count << " lights) -- stalled the GPU to do it safely. "
                     "If this happens routinely, raise PATHTRACE_LIGHT_INITIAL_CAPACITY." << std::endl;
    }
    if (!ensureBufferCapacity(pathTraceLightBuffers_[slotIndex], requiredSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        ctx_.setError("RayTracingScene::setPathTraceLights: light buffer allocation failed.");
        return false;
    }
    void* mapped = nullptr;
    if (vkMapMemory(ctx_.device(), pathTraceLightBuffers_[slotIndex].memory, 0, requiredSize, 0, &mapped) != VK_SUCCESS) {
        ctx_.setError("RayTracingScene::setPathTraceLights: vkMapMemory failed.");
        return false;
    }
    std::memcpy(mapped, packed, static_cast<size_t>(count) * strideFloats * sizeof(float));
    vkUnmapMemory(ctx_.device(), pathTraceLightBuffers_[slotIndex].memory);
    pathTraceLightCounts_[slotIndex] = count;
    return true;
}

bool RayTracingScene::isPathTraceTextureRegistered(uint32_t textureGlId, uint32_t* outIndex) const {
    if (textureGlId == 0) return false;
    auto found = pathTraceTextureIndices_.find(textureGlId);
    if (found == pathTraceTextureIndices_.end()) return false;
    if (outIndex) *outIndex = found->second;
    return true;
}

bool RayTracingScene::registerPathTraceTexture(uint32_t textureGlId, uint32_t width, uint32_t height,
                                                  const unsigned char* rgba8, size_t byteCount, uint32_t clamp, uint32_t* outIndex,
                                                  bool preMipped) {
    if (pathTraceTextures_.empty() && textureGlId != 0) {
        uint32_t dummyIdx = 0;
        if (!registerPathTraceTexture(0, 1, 1, nullptr, 0, 0, &dummyIdx)) return false;
    }
    if (textureGlId != 0) {
        auto found = pathTraceTextureIndices_.find(textureGlId);
        if (found != pathTraceTextureIndices_.end()) { if (outIndex) *outIndex=found->second; return true; }
    } else if (!pathTraceTextures_.empty()) {
        if (outIndex) *outIndex=0; return true;
    }
    if (pathTraceTextures_.size() >= kMaxPathTraceTextures) { ctx_.setError("RayTracingScene::registerPathTraceTexture: texture table full (" + std::to_string(kMaxPathTraceTextures) + " textures)."); return false; }
    const unsigned char fallback[4]={188,188,188,255};
    if (textureGlId == 0) { width=1; height=1; rgba8=fallback; byteCount=sizeof(fallback); preMipped=false; }
    size_t expectedBytes = size_t(width)*size_t(height)*4u;
    if (preMipped && width > 0 && height > 0) {
        const uint32_t levels = 1u + static_cast<uint32_t>(std::floor(std::log2(static_cast<double>(std::max(width, height)))));
        expectedBytes = 0;
        for (uint32_t l = 0; l < levels; ++l) expectedBytes += size_t(std::max(1u, width >> l)) * size_t(std::max(1u, height >> l)) * 4u;
    }
    if (!rgba8 || width == 0 || height == 0 || byteCount != expectedBytes) {
        ctx_.setError("RayTracingScene::registerPathTraceTexture: invalid RGBA8 texture data."); return false;
    }
    PathTraceTexture tex{};
    tex.glId=textureGlId; tex.width=width; tex.height=height; tex.clamp=clamp; tex.preMipped=preMipped;
    // full mip chain down to 1x1
    tex.mipLevels = 1u + static_cast<uint32_t>(std::floor(std::log2(static_cast<double>(std::max(width, height)))));
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter=VK_FILTER_LINEAR;
    samplerInfo.minFilter=VK_FILTER_LINEAR;
    // trilinear filtering across the generated mip chain below
    samplerInfo.mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR;
    if (preMipped) {  // max-mip data
        samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    } else {
        VkPhysicalDeviceFeatures pdf{};
        vkGetPhysicalDeviceFeatures(ctx_.physicalDevice(), &pdf);
        if (pdf.samplerAnisotropy == VK_TRUE) {
            VkPhysicalDeviceProperties pdp{};
            vkGetPhysicalDeviceProperties(ctx_.physicalDevice(), &pdp);
            samplerInfo.anisotropyEnable = VK_TRUE;
            samplerInfo.maxAnisotropy = std::min(16.0f, pdp.limits.maxSamplerAnisotropy);
        }
    }
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(tex.mipLevels - 1);
    samplerInfo.addressModeU = (clamp & 1u) ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE :
                      ((clamp & 0x100u) ? VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT : VK_SAMPLER_ADDRESS_MODE_REPEAT);
    samplerInfo.addressModeV = (clamp & 2u) ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE :
                      ((clamp & 0x200u) ? VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT : VK_SAMPLER_ADDRESS_MODE_REPEAT);
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    if (vkCreateSampler(ctx_.device(), &samplerInfo, nullptr, &tex.sampler) != VK_SUCCESS) {
        ctx_.setError("RayTracingScene::registerPathTraceTexture: sampler creation failed."); return false;
    }
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ii.imageType=VK_IMAGE_TYPE_2D; ii.format=VK_FORMAT_R8G8B8A8_UNORM; ii.extent={width,height,1}; ii.mipLevels=tex.mipLevels; ii.arrayLayers=1; ii.samples=VK_SAMPLE_COUNT_1_BIT; ii.tiling=VK_IMAGE_TILING_OPTIMAL; ii.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;
    if(vkCreateImage(ctx_.device(),&ii,nullptr,&tex.image)!=VK_SUCCESS){if(tex.sampler) vkDestroySampler(ctx_.device(),tex.sampler,nullptr);ctx_.setError("RayTracingScene::registerPathTraceTexture: vkCreateImage failed.");return false;}
    VkMemoryRequirements req{}; vkGetImageMemoryRequirements(ctx_.device(),tex.image,&req); VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(ctx_.physicalDevice(),&mp); uint32_t mt=UINT32_MAX; for(uint32_t i=0;i<mp.memoryTypeCount;++i) if((req.memoryTypeBits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)){mt=i;break;} if(mt==UINT32_MAX){vkDestroyImage(ctx_.device(),tex.image,nullptr);if(tex.sampler)vkDestroySampler(ctx_.device(),tex.sampler,nullptr);return false;}
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize=req.size; mai.memoryTypeIndex=mt; if(vkAllocateMemory(ctx_.device(),&mai,nullptr,&tex.memory)!=VK_SUCCESS){vkDestroyImage(ctx_.device(),tex.image,nullptr);return false;} if(vkBindImageMemory(ctx_.device(),tex.image,tex.memory,0)!=VK_SUCCESS){vkFreeMemory(ctx_.device(),tex.memory,nullptr);vkDestroyImage(ctx_.device(),tex.image,nullptr);if(tex.sampler)vkDestroySampler(ctx_.device(),tex.sampler,nullptr);return false;}
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image=tex.image; vi.viewType=VK_IMAGE_VIEW_TYPE_2D; vi.format=VK_FORMAT_R8G8B8A8_UNORM; vi.subresourceRange.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT; vi.subresourceRange.levelCount=tex.mipLevels; vi.subresourceRange.layerCount=1; if(vkCreateImageView(ctx_.device(),&vi,nullptr,&tex.view)!=VK_SUCCESS){vkFreeMemory(ctx_.device(),tex.memory,nullptr);vkDestroyImage(ctx_.device(),tex.image,nullptr);if(tex.sampler)vkDestroySampler(ctx_.device(),tex.sampler,nullptr);return false;}
    Buffer st{}; if(!createBuffer(byteCount,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,st)){vkDestroyImageView(ctx_.device(),tex.view,nullptr);vkFreeMemory(ctx_.device(),tex.memory,nullptr);vkDestroyImage(ctx_.device(),tex.image,nullptr);if(tex.sampler)vkDestroySampler(ctx_.device(),tex.sampler,nullptr);return false;} void* mapped=nullptr; if(vkMapMemory(ctx_.device(),st.memory,0,byteCount,0,&mapped)!=VK_SUCCESS){destroyBuffer(st);vkDestroyImageView(ctx_.device(),tex.view,nullptr);vkFreeMemory(ctx_.device(),tex.memory,nullptr);vkDestroyImage(ctx_.device(),tex.image,nullptr);if(tex.sampler)vkDestroySampler(ctx_.device(),tex.sampler,nullptr);return false;} std::memcpy(mapped,rgba8,byteCount);vkUnmapMemory(ctx_.device(),st.memory);
    uint32_t idx=static_cast<uint32_t>(pathTraceTextures_.size()); pathTraceTextures_.push_back(tex); if(textureGlId!=0) pathTraceTextureIndices_[textureGlId]=idx; if(outIndex)*outIndex=idx;
    PendingTextureUpload up;
    up.textureIndex = idx;
    up.staging = st;
    up.width = width;
    up.height = height;
    pendingTextureUploads_.push_back(up);
    ++pathTraceStaticDescriptorGeneration_;
    return true;
}

bool RayTracingScene::buildPathTraceMaterialTable(std::string* errorOut) {
    flushChunkBuilds();  // the table reads chunk buffer addresses
    uint32_t maxSlot = 0;
    bool any = false;
    for (auto& kv : chunkMeshes_) {
        if (!any || kv.second.materialSlot > maxSlot) maxSlot = kv.second.materialSlot;
        any = true;
    }
    const uint32_t chunkEntryCount = any ? (maxSlot + 1) : 0;
    waterPatchMaterialSlot_ = chunkEntryCount;
    dynamicMaterialBase_ = chunkEntryCount + 1;
    const uint32_t entryCount = chunkEntryCount + 1 + kMaxDynamicMaterialSlots;
    std::vector<GpuPathTraceMaterialEntry> table(entryCount);
    for (auto& e : table) { e.vertexAddr = 0; e.indexAddr = 0; e.albedo[0] = e.albedo[1] = e.albedo[2] = 0.6f; e.opaqueIndexCount = 0.0f; }
    for (auto& kv : chunkMeshes_) {
        const ChunkMesh& m = kv.second;
        if (m.materialSlot >= chunkEntryCount) continue;
        GpuPathTraceMaterialEntry& e = table[m.materialSlot];
        e.vertexAddr = static_cast<uint64_t>(m.vertex.address);
        e.indexAddr = static_cast<uint64_t>(m.index.address);
        e.albedo[0] = m.avgAlbedo[0]; e.albedo[1] = m.avgAlbedo[1]; e.albedo[2] = m.avgAlbedo[2];
        e.opaqueIndexCount = (m.opaqueIndexCount < static_cast<uint32_t>(m.indices.size())) ? static_cast<float>(m.opaqueIndexCount) : 0.0f;
    }
    const VkDeviceSize requiredSize = static_cast<VkDeviceSize>(entryCount) * sizeof(GpuPathTraceMaterialEntry);
    if (pathTraceMaterialsBuffer_.size < requiredSize) {
        VkResult idle = vkDeviceWaitIdle(ctx_.device());
        if (idle != VK_SUCCESS) {
            if (errorOut) *errorOut = "RayTracingScene::buildPathTraceMaterialTable: vkDeviceWaitIdle failed with VkResult " + std::to_string(idle);
            return false;
        }
    }
    if (!ensureBufferCapacity(pathTraceMaterialsBuffer_, requiredSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        if (errorOut) *errorOut = "RayTracingScene::buildPathTraceMaterialTable: buffer allocation failed.";
        return false;
    }
    void* mapped = nullptr;
    if (vkMapMemory(ctx_.device(), pathTraceMaterialsBuffer_.memory, 0, requiredSize, 0, &mapped) != VK_SUCCESS) {
        if (errorOut) *errorOut = "RayTracingScene::buildPathTraceMaterialTable: vkMapMemory failed.";
        return false;
    }
    std::memcpy(mapped, table.data(), static_cast<size_t>(requiredSize));
    vkUnmapMemory(ctx_.device(), pathTraceMaterialsBuffer_.memory);
    pathTraceMaterialCount_ = entryCount;
    pathTraceMaterialMirror_.assign(reinterpret_cast<const uint8_t*>(table.data()),
                                    reinterpret_cast<const uint8_t*>(table.data()) + static_cast<size_t>(requiredSize));
    ++pathTraceStaticDescriptorGeneration_;
    return true;
}

bool RayTracingScene::updateDynamicMaterialEntries(const std::vector<DynamicMaterialUpdate>& entries, std::string* errorOut) {
    if (entries.empty()) return true;
    if (pathTraceMaterialMirror_.empty()) return true;
    // patches the CPU mirror only
    GpuPathTraceMaterialEntry* table = reinterpret_cast<GpuPathTraceMaterialEntry*>(pathTraceMaterialMirror_.data());
    for (const DynamicMaterialUpdate& u : entries) {
        if (u.ordinal >= kMaxDynamicMaterialSlots) continue;
        const uint32_t index = dynamicMaterialBase_ + u.ordinal;
        if (index >= pathTraceMaterialCount_) continue;
        GpuPathTraceMaterialEntry& e = table[index];
        e.vertexAddr = u.vertexAddr;
        e.indexAddr = u.indexAddr;
        e.prevVertexAddr = u.prevVertexAddr;
        e.albedo[0] = e.albedo[1] = e.albedo[2] = 0.6f;
        e.opaqueIndexCount = 0.0f;  // no glass concept for a skinned instance
    }
    return true;
}

// pbr materials + light transport

bool RayTracingScene::setPathTraceMaterials(const PathTraceMaterialInput* materials, uint32_t count, std::string* errorOut) {
    if (count == 0 || !materials) {
        // cheap no-op mirrors buildPathTraceMaterialTable()'s own "no chunk geometry yet" early- out
        return true;
    }
    std::vector<GpuPathTraceMaterialInfo> table(count);
    for (uint32_t i = 0; i < count; ++i) {
        const PathTraceMaterialInput& in = materials[i];
        GpuPathTraceMaterialInfo& out = table[i];
        auto resolve = [&](uint32_t glId) -> uint32_t {
            if (glId == 0) return 0;
            auto it = pathTraceTextureIndices_.find(glId);
            return it != pathTraceTextureIndices_.end() ? it->second : 0u;
        };
        out.diffuseTexIndex = resolve(in.diffuseTexGlId);
        if (in.isWater) {
            float thickness = in.waterReferenceThickness;
            std::memcpy(&out.normalTexIndex, &thickness, sizeof(out.normalTexIndex));
        } else {
            out.normalTexIndex = in.hasNormal ? resolve(in.normalTexGlId) : 0u;
        }
        out.glowTexIndex = in.hasGlow ? resolve(in.glowTexGlId) : 0u;
        constexpr float kSpecScaleMax = 4.0f;
        const uint32_t specScaleQuant = static_cast<uint32_t>(std::clamp(in.specScale, 0.0f, kSpecScaleMax) / kSpecScaleMax * 255.0f + 0.5f);
        out.flags = (in.hasSpec ? 1u : 0u) | (in.hasNormal ? 2u : 0u) | (in.hasGlow ? 4u : 0u) | (in.isGlass ? 8u : 0u) | (in.isWater ? 16u : 0u) | (in.uniformSpec ? 32u : 0u) | (in.parallax ? 64u : 0u) | (in.isLava ? 128u : 0u) | (specScaleQuant << 8);
        out.parallax[0] = in.parallaxScale; out.parallax[1] = in.parallaxBias; out.parallax[3] = 0.0f;
        out.scroll[0] = in.scroll[0]; out.scroll[1] = in.scroll[1];
        out.scroll[2] = in.roughness >= 0.0f ? 1.0f + std::clamp(in.roughness, 0.0f, 1.0f) : 0.0f;  // 0 = automatic (the fallback material's zero too)
        out.scroll[3] = in.glowSelectLum;  // the fog's glow MIS (volfog_inject.comp)
        {
            const uint32_t heightIndex = in.parallax ? resolve(in.heightTexGlId) : 0u;
            std::memcpy(&out.parallax[2], &heightIndex, sizeof(float));
        }
        // metalness, packed into colorscale.w
        out.colorscale[0] = in.colorscale[0]; out.colorscale[1] = in.colorscale[1]; out.colorscale[2] = in.colorscale[2];
        out.colorscale[3] = std::clamp(in.metalness, 0.0f, 1.0f);
        out.glowcolor[0] = in.glowcolor[0]; out.glowcolor[1] = in.glowcolor[1]; out.glowcolor[2] = in.glowcolor[2];
        out.glowcolor[3] = std::clamp(in.glassAlpha, 0.0f, 1.0f);
    }
    const VkDeviceSize requiredSize = static_cast<VkDeviceSize>(count) * sizeof(GpuPathTraceMaterialInfo);
    if (pathTraceMaterialInfoBuffer_.size < requiredSize) {
        VkResult idle = vkDeviceWaitIdle(ctx_.device());
        if (idle != VK_SUCCESS) {
            if (errorOut) *errorOut = "RayTracingScene::setPathTraceMaterials: vkDeviceWaitIdle failed with VkResult " + std::to_string(idle);
            return false;
        }
    }
    if (!ensureBufferCapacity(pathTraceMaterialInfoBuffer_, requiredSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        if (errorOut) *errorOut = "RayTracingScene::setPathTraceMaterials: buffer allocation failed.";
        return false;
    }
    void* mapped = nullptr;
    if (vkMapMemory(ctx_.device(), pathTraceMaterialInfoBuffer_.memory, 0, requiredSize, 0, &mapped) != VK_SUCCESS) {
        if (errorOut) *errorOut = "RayTracingScene::setPathTraceMaterials: vkMapMemory failed.";
        return false;
    }
    std::memcpy(mapped, table.data(), static_cast<size_t>(requiredSize));
    vkUnmapMemory(ctx_.device(), pathTraceMaterialInfoBuffer_.memory);
    pathTraceMaterialInfoCount_ = count;
    ++pathTraceStaticDescriptorGeneration_;
    return true;
}

bool RayTracingScene::setPathTraceEmissiveTriangles(const float* packedTriangles, uint32_t triangleCount, std::string* errorOut) {
    if (triangleCount == 0 || !packedTriangles) {
        pathTraceEmissiveCount_ = 0;
        pathTraceEmissiveTotalPower_ = 0.0f;
        return true;
    }
    std::vector<GpuEmissiveTriangle> table(triangleCount);
    // per-triangle "power"
    std::vector<double> power(triangleCount, 0.0);
    double totalPower = 0.0;
    for (uint32_t i = 0; i < triangleCount; ++i) {
        const float* src = packedTriangles + static_cast<size_t>(i) * 13;
        GpuEmissiveTriangle& out = table[i];
        out.v0[0]=src[0]; out.v0[1]=src[1]; out.v0[2]=src[2]; out.v0[3]=0.0f;
        out.v1[0]=src[3]; out.v1[1]=src[4]; out.v1[2]=src[5]; out.v1[3]=0.0f;
        out.v2[0]=src[6]; out.v2[1]=src[7]; out.v2[2]=src[8]; out.v2[3]=0.0f;
        out.radiance[0]=src[9]; out.radiance[1]=src[10]; out.radiance[2]=src[11]; out.radiance[3]=src[12];  // procedural flag (path-traced lava)

        const double e0[3] = { src[3]-src[0], src[4]-src[1], src[5]-src[2] };
        const double e1[3] = { src[6]-src[0], src[7]-src[1], src[8]-src[2] };
        const double cr[3] = { e0[1]*e1[2]-e0[2]*e1[1], e0[2]*e1[0]-e0[0]*e1[2], e0[0]*e1[1]-e0[1]*e1[0] };
        const double area = 0.5 * std::sqrt(cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2]);
        const double luminance = 0.2126 * static_cast<double>(src[9]) + 0.7152 * static_cast<double>(src[10]) + 0.0722 * static_cast<double>(src[11]);
        power[i] = std::max(luminance * area, 1e-8);
        totalPower += power[i];
    }
    double running = 0.0;
    for (uint32_t i = 0; i < triangleCount; ++i) {
        running += power[i];
        table[i].v2[3] = static_cast<float>(running / totalPower);
    }
    table[triangleCount - 1].v2[3] = 1.0f;
    pathTraceEmissiveTotalPower_ = static_cast<float>(totalPower);
    const VkDeviceSize requiredSize = static_cast<VkDeviceSize>(triangleCount) * sizeof(GpuEmissiveTriangle);
    if (pathTraceEmissiveBuffer_.size < requiredSize) {
        VkResult idle = vkDeviceWaitIdle(ctx_.device());
        if (idle != VK_SUCCESS) {
            if (errorOut) *errorOut = "RayTracingScene::setPathTraceEmissiveTriangles: vkDeviceWaitIdle failed with VkResult " + std::to_string(idle);
            return false;
        }
    }
    if (!ensureBufferCapacity(pathTraceEmissiveBuffer_, requiredSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        if (errorOut) *errorOut = "RayTracingScene::setPathTraceEmissiveTriangles: buffer allocation failed.";
        return false;
    }
    void* mapped = nullptr;
    if (vkMapMemory(ctx_.device(), pathTraceEmissiveBuffer_.memory, 0, requiredSize, 0, &mapped) != VK_SUCCESS) {
        if (errorOut) *errorOut = "RayTracingScene::setPathTraceEmissiveTriangles: vkMapMemory failed.";
        return false;
    }
    std::memcpy(mapped, table.data(), static_cast<size_t>(requiredSize));
    vkUnmapMemory(ctx_.device(), pathTraceEmissiveBuffer_.memory);
    pathTraceEmissiveCount_ = triangleCount;
    ++pathTraceStaticDescriptorGeneration_;
    return true;
}

}  // namespace interop
