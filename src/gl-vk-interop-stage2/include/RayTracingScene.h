#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>  // causticHashSizeFor
#include <algorithm>
#include <unordered_map>
#include <cstddef>
#include <memory>

namespace interop {
class InteropContext;
class SharedTexture;
class SharedTexture3D;
class InteropFrame;

struct RtaoConstants {
    float invViewProj[16]{};
    float invView[16]{};
    float currentViewProjUnjittered[16]{};
    float prevViewProjJittered[16]{};
    float prevViewProjUnjittered[16]{};
    float radius = 2.0f;
    float intensity = 1.0f;
    float bias = 0.01f;
    uint32_t frameIndex = 0;
    uint32_t raysPerPixel = 8;
    bool historyValid = false;
    float lightDirection[3] = {0.0f, -1.0f, 0.0f};
    float lightRadius = 0.02f;
    uint32_t shadowRaysPerPixel = 1;
    bool shadowEnabled = false;
    uint64_t resourceEpoch = UINT64_MAX;
};

struct ReflectionConstants {
    float invViewProj[16]{};
    float invView[16]{};
    float currentViewProjUnjittered[16]{};
    float prevViewProjJittered[16]{};
    float prevViewProjUnjittered[16]{};
    float bias = 0.01f;
    float maxRayDistance = 500.0f;
    float maxConeAngleRadians = 0.15f;
    uint32_t frameIndex = 0;
    uint32_t raysPerPixel = 4;  // jittered rays per pixel within the roughness-driven cone
    bool historyValid = false;
    // false on this renderer's first dispatch after being (re-)enabled or resized
    bool prevColorValid = false;
    float skyHorizonColor[3] = {0.55f, 0.72f, 0.90f};
    float skyZenithColor[3] = {0.15f, 0.38f, 0.72f};
    float skyGroundColor[3] = {0.30f, 0.28f, 0.25f};
};

struct RtdiDebugConstants {
    float invViewProj[16]{};
    float invView[16]{};
    uint32_t frameIndex = 0;
};

struct RtdiConstants {
    float invViewProj[16]{};
    float invView[16]{};
    uint32_t frameIndex = 0;
    // candidates drawn per pixel for streaming RIS. 16-32 per the project's spec
    uint32_t candidateCount = 24;
    float bias = 0.01f;
    float maxRayDistance = 500.0f;
    // temporal reservoir reuse
    float previousViewProj[16]{};
    // caps how many "effective samples" a reservoir's M can accumulate across frames
    float maxHistoryM = 480.0f;
    float depthRejectThreshold = 0.006f;
    float pad0 = 0.0f;
    float pad1 = 0.0f;
};

struct PathTraceConstants {
    float invViewProj[16]{};
    // the forward (non-inverted) counterpart to invViewProj
    float viewProj[16]{};
    float invView[16]{};
    float prevViewProj[16]{};
    // per-pixel-precise reprojection target
    float prevViewProjJittered[16]{};
    uint32_t frameIndex = 0;
    uint32_t bounceCount = 2;
    bool historyValid = false;
    uint32_t lightCount = 0;
    float sunDirection[3] = {0.0f, 0.0f, 1.0f};
    float sunAngularRadius = 0.02f;
    float sunColor[3] = {1.0f, 1.0f, 1.0f};
    float maxHistorySamples = 32.0f;
    // flat "sky dome" ambient term a bounce ray uses when it misses all geometry
    float skyAmbient[3] = {0.3f, 0.3f, 0.3f};
    float ambientFloor[3] = {0.1f, 0.1f, 0.1f};
    // tunable emissive intensity multiplier
    float glowScale = 1.0f;
    // MIPMAPPING the primary camera ray's angular footprint per pixel (radians), i.e
    float pixelSpreadAngle = 0.0f;
    // tunable "how thin does glass act" dial
    float glassThinness = 1.0f;
    uint64_t resourceEpoch = UINT64_MAX;
    // sky hit sampling
    uint32_t skyFaceTexIndex[6] = {0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
    // false whenever the map has no skybox loaded, or any of the 6 faces failed to register
    bool skyValid = false;
    float skyRotationRadians = 0.0f;
    uint32_t blueNoiseTexIndex = 0xFFFFFFFFu;
    float skyLightScale = 1.0f;
    // cap sun + map-light irradiance at the lightmapper's 1.0 per channel (skyRotationPad.z)
    bool lightmapClamp = false;
    // specular weight for materials with no spec map and no metalness (skyRotationPad.w)
    float plainSpecular = 0.0f;
    // parallax occlusion mapping (pomParams)
    bool pomEnabled = true;
    uint32_t pomMaxIterations = 64;
    float pomDepthScale = 1.0f;
    uint32_t pomMode = 1;  // 0 = raymarch, 1 = quadtree (max-mip) where a height chain exists
    float glassVolumeTint = 0.6f;
    // how far alpha-geometry (textured) glass tints toward its texture colour
    float alphaGlassTint = 1.0f;
    // strength of a model's masks-texture blue channel (Sauerbraten's envmap mask) as environment
    // reflectivity
    float modelEnvReflect = 0.5f;
    float textureGradScale = 1.0f;
    bool causticsEnabled = false;
    // the trace pass counts its rays by kind into rayStatsBuffer(slot), see pathtrace_trace.comp's ray
    // stats
    bool rayStats = false;
    bool rayStatsProfile = false;  // + the trace pass's per-section shader-clock profile (ptprofile)
    // see pathtrace_trace.comp's directLighting()
    bool lightSampleBounces = true;
    // light splitting (ptlightexact / ptlightrandom)
    uint32_t lightSplitExact = 4;
    uint32_t lightSplitRandom = 0;
    // RIS light sampling (pathtrace_trace.comp's directLighting, ptris)
    bool risEnabled = false;
    bool risImportance = true;
    uint32_t risCandidates = 8, risSamples = 1;
    bool perfSkipShadow = false;  // no point-light shadow rays at the camera hit
    uint32_t perfCut = 0;  // 1 = stop after the primary hit, 2 = after its direct light
    bool cameraInWater = false;  // the camera is inside a water volume this frame
    float cameraWaterColor[3] = {0.0f, 0.0f, 0.0f};  // that water's colour (getwatercolor)
    float cameraWaterDepth = 150.0f;
    float cameraWaterSurfaceZ = 0.0f;
    float waterHistory = 6.0f;  // ptwaterhistory
    float waterDebug = 0.0f;  // ptwaterdebug
    // DLSS-RR specular motion vectors
    bool rrSpecularMotion = false;
    // the camera hit's static lights by reservoir reuse, see pathtrace_trace.comp's restirDirect()
    bool restirEnabled = false;
    uint32_t restirMinLights = 0;  // only where more static lights than this are in range
    uint32_t traceMode = 0;
    uint32_t restirSpatial = 2;  // last-frame neighbours merged per pixel
    uint32_t restirCandidates = 4;  // new light candidates per pixel
    uint32_t risBounceSamples = 1;  // RIS at bounce hits
    uint32_t risBounceCandidates = 4;
    float waterRays = 0.0f, waterRaysG = 0.7f, waterRaysFocus = 1.0f;
    float waterFlowDt = 0.0f;
    float lavaSize = 24.0f, lavaFlow = 1.0f, lavaRelief = 2.0f, lavaCrack = 0.08f;
    float lavaTemp = 1150.0f, lavaCrustTemp = 650.0f, lavaSpeed = 1.0f;
    float lavaCrustAmount = 0.45f;  // fraction of the surface covered by crust islands
    float lavaTintHue = 0.0f, lavaTintSat = 1.0f;
    bool cacheEnabled = true;  // ptcache
    float cacheCell = 4.0f;  // ptcachecell
    float cacheFrames = 32.0f;
    float cacheUpdate = 16.0f;  // ptcacheupdate
    float cacheFloor = 0.35f;  // ptcachefloor
    float cacheMinFrames = 2.0f;  // frames of data a cell needs before it's used
    bool cacheFallback = false;
    bool cacheCoherent = false;  // pick update paths per 8x4 block
    float cacheEvict = 256.0f;  // frames untouched before a cell is evicted
    float particleReflect = 1.0f;  // ptpartreflect
    bool skyboxLight = true;
    // see pathtrace_trace.comp's materialRoughness()
    bool physicalMaterials = false;
    // see pathtrace_trace.comp's transportAlbedo()
    bool transportAlbedo = false;
    float transportAlbedoGamma = 1.6f;
    float waterClarity = 2.0f;  // ptwaterclarity
    float waterWaves = 1.0f;  // ptwaterwaves
    float waterScatter = 0.5f;  // ptwaterscatter
    float timeSeconds = 0.0f;  // wave animation clock
    float causticRadius = 1.5f;
    float causticRange = 128.0f;
    float causticRipple = 1.0f;  // fine ripple strength the photons refract through
    uint32_t causticPhotons = 131072;
};

struct GpuVolFogConstants {
    float invViewProj[16]{};
    float prevViewProj[16]{};
    uint32_t frameIndex = 0;
    uint32_t shadowRefreshFreq = 4;  // volfogshadowfreq cvar, ~1/N froxels trace a shadow ray per frame
    uint32_t historyValid = 0;
    uint32_t reservedFrameData = 0;
    float sunDirectionRadius[4]{0.0f, 0.0f, 1.0f, 0.02f};
    float sunColorPad[4]{1.0f, 1.0f, 1.0f, 0.0f};
    float skyAmbientPad[4]{0.3f, 0.3f, 0.3f, 0.0f};
    // reservedGridDim
    uint32_t gridDimX = 160, gridDimY = 90, gridDimZ = 128, reservedGridDim = 0;
    float nearPlane = 0.1f, volFogDist = 1024.0f, reservedNearFar0 = 0.0f, reservedNearFar1 = 0.0f;
    float baseDensity = 0.02f, fogBaseHeight = 0.0f, heightFalloff = 0.01f, ambientShadowFloor = 0.15f;
    float phaseG = 0.2f, noiseScale = 1.0f, noiseEnabled = 0.0f, noiseIntensity = 0.5f;
    float camWorldPos[4]{};
    float camForward[4]{0.0f, 0.0f, 1.0f, 0.0f};
    float mediumOverride[4]{};
    float localLightParams[4]{1.0f, 1.0f, 0.0f, 0.0f};
    float ambientFloor[4]{};
    // the lava field (rt_lava.glsl, = PathTraceConstants' lava params) so the fog evaluates the lava's
    // emission
    float lava0[4]{24, 1, 2, 0.08f};
    float lava1[4]{1150, 650, 0, 0.45f};
    float lavaFog[4]{-1, 0, 1, 0};
    float emissiveFog[4]{};
    // xy = hue rotation (radians) / saturation scale for the lava's glow gradient, from the map's
    // lavacolour
    float lavaTint[4]{0.0f, 1.0f, 0.0f, 0.0f};
};

class RayTracingScene {
public:
    explicit RayTracingScene(InteropContext& ctx, const std::string& shaderDirectory, uint32_t frameSlots = 3, uint32_t dispatchSlots = 3);
    ~RayTracingScene();
    RayTracingScene(const RayTracingScene&) = delete;
    RayTracingScene& operator=(const RayTracingScene&) = delete;

    bool addMesh(const float* positions, uint32_t vertexCount,
                 const uint32_t* indices, uint32_t indexCount, uint32_t materialSlot = 0);
    // separate pool for geometry that changes every frame (an animated Object Placer instance's CPU-
    // skinned mesh)
    bool addDynamicMesh(const float* positions, uint32_t vertexCount,
                         const uint32_t* indices, uint32_t indexCount, uint32_t materialSlot = 0);


    struct SkinSourceVertex {
        float px, py, pz;
        float nx, ny, nz;
        float tx, ty, tz, tw;
        float u, v;
        uint32_t blendBone0, blendBone1, blendBone2, blendBone3;
        float blendWeight0, blendWeight1, blendWeight2, blendWeight3;
        uint32_t textureIndex;
        uint32_t materialIndex;
    };

    bool registerSkinSource(uint64_t modelKey, const SkinSourceVertex* vertices, uint32_t vertexCount,
                             const uint32_t* indices, uint32_t indexCount, uint32_t boneCount,
                             std::string* errorOut = nullptr);
    bool hasSkinSource(uint64_t modelKey, uint32_t boneCount) const;

    uint64_t geometryEpoch() const { return geometryEpoch_; }

    // instanceMask
    bool addSkinnedDynamicMesh(uint64_t modelKey, const float* boneDualQuats, uint32_t boneCount,
                                const float worldMatrix[12], uint32_t materialSlot = 0, uint32_t instanceMask = 0xFFu);

    // chunk tier ---- A third geometry lifecycle, distinct from both of the above

    bool addChunkMesh(uint64_t chunkId, const float* positions, uint32_t vertexCount,
                       const uint32_t* indices, uint32_t indexCount, uint32_t materialSlot = 0,
                       const float* avgAlbedo = nullptr);
    bool addTexturedChunkMesh(uint64_t chunkId, const float* positions, const float* uvs,
                              const uint32_t* textureGlIds, uint32_t vertexCount,
                              const uint32_t* indices, uint32_t indexCount, uint32_t materialSlot = 0,
                              const float* avgAlbedo = nullptr,
                              const float* normals = nullptr, const float* tangents = nullptr,
                              const uint32_t* materialIndices = nullptr,
                              uint32_t opaqueIndexCount = 0xFFFFFFFFu,
                              const uint32_t* blendMaterialIndices = nullptr,
                              const float* blendAlphas = nullptr,
                              const float* blendUvs = nullptr,
                              const float* alphaTestCutoffs = nullptr,
                              const float* blendLmUvs = nullptr,
                              const uint32_t* blendLmTexGlIds = nullptr);
    bool addWaterMesh(const float* positions, const float* normals, const float* uvs,
                       const uint32_t* materialIndices, uint32_t vertexCount,
                       const uint32_t* indices, uint32_t indexCount);
    bool addTexturedDynamicMesh(const SkinSourceVertex* vertices, uint32_t vertexCount,
                                 const uint32_t* indices, uint32_t indexCount);
    // this frame's glowing sprites as one dynamic mesh
    bool addParticleSpriteMesh(const SkinSourceVertex* vertices, uint32_t vertexCount,
                               const uint32_t* indices, uint32_t indexCount);
    // this frame's grass blades as one dynamic mesh
    bool addGrassMesh(const SkinSourceVertex* vertices, const float* alphaCutoffs, uint32_t vertexCount,
                      const uint32_t* indices, uint32_t indexCount);

    // path-traced water waves all on the GPU, per frame the CPU only supplies these numbers
    bool setWaterSurfaces(const float* rects, uint32_t count, uint64_t flatWaterChunkId, uint32_t waterTopTris,
                          uint64_t flatLavaChunkId, uint32_t lavaTopTris);
    struct WaterPatchSettings {
        bool waves = false;  // the FFT sea runs (path-traced water present)
        bool enabled = false;  // + the displaced patch geometry (ptwaves)
        float camX = 0.0f, camY = 0.0f, camZ = 0.0f;
        float time = 0.0f;  // wave clock (s)
        float speed = 1.0f;  // time scale (ptwavespeed)
        float strength = 1.0f;  // wave amplitude (ptwaterwaves), geometry and normals
        float detail = 1.0f;  // finest cascade multiplier (ptwavedetail)
        float height = 1.0f;  // displacement scale of the patch geometry only (ptwaveheight)
        float range = 128.0f;  // patch half-size, world units (ptwaverange)
        float cellSize = 2.0f;  // grid spacing, a power of two (ptwavegrid)
        float fadeWidth = 24.0f;  // edge fade to the flat water, world units (ptwavefade)
        float windSpeed = 5.0f;  // m/s (ptwind)
        float windDir = 30.0f;  // degrees (ptwinddir)
        float fetch = 4000.0f;  // open-water fetch, metres (ptfetch)
        float spread = 6.0f;
        float chop = 0.7f;  // choppiness, horizontal displacement, sharper crests (ptchop)
        float sizeMeters = 50.0f;  // largest cascade's tile size, metres (ptwavesize)
        bool depthAware = true;  // depth + fetch aware wave map (ptwavedepth)
        float worldSize = 0.0f;  // the map's world size (0 = unknown)
        bool mapStats = false;  // log what each wave-map bake produced (ptwavestats)
        float lava0[4] = {24, 1, 2.0f, 0.08f};  // = PathTraceConstants' lava params (rt_lava.glsl lava0/lava1)
        float lava1[4] = {1150, 650, 0, 0.45f};
    };
    void setWaterPatch(const WaterPatchSettings& s) { waterPatchSettings_ = s; }

    bool hasRigidSource(uint64_t modelKey) const { return rigidSources_.find(modelKey) != rigidSources_.end(); }
    bool registerRigidSource(uint64_t modelKey, const SkinSourceVertex* vertices, uint32_t vertexCount,
                              const uint32_t* indices, uint32_t indexCount, std::string* errorOut = nullptr);
    bool addRigidInstance(uint64_t modelKey, const float worldMatrix[12]);
    bool registerPathTraceTexture(uint32_t textureGlId, uint32_t width, uint32_t height,
                                  const unsigned char* rgba8, size_t byteCount, uint32_t clamp = 0, uint32_t* outIndex = nullptr,
                                  bool preMipped = false);
    // see registerPathTraceTexture()'s GL-id dedupe (pathTraceTextureIndices_) for the mechanism this
    // exposes
    bool isPathTraceTextureRegistered(uint32_t textureGlId, uint32_t* outIndex = nullptr) const;
    // reverse lookup
    uint32_t pathTraceTextureGlId(uint32_t index) const {
        return index < pathTraceTextures_.size() ? pathTraceTextures_[index].glId : 0u;
    }
    bool updateChunkMesh(uint64_t chunkId, const float* positions, uint32_t vertexCount,
                          const uint32_t* indices, uint32_t indexCount, uint32_t materialSlot = 0,
                          const float* avgAlbedo = nullptr);

    struct PathTraceMaterialInput {
        uint32_t diffuseTexGlId = 0;
        uint32_t normalTexGlId = 0;
        uint32_t glowTexGlId = 0;
        bool hasNormal = false;
        bool hasSpec = false;
        bool hasGlow = false;
        bool uniformSpec = false;  // spec shader without a spec map, see RtMaterialDesc::uniformSpec
        bool parallax = false;
        float parallaxScale = 0.0f, parallaxBias = 0.0f;
        uint32_t heightTexGlId = 0;
        float colorscale[3] = { 1.0f, 1.0f, 1.0f };
        float glowcolor[3] = { 1.0f, 1.0f, 1.0f };
        float metalness = 0.0f;
        // Sauerbraten VSlot::specscale, sourced from the native RtMaterialDesc::specScale
        float specScale = 1.0f;
        float roughness = -1.0f;
        // Sauerbraten MAT_ALPHA material-volume flag, sourced from the native RtMaterialDesc::isGlass
        bool isGlass = false;
        // per-material opacity, sourced from the native sauerinterop::RtMaterialDesc::glassAlpha
        float glassAlpha = 1.0f;
        // a separate dielectric flag from isGlass
        bool isWater = false;
        bool isLava = false;  // flags bit 7 (MATFLAG_LAVA)
        float glowSelectLum = 0.0f;
        // per-material Beer-Lambert absorption reference distance (world units)
        float waterReferenceThickness = 64.0f;
        // UV units per second (MaterialInfo::scroll)
        float scroll[2] = { 0.0f, 0.0f };
    };
    bool setPathTraceMaterials(const PathTraceMaterialInput* materials, uint32_t count, std::string* errorOut);

    bool setPathTraceEmissiveTriangles(const float* packedTriangles, uint32_t triangleCount, std::string* errorOut);
    bool removeChunkMesh(uint64_t chunkId);
    // Number of chunks currently tracked, for diagnostics/logging
    size_t chunkMeshCount() const { return chunkMeshes_.size(); }
    bool build(bool reflectionsActive, bool rtdiActive, bool pathTraceActive = false);
    bool buildWithRtao(bool rtaoActive, bool reflectionsActive, bool rtdiActive, bool pathTraceActive = false);
    bool isBuilt() const { return tlas() != VK_NULL_HANDLE; }
    // destroys every currently-built BLAS/TLAS (and their backing buffers) and clears all three mesh
    // pools
    void resetGeometry();
    void resetDynamicGeometry();
    uint64_t lastDynamicRebuildMicros() const { return lastDynamicRebuildMicros_; }
    uint64_t lastTlasRebuildMicros() const { return lastTlasRebuildMicros_; }
    std::string lastRigidDiag() const { return lastRigidDiag_; }

    // diagnostic
    void setForceAccelRebuild(bool force) { forceAccelRebuild_ = force; }
    VkAccelerationStructureKHR tlas() const {
        return dynamicGenerations_.empty() ? VK_NULL_HANDLE : dynamicGenerations_[dynamicGenerationIndex_].tlas;
    }
    VkDescriptorSetLayout descriptorSetLayout() const { return descriptorSetLayout_; }    VkPipeline pipeline() const { return rtaoPipeline_; }
    VkPipelineLayout pipelineLayout() const { return pipelineLayout_; }
    bool updateDescriptorSet(VkDescriptorSet set, SharedTexture& depth, SharedTexture& normal, SharedTexture& motion,
                             SharedTexture& output, SharedTexture& historyPrev, SharedTexture& historyNext,
                             SharedTexture& shadowOutput, SharedTexture& shadowHistoryPrev, SharedTexture& shadowHistoryNext,
                             SharedTexture& momentsHistoryPrev, SharedTexture& momentsHistoryNext,
                             VkImageView depthView, VkImageView normalView, VkImageView motionView,
                             VkImageView outputView, VkImageView historyPrevView, VkImageView historyNextView,
                             VkImageView shadowOutputView, VkImageView shadowHistoryPrevView, VkImageView shadowHistoryNextView,
                             VkImageView momentsHistoryPrevView, VkImageView momentsHistoryNextView,
                             uint32_t slotIndex, const RtaoConstants& constants, std::string* errorOut);
    bool createDescriptorSets(uint32_t count, std::vector<VkDescriptorSet>& sets, std::string* errorOut);

    VkDescriptorSetLayout descriptorSetLayoutReflection() const { return descriptorSetLayoutReflection_; }
    VkPipeline pipelineReflection() const { return reflectionPipeline_; }
    VkPipelineLayout pipelineLayoutReflection() const { return pipelineLayoutReflection_; }
    // resources in order
    bool updateReflectionDescriptorSet(VkDescriptorSet set, SharedTexture& depth, SharedTexture& normalRoughness,
                             SharedTexture& motion, SharedTexture& prevSceneColor,
                             SharedTexture& reflOutput, SharedTexture& reflHistoryPrev, SharedTexture& reflHistoryNext,
                             SharedTexture& reflGeoHistoryPrev, SharedTexture& reflGeoHistoryNext,
                             VkImageView depthView, VkImageView normalRoughnessView, VkImageView motionView,
                             VkImageView prevSceneColorView, VkImageView reflOutputView,
                             VkImageView reflHistoryPrevView, VkImageView reflHistoryNextView,
                             VkImageView reflGeoHistoryPrevView, VkImageView reflGeoHistoryNextView,
                             uint32_t slotIndex, const ReflectionConstants& constants, std::string* errorOut);
    bool createReflectionDescriptorSets(uint32_t count, std::vector<VkDescriptorSet>& sets, std::string* errorOut);

    bool ensureRtdiDebugPipeline();
    VkPipeline pipelineRtdiDebug() const { return rtdiDebugPipeline_; }
    VkPipelineLayout pipelineLayoutRtdiDebug() const { return pipelineLayoutRtdiDebug_; }
    VkDescriptorSetLayout descriptorSetLayoutRtdiDebug() const { return descriptorSetLayoutRtdiDebug_; }
    bool setRtdiLights(uint32_t slotIndex, const float* packed, uint32_t count);
    bool createRtdiDebugDescriptorSets(uint32_t count, std::vector<VkDescriptorSet>& sets, std::string* errorOut);
    bool updateRtdiDebugDescriptorSet(VkDescriptorSet set, SharedTexture& depth, SharedTexture& normalRoughness,
                             SharedTexture& output, VkImageView depthView, VkImageView normalRoughnessView,
                             VkImageView outputView, uint32_t slotIndex, const RtdiDebugConstants& constants,
                             std::string* errorOut);

    VkPipeline pipelineRtdi() const { return rtdiPipeline_; }
    VkPipelineLayout pipelineLayoutRtdi() const { return pipelineLayoutRtdi_; }
    VkDescriptorSetLayout descriptorSetLayoutRtdi() const { return descriptorSetLayoutRtdi_; }
    bool createRtdiDescriptorSets(uint32_t count, std::vector<VkDescriptorSet>& sets, std::string* errorOut);
    bool updateRtdiDescriptorSet(VkDescriptorSet set, SharedTexture& depth, SharedTexture& normalRoughness,
                             SharedTexture& output, SharedTexture& reservoirA, SharedTexture& reservoirB,
                             SharedTexture& historyReservoirA, SharedTexture& historyReservoirB,
                             SharedTexture& historyDepth, SharedTexture& historyNormalRoughness,
                             VkImageView depthView, VkImageView normalRoughnessView, VkImageView outputView,
                             VkImageView reservoirAView, VkImageView reservoirBView,
                             VkImageView historyReservoirAView, VkImageView historyReservoirBView,
                             VkImageView historyDepthView, VkImageView historyNormalRoughnessView,
                             uint32_t slotIndex, const RtdiConstants& constants, std::string* errorOut);
    VkSemaphore currentGenerationRtdiReadySemaphore();

    VkPipeline pipelinePathTrace() const { return pathTracePipeline_; }
    // pass 2 of the trace/resolve split
    VkPipeline pipelinePathTraceResolve() const { return pathTraceResolvePipeline_; }
    bool recordPathTraceRays(VkCommandBuffer cmd, VkDescriptorSet set, uint32_t width, uint32_t height, bool ser);
    static constexpr uint32_t kMaxPathTraceTextures = 4096;
    static constexpr uint32_t kCausticBatches = 8;  // keep in sync with CAUSTIC_BATCHES (shader)
    static constexpr uint32_t kCausticMaxPhotons = 2048u * 1024u;
    static uint32_t causticHashSizeFor(uint32_t photons) {
        const uint32_t g = static_cast<uint32_t>(std::sqrt(static_cast<float>(photons)));
        uint32_t n = (std::max)(2u * g * g, 1u << 19), size = 1;
        while (size < n) size <<= 1;
        return size;
    }
    VkPipeline pipelinePathTraceCaustics() const { return causticPipeline_; }
    static constexpr uint32_t kRadianceCacheSize = 1u << 20;  // keep in sync with RC_SIZE (shader)
    VkPipeline pipelinePathTraceCache() const { return cachePipeline_; }
    // zero every cell (map change, or the cache switched back on)
    void clearRadianceCache();
    static constexpr uint32_t kSkyCdfCells = 64u * 32u;  // keep in sync with SKY_CELLS (shader)
    VkPipeline pipelinePathTraceSkyCdf() const { return skyCdfPipeline_; }
    uint32_t causticPhotonDispatchCount(const PathTraceConstants& c) const {
        if (!c.causticsEnabled || causticPipeline_ == VK_NULL_HANDLE) return 0;
        return c.causticPhotons < kCausticMaxPhotons ? c.causticPhotons : kCausticMaxPhotons;
    }
    static constexpr uint32_t kRayStatCount = 64;
    VkBuffer rayStatsBuffer(uint32_t slot) const { return slot < rayStatsSlots_.size() ? rayStatsSlots_[slot].buffer : VK_NULL_HANDLE; }
    const uint32_t* rayStatsMapped(uint32_t slot) const { return slot < rayStatsSlots_.size() ? static_cast<const uint32_t*>(rayStatsSlots_[slot].mapped) : nullptr; }
    VkBuffer causticPhotonBuffer(uint32_t /*slot*/) const { return causticPhotonBuffers_.empty() ? VK_NULL_HANDLE : causticPhotonBuffers_[0].buffer; }
    VkBuffer causticHashBuffer(uint32_t /*slot*/) const { return causticHashBuffers_.empty() ? VK_NULL_HANDLE : causticHashBuffers_[0].buffer; }
    bool setCausticEmitters(const std::vector<float>& triangleCorners);
    VkPipelineLayout pipelineLayoutPathTrace() const { return pipelineLayoutPathTrace_; }
    VkDescriptorSetLayout descriptorSetLayoutPathTrace() const { return descriptorSetLayoutPathTrace_; }
    bool createPathTraceDescriptorSets(uint32_t count, std::vector<VkDescriptorSet>& sets, std::string* errorOut);
    bool updatePathTraceDescriptorSet(VkDescriptorSet set, SharedTexture& depth, SharedTexture& normal,
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
                             uint32_t slotIndex, const PathTraceConstants& constants, std::string* errorOut);

    VkPipeline pipelineVolFogInject() const { return volFogInjectPipeline_; }
    VkPipeline pipelineVolFogIntegrate() const { return volFogIntegratePipeline_; }
    VkPipeline pipelineVolFogDepth() const { return volFogDepthPipeline_; }  // volfogcull column depth pass
    VkPipelineLayout pipelineLayoutVolFog() const { return pipelineLayoutVolFog_; }
    VkDescriptorSetLayout descriptorSetLayoutVolFog() const { return descriptorSetLayoutVolFog_; }
    bool ensureVolumetricFogPipeline();
    bool createVolFogDescriptorSets(uint32_t count, std::vector<VkDescriptorSet>& sets, std::string* errorOut);
    // allocates (or resizes, on a grid-dimension change) the four persistent froxel-grid textures
    bool ensureVolFogTextures(uint32_t gridX, uint32_t gridY, uint32_t gridZ, std::string* errorOut);
    SharedTexture3D& volFogVisibilityA() { return *volFogVisibilityA_; }
    SharedTexture3D& volFogVisibilityB() { return *volFogVisibilityB_; }
    SharedTexture3D& volFogScatterExtinction() { return *volFogScatterExtinction_; }
    SharedTexture3D& volFogIntegrated() { return *volFogIntegrated_; }
    SharedTexture3D* volFogPartLightCurrent() { return volFogPingPong_ ? volFogPartLightA_.get() : volFogPartLightB_.get(); }
    uint32_t volFogGridX() const { return volFogGridX_; }
    uint32_t volFogGridY() const { return volFogGridY_; }
    uint32_t volFogGridZ() const { return volFogGridZ_; }
    bool updateVolFogDescriptorSet(VkDescriptorSet set, uint32_t slotIndex,
                             const GpuVolFogConstants& constants, std::string* errorOut);

    bool setPathTraceLights(uint32_t slotIndex, const float* packed, uint32_t count);
    // map/dynamic lights for the volumetric fog
    void setVolFogLights(const float* packed, uint32_t count);
    bool setLightGrid(const uint32_t* data, size_t count, std::string* errorOut);

    bool buildPathTraceMaterialTable(std::string* errorOut);
    VkSemaphore currentGenerationPathTraceReadySemaphore();

    bool setEmissiveMaterials(const float* packed, uint32_t count);
    // VK_NULL_HANDLE before the first successful setEmissiveMaterials call
    VkBuffer emissiveMaterialsBuffer() const { return emissiveMaterialsBuffer_.buffer; }
    uint32_t emissiveMaterialCount() const { return emissiveMaterialCount_; }

private:
    // owned = false
    struct Buffer { VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; VkDeviceAddress address = 0; VkDeviceSize size = 0;
                    VkBufferUsageFlags usage = 0;
                    VkDeviceSize offset = 0; bool owned = true; };
    struct Mesh { std::vector<float> positions; std::vector<uint32_t> indices; Buffer vertex; Buffer index; VkAccelerationStructureKHR blas = VK_NULL_HANDLE; Buffer blasBuffer;
        uint32_t materialSlot = 0;
    };
    struct ChunkRtVertex {
        float px, py, pz;
        float nx, ny, nz;
        float tx, ty, tz, tw;  // xyz tangent, w = handedness sign (+-1)
        float u, v;
        uint32_t textureIndex;
        uint32_t materialIndex;
        uint32_t blendMaterialIndex = 0xFFFFFFFFu;
        float blendAlpha = 0.0f;
        float blendU = 0.0f, blendV = 0.0f;
        // Sauerbraten skin::alphatest cutout value (WorldGeometryExtract.h)
        float alphaTestCutoff = 0.0f;
        // the blend layer's lightmap UV and its bindless index (0 = none -> blendAlpha above is used)
        float blendLmU = 0.0f, blendLmV = 0.0f;
        uint32_t blendLmTexIndex = 0;
    };
    struct ChunkMesh {
        std::vector<ChunkRtVertex> vertices;
        std::vector<uint32_t> indices;
        Buffer vertex, index, blasBuffer;
        VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
        uint32_t materialSlot = 0;
        float avgAlbedo[3] = { 0.6f, 0.6f, 0.6f };
        uint32_t opaqueIndexCount = 0xFFFFFFFFu;
        // added, BLAS not built yet, see flushChunkBuilds()
        bool pendingBuild = false;
    };
    // persistent, one-time-per-model gpu upload of a skeletal model's bind-pose vertex/index data
    struct SkinSource {
        Buffer vertexBuffer, indexBuffer;
        uint32_t vertexCount = 0, indexCount = 0, boneCount = 0;
    };
    std::unordered_map<uint64_t, SkinSource> skinSources_;

    struct PendingSkinUpload {
        uint64_t modelKey = 0;
        Buffer stagingVertex, stagingIndex;
        VkDeviceSize vertexBytes = 0, indexBytes = 0;
    };
    // session-lifetime-scratch queue
    std::vector<PendingSkinUpload> pendingSkinUploads_;

    struct RigidMeshSource {
        Buffer vertexBuffer, indexBuffer;
        VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
        Buffer blasBuffer;
        uint32_t vertexCount = 0, indexCount = 0;
        uint32_t materialSlot = 0;
        bool blasBuilt = false;
    };
    std::unordered_map<uint64_t, RigidMeshSource> rigidSources_;
    // session-lifetime, never-reset/reused counter
    uint32_t nextRigidMaterialOrdinal_ = 0;
    static const uint32_t kMaxRigidMaterialSlots = 64;

    struct PendingRigidUpload {
        uint64_t modelKey = 0;
        Buffer stagingVertex, stagingIndex;
        VkDeviceSize vertexBytes = 0, indexBytes = 0;
    };
    // session-lifetime-scratch queue
    std::vector<PendingRigidUpload> pendingRigidUploads_;

    // per-frame instance list
    struct RigidInstanceDraw {
        uint64_t modelKey = 0;
        float worldMatrix[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    };
    std::vector<RigidInstanceDraw> pendingRigidInstances_;

    bool createSkinPipeline();
    VkShaderModule skinShader_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayoutSkin_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayoutSkin_ = VK_NULL_HANDLE;
    VkPipeline skinPipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPoolSkin_ = VK_NULL_HANDLE;

    struct DynamicMeshSlot {
        uint32_t instanceMask = 0xFFu;  // TLAS mask, set by addSkinnedDynamicMesh only
        std::vector<float> positions;
        std::vector<uint32_t> indices;
        Buffer vertex, index, stagingVertex, stagingIndex, blasBuffer, scratch;
        uint32_t vertexCapacity = 0, indexCapacity = 0;
        VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
        bool blasBuilt = false;
        uint32_t lastBuiltVertexCount = 0;
        uint32_t lastBuiltIndexCount = 0;
        uint32_t materialSlot = 0;

        bool skinned = false;
        uint64_t skinSourceKey = 0;
        // current-frame bone dual quaternions for this instance
        std::vector<float> boneMatrixData;
        float worldMatrix[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        Buffer boneBuffer, boneStaging;
        Buffer skinParams, skinParamsStaging;
        VkDescriptorSet skinDescriptorSet = VK_NULL_HANDLE;
        VkBuffer skinDescriptorBoundSource = VK_NULL_HANDLE, skinDescriptorBoundBones = VK_NULL_HANDLE,
                 skinDescriptorBoundOutput = VK_NULL_HANDLE, skinDescriptorBoundParams = VK_NULL_HANDLE;

        bool water = false;
        // glowing sprite quads, non-opaque, TLAS mask 0x80 (only reflection/refraction rays see them)
        bool particles = false;
        bool alphaTested = false;  // non-opaque, per-vertex alpha-tested (addGrassMesh)
        std::vector<ChunkRtVertex> waterVertices;

        bool textured = false;
        std::vector<ChunkRtVertex> texturedVertices;
    };

    bool recordSkinDispatch(VkCommandBuffer cmd, DynamicMeshSlot& slot);

    struct DynamicGeneration {
        std::vector<DynamicMeshSlot> slots;
        size_t slotsUsed = 0;
        VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
        Buffer tlasBuffer, tlasInstanceBuffer, tlasScratch;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        Buffer materials;
        VkFence buildFence = VK_NULL_HANDLE;
        VkSemaphore readyForAo = VK_NULL_HANDLE;
        VkSemaphore readyForReflections = VK_NULL_HANDLE;
        VkSemaphore readyForRtdi = VK_NULL_HANDLE;
        VkSemaphore readyForPathTrace = VK_NULL_HANDLE;
        // via Vulkan validation
        bool aoReadyPending = false;
        bool reflectionsReadyPending = false;
        bool rtdiReadyPending = false;
        bool pathTraceReadyPending = false;
        uint32_t lastBuiltInstanceCount = 0;

        std::vector<Buffer> retiredStagingBuffers_;
    };

    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, Buffer& out, VkMemoryAllocateFlags allocFlags = 0);
    void destroyBuffer(Buffer& b);
    // grow-only helper shared by every persistent buffer below
    bool ensureBufferCapacity(Buffer& buf, VkDeviceSize requiredSize, VkBufferUsageFlags usage,
                              VkMemoryPropertyFlags properties, VkMemoryAllocateFlags allocFlags = 0);
    bool buildBlas(Mesh& mesh);
    // chunk tier equivalents of buildBlas() above
    bool buildChunkBlas(ChunkMesh& mesh);
public:
    bool flushChunkBuilds();
    // timing of the last flushChunkBuilds() (ms)
    struct FlushStats { double createMs = 0, buildMs = 0, compactMs = 0; uint32_t chunks = 0, batches = 0; };
    FlushStats lastFlushStats() const { return lastFlushStats_; }
    size_t pendingChunkBuildCount() const { size_t n = 0; for (const auto& kv : chunkMeshes_) if (kv.second.pendingBuild) ++n; return n; }
private:
    void destroyChunkBlas(ChunkMesh& mesh);
    bool compactBlas(VkAccelerationStructureKHR& blas, Buffer& blasBuffer);
    bool recordDynamicSlotBuild(VkCommandBuffer cmd, DynamicMeshSlot& slot, uint32_t ordinal);
    bool recordTlasBuild(VkCommandBuffer cmd, DynamicGeneration& generation);
    bool ensureDynamicGenerations();
    bool loadRtFunctions();
    bool ensureRtaoPipeline();
    bool createParamSlots();
    bool ensureReflectionPipeline();
    bool createReflectionParamSlots();
    bool ensureRtdiPipeline();
    bool ensureRtdiLightBuffers();
    bool ensureTimestampPool();
    bool ensurePathTracePipeline();
    bool ensurePathTraceLightBuffers();

    InteropContext& ctx_;
    std::vector<Mesh> meshes_;
    std::unordered_map<uint64_t, ChunkMesh> chunkMeshes_;
    FlushStats lastFlushStats_;  // see lastFlushStats()
    std::vector<Buffer> chunkArenas_;
    // true once meshes_' BLASes have been built the one time that ever happens
    bool staticBuilt_ = false;

    std::vector<DynamicGeneration> dynamicGenerations_;
    uint32_t dynamicGenerationIndex_ = 0;
    bool forceAccelRebuild_ = false;

    VkQueryPool timestampPool_ = VK_NULL_HANDLE;
    float timestampPeriodNs_ = 1.0f;
    uint64_t lastDynamicRebuildMicros_ = 0;
    uint64_t lastTlasRebuildMicros_ = 0;
    std::string lastRigidDiag_;

    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR_ = nullptr;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR_ = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR_ = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR_ = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR_ = nullptr;
    PFN_vkGetBufferDeviceAddress vkGetBufferDeviceAddress_ = nullptr;
    PFN_vkCmdWriteAccelerationStructuresPropertiesKHR vkCmdWriteAccelerationStructuresPropertiesKHR_ = nullptr;
    PFN_vkCmdCopyAccelerationStructureKHR vkCmdCopyAccelerationStructureKHR_ = nullptr;

    VkSampler sampler_ = VK_NULL_HANDLE;
    // sampler_ (above) is VK_FILTER_NEAREST
    VkSampler historySampler_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline rtaoPipeline_ = VK_NULL_HANDLE;
    VkShaderModule rtaoShader_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    struct ParamSlot { VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; void* mapped = nullptr; };
    std::vector<ParamSlot> paramSlots_;
    uint32_t frameSlots_ = 3;
    // separate size for every per-dispatch-kind resource ring in this class
    uint32_t dispatchSlots_ = 3;
    std::string shaderDirectory_;

    // reflection pipeline's layout/pipeline/pool/param-ring
    VkDescriptorSetLayout descriptorSetLayoutReflection_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayoutReflection_ = VK_NULL_HANDLE;
    VkPipeline reflectionPipeline_ = VK_NULL_HANDLE;
    VkShaderModule reflectionShader_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPoolReflection_ = VK_NULL_HANDLE;
    std::vector<ParamSlot> reflectionParamSlots_;

    VkDescriptorSetLayout descriptorSetLayoutRtdiDebug_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayoutRtdiDebug_ = VK_NULL_HANDLE;
    VkPipeline rtdiDebugPipeline_ = VK_NULL_HANDLE;
    VkShaderModule rtdiDebugShader_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPoolRtdiDebug_ = VK_NULL_HANDLE;
    std::vector<ParamSlot> rtdiDebugParamSlots_;
    std::vector<Buffer> rtdiLightBuffers_;
    // how many light records are currently valid in each ring slot's buffer
    std::vector<uint32_t> rtdiLightCounts_;
    Buffer emissiveMaterialsBuffer_;
    uint32_t emissiveMaterialCount_ = 0;
    VkDescriptorSetLayout descriptorSetLayoutRtdi_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayoutRtdi_ = VK_NULL_HANDLE;
    VkPipeline rtdiPipeline_ = VK_NULL_HANDLE;
    VkShaderModule rtdiShader_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPoolRtdi_ = VK_NULL_HANDLE;
    std::vector<ParamSlot> rtdiParamSlots_;

    VkDescriptorSetLayout descriptorSetLayoutPathTrace_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayoutPathTrace_ = VK_NULL_HANDLE;
    VkPipeline pathTracePipeline_ = VK_NULL_HANDLE;
    VkShaderModule pathTraceShader_ = VK_NULL_HANDLE;
    VkPipeline pathTraceResolvePipeline_ = VK_NULL_HANDLE;
    VkPipeline pathTraceRgenPipeline_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkShaderModule pathTraceRgenShader_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    Buffer pathTraceRgenSbt_[2];
    VkStridedDeviceAddressRegionKHR pathTraceRgenRegion_[2] = {};
    PFN_vkCmdTraceRaysKHR cmdTraceRays_ = nullptr;
    PFN_vkCmdSetRayTracingPipelineStackSizeKHR cmdSetRtStack_ = nullptr;
    uint32_t pathTraceRgenStack_[2] = {0, 0};  // driver-reported raygen stack
public:
    uint32_t pathTraceRgenStackMin_ = 0;  // stack floor (ptraygenstack), diagnostic
private:
    void createPathTraceRaygenPipelines();
    VkShaderModule pathTraceResolveShader_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPoolPathTrace_ = VK_NULL_HANDLE;
    std::vector<ParamSlot> pathTraceParamSlots_;
    std::vector<ParamSlot> rayStatsSlots_;  // see rayStatsBuffer()
    std::vector<Buffer> pathTraceLightBuffers_;
    // see pipelinePathTraceCaustics()
    VkShaderModule causticShader_ = VK_NULL_HANDLE;
    VkPipeline causticPipeline_ = VK_NULL_HANDLE;
    // see pipelinePathTraceCache()
    VkShaderModule cacheShader_ = VK_NULL_HANDLE;
    VkPipeline cachePipeline_ = VK_NULL_HANDLE;
    Buffer rcKeyBuffer_, rcAccumBuffer_, rcRadianceBuffer_;
    // RESTIR DI reservoirs (binding 33)
    Buffer restirBuffer_;
    void fillBufferNow(Buffer& buf, uint32_t value);  // one-off fill of a new buffer
    // see pipelinePathTraceSkyCdf()
    VkShaderModule skyCdfShader_ = VK_NULL_HANDLE;
    VkPipeline skyCdfPipeline_ = VK_NULL_HANDLE;
    Buffer skyCdfBuffer_;
    std::vector<Buffer> causticPhotonBuffers_;
    uint32_t causticCapacity_ = 0;  // photons per batch causticPhotonBuffers_[0] holds
    bool ensureCausticCapacity(uint32_t photonsPerBatch);
    std::vector<Buffer> causticHashBuffers_;
    Buffer causticEmitterBuffer_;
    uint32_t causticEmitterTriCount_ = 0;
    float causticEmitterArea_ = 0.0f;
    uint32_t causticValidMask_ = 0;
    uint64_t causticEmitterVersion_ = 0, causticMaskEmitterVersion_ = ~0ull;
    float causticMaskRadius_ = -1.0f;
    uint32_t causticMaskPhotons_ = 0;
    std::vector<uint32_t> pathTraceLightCounts_;

    VkDescriptorSetLayout descriptorSetLayoutVolFog_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayoutVolFog_ = VK_NULL_HANDLE;
    VkPipeline volFogInjectPipeline_ = VK_NULL_HANDLE;
    VkShaderModule volFogInjectShader_ = VK_NULL_HANDLE;
    VkPipeline volFogIntegratePipeline_ = VK_NULL_HANDLE;
    VkShaderModule volFogIntegrateShader_ = VK_NULL_HANDLE;
    VkPipeline volFogDepthPipeline_ = VK_NULL_HANDLE;
    VkShaderModule volFogDepthShader_ = VK_NULL_HANDLE;
    std::vector<Buffer> volFogDepthBuffers_;  // per dispatch slot, binding 16, volfogcull column depths
    VkDescriptorPool descriptorPoolVolFog_ = VK_NULL_HANDLE;
    std::vector<ParamSlot> volFogParamSlots_;
    std::vector<Buffer> volFogLightBuffers_;  // per dispatch slot, binding 8
    std::vector<float> volFogPendingLights_;  // setVolFogLights() staging
    Buffer lightGridBuffer_;  // see setLightGrid()
    bool ensureLightGridPlaceholder();
    // persistent froxel-grid textures
    std::unique_ptr<SharedTexture3D> volFogVisibilityA_;
    std::unique_ptr<SharedTexture3D> volFogVisibilityB_;
    // ping-ponged pair
    std::unique_ptr<SharedTexture3D> volFogScatterExtinction_;
    std::unique_ptr<SharedTexture3D> volFogScatterExtinctionB_;
    std::unique_ptr<SharedTexture3D> volFogIntegrated_;
    // ping-ponged like the visibility pair (prev sampled for temporal blending, next written)
    std::unique_ptr<SharedTexture3D> volFogPartLightA_, volFogPartLightB_;
    VkImageView volFogPartLightAView_ = VK_NULL_HANDLE, volFogPartLightBView_ = VK_NULL_HANDLE;
    std::unique_ptr<SharedTexture3D> volFogNoisePlaceholder_;
    uint32_t volFogGridX_ = 0, volFogGridY_ = 0, volFogGridZ_ = 0;
    // cached VkImageViews for the five textures above
    VkImageView volFogVisibilityAView_ = VK_NULL_HANDLE;
    VkImageView volFogVisibilityBView_ = VK_NULL_HANDLE;
    VkImageView volFogScatterExtinctionView_ = VK_NULL_HANDLE;
    VkImageView volFogScatterExtinctionBView_ = VK_NULL_HANDLE;
    VkImageView volFogIntegratedView_ = VK_NULL_HANDLE;
    VkImageView volFogNoisePlaceholderView_ = VK_NULL_HANDLE;
    bool volFogPingPong_ = false;
    uint64_t pathTraceStaticDescriptorGeneration_ = 0;
    std::vector<uint64_t> pathTraceStaticDescriptorWrittenGeneration_;
    Buffer pathTraceMaterialsBuffer_;
    uint32_t pathTraceMaterialCount_ = 0;
    std::vector<uint8_t> pathTraceMaterialMirror_;
    static const uint32_t kMaxDynamicMaterialSlots = 512;
    uint32_t dynamicMaterialBase_ = 0;
    uint64_t geometryEpoch_ = 0;

    struct WaterPatchGen {
        Buffer vertex, blasBuffer, scratch, params;
        VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        bool built = false, active = false;
        float originX = 0.0f, originY = 0.0f, cellSize = 0.0f;
        uint32_t cells = 0;
        // FFT sea output for this generation (6 layers, see rt_water.glsl) + its FFT scratch
        VkImage fftImage = VK_NULL_HANDLE;
        VkDeviceMemory fftMemory = VK_NULL_HANDLE;
        VkImageView fftView = VK_NULL_HANDLE;
        Buffer fftTemp, fftParams;
        VkDescriptorSet fftSet = VK_NULL_HANDLE;
        bool fftValid = false;  // this generation's FFT ran for its current frame
    };
    static constexpr uint32_t kWaveN = 256;  // FFT size per cascade (water_fft.comp's WAVE_N)
    static constexpr uint32_t kWaveFftChannels = 6;  // complex channels per cascade (water_fft.comp's WAVE_CH)
    static constexpr uint32_t kWaveFftLayers = 9;  // output layers
    static constexpr float kUnitsPerMetre = 8.0f;  // Sauerbraten
    bool wavesCreated_ = false, wavesFailed_ = false;
    Buffer waveSpectrum_, waveSpectrumParams_, waveMapParams_;
    VkDescriptorSet waveSpectrumSet_ = VK_NULL_HANDLE, waveMapSet_ = VK_NULL_HANDLE;
    float waveSpectrumKey_[6] = {-1, -1, -1, -1, -1, -1};
    float waveMapKey_[5] = {-1, -1, -1, -1, -1};
    VkImage waveMapImage_ = VK_NULL_HANDLE;
    VkDeviceMemory waveMapMemory_ = VK_NULL_HANDLE;
    VkImageView waveMapView_ = VK_NULL_HANDLE;
    uint32_t waveMapW_ = 1, waveMapH_ = 1;
    float waveMapOriginX_ = 0.0f, waveMapOriginY_ = 0.0f, waveMapTexel_ = 4.0f;
    bool waveMapValid_ = false;  // baked for the current map + settings
    bool waveMapNeedsImage_ = false;
    VkSampler waveFftSampler_ = VK_NULL_HANDLE, waveMapSampler_ = VK_NULL_HANDLE;
    VkShaderModule waveSpectrumShader_ = VK_NULL_HANDLE, waveFftShader_ = VK_NULL_HANDLE, waveMapShader_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout waveSpectrumSetLayout_ = VK_NULL_HANDLE, waveFftSetLayout_ = VK_NULL_HANDLE, waveMapSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout waveSpectrumLayout_ = VK_NULL_HANDLE, waveFftLayout_ = VK_NULL_HANDLE, waveMapLayout_ = VK_NULL_HANDLE;
    VkPipeline waveSpectrumPipeline_ = VK_NULL_HANDLE, waveFftPipeline_ = VK_NULL_HANDLE, waveMapPipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPoolWave_ = VK_NULL_HANDLE;
    bool ensureWaveResources();
    bool createWaveImage(uint32_t w, uint32_t h, uint32_t layers, VkImage& image, VkDeviceMemory& memory, VkImageView& view);
    bool recreateWaveMapImage();
    void waveCascadeSizes(float out[3]) const;
    void waveBands(float out[3]) const;
    bool recordWaveFft(VkCommandBuffer cmd, WaterPatchGen& gen);
    bool recordWaveMapBake(VkCommandBuffer cmd, VkAccelerationStructureKHR tlas);
    void logWaveMapStats();  // ptwavestats
    bool waveMapStatsPending_ = false;  // a bake to log once it has run
    // pipeline stats diagnostic (env SAUER_PIPELINE_STATS)
    void reportPipelineStats(const char* spvName, VkPipelineLayout layout);
    // rt_water.glsl's fft / map / amp vectors for a frame of generation gen
    void waveShaderParams(const WaterPatchGen& gen, float fft[4], float map[4], float amp[4]) const;
    std::vector<WaterPatchGen> waterPatchGens_;  // one per generation (frame in flight)
    WaterPatchSettings waterPatchSettings_;
    Buffer waterRectBuffer_, waterBucketBuffer_, waterItemBuffer_, waterPatchIndexBuffer_;
    uint32_t waterRectCount_ = 0, waterPatchIndexCells_ = 0;
    float waterBucketOriginX_ = 0.0f, waterBucketOriginY_ = 0.0f, waterBucketSize_ = 64.0f, waterMinZ_ = 0.0f;
    uint32_t waterBucketsX_ = 0, waterBucketsY_ = 0;
    uint64_t waterFlatChunkId_ = 0, lavaFlatChunkId_ = 0;
    std::vector<uint64_t> volFogTexGenWritten_;  // per fog descriptor set
    uint32_t waterTopTris_ = 0, lavaTopTris_ = 0;
    uint32_t waterPatchMaterialSlot_ = 0;  // the materials[] entry right before the dynamic range
    VkShaderModule waterPatchShader_ = VK_NULL_HANDLE;
    VkPipeline waterPatchPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayoutWaterPatch_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayoutWaterPatch_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPoolWaterPatch_ = VK_NULL_HANDLE;
    bool waterPatchPipelineFailed_ = false;
    bool createWaterPatchPipeline();
    bool recordWaterPatch(VkCommandBuffer cmd, WaterPatchGen& gen);
    struct DynamicMaterialUpdate { uint32_t ordinal; uint64_t vertexAddr; uint64_t indexAddr; uint64_t prevVertexAddr = 0; };
    bool updateDynamicMaterialEntries(const std::vector<DynamicMaterialUpdate>& entries, std::string* errorOut = nullptr);
    // pbr materials Sauerbraten-VSlot-indexed material table, and pbr light transport
    Buffer pathTraceMaterialInfoBuffer_;
    uint32_t pathTraceMaterialInfoCount_ = 0;
    Buffer pathTraceEmissiveBuffer_;
    uint32_t pathTraceEmissiveCount_ = 0;
    // sum, over every emissive triangle currently uploaded, of luminance(unscaled radiance) * that
    // triangle's area
    float pathTraceEmissiveTotalPower_ = 0.0f;
    struct PathTraceTexture {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        uint32_t glId = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t clamp = 0;
        uint32_t mipLevels = 1;
        // mip chain supplied by the caller, see registerPathTraceTexture()
        bool preMipped = false;
    };
    std::vector<PathTraceTexture> pathTraceTextures_;
    std::unordered_map<uint32_t, uint32_t> pathTraceTextureIndices_;

    struct PendingTextureUpload {
        uint32_t textureIndex = 0;  // index into pathTraceTextures_
        Buffer staging;
        uint32_t width = 0, height = 0;
    };
    std::vector<PendingTextureUpload> pendingTextureUploads_;

    void recordPendingUploads(VkCommandBuffer cmd, DynamicGeneration& generation);
public:
    VkSemaphore currentGenerationAoReadySemaphore();
    VkSemaphore currentGenerationReflectionsReadySemaphore();
};
}  // namespace interop
