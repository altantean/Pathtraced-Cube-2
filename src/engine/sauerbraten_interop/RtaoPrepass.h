#pragma once
// RtaoPrepass depth+normal+motion+albedo prepass

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace sauerinterop {

// one contiguous per-texture draw range
struct PrepassBatch {
    uint32_t textureGlId = 0;
    uint32_t indexOffset = 0;
    uint32_t indexCount = 0;
    uint32_t normalTexGlId = 0;
    uint32_t glowTexGlId = 0;
    bool hasNormal = false;
    bool hasSpec = false;
    bool hasGlow = false;
    float colorscale[3] = { 1.0f, 1.0f, 1.0f };
    float glowcolor[3] = { 1.0f, 1.0f, 1.0f };
    // per-batch metalness
    float metalness = 0.0f;
    // per-batch specscale render() feeds this to the fragment shader's uSpecScale uniform
    float specScale = 1.0f;
    float roughness = -1.0f;  // roughness override (vrough/texrough), -1 = automatic
    // mapmodel alpha-test see ChunkGlBatch::alphaTestCutoff
    float alphaTestCutoff = 0.0f;
    float uvScroll[2] = { 0.0f, 0.0f };
};

class RtaoPrepass {
public:
    RtaoPrepass() = default;
    ~RtaoPrepass();

    RtaoPrepass(const RtaoPrepass&) = delete;
    RtaoPrepass& operator=(const RtaoPrepass&) = delete;

    // compiles the prepass program
    bool initialize(std::string* error);

    bool uploadMesh(const float* interleavedPosNormalUv, uint32_t vertexCount,
                     const uint32_t* indices, uint32_t indexCount,
                     const PrepassBatch* batches, uint32_t batchCount, std::string* error);

    bool attachTargets(uint32_t depthGlTexture, uint32_t normalGlTexture, uint32_t motionGlTexture,
                        uint32_t albedoGlTexture, uint32_t glowGlTexture, uint32_t width, uint32_t height, std::string* error);

    bool render(uint32_t width, uint32_t height, const float* camProjFlat16, const float* camFlat16,
                const float* currentVpUnjitteredFlat16, const float* prevVpUnjitteredFlat16);

    uint32_t indexCount() const { return indexCount_; }

    // RtaoPrepass

    struct AnimatedSkinBatch {
        uint32_t glTexture = 0;
        uint32_t indexOffset = 0;
        uint32_t indexCount = 0;
        uint32_t masksGlTexture = 0;
        float spec = 0.0f;
        float glow = 0.0f;
    };
    bool uploadAnimatedSkinSource(uint64_t modelKey, const void* vertexData, uint32_t vertexStride,
                                   uint32_t vertexCount, const uint32_t* indices, uint32_t indexCount,
                                   const AnimatedSkinBatch* batches, uint32_t batchCount,
                                   std::string* error);
    bool hasAnimatedSkinSource(uint64_t modelKey) const;

    // one animated instance to draw this frame
    struct AnimatedInstance {
        uint64_t modelKey = 0;
        const float* currentBones = nullptr;
        const float* prevBones = nullptr;
        uint32_t boneCount = 0;
        float currentWorldMatrix[12] = {};
        float prevWorldMatrix[12] = {};
    };

    bool renderAnimated(uint32_t width, uint32_t height, const AnimatedInstance* instances, uint32_t instanceCount,
                         const float* camProjFlat16, const float* camFlat16, const float* currentVpUnjitteredFlat16,
                         const float* prevVpUnjitteredFlat16);

    struct ParticleRun { uint32_t glTexture, first, count; };
    bool renderParticleMotion(uint32_t width, uint32_t height, uint32_t motionGlTexture, uint32_t sceneDepthGlTexture,
                              const float* verts, uint32_t vertexCount, const ParticleRun* runs, uint32_t runCount,
                              const float* camProjFlat16, const float* currentVpUnjitteredFlat16,
                              const float* prevVpUnjitteredFlat16, float minCoverage);

    // ptmodelreflect, see kAnimatedFragmentSrc's uModelReflect
    void setModelReflect(float v) { modelReflect_ = v; }
    // ptmat, the guide roughness follows the path tracer's material model
    void setPtMat(bool on) { ptMat_ = on; }

private:
    bool initialized_ = false;
    uint32_t program_ = 0;
    uint32_t vao_ = 0, vbo_ = 0, ebo_ = 0;
    uint32_t fallbackTex_ = 0;
    uint32_t fbo_ = 0;
    uint32_t indexCount_ = 0;
    int uCamProjLoc_ = -1, uCamLoc_ = -1;
    int uCurrUnjitteredLoc_ = -1, uPrevUnjitteredLoc_ = -1;
    int uAlbedoTexLoc_ = -1;
    // new per-batch uniforms
    int uNormalTexLoc_ = -1, uGlowTexLoc_ = -1;
    int uHasNormalLoc_ = -1, uHasSpecLoc_ = -1, uHasGlowLoc_ = -1;
    int uColorScaleLoc_ = -1, uGlowColorLoc_ = -1;
    int uMetalnessLoc_ = -1;  // pbr light transport
    int uAlphaTestLoc_ = -1;  // see PrepassBatch::alphaTestCutoff
    int uUvOffsetLoc_ = -1, uUvStepLoc_ = -1;  // scrolling textures
    float scrollTime_ = 0.0f, scrollDt_ = 0.0f;
public:
    // this frame's scroll clock and step (seconds), see PrepassBatch::uvScroll
    void setScrollClock(float t, float dt) { scrollTime_ = t; scrollDt_ = dt; }
private:
    int uSpecScaleLoc_ = -1;  // on uSpecScale (RtaoPrepass.cpp)
    int uPtMatLoc_ = -1, animUPtMatLoc_ = -1;  // ptmat, see materialRoughnessForDlss
    int uRoughOverrideLoc_ = -1;  // per batch
    bool ptMat_ = false;
    bool haveTargets_ = false;
    std::vector<PrepassBatch> batches_;
    // see kSkyMotionFragmentSrc (RtaoPrepass.cpp)
    uint32_t skyProgram_ = 0;
    int skyUCurrUnjitteredLoc_ = -1, skyUPrevUnjitteredLoc_ = -1;

    bool animInitialized_ = false;
    uint32_t animProgram_ = 0;
    int animUCamProjLoc_ = -1, animUCurrUnjitteredLoc_ = -1, animUPrevUnjitteredLoc_ = -1;
    int animUCamLoc_ = -1, animUAlbedoTexLoc_ = -1;
    int animUMasksTexLoc_ = -1, animUHasMasksLoc_ = -1, animUModelSpecLoc_ = -1, animUModelGlowLoc_ = -1;
    int animUModelReflectLoc_ = -1;
    float modelReflect_ = 0.5f;
    int animUCurrWorldRow0Loc_ = -1, animUCurrWorldRow1Loc_ = -1, animUCurrWorldRow2Loc_ = -1;
    int animUPrevWorldRow0Loc_ = -1, animUPrevWorldRow1Loc_ = -1, animUPrevWorldRow2Loc_ = -1;
    uint32_t animCurrBoneSsbo_ = 0, animPrevBoneSsbo_ = 0;
    struct AnimatedSkinGlSource {
        uint32_t vao = 0, vbo = 0, ebo = 0;
        uint32_t indexCount = 0;
        std::vector<AnimatedSkinBatch> batches;
    };
    std::unordered_map<uint64_t, AnimatedSkinGlSource> animSkinSources_;
    bool initAnimatedProgram(std::string* error);

    // particle motion vectors
    int partState_ = 0;  // 0 untried, 1 ready, -1 failed
    uint32_t partProgram_ = 0, partVao_ = 0, partVbo_ = 0, partFbo_ = 0;
    int partUCamProjLoc_ = -1, partUCurrLoc_ = -1, partUPrevLoc_ = -1, partUTexLoc_ = -1, partUMinLoc_ = -1;
    bool initParticleMotion();
};

}  // namespace sauerinterop
