#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "model.h"  // SkinExtractSink, SkinSourceVertexData, dynent, vec

namespace sauerinterop {

// walks entities::getents() for ET_MAPMODEL entities with animation data
// (model::hasRealAnimationData())
void extractAnimatedMapModelInstances(SkinExtractSink& sink);

class AnimatedInstanceCollector : public SkinExtractSink {
public:
    bool hasSkinSource(uint64_t modelKey, uint32_t boneCount) override;
    void submitSkinSource(uint64_t modelKey, const SkinSourceVertexData* vertices, uint32_t vertexCount,
                           const uint32_t* indices, uint32_t indexCount, uint32_t boneCount) override;
    uint32_t registerTexture(uint32_t textureGlId) override;
    void submitPose(uint64_t modelKey, const float* boneDualQuats, uint32_t boneCount,
                     const float worldMatrix[12]) override;
    // override
    void submitDynamicMesh(uint64_t modelKey, const SkinSourceVertexData* vertices, uint32_t vertexCount,
                            const uint32_t* indices, uint32_t indexCount) override;
    bool hasRigidSource(uint64_t modelKey) override { return knownRigidModels_.find(modelKey) != knownRigidModels_.end(); }
    void registerRigidSource(uint64_t modelKey, const SkinSourceVertexData* vertices, uint32_t vertexCount,
                              const uint32_t* indices, uint32_t indexCount) override;
    void submitRigidInstance(uint64_t modelKey, const float worldMatrix[12]) override;
    void beginInstance(dynent* d) override {
        currentDynent_ = d;
        if (directPoseTarget_) directPoseTarget_->beginInstance(d);
    }
    // only a direct (same-frame, path-trace) target can have accepted an instance into this frame's
    // scene
    bool lastRigidInstanceAccepted() override {
        return directPoseTarget_ && directPoseTarget_->lastRigidInstanceAccepted();
    }
    bool lastPoseAccepted() override {
        return directPoseTarget_ && directPoseTarget_->lastPoseAccepted();
    }

    void setDirectPoseTarget(SkinExtractSink* target) { directPoseTarget_ = target; }

    void flush(SkinExtractSink& target);

    void invalidateKnownModels() { knownModels_.clear(); knownRigidModels_.clear(); }

private:
    // session-lifetime map (modelKey -> the boneCount it was last buffered/registered with)
    std::unordered_map<uint64_t, uint32_t> knownModels_;
    dynent* currentDynent_ = nullptr;
    SkinExtractSink* directPoseTarget_ = nullptr;

    struct PendingSkinSource {
        uint64_t modelKey = 0;
        std::vector<SkinSourceVertexData> vertices;
        std::vector<uint32_t> indices;
        uint32_t boneCount = 0;
    };
    std::vector<PendingSkinSource> pendingSkinSources_;

    struct PendingDynamicMesh {
        uint64_t modelKey = 0;
        std::vector<SkinSourceVertexData> vertices;
        std::vector<uint32_t> indices;
    };
    std::vector<PendingDynamicMesh> pendingDynamicMeshes_;

    struct PendingPose {
        uint64_t modelKey = 0;
        std::vector<float> boneData;
        float worldMatrix[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        dynent* d = nullptr;
    };
    std::vector<PendingPose> pendingPoses_;

    std::unordered_set<uint64_t> knownRigidModels_;
    struct PendingRigidRegistration {
        uint64_t modelKey = 0;
        std::vector<SkinSourceVertexData> vertices;
        std::vector<uint32_t> indices;
    };
    std::vector<PendingRigidRegistration> pendingRigidRegistrations_;
    struct PendingRigidInstance {
        uint64_t modelKey = 0;
        float worldMatrix[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    };
    std::vector<PendingRigidInstance> pendingRigidInstances_;
};

}  // namespace sauerinterop
