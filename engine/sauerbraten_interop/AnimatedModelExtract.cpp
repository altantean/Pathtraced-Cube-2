#include <new>
#define SAUERINTEROP_STL_PLACEMENT_NEW
#include "engine.h"  // entities::getents/animatemapmodel/extentity, model/loadmapmodel

#include "AnimatedModelExtract.h"  // model.h (already fully defined above) -> SkinExtractSink, dynent, vec

namespace sauerinterop {

void extractAnimatedMapModelInstances(SkinExtractSink& sink)
{
    vector<extentity*>& ents = entities::getents();
    loopv(ents)
    {
        extentity& e = *ents[i];
        if (e.type != ET_MAPMODEL || e.attr2 < 0) continue;
        if (e.flags & (EF_NOVIS | EF_ANIM)) continue;

        model* m = loadmapmodel(e.attr2);
        if (!m) continue;

        // the complement of MapModelExtract's static gate
        if (!m->mapmodelAnimates()) continue;

        int anim = 0, basetime = 0;
        entities::animatemapmodel(e, anim, basetime);

        m->extractSkinPose(anim, basetime, 0, e.o, float(e.attr1), 0.0f, nullptr, sink);
    }
}

bool AnimatedInstanceCollector::hasSkinSource(uint64_t modelKey, uint32_t boneCount)
{
    auto it = knownModels_.find(modelKey);
    return it != knownModels_.end() && it->second == boneCount;
}

void AnimatedInstanceCollector::submitSkinSource(uint64_t modelKey, const SkinSourceVertexData* vertices, uint32_t vertexCount,
                                                  const uint32_t* indices, uint32_t indexCount, uint32_t boneCount)
{
    auto it = knownModels_.find(modelKey);
    if (it != knownModels_.end() && it->second == boneCount) return;
    knownModels_[modelKey] = boneCount;
    PendingSkinSource pending;
    pending.modelKey = modelKey;
    pending.vertices.assign(vertices, vertices + vertexCount);
    pending.indices.assign(indices, indices + indexCount);
    pending.boneCount = boneCount;
    pendingSkinSources_.push_back(std::move(pending));
}

uint32_t AnimatedInstanceCollector::registerTexture(uint32_t textureGlId)
{
    return textureGlId;
}

void AnimatedInstanceCollector::submitPose(uint64_t modelKey, const float* boneDualQuats, uint32_t boneCount,
                                            const float worldMatrix[12])
{
    if (directPoseTarget_) {
        directPoseTarget_->submitPose(modelKey, boneDualQuats, boneCount, worldMatrix);
        return;
    }
    PendingPose pending;
    pending.modelKey = modelKey;
    pending.boneData.assign(boneDualQuats, boneDualQuats + static_cast<size_t>(boneCount) * 8);
    std::memcpy(pending.worldMatrix, worldMatrix, sizeof(float) * 12);
    pending.d = currentDynent_;
    pendingPoses_.push_back(std::move(pending));
}

void AnimatedInstanceCollector::submitDynamicMesh(uint64_t modelKey, const SkinSourceVertexData* vertices, uint32_t vertexCount,
                                                    const uint32_t* indices, uint32_t indexCount)
{
    PendingDynamicMesh pending;
    pending.modelKey = modelKey;
    pending.vertices.assign(vertices, vertices + vertexCount);
    pending.indices.assign(indices, indices + indexCount);
    pendingDynamicMeshes_.push_back(std::move(pending));
}

void AnimatedInstanceCollector::registerRigidSource(uint64_t modelKey, const SkinSourceVertexData* vertices, uint32_t vertexCount,
                                                      const uint32_t* indices, uint32_t indexCount)
{
    // always buffered, even when directPoseTarget_ is set
    if (!vertices || !indices || vertexCount == 0 || indexCount == 0 || knownRigidModels_.find(modelKey) != knownRigidModels_.end()) return;
    knownRigidModels_.insert(modelKey);
    PendingRigidRegistration reg;
    reg.modelKey = modelKey;
    reg.vertices.assign(vertices, vertices + vertexCount);
    reg.indices.assign(indices, indices + indexCount);
    pendingRigidRegistrations_.push_back(std::move(reg));
}

void AnimatedInstanceCollector::submitRigidInstance(uint64_t modelKey, const float worldMatrix[12])
{
    if (directPoseTarget_) {
        directPoseTarget_->submitRigidInstance(modelKey, worldMatrix);
        return;
    }
    PendingRigidInstance inst;
    inst.modelKey = modelKey;
    std::memcpy(inst.worldMatrix, worldMatrix, sizeof(float) * 12);
    pendingRigidInstances_.push_back(std::move(inst));
}

void AnimatedInstanceCollector::flush(SkinExtractSink& target)
{
    for (auto& pending : pendingSkinSources_) {
        for (auto& v : pending.vertices) {
            if (v.textureIndex != 0) v.textureIndex = target.registerTexture(v.textureIndex);
        }
        target.submitSkinSource(pending.modelKey, pending.vertices.data(), static_cast<uint32_t>(pending.vertices.size()),
                                 pending.indices.data(), static_cast<uint32_t>(pending.indices.size()), pending.boneCount);
    }
    pendingSkinSources_.clear();

    for (auto& pending : pendingPoses_) {
        target.beginInstance(pending.d);
        target.submitPose(pending.modelKey, pending.boneData.data(),
                           static_cast<uint32_t>(pending.boneData.size() / 8), pending.worldMatrix);
    }
    pendingPoses_.clear();

    for (auto& pending : pendingDynamicMeshes_) {
        for (auto& v : pending.vertices) {
            if (v.textureIndex != 0) v.textureIndex = target.registerTexture(v.textureIndex);
        }
        target.submitDynamicMesh(pending.modelKey, pending.vertices.data(), static_cast<uint32_t>(pending.vertices.size()),
                                  pending.indices.data(), static_cast<uint32_t>(pending.indices.size()));
    }
    pendingDynamicMeshes_.clear();

    for (auto& pending : pendingRigidRegistrations_) {
        for (auto& v : pending.vertices) {
            if (v.textureIndex != 0) v.textureIndex = target.registerTexture(v.textureIndex);
        }
        target.registerRigidSource(pending.modelKey, pending.vertices.data(), static_cast<uint32_t>(pending.vertices.size()),
                                    pending.indices.data(), static_cast<uint32_t>(pending.indices.size()));
    }
    pendingRigidRegistrations_.clear();

    for (auto& pending : pendingRigidInstances_) {
        target.submitRigidInstance(pending.modelKey, pending.worldMatrix);
    }
    pendingRigidInstances_.clear();
}

}  // namespace sauerinterop
