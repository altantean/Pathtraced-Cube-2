#include "MapModelExtract.h"  // pulls in WorldGeometryExtract.h -> <vector>

#include <new>
#include <string>
#define SAUERINTEROP_STL_PLACEMENT_NEW
#include "engine.h"
#include <cmath>

namespace sauerinterop {

    uint64_t makeMapModelChunkId(int entityIndex)
    {
        return (uint64_t(1) << 63) | uint64_t(uint32_t(entityIndex));
    }

    namespace {

        // same flat normal / UV-gradient tangent construction as WorldGeometryExtract.cpp
        void computeFlatNormalAndTangent(const vec& p0, const vec& p1, const vec& p2,
            const vec2& uv0, const vec2& uv1, const vec2& uv2,
            vec& outNormal, vec& outTangent, float& outHandedness)
        {
            outNormal = vec(0, 0, 1);
            outTangent = vec(1, 0, 0);
            outHandedness = 1.0f;

            const vec e1(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
            const vec e2(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
            vec n; n.cross(e1, e2);
            if (n.magnitude() > 1e-8f) { n.normalize(); outNormal = n; }

            const float du1 = uv1.x - uv0.x, dv1 = uv1.y - uv0.y;
            const float du2 = uv2.x - uv0.x, dv2 = uv2.y - uv0.y;
            const float det = du1 * dv2 - du2 * dv1;
            vec tangent(0, 0, 0), bitangent(0, 0, 0);
            if (fabs(det) > 1e-8f)
            {
                const float invDet = 1.0f / det;
                tangent = vec(e1).mul(dv2).sub(vec(e2).mul(dv1)).mul(invDet);
                bitangent = vec(e2).mul(du1).sub(vec(e1).mul(du2)).mul(invDet);
            }
            tangent.sub(vec(outNormal).mul(outNormal.dot(tangent)));
            if (tangent.magnitude() > 1e-6f)
            {
                tangent.normalize();
                outTangent = tangent;
                vec cross; cross.cross(outNormal, outTangent);
                outHandedness = cross.dot(bitangent) < 0 ? -1.0f : 1.0f;
            }
            else
            {
                vec up = fabs(outNormal.z) < 0.999f ? vec(0, 0, 1) : vec(1, 0, 0);
                vec fallback; fallback.cross(up, outNormal);
                if (fallback.magnitude() > 1e-6f) { fallback.normalize(); outTangent = fallback; }
            }
        }

        // shared per-mesh triangle-push logic
        void processMapModelMesh(ChunkMeshData& chunk, const BIH::mesh& bm,
            const vec2& yawRot, bool rotate, const vec& entityPos, float cutoff)
        {
            Texture* tex = bm.tex ? bm.tex : notexture;
            const uint32_t texGlId = tex ? static_cast<uint32_t>(tex->id) : 0u;

            const uint32_t batchIndexOffset = static_cast<uint32_t>(chunk.glIndices.size());

            for (int ti = 0; ti < bm.numtris; ++ti)
            {
                const BIH::tri& t = bm.tris[ti];

                vec worldPos[3];
                vec2 uv[3];
                loopk(3)
                {
                    const int idx = t.vert[k];
                    vec local = bm.xform.transform(bm.getpos(idx));
                    if (rotate) local.rotate_around_z(yawRot);
                    worldPos[k] = vec(local).add(entityPos);
                    uv[k] = bm.gettc(idx);
                }

                vec triNormal, triTangent;
                float triHandedness;
                computeFlatNormalAndTangent(worldPos[0], worldPos[1], worldPos[2],
                    uv[0], uv[1], uv[2], triNormal, triTangent, triHandedness);

                const uint32_t base = static_cast<uint32_t>(chunk.rtPositions.size() / 3);
                loopk(3)
                {
                    chunk.rtPositions.push_back(worldPos[k].x);
                    chunk.rtPositions.push_back(worldPos[k].y);
                    chunk.rtPositions.push_back(worldPos[k].z);
                    chunk.rtUvs.push_back(uv[k].x);
                    chunk.rtUvs.push_back(uv[k].y);
                    chunk.rtTextureGlIds.push_back(texGlId);
                    chunk.rtNormals.push_back(triNormal.x);
                    chunk.rtNormals.push_back(triNormal.y);
                    chunk.rtNormals.push_back(triNormal.z);
                    chunk.rtTangents.push_back(triTangent.x);
                    chunk.rtTangents.push_back(triTangent.y);
                    chunk.rtTangents.push_back(triTangent.z);
                    chunk.rtTangents.push_back(triHandedness);
                    chunk.rtMaterialIndices.push_back(0xFFFFFFFFu);
                    chunk.rtBlendMaterialIndices.push_back(0xFFFFFFFFu);
                    chunk.rtBlendAlphas.push_back(0.0f);
                    chunk.rtBlendUvs.push_back(0.0f);
                    chunk.rtBlendUvs.push_back(0.0f);
                    chunk.rtAlphaTestCutoffs.push_back(cutoff);

                    chunk.glVertexData.push_back(worldPos[k].x);
                    chunk.glVertexData.push_back(worldPos[k].y);
                    chunk.glVertexData.push_back(worldPos[k].z);
                    chunk.glVertexData.push_back(triNormal.x);
                    chunk.glVertexData.push_back(triNormal.y);
                    chunk.glVertexData.push_back(triNormal.z);
                    chunk.glVertexData.push_back(triTangent.x);
                    chunk.glVertexData.push_back(triTangent.y);
                    chunk.glVertexData.push_back(triTangent.z);
                    chunk.glVertexData.push_back(triHandedness);
                    chunk.glVertexData.push_back(uv[k].x);
                    chunk.glVertexData.push_back(uv[k].y);
                }
                chunk.rtIndices.push_back(base + 0);
                chunk.rtIndices.push_back(base + 1);
                chunk.rtIndices.push_back(base + 2);
                chunk.glIndices.push_back(base + 0);
                chunk.glIndices.push_back(base + 1);
                chunk.glIndices.push_back(base + 2);
            }

            ChunkGlBatch batch;
            batch.textureGlId = texGlId;
            batch.textureClamp = tex ? static_cast<uint32_t>(tex->clamp) : 0u;
            batch.indexOffset = batchIndexOffset;
            batch.indexCount = static_cast<uint32_t>(chunk.glIndices.size()) - batchIndexOffset;
            batch.materialIndex = 0xFFFFFFFFu;
            batch.alphaTestCutoff = cutoff;
            if (batch.indexCount > 0) chunk.glBatches.push_back(batch);
        }

    }  // namespace

    void extractStaticMapModelChunks(std::vector<ChunkMeshData>& out)
    {
        out.clear();
        vector<extentity*>& ents = entities::getents();
        std::vector<std::string> skipped;
        auto skip = [&](int i, const extentity& e, const model* m, const char* why) {
            skipped.push_back("#" + std::to_string(i) + " " + (m && m->name ? m->name : "?") + " (" + why + ")");
        };
        struct SkipLog { std::vector<std::string>& s; ~SkipLog() {
            if (s.empty()) return;
            std::string line = "mapmodels left out of the path tracer: " + std::to_string(s.size());
            for (size_t k = 0; k < s.size() && k < 24; ++k) line += (k ? ", " : ": ") + s[k];
            if (s.size() > 24) line += ", ...";
            conoutf("%s", line.c_str());
        } } skipLog{ skipped };
        loopv(ents)
        {
            extentity& e = *ents[i];
            if (e.type == ET_MAPMODEL) e.flags &= ~EF_RTBAKED;
            if (e.type != ET_MAPMODEL || e.attr2 < 0) continue;
            if (e.flags & EF_NOVIS) continue;

            model* m = loadmapmodel(e.attr2);
            if (!m) { skip(i, e, nullptr, "no model"); continue; }

            if (m->mapmodelAnimates()) { skip(i, e, m, "animated"); continue; }
            if (m->spinyaw != 0 || m->spinpitch != 0) { skip(i, e, m, "spins"); continue; }
            if (e.flags & EF_ANIM) { skip(i, e, m, "EF_ANIM"); continue; }

            BIH* bih = m->bih ? m->bih : m->setBIH();
            if (!bih || !bih->meshes || bih->nummeshes <= 0) { skip(i, e, m, "no BIH"); continue; }

            const vec2& yawRot = sincosmod360(int(e.attr1));

            ChunkMeshData chunk;
            chunk.chunkId = makeMapModelChunkId(i);

            // two-pass emission
            std::vector<int> opaqueMeshIndices, alphaTestMeshIndices;
            std::vector<float> meshCutoffs(static_cast<size_t>(bih->nummeshes), 0.0f);
            for (int mi = 0; mi < bih->nummeshes; ++mi)
            {
                const BIH::mesh& bm = bih->meshes[mi];
                if (bm.numtris <= 0 || !bm.tris) continue;
                const float cutoff = findMapModelAlphaTest(m, bm.tex);
                meshCutoffs[static_cast<size_t>(mi)] = cutoff;
                if (cutoff > 0.0f) alphaTestMeshIndices.push_back(mi);
                else opaqueMeshIndices.push_back(mi);
            }
            for (int mi : opaqueMeshIndices)
                processMapModelMesh(chunk, bih->meshes[mi], yawRot, e.attr1 != 0, e.o, 0.0f);
            if (!alphaTestMeshIndices.empty())
            {
                chunk.opaqueIndexCount = static_cast<uint32_t>(chunk.rtIndices.size());
                for (int mi : alphaTestMeshIndices)
                    processMapModelMesh(chunk, bih->meshes[mi], yawRot, e.attr1 != 0, e.o, meshCutoffs[static_cast<size_t>(mi)]);
            }

            if (chunk.rtIndices.empty()) { skip(i, e, m, "no triangles"); continue; }

            e.flags |= EF_RTBAKED;

            out.push_back(std::move(chunk));
        }
    }

}  // namespace sauerinterop
