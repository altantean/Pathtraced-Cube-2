#include "WaterExtract.h"  // pulls in <vector>, WorldGeometryExtract.h

#include <new>
#define SAUERINTEROP_STL_PLACEMENT_NEW

#include "engine.h"
#include <unordered_map>

extern vector<vtxarray *> valist;

namespace sauerinterop {

namespace {

    // one water material per distinct MAT_WATER index
    uint32_t ensureWaterMaterial(int material, std::vector<RtMaterialDesc>& materials,
                                 std::unordered_map<int, uint32_t>& seen)
    {
        auto it = seen.find(material);
        if(it != seen.end()) return it->second;
        RtMaterialDesc desc;
        desc.vslotIndex = static_cast<uint32_t>(material);
        desc.isWater = true;
        desc.diffuseTexGlId = 0;
        desc.hasNormal = desc.hasSpec = desc.hasGlow = false;
        const bvec &wcol = getwatercolor(material);
        desc.colorscale[0] = wcol.x / 255.0f;
        desc.colorscale[1] = wcol.y / 255.0f;
        desc.colorscale[2] = wcol.z / 255.0f;
        desc.glowcolor[0] = desc.glowcolor[1] = desc.glowcolor[2] = 1.0f;
        desc.metalness = 0.0f;
        desc.glassAlpha = 0.0f;
        const int wfog = getwaterfog(material);
        desc.waterReferenceThickness = wfog > 0 ? float(wfog) : 150.0f;
        const uint32_t index = static_cast<uint32_t>(materials.size());
        materials.push_back(desc);
        seen.emplace(material, index);
        return index;
    }

    // same quad layout/winding as GlassExtract's pushGlassQuad (outward normals), with water's offsets
    void pushWaterQuad(ChunkMeshData& chunk, const materialsurface& m, uint32_t materialIndex)
    {
        const int orient = m.orient;
        const int dim = dimension(orient);
        const int c = C[dim], r = R[dim];
        const bool positive = dimcoord(orient) != 0;
        const int csize = m.csize, rsize = m.rsize;
        vec p[4];
        loopk(4) p[k][dim] = float(m.o[dim]);
        p[0][c] = float(m.o[c]);           p[0][r] = float(m.o[r]);
        p[1][c] = float(m.o[c] + csize);   p[1][r] = float(m.o[r]);
        p[2][c] = float(m.o[c] + csize);   p[2][r] = float(m.o[r] + rsize);
        p[3][c] = float(m.o[c]);           p[3][r] = float(m.o[r] + rsize);
        if(dim == 2)
        {
            if(orient == O_TOP) loopk(4) p[k].z -= WATER_OFFSET;
        }
        else
        {
            float zlo = 1e16f, zhi = -1e16f;
            loopk(4) { zlo = min(zlo, p[k].z); zhi = max(zhi, p[k].z); }
            loopk(4)
            {
                if((m.ends & 2) && p[k].z == zhi) p[k].z -= WATER_OFFSET;
                else if((m.ends & 1) && p[k].z == zlo) p[k].z -= WATER_OFFSET;
            }
        }
        int idx[6];
        if(positive) { idx[0]=0; idx[1]=2; idx[2]=1; idx[3]=0; idx[4]=3; idx[5]=2; }
        else         { idx[0]=0; idx[1]=1; idx[2]=2; idx[3]=0; idx[4]=2; idx[5]=3; }
        vec n(0, 0, 0);
        n[dim] = positive ? 1.0f : -1.0f;  // outward from the water, into the air
        vec tangent(0, 0, 0);
        tangent[c] = 1.0f;
        loopi(2)
        {
            const uint32_t base = static_cast<uint32_t>(chunk.rtPositions.size() / 3);
            loopk(3)
            {
                const vec &v = p[idx[i*3+k]];
                chunk.rtPositions.push_back(v.x);
                chunk.rtPositions.push_back(v.y);
                chunk.rtPositions.push_back(v.z);
                chunk.rtUvs.push_back(0.0f);
                chunk.rtUvs.push_back(0.0f);
                chunk.rtTextureGlIds.push_back(0u);
                chunk.rtNormals.push_back(n.x);
                chunk.rtNormals.push_back(n.y);
                chunk.rtNormals.push_back(n.z);
                chunk.rtTangents.push_back(tangent.x);
                chunk.rtTangents.push_back(tangent.y);
                chunk.rtTangents.push_back(tangent.z);
                chunk.rtTangents.push_back(1.0f);
                chunk.rtMaterialIndices.push_back(materialIndex);
                chunk.rtBlendMaterialIndices.push_back(0xFFFFFFFFu);
                chunk.rtBlendAlphas.push_back(0.0f);
                chunk.rtBlendUvs.push_back(0.0f);
                chunk.rtBlendUvs.push_back(0.0f);
                chunk.rtAlphaTestCutoffs.push_back(0.0f);
            }
            chunk.rtIndices.push_back(base + 0);
            chunk.rtIndices.push_back(base + 1);
            chunk.rtIndices.push_back(base + 2);
        }
    }

    bool keepSideFace(const materialsurface& m)
    {
        (void)m;
        return true;
    }

}  // anonymous namespace

namespace {

    struct LavaMaterial { uint32_t index; Texture *tex; float texScale; vec avgColor; };

    // one material per (lava index, top/side)
    bool ensureLavaMaterial(int material, bool side, float glow, bool scroll, bool procedural, std::vector<RtMaterialDesc>& materials,
                            std::unordered_map<int, LavaMaterial>& seen, LavaMaterial& out)
    {
        const int key = material * 2 + (side ? 1 : 0);
        auto it = seen.find(key);
        if(it != seen.end()) { out = it->second; return out.tex != NULL; }
        LavaMaterial lm = { 0, NULL, 1.0f, vec(0, 0, 0) };
        MSlot &mslot = lookupmaterialslot(material, false);
        const int sub = side ? 1 : 0;
        if(mslot.loaded && mslot.sts.inrange(sub) && mslot.sts[sub].t && mslot.sts[sub].t != notexture)
        {
            RtMaterialDesc desc;
            desc.vslotIndex = static_cast<uint32_t>(material);
            Texture *t = mslot.sts[sub].t;
            desc.diffuseTexGlId = static_cast<uint32_t>(t->id);
            desc.glowTexGlId = static_cast<uint32_t>(t->id);
            desc.hasGlow = true;
            desc.hasNormal = desc.hasSpec = false;
            desc.colorscale[0] = desc.colorscale[1] = desc.colorscale[2] = 0.2f;
            desc.glowcolor[0] = desc.glowcolor[1] = desc.glowcolor[2] = glow;
            desc.metalness = 0.0f;
            desc.isLava = procedural;  // the shader's procedural blackbody lava
            lm.index = static_cast<uint32_t>(materials.size());
            materials.push_back(desc);
            lm.tex = t;
            lm.texScale = mslot.scale > 0 ? mslot.scale : 1.0f;
            // lava flow (material.cpp / water.cpp setuplava)
            if(scroll)
            {
                const float xk = TEX_SCALE / (t->xs * lm.texScale), yk = TEX_SCALE / (t->ys * lm.texScale);
                if(side) { desc.scroll[0] = 0.0f; desc.scroll[1] = -yk * 16.0f / 3.0f; }
                else { desc.scroll[0] = xk; desc.scroll[1] = yk; }
                materials.back().scroll[0] = desc.scroll[0];
                materials.back().scroll[1] = desc.scroll[1];
            }
            float avg[3];
            textureAverageColor(t, avg);
            lm.avgColor = vec(avg[0], avg[1], avg[2]).mul(glow);
            if(!procedural) materials[lm.index].glowSelectLum = 0.2126f * lm.avgColor.x + 0.7152f * lm.avgColor.y + 0.0722f * lm.avgColor.z;
        }
        seen.emplace(key, lm);
        out = lm;
        return lm.tex != NULL;
    }

}  // anonymous namespace

void extractLavaVolumes(std::vector<RtMaterialDesc>& materials, ChunkMeshData& out,
                        std::vector<EmissiveTriangleDesc>& emissive, float glow, bool scroll, bool procedural,
                        std::vector<float>* topRects)
{
    std::unordered_map<int, LavaMaterial> seen;
    // topRects (displaced lava)
    uint32_t sideIndexCount = 0;
    loopk(topRects ? 2 : 1) loopv(valist)
    {
        vtxarray *va = valist[i];
        if(!va || !va->matbuf) continue;
        loopj(va->matsurfs)
        {
            const materialsurface &m = va->matbuf[j];
            if((m.material & MATF_VOLUME) != MAT_LAVA) continue;
            if(topRects && (m.orient == O_TOP) != (k == 1)) continue;
            if(!keepSideFace(m)) continue;
            const bool side = dimension(m.orient) != 2;
            LavaMaterial lm;
            if(!ensureLavaMaterial(m.material, side, glow, scroll, procedural, materials, seen, lm)) continue;
            const uint32_t firstVert = static_cast<uint32_t>(out.rtPositions.size() / 3);
            pushWaterQuad(out, m, lm.index);
            if(topRects && m.orient == O_TOP)
            {
                const float r[8] = { float(m.o.x), float(m.o.y), float(m.o.x + m.rsize), float(m.o.y + m.csize),
                                     float(m.o.z) - WATER_OFFSET, float(lm.index), 1.0f, glow };  // kind 1 = lava
                topRects->insert(topRects->end(), r, r + 8);
            }
            else if(topRects) sideIndexCount = static_cast<uint32_t>(out.rtIndices.size());
            // raster lava texgen (glsl.cfg "lava")
            const float xk = TEX_SCALE / (lm.tex->xs * lm.texScale), yk = TEX_SCALE / (lm.tex->ys * lm.texScale);
            const int dim = dimension(m.orient);
            const uint32_t lastVert = static_cast<uint32_t>(out.rtPositions.size() / 3);
            for(uint32_t v = firstVert; v < lastVert; ++v)
            {
                const float px = out.rtPositions[v*3], py = out.rtPositions[v*3+1], pz = out.rtPositions[v*3+2];
                float u, w;
                if(dim == 2) { u = px; w = py; }
                else if(dim == 0) { u = py; w = pz; }
                else { u = px; w = pz; }
                out.rtUvs[v*2] = u * xk;
                out.rtUvs[v*2+1] = w * (dim == 2 ? yk : -yk);
                out.rtTextureGlIds[v] = static_cast<uint32_t>(lm.tex->id);
                loopk(3) out.glVertexData.push_back(out.rtPositions[v*3+k]);
                loopk(3) out.glVertexData.push_back(out.rtNormals[v*3+k]);
                loopk(4) out.glVertexData.push_back(out.rtTangents[v*4+k]);
                out.glVertexData.push_back(out.rtUvs[v*2]);
                out.glVertexData.push_back(out.rtUvs[v*2+1]);
                out.glIndices.push_back(v);
            }
            if(out.glBatches.empty() || out.glBatches.back().materialIndex != lm.index)
            {
                ChunkGlBatch batch;
                batch.textureGlId = static_cast<uint32_t>(lm.tex->id);
                batch.indexOffset = static_cast<uint32_t>(out.glIndices.size()) - (lastVert - firstVert);
                batch.indexCount = 0;
                batch.materialIndex = lm.index;
                out.glBatches.push_back(batch);
            }
            out.glBatches.back().indexCount += lastVert - firstVert;
            for(uint32_t v = firstVert; v + 2 < lastVert; v += 3)
            {
                EmissiveTriangleDesc tri;
                loopk(3)
                {
                    tri.v0[k] = out.rtPositions[v*3 + k];
                    tri.v1[k] = out.rtPositions[(v+1)*3 + k];
                    tri.v2[k] = out.rtPositions[(v+2)*3 + k];
                }
                if(procedural)
                {
                    // nominal value (light-selection weight only), the shader evaluates the emission
                    // where it samples
                    tri.radiance[0] = glow * kLavaNominal[0]; tri.radiance[1] = glow * kLavaNominal[1]; tri.radiance[2] = glow * kLavaNominal[2];
                    tri.procedural = 1.0f;
                }
                else { tri.radiance[0] = lm.avgColor.x; tri.radiance[1] = lm.avgColor.y; tri.radiance[2] = lm.avgColor.z; }
                emissive.push_back(tri);
            }
        }
    }
    if(!out.rtIndices.empty())
    {
        out.chunkId = kLavaChunkId;  // opaque
        if(topRects) out.opaqueIndexCount = sideIndexCount;
        out.avgAlbedo[0] = out.avgAlbedo[1] = out.avgAlbedo[2] = 0.2f;
    }
}

void extractWaterVolumes(std::vector<RtMaterialDesc>& materials, ChunkMeshData& out, std::vector<float>* topRects)
{
    std::unordered_map<int, uint32_t> seen;
    // two passes
    loopk(2) loopv(valist)
    {
        vtxarray *va = valist[i];
        if(!va || !va->matbuf) continue;
        loopj(va->matsurfs)
        {
            const materialsurface &m = va->matbuf[j];
            if((m.material & MATF_VOLUME) != MAT_WATER) continue;
            if((m.orient == O_TOP) != (k == 0)) continue;
            if(!keepSideFace(m)) continue;
            const uint32_t matIndex = ensureWaterMaterial(m.material, materials, seen);
            pushWaterQuad(out, m, matIndex);
            // top face on x (rsize) by y (csize), pushWaterQuad's layout for dim 2
            if(topRects && m.orient == O_TOP)
            {
                const float r[8] = { float(m.o.x), float(m.o.y), float(m.o.x + m.rsize), float(m.o.y + m.csize),
                                     float(m.o.z) - WATER_OFFSET, float(matIndex), 0.0f, 0.0f };  // kind 0 = water
                topRects->insert(topRects->end(), r, r + 8);
            }
        }
    }
    if(!out.rtIndices.empty())
    {
        out.chunkId = kWaterChunkId;
        out.opaqueIndexCount = 0;  // all water
        if(!materials.empty() && !seen.empty())
        {
            const RtMaterialDesc &w = materials[seen.begin()->second];
            loopk(3) out.avgAlbedo[k] = w.colorscale[k];
        }
    }
}

}  // namespace sauerinterop
