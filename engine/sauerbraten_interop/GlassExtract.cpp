#include "GlassExtract.h"  // pulls in <vector>, WorldGeometryExtract.h

#include <new>
#define SAUERINTEROP_STL_PLACEMENT_NEW

#include "engine.h"
#include <unordered_map>

extern vector<vtxarray *> valist;

namespace sauerinterop {

namespace {

    // one-time-per-distinct-glass-variant registration
    uint32_t ensureGlassVolumeMaterial(int material, std::vector<sauerinterop::RtMaterialDesc>& materials,
                                        std::unordered_map<int, uint32_t>& seen)
    {
        auto it = seen.find(material);
        if(it != seen.end()) return it->second;

        sauerinterop::RtMaterialDesc desc;
        desc.vslotIndex = static_cast<uint32_t>(material);
        desc.isGlass = true;
        const bvec &gcol = getglasscolor(material);
        desc.colorscale[0] = gcol.x / 255.0f;
        desc.colorscale[1] = gcol.y / 255.0f;
        desc.colorscale[2] = gcol.z / 255.0f;
        desc.glowcolor[0] = desc.glowcolor[1] = desc.glowcolor[2] = 1.0f;
        desc.metalness = 0.0f;
        // no Sauerbraten per-material glass opacity value exists the way water has getwaterfog()
        desc.glassAlpha = 0.92f;

        const uint32_t index = static_cast<uint32_t>(materials.size());
        materials.push_back(desc);
        seen.emplace(material, index);
        return index;
    }

    // pushes one materialsurface quad (2 triangles) into `chunk`'s rt arrays
    void pushGlassQuad(ChunkMeshData& chunk, const ivec& o, int csize, int rsize, int orient, uint32_t materialIndex)
    {
        int dim = dimension(orient);
        int c = C[dim], r = R[dim];
        bool positive = dimcoord(orient) != 0;

        vec p[4];
        loopk(4) p[k][dim] = float(o[dim]);
        p[0][c] = float(o[c]);           p[0][r] = float(o[r]);
        p[1][c] = float(o[c] + csize);   p[1][r] = float(o[r]);
        p[2][c] = float(o[c] + csize);   p[2][r] = float(o[r] + rsize);
        p[3][c] = float(o[c]);           p[3][r] = float(o[r] + rsize);

        // outward-facing (away from the glass volume, into the air) triangle winding
        int idx[6];
        if(positive) { idx[0]=0; idx[1]=2; idx[2]=1; idx[3]=0; idx[4]=3; idx[5]=2; }
        else         { idx[0]=0; idx[1]=1; idx[2]=2; idx[3]=0; idx[4]=2; idx[5]=3; }

        loopi(2)
        {
            vec v0 = p[idx[i*3+0]], v1 = p[idx[i*3+1]], v2 = p[idx[i*3+2]];
            vec e1(v1.x-v0.x, v1.y-v0.y, v1.z-v0.z);
            vec e2(v2.x-v0.x, v2.y-v0.y, v2.z-v0.z);
            vec n; n.cross(e1, e2);
            if(n.magnitude() > 1e-8f) n.normalize(); else n = vec(0, 0, 1);
            vec tangent(0, 0, 0);
            tangent[c] = 1.0f;

            const uint32_t base = static_cast<uint32_t>(chunk.rtPositions.size() / 3);
            const vec *tri[3] = { &v0, &v1, &v2 };
            loopk(3)
            {
                const vec &v = *tri[k];
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

}  // anonymous namespace

void extractGlassVolumes(std::vector<RtMaterialDesc>& materials, ChunkMeshData& out)
{
    std::unordered_map<int, uint32_t> seen;
    loopv(valist)
    {
        vtxarray *va = valist[i];
        if(!va || !va->matbuf) continue;
        loopj(va->matsurfs)
        {
            materialsurface &m = va->matbuf[j];
            if((m.material & MATF_VOLUME) != MAT_GLASS) continue;
            uint32_t materialIndex = ensureGlassVolumeMaterial(m.material, materials, seen);
            pushGlassQuad(out, m.o, static_cast<int>(m.csize), static_cast<int>(m.rsize), m.orient, materialIndex);
        }
    }
    if(!out.rtIndices.empty()) out.opaqueIndexCount = 0;
}

}  // namespace sauerinterop
