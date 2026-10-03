#include "WorldGeometryExtract.h"  // pulls in <vector>

#include <new>
#define SAUERINTEROP_STL_PLACEMENT_NEW

#include "engine.h"
#include <cmath>  // std::sqrt, for the computed face-normal length below
#include <cstring>
#include <unordered_map>  // texture-GLuint -> average-color cache

extern vector<vtxarray *> valist;

namespace sauerinterop {

void onMaterialsEdited();

// automatic roughness (path tracer + DLSS-RR guide)
VARF(ptautorough, 0, 0, 1, onMaterialsEdited());
FVARF(ptautoroughmin, 0.05f, 0.5f, 0.9f, onMaterialsEdited());  // smoothest an unrecognised texture gets

namespace {

    float getTextureDetail(Texture *tex)
    {
        static std::unordered_map<uint32_t, float> cache;
        if(!tex || !tex->id || tex == notexture) return -1.0f;
        auto it = cache.find(static_cast<uint32_t>(tex->id));
        if(it != cache.end()) return it->second;

        GLint prevTex = 0;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex);
        glBindTexture(GL_TEXTURE_2D, tex->id);
        int level = 0;
        if(tex->mipmap)
        {
            int guess = 0, dim = tex->w < tex->h ? tex->w : tex->h;
            while(dim > 128 && guess < 12) { dim >>= 1; ++guess; }
            GLint glw = 0;
            glGetTexLevelParameteriv(GL_TEXTURE_2D, guess, GL_TEXTURE_WIDTH, &glw);
            if(glw > 0) level = guess;
        }
        GLint lw = 0, lh = 0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, level, GL_TEXTURE_WIDTH, &lw);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, level, GL_TEXTURE_HEIGHT, &lh);
        float detail = -1.0f;
        if(lw >= 4 && lh >= 4 && lw <= 2048 && lh <= 2048)
        {
            std::vector<unsigned char> px(static_cast<size_t>(lw) * lh * 3);
            glGetTexImage(GL_TEXTURE_2D, level, GL_RGB, GL_UNSIGNED_BYTE, px.data());
            std::vector<float> l(static_cast<size_t>(lw) * lh);
            double sum = 0.0;
            for(size_t i = 0; i < l.size(); ++i)
            {
                l[i] = (0.2126f*px[i*3] + 0.7152f*px[i*3+1] + 0.0722f*px[i*3+2]) / 255.0f;
                sum += l[i];
            }
            double hp = 0.0;
            for(int y = 0; y < lh; ++y) for(int x = 0; x < lw; ++x)
            {
                float box = 0.0f;  // wrapping, world textures tile
                for(int dy = -1; dy <= 1; ++dy) for(int dx = -1; dx <= 1; ++dx)
                    box += l[static_cast<size_t>((y + dy + lh) % lh) * lw + (x + dx + lw) % lw];
                hp += std::fabs(l[static_cast<size_t>(y) * lw + x] - box / 9.0f);
            }
            const double n = double(l.size()), mean = sum / n;
            detail = float((hp / n) / (mean > 0.1 ? mean : 0.1));
        }
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex));
        cache[static_cast<uint32_t>(tex->id)] = detail;
        return detail;
    }

    float autoRoughness(Slot &slot, Texture *diffuse)
    {
        const char *sh = slot.shader && slot.shader->name ? slot.shader->name : "";
        if(strstr(sh, "env")) return 0.25f;
        const float detail = getTextureDetail(diffuse);
        if(detail < 0.0f) return -1.0f;
        string base;
        const char *slash = max(strrchr(diffuse->name, '/'), strrchr(diffuse->name, '\\'));
        copystring(base, slash ? slash + 1 : diffuse->name);
        for(char *c = base; *c; ++c) *c = tolower(*c);
        struct Range { const char *const *words; float lo, hi; };
        static const char *const rough[] = { "conc", "paper","rock", "stone", "sand", "dirt", "grass", "brick", "gravel", "mud", "ter_", "soil", "moss", "bark", "plaster", "carpet", "cloth", "cliff", "ground", "snow", "rubble", "asphalt", NULL };
        static const char *const wood[] = { "wood", "plank", "crate", "rubber", NULL };
        static const char *const metal[] = { "metal", "met_", "steel", "iron", "chrome", "plate", "panel", "pipe", "grate", "grill", "tech", NULL };
        static const char *const gloss[] = { "tile", "marble", "plastic", "ceramic", "glass", "polish", "paint", NULL };
        static const Range ranges[] = { { rough, 0.85f, 0.9f }, { wood, 0.6f, 0.85f }, { metal, 0.35f, 0.75f }, { gloss, 0.25f, 0.5f } };
        float lo = ptautoroughmin, hi = 0.9f;  // unknown name
        bool found = false;
        for(const Range &rg : ranges)
        {
            for(const char *const *w = rg.words; *w && !found; ++w) if(strstr(base, *w)) { lo = rg.lo; hi = rg.hi; found = true; }
            if(found) break;
        }
        // relative detail 0.02 (smooth) .. 0.12 (grainy) -> lo .. hi
        const float t = clamp((detail - 0.02f) / 0.10f, 0.0f, 1.0f);
        const float r = lo + (hi - lo) * t;
        // rough results leave the slot alone
        if(r >= 0.7f)
        {
            logoutf("ptautorough: %s (%s) detail %.3f -> %.2f, left matte", diffuse->name, sh, detail, r);
            return -1.0f;
        }
        logoutf("ptautorough: %s (%s) detail %.3f -> roughness %.2f", diffuse->name, sh, detail, r);
        return r;
    }

    std::unordered_map<uint32_t, vec>& averageColorCache()
    {
        static std::unordered_map<uint32_t, vec> cache;
        return cache;
    }

    vec getTextureAverageColor(Texture *tex)
    {
        if(!tex || !tex->id) return vec(0.6f, 0.6f, 0.6f);  // neutral fallback

        auto &cache = averageColorCache();
        auto it = cache.find(static_cast<uint32_t>(tex->id));
        if(it != cache.end()) return it->second;

        GLint prevTex = 0;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex);
        glBindTexture(GL_TEXTURE_2D, tex->id);

        // pick a coarse mip level to keep the readback small
        int level = 0;
        if(tex->mipmap)
        {
            int guess = 0;
            int dim = tex->w < tex->h ? tex->w : tex->h;
            while(dim > 8 && guess < 12) { dim >>= 1; ++guess; }
            GLint glw = 0, glh = 0;
            glGetTexLevelParameteriv(GL_TEXTURE_2D, guess, GL_TEXTURE_WIDTH, &glw);
            glGetTexLevelParameteriv(GL_TEXTURE_2D, guess, GL_TEXTURE_HEIGHT, &glh);
            if(glw > 0 && glh > 0) level = guess;
        }

        GLint lw = 0, lh = 0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, level, GL_TEXTURE_WIDTH, &lw);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, level, GL_TEXTURE_HEIGHT, &lh);

        vec avg(0.6f, 0.6f, 0.6f);
        // defensive upper bound a resident level should never be this large by the time `level` was
        // picked above
        if(lw > 0 && lh > 0 && lw <= 2048 && lh <= 2048)
        {
            std::vector<unsigned char> pixels(static_cast<size_t>(lw) * static_cast<size_t>(lh) * 3);
            glGetTexImage(GL_TEXTURE_2D, level, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
            double r = 0.0, g = 0.0, b = 0.0;
            const size_t count = static_cast<size_t>(lw) * static_cast<size_t>(lh);
            for(size_t p = 0; p < count; ++p)
            {
                r += std::pow(double(pixels[p*3 + 0]) / 255.0, 2.2);
                g += std::pow(double(pixels[p*3 + 1]) / 255.0, 2.2);
                b += std::pow(double(pixels[p*3 + 2]) / 255.0, 2.2);
            }
            if(count)
            {
                avg = vec(float(r / double(count)), float(g / double(count)), float(b / double(count)));
            }
        }

        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex));
        cache[static_cast<uint32_t>(tex->id)] = avg;
        return avg;
    }

    Slot::Tex *findSlotTex(Slot &slot, int type)
    {
        loopv(slot.sts) if(slot.sts[i].type == type) return &slot.sts[i];
        return nullptr;
    }

    // builds (or returns the already-built) RtMaterialDesc for a VSlot index
    struct TriPosKey {
        uint32_t bits[9];
        bool operator==(const TriPosKey &o) const { return std::memcmp(bits, o.bits, sizeof(bits)) == 0; }
    };
    struct TriPosKeyHash {
        size_t operator()(const TriPosKey &k) const {
            uint64_t h = 1469598103934665603ull;  // FNV-1a
            for(uint32_t b : k.bits) { h ^= b; h *= 1099511628211ull; }
            return static_cast<size_t>(h);
        }
    };
    TriPosKey makeTriPosKey(const vec &a, const vec &b, const vec &c)
    {
        TriPosKey k;
        float f[9] = { a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z };
        std::memcpy(k.bits, f, sizeof(k.bits));
        return k;
    }

    // one "top" (blend-layer) triangle's data, gathered by the pre-pass in extractSolidCubeChunks()
    // below
    struct BlendTopTriInfo {
        int topVslotIndex = 0;
        float alpha[3] = { 0, 0, 0 };
        float u[3] = { 0, 0, 0 };
        float v[3] = { 0, 0, 0 };
        // per-pixel blend source, see ChunkMeshData::rtBlendLmUvs
        float lmU[3] = { 0, 0, 0 };
        float lmV[3] = { 0, 0, 0 };
        uint32_t lmTexGlId = 0;
    };

    uint32_t ensureMaterial(int vslotIndex, VSlot &vslot, bool isGlass,
                             std::unordered_map<int64_t, uint32_t> &seen,
                             std::vector<sauerinterop::RtMaterialDesc> &materials)
    {
        const int64_t key = (static_cast<int64_t>(vslotIndex) << 1) | (isGlass ? 1 : 0);
        auto it = seen.find(key);
        if(it != seen.end()) return it->second;

        sauerinterop::RtMaterialDesc desc;
        desc.vslotIndex = static_cast<uint32_t>(vslotIndex);
        desc.isGlass = isGlass;
        Slot *slot = vslot.slot;
        if(slot)
        {
            Texture *diffuse = !slot->sts.empty() && slot->sts[0].t ? slot->sts[0].t : notexture;
            desc.diffuseTexGlId = diffuse ? static_cast<uint32_t>(diffuse->id) : 0u;

            desc.hasNormal = (slot->texmask & (1<<TEX_NORMAL)) != 0;
            desc.hasSpec   = (slot->texmask & (1<<TEX_SPEC))   != 0;
            desc.hasGlow   = (slot->texmask & (1<<TEX_GLOW))   != 0;
            desc.uniformSpec = !desc.hasSpec && slot->shader && slot->shader->name && strstr(slot->shader->name, "spec");
            desc.parallax = desc.hasNormal && slot->shader && slot->shader->name && strstr(slot->shader->name, "parallax") &&
                            vslot.parallaxscale.x != 0.0f;
            desc.parallaxScale = vslot.parallaxscale.x;
            desc.parallaxBias = vslot.parallaxscale.y;

            if(desc.hasNormal)
            {
                Slot::Tex *nt = findSlotTex(*slot, TEX_NORMAL);
                if(nt && nt->t) desc.normalTexGlId = static_cast<uint32_t>(nt->t->id);
                else desc.hasNormal = false;
            }
            if(desc.hasGlow)
            {
                Slot::Tex *gt = findSlotTex(*slot, TEX_GLOW);
                if(gt && gt->t) desc.glowTexGlId = static_cast<uint32_t>(gt->t->id);
                else desc.hasGlow = false;
            }

            desc.colorscale[0] = vslot.colorscale.x;
            desc.colorscale[1] = vslot.colorscale.y;
            desc.colorscale[2] = vslot.colorscale.z;
            desc.glowcolor[0] = vslot.glowcolor.x;
            desc.glowcolor[1] = vslot.glowcolor.y;
            desc.glowcolor[2] = vslot.glowcolor.z;
            // VSlot::metalness (texture.h's new field, 0 default)
            desc.metalness = vslot.metalness;
            desc.roughness = vslot.roughness;  // roughness override
            // only where nothing authored says otherwise
            if(ptautorough && desc.roughness < 0.0f && !desc.hasSpec && !desc.uniformSpec && !isGlass)
                desc.roughness = autoRoughness(*slot, diffuse);
            desc.specScale = (vslot.specscale.x + vslot.specscale.y + vslot.specscale.z) / 3.0f;
            // the same per-slot scroll changetexgen() (renderva.cpp) adds to the world texcoords
            if(!vslot.scroll.iszero() && !slot->sts.empty() && slot->sts[0].t)
            {
                Texture *tex = slot->sts[0].t;
                const texrotation &r = texrotations[vslot.rotation];
                const float xs = r.flipx ? -tex->xs : tex->xs, ys = r.flipy ? -tex->ys : tex->ys;
                float sx = vslot.scroll.x, sy = vslot.scroll.y;
                if(r.swapxy) { const float t = sx; sx = sy; sy = t; }
                desc.scroll[0] = sx * 1000.0f * tex->xs / xs;
                desc.scroll[1] = sy * 1000.0f * tex->ys / ys;
            }
        }
        if(isGlass) desc.glassAlpha = max(vslot.alphafront, vslot.alphaback);

        const uint32_t index = static_cast<uint32_t>(materials.size());
        materials.push_back(desc);
        seen.emplace(key, index);
        return index;
    }

    // shared per-triangle geometric processing
    void pushTriangle(const vertex& v0, const vertex& v1, const vertex& v2,
                       uint32_t materialIndex, Texture* tex, bool addToGl,
                       sauerinterop::ChunkMeshData& chunk,
                       const BlendTopTriInfo* blend = nullptr, uint32_t blendMaterialIndex = 0xFFFFFFFFu)
    {
        vec flatNormal(0, 0, 1);
        vec flatTangent(1, 0, 0);
        float flatHandedness = 1.0f;
        {
            const vec e1(v1.pos.x - v0.pos.x, v1.pos.y - v0.pos.y, v1.pos.z - v0.pos.z);
            const vec e2(v2.pos.x - v0.pos.x, v2.pos.y - v0.pos.y, v2.pos.z - v0.pos.z);
            vec n; n.cross(e1, e2);
            if(n.magnitude() > 1e-8f) { n.normalize(); flatNormal = n; }

            const float du1 = v1.tc.x - v0.tc.x, dv1 = v1.tc.y - v0.tc.y;
            const float du2 = v2.tc.x - v0.tc.x, dv2 = v2.tc.y - v0.tc.y;
            const float det = du1 * dv2 - du2 * dv1;
            vec tangent(0, 0, 0), bitangent(0, 0, 0);
            if(fabs(det) > 1e-8f)
            {
                const float invDet = 1.0f / det;
                tangent = vec(e1).mul(dv2).sub(vec(e2).mul(dv1)).mul(invDet);
                bitangent = vec(e2).mul(du1).sub(vec(e1).mul(du2)).mul(invDet);
            }
            tangent.sub(vec(flatNormal).mul(flatNormal.dot(tangent)));
            if(tangent.magnitude() > 1e-6f)
            {
                tangent.normalize();
                flatTangent = tangent;
                vec cross; cross.cross(flatNormal, flatTangent);
                flatHandedness = cross.dot(bitangent) < 0 ? -1.0f : 1.0f;
            }
            else
            {
                vec up = fabs(flatNormal.z) < 0.999f ? vec(0, 0, 1) : vec(1, 0, 0);
                vec fallback; fallback.cross(up, flatNormal);
                if(fallback.magnitude() > 1e-6f) { fallback.normalize(); flatTangent = fallback; }
            }
        }

        const uint32_t base = static_cast<uint32_t>(chunk.rtPositions.size() / 3);
        // 12 floats/vertex
        const uint32_t glBase = static_cast<uint32_t>(chunk.glVertexData.size() / 12);
        const vertex* tri[3] = { &v0, &v1, &v2 };
        for(int k = 0; k < 3; ++k)
        {
            const vertex* v = tri[k];
            // same flat normal/tangent for all 3 corners of this triangle
            const vec smoothNormal = flatNormal;
            const vec tangentXyz = flatTangent;
            const float handedness = flatHandedness;

            chunk.rtPositions.push_back(v->pos.x);
            chunk.rtPositions.push_back(v->pos.y);
            chunk.rtPositions.push_back(v->pos.z);
            chunk.rtUvs.push_back(v->tc.x);
            chunk.rtUvs.push_back(v->tc.y);
            chunk.rtTextureGlIds.push_back(tex ? static_cast<uint32_t>(tex->id) : 0u);
            chunk.rtNormals.push_back(smoothNormal.x);
            chunk.rtNormals.push_back(smoothNormal.y);
            chunk.rtNormals.push_back(smoothNormal.z);
            chunk.rtTangents.push_back(tangentXyz.x);
            chunk.rtTangents.push_back(tangentXyz.y);
            chunk.rtTangents.push_back(tangentXyz.z);
            chunk.rtTangents.push_back(handedness);
            chunk.rtMaterialIndices.push_back(materialIndex);

            if(blend)
            {
                chunk.rtBlendMaterialIndices.push_back(blendMaterialIndex);
                chunk.rtBlendAlphas.push_back(blend->alpha[k]);
                chunk.rtBlendUvs.push_back(blend->u[k]);
                chunk.rtBlendUvs.push_back(blend->v[k]);
                chunk.rtBlendLmUvs.push_back(blend->lmU[k]);
                chunk.rtBlendLmUvs.push_back(blend->lmV[k]);
                chunk.rtBlendLmTexGlIds.push_back(blend->lmTexGlId);
            }
            else
            {
                chunk.rtBlendMaterialIndices.push_back(0xFFFFFFFFu);
                chunk.rtBlendAlphas.push_back(0.0f);
                chunk.rtBlendUvs.push_back(0.0f);
                chunk.rtBlendUvs.push_back(0.0f);
                chunk.rtBlendLmUvs.push_back(0.0f);
                chunk.rtBlendLmUvs.push_back(0.0f);
                chunk.rtBlendLmTexGlIds.push_back(0u);
            }

            chunk.rtAlphaTestCutoffs.push_back(0.0f);

            if(addToGl)
            {
                chunk.glVertexData.push_back(v->pos.x);
                chunk.glVertexData.push_back(v->pos.y);
                chunk.glVertexData.push_back(v->pos.z);
                chunk.glVertexData.push_back(smoothNormal.x);
                chunk.glVertexData.push_back(smoothNormal.y);
                chunk.glVertexData.push_back(smoothNormal.z);
                chunk.glVertexData.push_back(tangentXyz.x);
                chunk.glVertexData.push_back(tangentXyz.y);
                chunk.glVertexData.push_back(tangentXyz.z);
                chunk.glVertexData.push_back(handedness);
                chunk.glVertexData.push_back(v->tc.x);
                chunk.glVertexData.push_back(v->tc.y);
            }
        }
        chunk.rtIndices.push_back(base + 0);
        chunk.rtIndices.push_back(base + 1);
        chunk.rtIndices.push_back(base + 2);
        if(addToGl)
        {
            chunk.glIndices.push_back(glBase + 0);
            chunk.glIndices.push_back(glBase + 1);
            chunk.glIndices.push_back(glBase + 2);
        }
    }

}  // namespace

void textureAverageColor(Texture *tex, float out[3])
{
    const vec c = getTextureAverageColor(tex);
    out[0] = c.x; out[1] = c.y; out[2] = c.z;
}

uint64_t makeChunkId(int x, int y, int z, int size)
{
    // 18 bits each for x/y/z, 10 bits for size
    const uint64_t ux = static_cast<uint64_t>(x) & 0x3FFFF;
    const uint64_t uy = static_cast<uint64_t>(y) & 0x3FFFF;
    const uint64_t uz = static_cast<uint64_t>(z) & 0x3FFFF;
    // size is stored as log2
    uint64_t lg = 0;
    while(lg < 63 && (uint64_t(1) << lg) < static_cast<uint64_t>(size > 0 ? size : 1)) ++lg;
    return (ux << 46) | (uy << 28) | (uz << 10) | (lg & 0x3FF);
}

void extractSolidCubeChunks(std::vector<ChunkMeshData>& out,
                             std::vector<RtMaterialDesc>& outMaterials,
                             std::vector<EmissiveTriangleDesc>& outEmissive)
{
    out.clear();
    outMaterials.clear();
    outEmissive.clear();
    std::unordered_map<int64_t, uint32_t> materialsSeen;  // (vslotIndex, isGlass) composite key -> outMaterials[] index

    BlendMapCache *blendCache = newblendmapcache();

    for(int i = 0; i < valist.length(); ++i)
    {
        vtxarray* va = valist[i];
        if(!va || va->verts <= 0) continue;
        if(va->tris <= 0 && va->alphaback <= 0 && va->alphafront <= 0) continue;

        ushort* edata = nullptr;
        vertex* vdata = nullptr;
        const int alphaTriCountForRead = va->blendtris + va->alphabacktris + va->alphafronttris;
        if(!readva(va, edata, vdata, alphaTriCountForRead)) continue;
        // the tail (blend+alpha combined) starts immediately after the opaque portion
        ushort* tailEdataPtr = edata + 3 * static_cast<size_t>(va->tris);

        ChunkMeshData chunk;
        chunk.chunkId = makeChunkId(va->o.x, va->o.y, va->o.z, va->size);
        chunk.rtPositions.reserve(static_cast<size_t>(va->tris) * 9);
        chunk.rtUvs.reserve(static_cast<size_t>(va->tris) * 6);
        chunk.rtTextureGlIds.reserve(static_cast<size_t>(va->tris) * 3);
        chunk.rtNormals.reserve(static_cast<size_t>(va->tris) * 9);
        chunk.rtTangents.reserve(static_cast<size_t>(va->tris) * 12);
        chunk.rtMaterialIndices.reserve(static_cast<size_t>(va->tris) * 3);
        chunk.rtBlendMaterialIndices.reserve(static_cast<size_t>(va->tris) * 3);
        chunk.rtBlendAlphas.reserve(static_cast<size_t>(va->tris) * 3);
        chunk.rtBlendUvs.reserve(static_cast<size_t>(va->tris) * 6);
        chunk.rtAlphaTestCutoffs.reserve(static_cast<size_t>(va->tris) * 3);
        chunk.glVertexData.reserve(static_cast<size_t>(va->tris) * 36);  // 12 floats * 3 verts/tri
        chunk.rtIndices.reserve(static_cast<size_t>(va->tris) * 3);
        chunk.glIndices.reserve(static_cast<size_t>(va->tris) * 3);

        double albedoR = 0.0, albedoG = 0.0, albedoB = 0.0;
        uint32_t albedoWeight = 0;

        // per-vertex blend data for this vtxarray's own "top" (blend-layer) triangles
        setblendmaporigin(blendCache, va->o, va->size);
        std::unordered_map<TriPosKey, BlendTopTriInfo, TriPosKeyHash> blendTopByPos;
        struct BlendTopTri { vec p[3]; float u[3], v[3]; float lmu[3], lmv[3]; uint32_t lmTex; int vslot; vec n; float d; };
        std::vector<BlendTopTri> blendTopTris;
        {
            const uint32_t blendTailEntryStart = static_cast<uint32_t>(va->texs);
            const uint32_t blendTailEntryCount = static_cast<uint32_t>(va->blends) + static_cast<uint32_t>(va->alphaback) + static_cast<uint32_t>(va->alphafront);
            if(blendTailEntryCount > 0)
            {
                ushort *idxBlendPre = tailEdataPtr;
                const ushort *blendTailEdataEnd = edata + 3 * (static_cast<size_t>(va->tris) + static_cast<size_t>(alphaTriCountForRead));
                for(uint32_t j = blendTailEntryStart; j < blendTailEntryStart + blendTailEntryCount; ++j)
                {
                    elementset &e = va->eslist[j];
                    const int entryIndexCount = int(e.length[1]);
                    if(entryIndexCount <= 0) { idxBlendPre += e.length[1]; continue; }
                    if(e.layer != LAYER_BLEND) { idxBlendPre += e.length[1]; continue; }
                    if(idxBlendPre + entryIndexCount > blendTailEdataEnd) break;

                    const int triCount = entryIndexCount / 3;
                    for(int te = 0; te < triCount; ++te)
                    {
                        const int k = te * 3;
                        const uint32_t i0 = static_cast<uint32_t>(idxBlendPre[k + 0]) - static_cast<uint32_t>(va->voffset);
                        const uint32_t i1 = static_cast<uint32_t>(idxBlendPre[k + 1]) - static_cast<uint32_t>(va->voffset);
                        const uint32_t i2 = static_cast<uint32_t>(idxBlendPre[k + 2]) - static_cast<uint32_t>(va->voffset);
                        const vertex &tv0 = vdata[i0];
                        const vertex &tv1 = vdata[i1];
                        const vertex &tv2 = vdata[i2];

                        BlendTopTriInfo info;
                        info.topVslotIndex = e.texture;
                        info.alpha[0] = float(lookupblendmap(blendCache, tv0.pos)) / 255.0f;
                        info.alpha[1] = float(lookupblendmap(blendCache, tv1.pos)) / 255.0f;
                        info.alpha[2] = float(lookupblendmap(blendCache, tv2.pos)) / 255.0f;
                        info.u[0] = tv0.tc.x; info.v[0] = tv0.tc.y;
                        info.u[1] = tv1.tc.x; info.v[1] = tv1.tc.y;
                        info.u[2] = tv2.tc.x; info.v[2] = tv2.tc.y;
                        if(e.lmid >= LMID_RESERVED && e.lmid < lightmaptexs.length() &&
                           (lightmaptexs[e.lmid].type & LM_ALPHA) && lightmaptexs[e.lmid].id)
                        {
                            info.lmTexGlId = static_cast<uint32_t>(lightmaptexs[e.lmid].id);
                            const float lmScale = 1.0f / 32767.0f;
                            info.lmU[0] = tv0.lm.x * lmScale; info.lmV[0] = tv0.lm.y * lmScale;
                            info.lmU[1] = tv1.lm.x * lmScale; info.lmV[1] = tv1.lm.y * lmScale;
                            info.lmU[2] = tv2.lm.x * lmScale; info.lmV[2] = tv2.lm.y * lmScale;
                        }

                        blendTopByPos.emplace(makeTriPosKey(tv0.pos, tv1.pos, tv2.pos), info);
                        {
                            BlendTopTri t;
                            t.p[0] = tv0.pos; t.p[1] = tv1.pos; t.p[2] = tv2.pos;
                            t.u[0] = tv0.tc.x; t.u[1] = tv1.tc.x; t.u[2] = tv2.tc.x;
                            t.v[0] = tv0.tc.y; t.v[1] = tv1.tc.y; t.v[2] = tv2.tc.y;
                            t.vslot = e.texture;
                            t.lmTex = info.lmTexGlId;
                            loopk(3) { t.lmu[k] = info.lmU[k]; t.lmv[k] = info.lmV[k]; }
                            t.n.cross(vec(tv1.pos).sub(tv0.pos), vec(tv2.pos).sub(tv0.pos));
                            if(t.n.magnitude() > 1e-6f) { t.n.normalize(); t.d = t.n.dot(tv0.pos); blendTopTris.push_back(t); }
                        }
                    }
                    idxBlendPre += e.length[1];
                }
            }
        }

        ushort *idx = edata;
        for(int j = 0; j < va->texs; ++j)
        {
            elementset &e = va->eslist[j];
            const int entryIndexCount = int(e.length[1]);
            if(entryIndexCount <= 0)
            {
                idx += e.length[1];
                continue;
            }

            Texture *tex = notexture;
            VSlot &vslot = lookupvslot(e.texture, true);
            if(vslot.slot && !vslot.slot->sts.empty() && vslot.slot->sts[0].t)
                tex = vslot.slot->sts[0].t;
            const vec avgColor = getTextureAverageColor(tex);

            // material description for this vslot-run, built/deduped once per unique VSlot
            const uint32_t materialIndex = ensureMaterial(e.texture, vslot, false, materialsSeen, outMaterials);
            const RtMaterialDesc &materialDesc = outMaterials[materialIndex];
            vec glowTint(materialDesc.glowcolor[0], materialDesc.glowcolor[1], materialDesc.glowcolor[2]);
            vec avgGlowColor(0, 0, 0);
            bool triangleIsEmissive = false;
            if(materialDesc.hasGlow && vslot.slot)
            {
                Slot::Tex *glowTex = findSlotTex(*vslot.slot, TEX_GLOW);
                if(glowTex && glowTex->t)
                {
                    avgGlowColor = getTextureAverageColor(glowTex->t);
                    avgGlowColor.mul(glowTint);
                    triangleIsEmissive = avgGlowColor.x > 0 || avgGlowColor.y > 0 || avgGlowColor.z > 0;
                    outMaterials[materialIndex].glowSelectLum = 0.2126f * avgGlowColor.x + 0.7152f * avgGlowColor.y + 0.0722f * avgGlowColor.z;  // see RtMaterialDesc::glowSelectLum
                }
            }

            const uint32_t batchIndexOffset = static_cast<uint32_t>(chunk.glIndices.size());

            // writeobj() emits every index in es.length[1] from the same contiguous idx stream
            const int triangleCount = entryIndexCount / 3;
            for(int te = 0; te < triangleCount; ++te)
            {
                const int k = te * 3;
                const uint32_t i0 = static_cast<uint32_t>(idx[k + 0]) - static_cast<uint32_t>(va->voffset);
                const uint32_t i1 = static_cast<uint32_t>(idx[k + 1]) - static_cast<uint32_t>(va->voffset);
                const uint32_t i2 = static_cast<uint32_t>(idx[k + 2]) - static_cast<uint32_t>(va->voffset);
                const vertex &v0 = vdata[i0];
                const vertex &v1 = vdata[i1];
                const vertex &v2 = vdata[i2];

                std::vector<const BlendTopTri*> blendCands;
                vec bottomN; bottomN.cross(vec(v1.pos).sub(v0.pos), vec(v2.pos).sub(v0.pos));
                if(!blendTopTris.empty() && bottomN.magnitude() > 1e-6f)
                {
                    bottomN.normalize();
                    vec bmin = vec(v0.pos).min(v1.pos).min(v2.pos), bmax = vec(v0.pos).max(v1.pos).max(v2.pos);
                    for(const BlendTopTri &t : blendTopTris)
                    {
                        if(fabs(t.n.dot(bottomN)) < 0.999f || fabs(t.n.dot(v0.pos) - t.d) > 0.05f) continue;
                        if(lookupvslot(t.vslot, false).layer != int(e.texture)) continue;
                        vec tmin = vec(t.p[0]).min(t.p[1]).min(t.p[2]), tmax = vec(t.p[0]).max(t.p[1]).max(t.p[2]);
                        if(tmin.x > bmax.x + 0.01f || tmin.y > bmax.y + 0.01f || tmin.z > bmax.z + 0.01f ||
                           tmax.x < bmin.x - 0.01f || tmax.y < bmin.y - 0.01f || tmax.z < bmin.z - 0.01f) continue;
                        blendCands.push_back(&t);
                    }
                }
                if(blendCands.empty())
                {
                    pushTriangle(v0, v1, v2, materialIndex, tex, /*addToGl=*/true, chunk);
                }
                else
                {
                    typedef std::vector<vec> ClipPoly;
                    auto polyArea = [](const ClipPoly &q) -> float {
                        if(q.size() < 3) return 0.0f;
                        vec acc(0, 0, 0);
                        for(size_t i = 1; i + 1 < q.size(); ++i) { vec c; c.cross(vec(q[i]).sub(q[0]), vec(q[i + 1]).sub(q[0])); acc.add(c); }
                        return 0.5f * acc.magnitude();
                    };
                    // sutherland-Hodgman split of a convex polygon by the plane sgn*dot(en, P - a) >=
                    // 0 (inside) / < 0 (outside)
                    auto splitPoly = [](const ClipPoly &in, const vec &a, const vec &en, float sgn, ClipPoly &inside, ClipPoly &outside) {
                        inside.clear(); outside.clear();
                        for(size_t i = 0; i < in.size(); ++i)
                        {
                            const vec &P = in[i], &Q = in[(i + 1) % in.size()];
                            const float dp = sgn * en.dot(vec(P).sub(a)), dq = sgn * en.dot(vec(Q).sub(a));
                            if(dp >= 0) inside.push_back(P); else outside.push_back(P);
                            if((dp >= 0) != (dq >= 0))
                            {
                                const float t = dp / (dp - dq);
                                const vec X = vec(Q).sub(P).mul(t).add(P);
                                inside.push_back(X); outside.push_back(X);
                            }
                        }
                    };
                    auto bary = [](const vec &pt, const vec *tp) -> vec {
                        const vec e0 = vec(tp[1]).sub(tp[0]), e1 = vec(tp[2]).sub(tp[0]), e2 = vec(pt).sub(tp[0]);
                        const float d00 = e0.dot(e0), d01 = e0.dot(e1), d11 = e1.dot(e1), d20 = e2.dot(e0), d21 = e2.dot(e1);
                        const float den = d00 * d11 - d01 * d01;
                        if(fabs(den) < 1e-12f) return vec(1, 0, 0);
                        const float bv = (d11 * d20 - d01 * d21) / den, bw = (d00 * d21 - d01 * d20) / den;
                        return vec(1.0f - bv - bw, bv, bw);
                    };
                    const vec bottomP[3] = { v0.pos, v1.pos, v2.pos };
                    // A vertex at P on this triangle
                    auto makeVert = [&](const vec &P) -> vertex {
                        const vec b = bary(P, bottomP);
                        vertex r = v0;
                        r.pos = P;
                        r.tc.x = b.x * v0.tc.x + b.y * v1.tc.x + b.z * v2.tc.x;
                        r.tc.y = b.x * v0.tc.y + b.y * v1.tc.y + b.z * v2.tc.y;
                        return r;
                    };
                    const float invWorld = 1.0f / float(worldsize);
                    auto emitPiece = [&](const ClipPoly &q, const BlendTopTri *T) {
                        if(q.size() < 3 || polyArea(q) < 1e-5f) return;
                        uint32_t pieceBlendMat = 0xFFFFFFFFu;
                        if(T) pieceBlendMat = ensureMaterial(T->vslot, lookupvslot(T->vslot, true), false, materialsSeen, outMaterials);
                        for(size_t i = 1; i + 1 < q.size(); ++i)
                        {
                            const vec corner[3] = { q[0], q[i], q[i + 1] };
                            const vertex a = makeVert(corner[0]), b = makeVert(corner[1]), c = makeVert(corner[2]);
                            if(!T) { pushTriangle(a, b, c, materialIndex, tex, /*addToGl=*/true, chunk); continue; }
                            BlendTopTriInfo info;
                            info.topVslotIndex = T->vslot;
                            loopk(3)
                            {
                                const vec bt = bary(corner[k], T->p);
                                info.u[k] = bt.x * T->u[0] + bt.y * T->u[1] + bt.z * T->u[2];
                                info.v[k] = bt.x * T->v[0] + bt.y * T->v[1] + bt.z * T->v[2];
                                info.alpha[k] = float(lookupblendmap(blendCache, corner[k])) / 255.0f;
                                if(T->lmTex)
                                {
                                    info.lmU[k] = bt.x * T->lmu[0] + bt.y * T->lmu[1] + bt.z * T->lmu[2];
                                    info.lmV[k] = bt.x * T->lmv[0] + bt.y * T->lmv[1] + bt.z * T->lmv[2];
                                }
                                else
                                {
                                    info.lmU[k] = corner[k].x * invWorld;
                                    info.lmV[k] = corner[k].y * invWorld;
                                }
                            }
                            info.lmTexGlId = T->lmTex ? T->lmTex : kWorldBlendmapGlId;
                            pushTriangle(a, b, c, materialIndex, tex, /*addToGl=*/true, chunk, &info, pieceBlendMat);
                        }
                    };
                    std::vector<ClipPoly> remaining(1, ClipPoly{ v0.pos, v1.pos, v2.pos });
                    for(const BlendTopTri *T : blendCands)
                    {
                        std::vector<ClipPoly> next;
                        for(const ClipPoly &poly : remaining)
                        {
                            ClipPoly cur = poly;
                            for(int ei = 0; ei < 3 && !cur.empty(); ++ei)
                            {
                                const vec &ea = T->p[ei], &eb = T->p[(ei + 1) % 3], &ec = T->p[(ei + 2) % 3];
                                vec en; en.cross(T->n, vec(eb).sub(ea));
                                const float sgn = en.dot(vec(ec).sub(ea)) >= 0 ? 1.0f : -1.0f;
                                ClipPoly in, out;
                                splitPoly(cur, ea, en, sgn, in, out);
                                if(polyArea(out) >= 1e-5f) next.push_back(out);
                                cur = polyArea(in) >= 1e-5f ? in : ClipPoly();
                            }
                            if(!cur.empty()) emitPiece(cur, T);
                        }
                        remaining = std::move(next);  // not .swap()
                        if(remaining.empty()) break;
                    }
                    for(const ClipPoly &poly : remaining) emitPiece(poly, nullptr);
                }

                if(triangleIsEmissive)
                {
                    EmissiveTriangleDesc tri;
                    tri.v0[0] = v0.pos.x; tri.v0[1] = v0.pos.y; tri.v0[2] = v0.pos.z;
                    tri.v1[0] = v1.pos.x; tri.v1[1] = v1.pos.y; tri.v1[2] = v1.pos.z;
                    tri.v2[0] = v2.pos.x; tri.v2[1] = v2.pos.y; tri.v2[2] = v2.pos.z;
                    tri.radiance[0] = avgGlowColor.x;
                    tri.radiance[1] = avgGlowColor.y;
                    tri.radiance[2] = avgGlowColor.z;
                    outEmissive.push_back(tri);
                }
            }

            ChunkGlBatch batch;
            batch.textureGlId = tex ? static_cast<uint32_t>(tex->id) : 0;
            batch.textureClamp = tex ? static_cast<uint32_t>(tex->clamp) : 0u;
            batch.indexOffset = batchIndexOffset;
            batch.indexCount = static_cast<uint32_t>(chunk.glIndices.size()) - batchIndexOffset;
            batch.materialIndex = materialIndex;
            if(batch.indexCount > 0) chunk.glBatches.push_back(batch);

            albedoR += double(avgColor.x) * double(triangleCount);
            albedoG += double(avgColor.y) * double(triangleCount);
            albedoB += double(avgColor.z) * double(triangleCount);
            albedoWeight += static_cast<uint32_t>(triangleCount);

            idx += e.length[1];
        }

        chunk.opaqueIndexCount = static_cast<uint32_t>(chunk.rtIndices.size());

        // MAT_ALPHA-flagged geometry (glass)
        const uint32_t tailEntryStart = static_cast<uint32_t>(va->texs);
        const uint32_t tailEntryCount = static_cast<uint32_t>(va->blends) + static_cast<uint32_t>(va->alphaback) + static_cast<uint32_t>(va->alphafront);
        if(tailEntryCount > 0)
        {
            ushort *idxTail = tailEdataPtr;
            // defense in depth
            const ushort *tailEdataEnd = edata + 3 * (static_cast<size_t>(va->tris) + static_cast<size_t>(alphaTriCountForRead));
            for(uint32_t j = tailEntryStart; j < tailEntryStart + tailEntryCount; ++j)
            {
                elementset &e = va->eslist[j];
                const int entryIndexCount = int(e.length[1]);
                if(entryIndexCount <= 0)
                {
                    idxTail += e.length[1];
                    continue;
                }
                if(e.layer == LAYER_BLEND)
                {
                    idxTail += e.length[1];
                    continue;
                }
                if(idxTail + entryIndexCount > tailEdataEnd)
                {
                    break;
                }

                Texture *tex = notexture;
                VSlot &vslot = lookupvslot(e.texture, true);
                if(vslot.slot && !vslot.slot->sts.empty() && vslot.slot->sts[0].t)
                    tex = vslot.slot->sts[0].t;

                const uint32_t materialIndex = ensureMaterial(e.texture, vslot, true, materialsSeen, outMaterials);
                const RtMaterialDesc &materialDesc = outMaterials[materialIndex];
                vec glowTint(materialDesc.glowcolor[0], materialDesc.glowcolor[1], materialDesc.glowcolor[2]);
                vec avgGlowColor(0, 0, 0);
                bool triangleIsEmissive = false;
                if(materialDesc.hasGlow && vslot.slot)
                {
                    Slot::Tex *glowTex = findSlotTex(*vslot.slot, TEX_GLOW);
                    if(glowTex && glowTex->t)
                    {
                        avgGlowColor = getTextureAverageColor(glowTex->t);
                        avgGlowColor.mul(glowTint);
                        triangleIsEmissive = avgGlowColor.x > 0 || avgGlowColor.y > 0 || avgGlowColor.z > 0;
                        outMaterials[materialIndex].glowSelectLum = 0.2126f * avgGlowColor.x + 0.7152f * avgGlowColor.y + 0.0722f * avgGlowColor.z;  // see RtMaterialDesc::glowSelectLum
                    }
                }

                const int triangleCount = entryIndexCount / 3;
                for(int te = 0; te < triangleCount; ++te)
                {
                    const int k = te * 3;
                    const uint32_t i0 = static_cast<uint32_t>(idxTail[k + 0]) - static_cast<uint32_t>(va->voffset);
                    const uint32_t i1 = static_cast<uint32_t>(idxTail[k + 1]) - static_cast<uint32_t>(va->voffset);
                    const uint32_t i2 = static_cast<uint32_t>(idxTail[k + 2]) - static_cast<uint32_t>(va->voffset);
                    const uint32_t vertCount = static_cast<uint32_t>(va->verts);
                    if(i0 >= vertCount || i1 >= vertCount || i2 >= vertCount) continue;
                    const vertex &v0 = vdata[i0];
                    const vertex &v1 = vdata[i1];
                    const vertex &v2 = vdata[i2];

                    pushTriangle(v0, v1, v2, materialIndex, tex, /*addToGl=*/false, chunk);

                    if(triangleIsEmissive)
                    {
                        EmissiveTriangleDesc tri;
                        tri.v0[0] = v0.pos.x; tri.v0[1] = v0.pos.y; tri.v0[2] = v0.pos.z;
                        tri.v1[0] = v1.pos.x; tri.v1[1] = v1.pos.y; tri.v1[2] = v1.pos.z;
                        tri.v2[0] = v2.pos.x; tri.v2[1] = v2.pos.y; tri.v2[2] = v2.pos.z;
                        tri.radiance[0] = avgGlowColor.x;
                        tri.radiance[1] = avgGlowColor.y;
                        tri.radiance[2] = avgGlowColor.z;
                        outEmissive.push_back(tri);
                    }
                }


                idxTail += e.length[1];
            }
        }

        if(albedoWeight > 0)
        {
            chunk.avgAlbedo[0] = static_cast<float>(albedoR / double(albedoWeight));
            chunk.avgAlbedo[1] = static_cast<float>(albedoG / double(albedoWeight));
            chunk.avgAlbedo[2] = static_cast<float>(albedoB / double(albedoWeight));
        }

        chunk.solidLeafCubes = static_cast<uint32_t>(va->tris);

        delete[] edata;
        delete[] vdata;

        if(!chunk.rtIndices.empty()) out.push_back(std::move(chunk));
    }

    freeblendmapcache(blendCache);
}

}  // namespace sauerinterop
