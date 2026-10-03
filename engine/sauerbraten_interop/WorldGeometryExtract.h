#pragma once

// texture -> average color

// pbr materials + light transport additions

#include <cstdint>
#include <vector>

struct Texture;  // engine/texture.h

namespace sauerinterop {

// fake GL id of the world blend map texture
constexpr uint32_t kWorldBlendmapGlId = 0x7FFFB1E2u;

// one contiguous run of `glIndices`, all using the same Sauerbraten GL texture
struct ChunkGlBatch {
    uint32_t textureGlId = 0;
    uint32_t textureClamp = 0;
    uint32_t indexOffset = 0;
    uint32_t indexCount = 0;
    uint32_t materialIndex = 0;
    // this batch's skin::alphatest cutoff (0 = opaque)
    float alphaTestCutoff = 0.0f;
};

struct RtMaterialDesc {
    uint32_t vslotIndex = 0;  // VSlot index, matches rtMaterialIndices / ChunkGlBatch::materialIndex
    uint32_t diffuseTexGlId = 0;
    uint32_t normalTexGlId = 0;  // the TEX_NORMAL entry's t->id, or 0 if hasNormal is false
    uint32_t glowTexGlId = 0;  // the TEX_GLOW entry's t->id, or 0 if hasGlow is false
    bool hasNormal = false;  // slot::texmask & (1<<TEX_NORMAL)
    bool hasSpec = false;
    bool hasGlow = false;  // slot::texmask & (1<<TEX_GLOW)
    // shader is a specular variant (specworld, bumpspecworld, ...) but the slot has no spec map
    bool uniformSpec = false;
    bool parallax = false;
    float parallaxScale = 0.0f;  // VSlot::parallaxscale.x, depth range, UV units
    float parallaxBias = 0.0f;  // VSlot::parallaxscale.y
    float colorscale[3] = { 1.0f, 1.0f, 1.0f };  // VSlot::colorscale (diffuse tint)
    float glowcolor[3] = { 1.0f, 1.0f, 1.0f };
    // VSlot::metalness (texture.h/texture.cpp's new texmetal command)
    float metalness = 0.0f;
    // VSlot::roughness (vrough/texrough), -1 = automatic
    float roughness = -1.0f;
    // Sauerbraten VSlot::specscale (texture.h/shader.cpp's new linkvslotshader()-resolved field)
    float specScale = 1.0f;
    bool isGlass = false;
    float glassAlpha = 1.0f;
    // separate dielectric flag from isGlass
    bool isWater = false;
    bool isLava = false;
    // luminance of this material's emissive triangles' stored radiance (average linear glow texel x
    // glow tint)
    float glowSelectLum = 0.0f;
    // per-material Beer-Lambert absorption reference distance (world units)
    float waterReferenceThickness = 64.0f;
    float scroll[2] = { 0.0f, 0.0f };
};

struct EmissiveTriangleDesc {
    float v0[3] = { 0, 0, 0 };
    float v1[3] = { 0, 0, 0 };
    float v2[3] = { 0, 0, 0 };
    float radiance[3] = { 0, 0, 0 };
    // 1 = procedural emitter (path-traced lava)
    float procedural = 0.0f;
};

struct ChunkMeshData {
    uint64_t chunkId = 0;

    std::vector<float> rtPositions;
    std::vector<float> rtUvs;
    std::vector<uint32_t> rtTextureGlIds;
    std::vector<uint32_t> rtIndices;

    // per-vertex smoothed normal/tangent, straight out of Sauerbraten's vertex::norm/vertex::tangent
    std::vector<float> rtNormals;  // 3 floats/vertex, world-space, Sauerbraten smoothing
    std::vector<float> rtTangents;  // 4 floats/vertex (xyz + handedness sign in .w), world-space
    std::vector<uint32_t> rtMaterialIndices;

    // Sauerbraten's "blendmap" painted-texture feature (engine/blend.cpp, texlayer/vlayer commands)
    std::vector<uint32_t> rtBlendMaterialIndices;  // 1 uint/vertex, 0xFFFFFFFFu = no blend
    std::vector<float> rtBlendAlphas;
    std::vector<float> rtBlendUvs;

    std::vector<float> rtAlphaTestCutoffs;  // 1 float/vertex
    // the blend layer's lightmap
    std::vector<float> rtBlendLmUvs;
    std::vector<uint32_t> rtBlendLmTexGlIds;

    // interleaved (px,py,pz, nx,ny,nz, tx,ty,tz,tw, u,v) per vertex
    std::vector<float> glVertexData;
    std::vector<uint32_t> glIndices;

    std::vector<ChunkGlBatch> glBatches;

    float avgAlbedo[3] = { 0.6f, 0.6f, 0.6f };

    uint32_t solidLeafCubes = 0;

    uint32_t opaqueIndexCount = 0xFFFFFFFFu;
};

// stable packing
uint64_t makeChunkId(int x, int y, int z, int size);

void textureAverageColor(Texture *tex, float out[3]);

void extractSolidCubeChunks(std::vector<ChunkMeshData>& out,
                             std::vector<RtMaterialDesc>& outMaterials,
                             std::vector<EmissiveTriangleDesc>& outEmissive);

}  // namespace sauerinterop
