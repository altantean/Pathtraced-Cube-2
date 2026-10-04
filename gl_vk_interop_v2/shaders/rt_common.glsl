// shared ray-tracing scene access

#ifndef RT_COMMON_GLSL
#define RT_COMMON_GLSL

// per-chunk hit-attribute table entry
struct MaterialEntry {
    uint64_t vertexAddr;
    uint64_t indexAddr;
    // this instance's previous-frame vertex buffer address
    uint64_t prevVertexAddr;
    uint64_t pad0;
    vec4 albedoPad;
};
layout(std430, set = 0, binding = RT_MATERIALS_BINDING) readonly buffer MaterialBuffer {
    MaterialEntry materials[];
};

// raw per-triangle geometry, fetched via buffer_reference from the addresses stored in MaterialEntry
// above
struct ChunkRtVertexGpu {
    vec3 pos;
    vec3 normal;
    vec4 tangent;
    vec2 uv;
    uint textureIndex;
    uint materialIndex;
    uint blendMaterialIndex;
    float blendAlpha;
    vec2 blendUv;
    // Sauerbraten skin::alphatest cutoff for this triangle's cutout material
    float alphaTestCutoff;
    // the blend layer's lightmap UV + bindless index (0 = none -> use blendAlpha)
    vec2 blendLmUv;
    uint blendLmTexIndex;
};
layout(buffer_reference, buffer_reference_align = 4, scalar) readonly buffer VertexBuf { ChunkRtVertexGpu v[]; };
layout(buffer_reference, buffer_reference_align = 4, scalar) readonly buffer IndexBuf  { uint  i[]; };

// bindless texture table size, keep in sync with RayTracingScene::kMaxPathTraceTextures
#define RT_MAX_TEXTURES 4096
layout(set = 0, binding = RT_TEXTURES_BINDING) uniform sampler2D uMapTextures[RT_MAX_TEXTURES];

// Sauerbraten skin::alphatest cutout support
bool gCandidateIsGrass = false;
bool resolveAlphaTestCandidate(rayQueryEXT rq, out bool passed, out MaterialEntry outEntry) {
    passed = false;
    gCandidateIsGrass = false;
    outEntry.vertexAddr = 0ul;
    outEntry.indexAddr = 0ul;
    int instanceId = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false);
    if (instanceId < 0 || instanceId >= materials.length()) return false;
    outEntry = materials[instanceId];
    MaterialEntry m = outEntry;
    if (m.vertexAddr == 0ul || m.indexAddr == 0ul) return false;
    int localPrimId = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
    uint primId = uint(localPrimId) + uint(m.albedoPad.w) / 3u;
    VertexBuf vb = VertexBuf(m.vertexAddr);
    IndexBuf ib = IndexBuf(m.indexAddr);
    uint i0 = ib.i[primId * 3 + 0];
    float cutoff = vb.v[i0].alphaTestCutoff;
    if (cutoff <= 0.0) return false;
    gCandidateIsGrass = cutoff >= 2.0;
    if (gCandidateIsGrass) cutoff -= 2.0;
    uint i1 = ib.i[primId * 3 + 1];
    uint i2 = ib.i[primId * 3 + 2];
    vec2 bc = rayQueryGetIntersectionBarycentricsEXT(rq, false);
    vec3 bary = vec3(1.0 - bc.x - bc.y, bc.x, bc.y);
    vec2 uv = vb.v[i0].uv * bary.x + vb.v[i1].uv * bary.y + vb.v[i2].uv * bary.z;
    float alphaSample = textureLod(uMapTextures[nonuniformEXT(vb.v[i0].textureIndex)], uv, 0.0).a;
    passed = alphaSample >= cutoff;
    return true;
}

// occlusion-only resolution of a candidate triangle (AO / hard visibility)
bool rtCandidateOccludes(rayQueryEXT rq) {
    bool passed;
    MaterialEntry entry;
    if (resolveAlphaTestCandidate(rq, passed, entry)) return passed;
    return entry.vertexAddr == 0ul || entry.indexAddr == 0ul;
}

#endif  // RT_COMMON_GLSL
