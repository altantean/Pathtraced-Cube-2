#pragma once
// WaterExtract path-traced water geometry

#include <cstdint>
#include <vector>
#include "WorldGeometryExtract.h"

namespace sauerinterop {

constexpr uint64_t kWaterChunkId = (uint64_t(1) << 63) | (uint64_t(1) << 62);

void extractWaterVolumes(std::vector<RtMaterialDesc>& materials, ChunkMeshData& out, std::vector<float>* topRects = nullptr);

constexpr uint64_t kLavaChunkId = (uint64_t(1) << 63) | (uint64_t(1) << 61);
// procedural
constexpr float kLavaNominal[3] = { 1.2f, 0.25f, 0.01f };  // = pathtrace_trace.comp's LAVA_NOMINAL
void extractLavaVolumes(std::vector<RtMaterialDesc>& materials, ChunkMeshData& out,
                        std::vector<EmissiveTriangleDesc>& emissive, float glow, bool scroll, bool procedural,
                        std::vector<float>* topRects = nullptr);  // 8 floats per top, kind 1, glow

}  // namespace sauerinterop
