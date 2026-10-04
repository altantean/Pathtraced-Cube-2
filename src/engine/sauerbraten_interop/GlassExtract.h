#pragma once

#include <cstdint>
#include <vector>
#include "WorldGeometryExtract.h"

namespace sauerinterop {

void extractGlassVolumes(std::vector<RtMaterialDesc>& materials, ChunkMeshData& out);

}  // namespace sauerinterop
