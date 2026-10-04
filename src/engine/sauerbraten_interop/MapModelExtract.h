
#include <cstdint>
#include <vector>
#include "WorldGeometryExtract.h"

namespace sauerinterop {

// walks Sauerbraten's entities::getents() for ET_MAPMODEL entities, extracts eligible ones into `out`
void extractStaticMapModelChunks(std::vector<ChunkMeshData>& out);

// stable packing distinct from makeChunkId()'s world-chunk IDs
uint64_t makeMapModelChunkId(int entityIndex);

}  // namespace sauerinterop
