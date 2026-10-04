#ifndef RT_WATER_RECTS_GLSL
#define RT_WATER_RECTS_GLSL
// the map's water top rectangles (RayTracingScene::setWaterSurfaces, once per map) with a bucket grid
// over them
struct WaterRect { vec4 rect; vec4 zmat; };
layout(std430, binding = 0) readonly buffer RectBuf { WaterRect rects[]; };
// heads[b] = (first item, count)
layout(std430, binding = 1) readonly buffer BucketBuf { uvec2 heads[]; };
layout(std430, binding = 2) readonly buffer ItemBuf { uint items[]; };

// buckets
bool waterAt(vec2 xy, vec4 buckets, float bucketsY, out float z, out uint mat, out float kind, out float glow) {
    z = 0.0; mat = 0u; kind = 0.0; glow = 0.0;
    ivec2 b = ivec2(floor((xy - buckets.xy) / buckets.z));
    if (b.x < 0 || b.y < 0 || b.x >= int(buckets.w) || b.y >= int(bucketsY)) return false;
    uvec2 h = heads[uint(b.y) * uint(buckets.w) + uint(b.x)];
    for (uint i = 0u; i < h.y; ++i) {
        WaterRect r = rects[items[h.x + i]];
        if (xy.x >= r.rect.x && xy.x < r.rect.z && xy.y >= r.rect.y && xy.y < r.rect.w) {
            z = r.zmat.x;
            mat = floatBitsToUint(r.zmat.y);
            kind = r.zmat.z;
            glow = r.zmat.w;
            return true;
        }
    }
    return false;
}
#endif  // RT_WATER_RECTS_GLSL
