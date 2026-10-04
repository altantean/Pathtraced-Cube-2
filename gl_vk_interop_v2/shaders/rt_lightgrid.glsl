// rt_lightgrid.glsl light grid

layout(std430, set = 0, binding = LIGHTGRID_BINDING) readonly buffer LightGridBuf { uint lightGrid[]; };

struct LightList {
    uint cellOffset, cellCount;  // this point's cell list
    uint globalOffset, globalCount;  // no-falloff static lights
    uint dynBegin, dynCount;  // lights the grid doesn't cover (dynamic lights)
};

LightList lightListAt(vec3 pos, uint lightCount) {
    LightList L;
    L.cellOffset = 0u; L.cellCount = 0u; L.globalOffset = 0u; L.globalCount = 0u;
    if (lightGrid[10] == 0u) { L.dynBegin = 0u; L.dynCount = lightCount; return L; }  // no grid
    uint gridLights = min(lightGrid[9], lightCount);
    L.dynBegin = gridLights;
    L.dynCount = lightCount - gridLights;
    L.globalOffset = lightGrid[7];
    L.globalCount = lightGrid[8];
    uvec3 dims = uvec3(lightGrid[0], lightGrid[1], lightGrid[2]);
    float cellSize = uintBitsToFloat(lightGrid[3]);
    vec3 origin = vec3(uintBitsToFloat(lightGrid[4]), uintBitsToFloat(lightGrid[5]), uintBitsToFloat(lightGrid[6]));
    ivec3 c = ivec3(floor((pos - origin) / cellSize));
    if (all(greaterThanEqual(c, ivec3(0))) && all(lessThan(c, ivec3(dims)))) {
        uint ci = (uint(c.z) * dims.y + uint(c.y)) * dims.x + uint(c.x);
        L.cellOffset = lightGrid[16u + 2u * ci];
        L.cellCount = lightGrid[17u + 2u * ci];
    }
    return L;
}

uint lightListSize(LightList L) { return L.cellCount + L.globalCount + L.dynCount; }

// k-th light index of the list (callers still check it against the frame's light count)
uint lightListGet(LightList L, uint k) {
    if (k < L.cellCount) return lightGrid[L.cellOffset + k];
    k -= L.cellCount;
    if (k < L.globalCount) return lightGrid[L.globalOffset + k];
    return L.dynBegin + (k - L.globalCount);
}

// the grid builder's per-cell alias tables (layout in SauerbratenInterop's updateLightGrid)
bool lightListHasImportance(LightList L) { return lightGrid[13] != 0u && L.cellCount > 0u; }

uint lightListImportancePick(LightList L, float u, out float pdfInCell) {
    uint k = min(uint(u), L.cellCount - 1u);
    uint t = lightGrid[11] + 3u * (L.cellOffset + k - lightGrid[12]);
    if (fract(u) >= uintBitsToFloat(lightGrid[t])) {
        k = lightGrid[t + 1u];
        t = lightGrid[11] + 3u * (L.cellOffset + k - lightGrid[12]);
    }
    pdfInCell = uintBitsToFloat(lightGrid[t + 2u]);
    return k;
}
