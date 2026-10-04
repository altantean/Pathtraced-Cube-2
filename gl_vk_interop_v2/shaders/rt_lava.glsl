#ifndef RT_LAVA_GLSL
#define RT_LAVA_GLSL
// shared by pathtrace_trace.comp (shading, emission, NEE) and water_patch.comp (the displaced lava
// geometry)
const float LAVA_EMISSION_SCALE = 0.75;
struct LavaSurfaceInfo { vec3 emission; float albedo; float roughness; vec3 normal; float height; float specWeight; };
uint lavaPcg(uint v) { uint st = v * 747796405u + 2891336453u; uint w = ((st >> ((st >> 28u) + 4u)) ^ st) * 277803737u; return (w >> 22u) ^ w; }

float lavaHash(ivec2 c) { return float(lavaPcg(uint(c.x) * 1973u ^ lavaPcg(uint(c.y) * 9277u + 0x68E31DA4u))) / 4294967296.0; }
vec2 lavaHash2(ivec2 c) { uint h = lavaPcg(uint(c.x) * 1973u ^ lavaPcg(uint(c.y) * 9277u + 0x68E31DA4u)); return vec2(float(h & 0xFFFFu), float(h >> 16u)) / 65536.0; }
float lavaNoise(vec2 x) {
    ivec2 i = ivec2(floor(x));
    vec2 f = fract(x);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(lavaHash(i), lavaHash(i + ivec2(1, 0)), u.x), mix(lavaHash(i + ivec2(0, 1)), lavaHash(i + ivec2(1, 1)), u.x), u.y);
}
float lavaPerlin(vec2 x) {
    ivec2 i = ivec2(floor(x));
    vec2 f = fract(x);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float h00 = 6.2831853 * lavaHash(i), h10 = 6.2831853 * lavaHash(i + ivec2(1, 0));
    float h01 = 6.2831853 * lavaHash(i + ivec2(0, 1)), h11 = 6.2831853 * lavaHash(i + ivec2(1, 1));
    float a = dot(vec2(cos(h00), sin(h00)), f), b = dot(vec2(cos(h10), sin(h10)), f - vec2(1.0, 0.0));
    float c = dot(vec2(cos(h01), sin(h01)), f - vec2(0.0, 1.0)), d = dot(vec2(cos(h11), sin(h11)), f - vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float lavaFbm(vec2 x) { return 0.5 + 0.55 * lavaPerlin(x) + 0.3 * lavaPerlin(x * 2.03 + 11.7) + 0.15 * lavaPerlin(x * 4.11 + 3.1); }

// planck spectral radiance at R/G/B wavelengths, normalised so a 1300 K body has green = 1
vec3 lavaBlackbody(float T) {
    const vec3 lambda = vec3(610.0, 550.0, 465.0);
    const vec3 inv5 = vec3(1.0 / 8.4460e13, 1.0 / 5.0328e13, 1.0 / 2.1730e13);  // lambda^-5
    vec3 x = 1.4388e7 / (lambda * max(T, 300.0));
    vec3 b = inv5 / max(exp(min(x, vec3(80.0))) - 1.0, vec3(1e-30));
    const float ref = (1.0 / 5.0328e13) / (5.4919e8 - 1.0);  // green at 1300 K: exp(20.124)
    return b / ref;
}

vec3 lavaTint(vec3 c, vec2 tint) {
    if (abs(tint.x) < 1e-4 && abs(tint.y - 1.0) < 1e-4) return c;
    float y = dot(c, vec3(0.299, 0.587, 0.114));
    vec2 iq = vec2(dot(c, vec3(0.596, -0.274, -0.322)), dot(c, vec3(0.211, -0.523, 0.312)));
    float cs = cos(tint.x), sn = sin(tint.x);
    iq = vec2(iq.x * cs - iq.y * sn, iq.x * sn + iq.y * cs) * tint.y;
    vec3 o = max(vec3(y + 0.956 * iq.x + 0.621 * iq.y, y - 0.272 * iq.x - 0.647 * iq.y, y - 1.106 * iq.x + 1.703 * iq.y), vec3(0.0));
    const vec3 L709 = vec3(0.2126, 0.7152, 0.0722);
    return o * (dot(c, L709) / max(dot(o, L709), 1e-8));
}

vec2 lavaShoreFromMap(sampler2D waveMap, vec4 map, vec2 xy) {
    if (map.z <= 0.0) return vec2(1e4);
    vec4 s = textureLod(waveMap, (xy - map.xy) * map.zw, 0.0);
    return s.x < -0.01 ? s.yz / -s.x : vec2(1e4);
}

// pos / n
float lavaMeltHeight(vec2 q, vec2 fdir, float t) {
    vec2 fq = vec2(dot(q, fdir) * 0.8, dot(q, vec2(-fdir.y, fdir.x)) * 1.4);
    float swell = lavaPerlin(fq * 0.9 + vec2(t * 0.02, t * 0.013));
    float fold = 1.0 - min(abs(lavaPerlin(vec2(fq.x * 2.2, fq.y * 4.0) + vec2(t * 0.2, 0.0))) * 2.5, 1.0);
    return 0.45 * swell + 0.25 * fold * fold;
}
LavaSurfaceInfo lavaSurfaceS(vec3 pos, vec3 n, float glow, vec4 lava0, vec4 lava1, vec2 shore) {
    float t = lava1.z;
    float size = max(lava0.x, 1.0);
    bool top = abs(n.z) >= 0.5;
    vec2 q, du, dv;  // field coordinates, and the world directions they run along
    if (top) {
        q = pos.xy / size;
        du = vec2(1.0, 0.0); dv = vec2(0.0, 1.0);
        q -= vec2(0.7071, 0.7071) * (t * lava0.y / size);  // flows like the raster lava's texture scroll
    } else {
        vec2 h = normalize(vec2(-n.y, n.x) + vec2(1e-6));
        q = vec2(dot(pos.xy, h) / size, pos.z / (size * 2.5));  // stretched down the fall
        q.y += t * (16.0 / 3.0) / (size * 2.5);  // falls fast (raster lavafall speed)
        du = h; dv = vec2(0.0);  // side bump
    }
    // slow domain warp
    q += 0.35 * vec2(lavaNoise(q * 0.5 + vec2(0.0, t * 0.05)), lavaNoise(q * 0.5 + vec2(17.3, -t * 0.05))) - 0.175;

    // streaky, stretched along the flow
    vec2 fdir = top ? vec2(0.7071, 0.7071) : vec2(0.0, 1.0);
    vec2 fq = vec2(dot(q, fdir) * 0.8, dot(q, vec2(-fdir.y, fdir.x)) * 1.4);
    float melt = clamp((lavaFbm(fq * 1.6 + vec2(t * 0.03, 0.0)) - 0.5) * 2.5, -1.0, 1.0);
    // thin cooler skin wrinkling on the flow (ridged noise)
    float skin = 1.0 - min(abs(lavaPerlin(vec2(fq.x * 2.2, fq.y * 4.0) + vec2(t * 0.2, 0.0))) * 2.5, 1.0);
    skin = skin * skin * skin * skin * smoothstep(0.35, -0.5, melt);
    // wide spread
    float meltT = lava1.x - (top ? 60.0 : 0.0) + 200.0 * melt - 220.0 * skin + 50.0 * (lavaNoise(q * 5.0 + vec2(t * 0.5, -t * 0.4)) - 0.5);

    float cover = clamp((lavaFbm(q * 0.33 + vec2(t * 0.015, -t * 0.011)) - 0.5) * 2.2 + 0.5, 0.0, 1.0);
    float amt = clamp(lava1.w, 0.0, 1.0);
    float island = top ? smoothstep(1.0 - amt - 0.06, 1.0 - amt + 0.06, cover) : 0.0;  // falling lava moves too fast to crust
    float bankWidth = 1.5 * size;
    float shoreRagged = shore.x + 0.45 * bankWidth * (lavaNoise(q * 1.7 + vec2(41.0, t * 0.01)) - 0.5);
    float bank = top ? 1.0 - smoothstep(0.25 * bankWidth, bankWidth, shoreRagged) : 0.0;
    float plunge = top ? 1.0 - smoothstep(0.2 * bankWidth, 1.2 * bankWidth, shore.y) : 0.0;
    bank *= 1.0 - plunge;
    island = max(island, bank) * (1.0 - plunge);
    meltT += 120.0 * plunge - 160.0 * bank;
    const float PLATE = 1.7;  // plates per field unit, smaller than the islands they break up
    // warped cell space
    vec2 qv = q * PLATE + 0.45 * vec2(lavaPerlin(q * 0.9 + vec2(3.7, t * 0.01)), lavaPerlin(q * 0.9 + vec2(-t * 0.01, 9.2)));
    ivec2 c = ivec2(floor(qv));
    float F1 = 9.0, F2 = 9.0;
    vec2 p1 = vec2(0.0), p2 = vec2(0.0);
    ivec2 id1 = c;
    for (int j = -1; j <= 1; ++j) for (int i = -1; i <= 1; ++i) {
        ivec2 cc = c + ivec2(i, j);
        vec2 hsh = lavaHash2(cc);
        vec2 fp = vec2(cc) + 0.5 + 0.38 * sin(t * 0.12 + 6.2831853 * hsh);
        float d = length(qv - fp);
        if (d < F1) { F2 = F1; p2 = p1; F1 = d; p1 = fp; id1 = cc; }
        else if (d < F2) { F2 = d; p2 = fp; }
    }
    float edge = F2 - F1;  // ~0 on a plate border
    float w = max(lava0.w, 1e-3) * (0.5 + lavaNoise(qv * 1.3 + 5.0));  // seams vary in width
    float crack = 1.0 - smoothstep(0.0, w, edge);
    float solid = island * (1.0 - crack);  // 1 = solid crust
    float plateHeat = lavaHash(id1);
    float rim = 1.0 - smoothstep(0.0, w * 4.0, edge);
    float islandEdge = 1.0 - smoothstep(0.0, 0.5, island);  // crust thins toward the island's edge
    float crustT = lava1.y + (220.0 * plateHeat * plateHeat + 320.0 * max(rim, islandEdge)) * (1.0 - 0.6 * bank);  // the shelf is old, cold crust
    // fresh melt welling up in the seams between plates is the hottest, yellowest lava on the surface
    meltT += 110.0 * crack * island * (1.0 - 0.7 * bank);  // the cold shelf's seams barely glow
    float T = mix(meltT, crustT, solid);

    LavaSurfaceInfo o;
    o.emission = lavaBlackbody(T) * (LAVA_EMISSION_SCALE * glow);
    o.albedo = mix(0.02, 0.025 * (0.7 + 0.6 * lavaNoise(qv * 5.0)), solid);
    o.roughness = mix(0.55, 0.95, solid);  // the melt has a glassy skin, softened
    o.specWeight = mix(0.5, 0.1, solid);
    vec2 g1 = (qv - p1) / max(F1, 1e-4), g2 = (qv - p2) / max(F2, 1e-4);
    float dome = 1.0 - 1.2 * F1 * F1;
    float x = clamp(edge / w, 0.0, 1.0);
    vec2 gCrack = -(6.0 * x * (1.0 - x) / w) * (g2 - g1);  // d crack / dq
    float hField = dome * (1.0 - crack) * island;
    vec2 gh = ((-2.4 * F1 * g1) * (1.0 - crack) - dome * gCrack) * island * PLATE;
    // the melt heaves too (lower than the crust, it sits beneath it)
    if (island < 0.99) {
        const float e = 0.02;
        float m0 = lavaMeltHeight(q, fdir, t);
        vec2 gm = vec2(lavaMeltHeight(q + vec2(e, 0.0), fdir, t) - m0, lavaMeltHeight(q + vec2(0.0, e), fdir, t) - m0) / e;
        hField += (1.0 - island) * (m0 - 0.3);
        gh += (1.0 - island) * gm;
    }
    // rough, clinkery crust surface (gradient noise, finite differences)
    vec2 gClinker = vec2(0.0);
    if (solid > 0.01) {
        vec2 rq = qv * 6.0;
        float r0 = lavaPerlin(rq);
        gClinker = solid * 0.25 * 6.0 * PLATE * vec2(lavaPerlin(rq + vec2(0.05, 0.0)) - r0, lavaPerlin(rq + vec2(0.0, 0.05)) - r0) / 0.05;
    }
    float hClinker = solid > 0.01 ? solid * 0.25 * lavaPerlin(qv * 6.0) : 0.0;  // ~0.25 world units, whatever the relief
    o.height = hField * lava0.z + hClinker;
    gh = (gh * lava0.z + gClinker) / size;  // world units of height per world unit
    // shading normals stay well above the horizon (steep plate rims read as rims, not as black holes)
    float ghLen = length(gh);
    if (ghLen > 1.2) gh *= 1.2 / ghLen;
    vec3 tu = vec3(du, 0.0), tv = top ? vec3(dv, 0.0) : vec3(0.0, 0.0, 1.0);
    vec3 nn = top ? vec3(0.0, 0.0, sign(n.z)) : normalize(n);
    o.normal = normalize(nn - (gh.x * tu + (top ? gh.y : 0.0) * tv) * (top ? sign(n.z) : 1.0));
    return o;
}
LavaSurfaceInfo lavaSurface(vec3 pos, vec3 n, float glow, vec4 lava0, vec4 lava1) { return lavaSurfaceS(pos, n, glow, lava0, lava1, vec2(1e4)); }

#endif  // RT_LAVA_GLSL
