#ifndef RT_WATER_GLSL
#define RT_WATER_GLSL
// path-traced water the one wave field every water consumer shares

vec3 waterCascadeWeights(vec2 xy, vec4 map, vec4 amp) {
    vec3 w = map.z > 0.0 ? textureLod(uWaveMap, (xy - map.xy) * map.zw, 0.0).xyz : vec3(1.0);
    return w * vec3(amp.x, amp.x, amp.x * amp.y);
}

// world-space displacement of the surface point that rests at xy
vec3 waterWaveDisplacement(vec2 xy, vec4 fft, vec4 map, vec4 amp) {
    if (amp.w < 0.5) return vec3(0.0);
    vec3 m = waterCascadeWeights(xy, map, amp);
    vec3 d = vec3(0.0);
    for (int c = 0; c < 3; ++c)
        if (m[c] != 0.0) d += m[c] * textureLod(uWaveFft, vec3(xy / fft[c], float(2 * c)), 0.0).xyz;
    d.xy *= fft.w;
    return d;
}

// surface normal (up-facing) at xy
vec3 waterWaveNormalAt(vec2 xy, vec4 fft, vec4 map, vec4 amp, float fineMul) {
    if (amp.w < 0.5) return vec3(0.0, 0.0, 1.0);
    vec3 m = waterCascadeWeights(xy, map, amp);
    m.z *= fineMul;
    vec4 s = vec4(0.0);
    for (int c = 0; c < 3; ++c)
        if (m[c] != 0.0) s += m[c] * textureLod(uWaveFft, vec3(xy / fft[c], float(2 * c + 1)), 0.0);
    // choppy waves
    vec2 jac = max(vec2(1.0) + fft.w * s.zw, vec2(0.25));
    return normalize(vec3(-s.xy / jac, 1.0));
}

void waterFlowTerms(vec2 xy, vec4 fft, vec4 map, vec4 amp, out vec2 sdot, out vec3 hes) {
    vec3 m = waterCascadeWeights(xy, map, amp);
    sdot = vec2(0.0);
    hes = vec3(0.0);  // xx, yy, xy
    for (int c = 0; c < 3; ++c) {
        if (m[c] == 0.0) continue;
        vec2 uv = xy / fft[c];
        vec4 f = textureLod(uWaveFft, vec3(uv, float(6 + c)), 0.0);
        float hxy = textureLod(uWaveFft, vec3(uv, float(2 * c)), 0.0).w;
        sdot += m[c] * f.xy;
        hes += m[c] * vec3(f.zw, hxy);
    }
}
vec2 waterWaveFlow(vec2 xy, vec4 fft, vec4 map, vec4 amp, float radius) {
    if (amp.w < 0.5) return vec2(0.0);
    mat2 A = mat2(0.0);
    vec2 b = vec2(0.0);
    for (int j = -1; j <= 1; ++j)
    for (int i = -1; i <= 1; ++i) {
        vec2 sdot; vec3 hes;
        waterFlowTerms(xy + vec2(i, j) * radius, fft, map, amp, sdot, hes);
        float w = (i == 0 ? 2.0 : 1.0) * (j == 0 ? 2.0 : 1.0);  // binomial 1-2-1
        mat2 J = mat2(hes.x, hes.z, hes.z, hes.y);  // symmetric
        A += w * (J * J);
        b += w * (J * sdot);
    }
    float lambda = 0.02 * (A[0][0] + A[1][1]) + 1e-12;
    A[0][0] += lambda; A[1][1] += lambda;
    float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (!(abs(det) > 1e-30)) return vec2(0.0);
    vec2 v = -(inverse(A) * b);
    if (any(isnan(v)) || any(isinf(v))) return vec2(0.0);
    float vmax = 2.0 * sqrt(9.81 * 8.0 * fft.x / 6.2831853);
    float len = length(v);
    return len > vmax ? v * (vmax / len) : v;
}
#endif  // RT_WATER_GLSL
