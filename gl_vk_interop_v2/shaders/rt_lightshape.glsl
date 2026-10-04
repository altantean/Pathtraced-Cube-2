// rt_lightshape.glsl, path-tracer light shapes

#ifndef RT_LIGHTSHAPE_GLSL
#define RT_LIGHTSHAPE_GLSL

bool lightShaped(vec4 colorFlags) { return (int(colorFlags.w + 0.5) & 4) != 0; }

// half the capsule's axis (zero for a point / sphere light)
vec3 lightHalfAxis(vec4 colorFlags, vec4 reserved) { return lightShaped(colorFlags) ? reserved.yzw : vec3(0.0); }

// A glowing model's emitter light
float lightShadowGap(vec4 colorFlags, vec4 reserved) { return lightShaped(colorFlags) ? 0.0 : reserved.x; }
vec3 lightDiscFacing(vec4 colorFlags, vec4 reserved) { return lightShaped(colorFlags) ? vec3(0.0) : reserved.yzw; }

float lightSourceRadius(vec4 posRadius, vec4 colorFlags, vec4 reserved) {
    float r = lightShaped(colorFlags) ? reserved.x : 0.0;
    return r > 0.0 ? r : clamp(posRadius.w * 0.05, 0.5, 8.0);
}

// nearest point of the light's segment to p (its centre for a point light)
vec3 lightNearestPoint(vec3 centre, vec3 halfAxis, vec3 p) {
    float l2 = dot(halfAxis, halfAxis);
    if (l2 < 1e-8) return centre;
    return centre + halfAxis * clamp(dot(p - centre, halfAxis) / l2, -1.0, 1.0);
}

// cosine toward the light for weighing it (no rays)
float lightWeighCosine(vec3 centre, vec3 halfAxis, vec3 p, vec3 N, vec3 dirToNearest) {
    float c = dot(N, dirToNearest);
    if (c > 0.0 || dot(halfAxis, halfAxis) < 1e-8) return c;
    return max(dot(N, centre + halfAxis - p), dot(N, centre - halfAxis - p)) > 0.0 ? 0.05 : c;
}

#endif
