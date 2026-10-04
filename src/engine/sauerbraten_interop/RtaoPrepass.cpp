#include "RtaoPrepass.h"
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <new>
#define SAUERINTEROP_STL_PLACEMENT_NEW

#include "engine.h"
#include <windows.h>

namespace sauerinterop {

namespace {

typedef void (APIENTRYP PFNGLGENVERTEXARRAYS_LOCAL)(GLsizei, GLuint*);
typedef void (APIENTRYP PFNGLBINDVERTEXARRAY_LOCAL)(GLuint);
typedef void (APIENTRYP PFNGLDELETEVERTEXARRAYS_LOCAL)(GLsizei, const GLuint*);

PFNGLGENVERTEXARRAYS_LOCAL localGenVertexArrays = nullptr;
PFNGLBINDVERTEXARRAY_LOCAL localBindVertexArray = nullptr;
PFNGLDELETEVERTEXARRAYS_LOCAL localDeleteVertexArrays = nullptr;
bool localVaoResolved = false;

bool resolveLocalVaoFunctions(std::string* error) {
    if (localVaoResolved) return true;
    localGenVertexArrays = reinterpret_cast<PFNGLGENVERTEXARRAYS_LOCAL>(wglGetProcAddress("glGenVertexArrays"));
    localBindVertexArray = reinterpret_cast<PFNGLBINDVERTEXARRAY_LOCAL>(wglGetProcAddress("glBindVertexArray"));
    localDeleteVertexArrays = reinterpret_cast<PFNGLDELETEVERTEXARRAYS_LOCAL>(wglGetProcAddress("glDeleteVertexArrays"));
    if (!localGenVertexArrays || !localBindVertexArray || !localDeleteVertexArrays) {
        if (error) *error = "RtaoPrepass: independently-resolved glGenVertexArrays/glBindVertexArray/"
                             "glDeleteVertexArrays came back null from wglGetProcAddress.";
        return false;
    }
    localVaoResolved = true;
    return true;
}

// same "don't trust an unproven function pointer" discipline as resolveLocalVaoFunctions() above
typedef void (APIENTRYP PFNGLBINDBUFFERBASE_LOCAL)(GLenum, GLuint, GLuint);
PFNGLBINDBUFFERBASE_LOCAL localBindBufferBase = nullptr;
bool localBindBufferBaseResolved = false;

bool resolveLocalBindBufferBase(std::string* error) {
    if (localBindBufferBaseResolved) return true;
    localBindBufferBase = reinterpret_cast<PFNGLBINDBUFFERBASE_LOCAL>(wglGetProcAddress("glBindBufferBase"));
    if (!localBindBufferBase) {
        if (error) *error = "RtaoPrepass: independently-resolved glBindBufferBase came back null from wglGetProcAddress.";
        return false;
    }
    localBindBufferBaseResolved = true;
    return true;
}

const char* kVertexSrc = R"GLSL(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;
layout(location = 3) in vec4 aTangent;

uniform mat4 uCamProj;
uniform mat4 uCam;
uniform mat4 uCurrUnjittered;
uniform mat4 uPrevUnjittered;

out vec3 vNormalView;
out vec4 vTangentView;  // xyz view-space tangent, w handedness passthrough
out vec4 vClipCurrUnjittered;
out vec4 vClipPrevUnjittered;
out vec2 vUv;
out vec3 vWorldPos;  // see kFragmentSrc

void main() {
    vec4 worldPos = vec4(aPos, 1.0);
    vWorldPos = aPos;
    gl_Position = uCamProj * worldPos;
    mat3 nmat = mat3(uCam);
    vNormalView = nmat * aNormal;
    vTangentView = vec4(nmat * aTangent.xyz, aTangent.w);
    vClipCurrUnjittered = uCurrUnjittered * worldPos;
    vClipPrevUnjittered = uPrevUnjittered * worldPos;
    vUv = aUv;
}
)GLSL";

// fragAlbedo (location 2) is new
const char* kFragmentSrc = R"GLSL(
#version 330 core
in vec3 vNormalView;
in vec4 vTangentView;
in vec4 vClipCurrUnjittered;
in vec4 vClipPrevUnjittered;
in vec2 vUv;
in vec3 vWorldPos;
uniform vec2 uUvOffset;
uniform vec2 uUvStep;
uniform mat4 uPrevUnjittered;
// explicit locations, several outputs
layout(location = 0) out vec4 fragNormal;
layout(location = 1) out vec2 fragMotion;
layout(location = 2) out vec4 fragAlbedo;
layout(location = 3) out vec4 fragGlow;  // pbr materials

uniform sampler2D uAlbedoTex;  // bound per-batch in render()
uniform sampler2D uNormalTex;  // bound per-batch, unit 1
uniform sampler2D uGlowTex;  // bound per-batch, unit 2
uniform float uHasNormal;  // this batch's Slot::texmask&(1<<TEX_NORMAL)
uniform float uHasSpec;  // this batch's Slot::texmask&(1<<TEX_SPEC)
uniform float uHasGlow;  // this batch's Slot::texmask&(1<<TEX_GLOW)
uniform vec3 uColorScale;  // VSlot::colorscale (diffuse tint)
uniform vec3 uGlowColor;  // VSlot::glowcolor (glow tint)
uniform float uMetalness;  // VSlot::metalness (texture.h's new field)
uniform float uSpecScale;  // VSlot::specscale (texture.h's new field)
uniform float uAlphaTest;  // cutoff, 0 = opaque (see PrepassBatch::alphaTestCutoff)

const float DLSS_ROUGHNESS_MIN = 0.05;
const float DLSS_ROUGHNESS_MAX = 0.9;
float specIntensityToRoughnessForDlss(float specIntensity) {
    return clamp(mix(DLSS_ROUGHNESS_MAX, DLSS_ROUGHNESS_MIN, clamp(specIntensity, 0.0, 1.0)), DLSS_ROUGHNESS_MIN, DLSS_ROUGHNESS_MAX);
}
uniform float uPtMat;
uniform float uRoughOverride;  // roughness override (vrough/texrough), < 0 = automatic
float materialRoughnessForDlss(float ks) {
    if (uPtMat < 0.5) return specIntensityToRoughnessForDlss(clamp(ks, 0.0, 1.0));
    float a = clamp(0.1 / sqrt(max(ks, 1e-6)), DLSS_ROUGHNESS_MIN * DLSS_ROUGHNESS_MIN, DLSS_ROUGHNESS_MAX * DLSS_ROUGHNESS_MAX);
    return sqrt(a);
}

void main() {
    vec3 nView = normalize(vNormalView);
    vec2 uv = vUv + uUvOffset;  // scrolling textures

    // tangent-space normal mapping
    if (uHasNormal > 0.5) {
        vec3 tView = normalize(vTangentView.xyz - nView * dot(vTangentView.xyz, nView));  // re-orthogonalize against the interpolated normal
        vec3 bView = cross(nView, tView) * (vTangentView.w >= 0.0 ? 1.0 : -1.0);
        vec3 tsNormal = texture(uNormalTex, uv).rgb * 2.0 - 1.0;
        nView = normalize(mat3(tView, bView, nView) * tsNormal);
    }

    vec2 currUv = vClipCurrUnjittered.xy / vClipCurrUnjittered.w * 0.5 + 0.5;
    // vClipPrevUnjittered.w can be non-positive
    vec4 clipPrev = vClipPrevUnjittered;
    if (uUvStep.x != 0.0 || uUvStep.y != 0.0) {
        vec3 dPx = dFdx(vWorldPos), dPy = dFdy(vWorldPos);
        vec2 dUx = dFdx(vUv), dUy = dFdy(vUv);
        float det = dUx.x * dUy.y - dUy.x * dUx.y;
        if (abs(det) > 1e-12) {
            vec2 ab = vec2(uUvStep.x * dUy.y - dUy.x * uUvStep.y, dUx.x * uUvStep.y - uUvStep.x * dUx.y) / det;
            clipPrev = uPrevUnjittered * vec4(vWorldPos + dPx * ab.x + dPy * ab.y, 1.0);
        }
    }
    if (clipPrev.w > 1e-4) {
        vec2 prevUv = clipPrev.xy / clipPrev.w * 0.5 + 0.5;
        fragMotion = prevUv - currUv;
    } else {
        fragMotion = vec2(1000.0, 1000.0);
    }
    // Sauerbraten's textures are plain 8-bit gamma-encoded values
    vec4 rawAlbedo = texture(uAlbedoTex, uv);
    if (uAlphaTest > 0.0 && rawAlbedo.a < uAlphaTest) discard;
    float specIntensity = uHasSpec > 0.5 ? rawAlbedo.a : 0.0;
    fragAlbedo = vec4(pow(rawAlbedo.rgb, vec3(2.2)) * uColorScale, specIntensity);

    fragNormal = vec4(nView, uRoughOverride >= 0.0 ? max(uRoughOverride, DLSS_ROUGHNESS_MIN) : materialRoughnessForDlss(specIntensity * uSpecScale));

    fragGlow = vec4(uHasGlow > 0.5 ? pow(texture(uGlowTex, uv).rgb, vec3(2.2)) * uGlowColor : vec3(0.0), clamp(uMetalness, 0.0, 1.0) * 0.49);
}
)GLSL";

const char* kAnimatedVertexSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aBlendBones;
layout(location = 2) in vec4 aBlendWeights;
layout(location = 3) in vec3 aNormal;
layout(location = 4) in vec2 aUv;

struct BoneDualQuat { vec4 real; vec4 dual; };
layout(std430, binding = 0) readonly buffer CurrentBones { BoneDualQuat b[]; } curBonesBuf;
layout(std430, binding = 1) readonly buffer PrevBones { BoneDualQuat b[]; } prevBonesBuf;

uniform mat4 uCamProj;
uniform mat4 uCurrUnjittered;
uniform mat4 uPrevUnjittered;
uniform vec4 uCurrWorldRow0;
uniform vec4 uCurrWorldRow1;
uniform vec4 uCurrWorldRow2;
uniform vec4 uPrevWorldRow0;
uniform vec4 uPrevWorldRow1;
uniform vec4 uPrevWorldRow2;
uniform mat4 uCam;  // world -> view, same as the static prepass's uCam

out vec4 vClipCurrUnjittered;
out vec4 vClipPrevUnjittered;
out vec3 vNormalView;
out vec2 vUv;

BoneDualQuat blendFour(BoneDualQuat b0, BoneDualQuat b1, BoneDualQuat b2, BoneDualQuat b3,
                       float w0, float w1, float w2, float w3) {
    BoneDualQuat d;
    d.real = b0.real * w0;
    d.dual = b0.dual * w0;
    {
        float k = w1;
        if (dot(d.real, b1.real) < 0.0) k = -k;
        d.real += b1.real * k;
        d.dual += b1.dual * k;
    }
    if (w2 > 0.0) {
        float k = w2;
        if (dot(d.real, b2.real) < 0.0) k = -k;
        d.real += b2.real * k;
        d.dual += b2.dual * k;
        if (w3 > 0.0) {
            float k2 = w3;
            if (dot(d.real, b3.real) < 0.0) k2 = -k2;
            d.real += b3.real * k2;
            d.dual += b3.dual * k2;
        }
    }
    return d;
}

void normalizeDualQuat(inout BoneDualQuat d) {
    float len = length(d.real);
    if (len > 1e-8) { d.real /= len; d.dual /= len; }
}

vec3 dqTransformPoint(BoneDualQuat d, vec3 p) {
    vec3 qv = d.real.xyz;
    float qw = d.real.w;
    vec3 rotated = p + 2.0 * cross(qv, cross(qv, p) + qw * p);
    vec3 t = 2.0 * (qw * d.dual.xyz - d.dual.w * qv + cross(qv, d.dual.xyz));
    return rotated + t;
}

vec3 applyWorldMatrix(vec3 p, vec4 row0, vec4 row1, vec4 row2) {
    return vec3(dot(row0.xyz, p) + row0.w, dot(row1.xyz, p) + row1.w, dot(row2.xyz, p) + row2.w);
}

void main() {
    ivec4 bi = ivec4(aBlendBones);

    BoneDualQuat dCur = blendFour(curBonesBuf.b[bi.x], curBonesBuf.b[bi.y], curBonesBuf.b[bi.z], curBonesBuf.b[bi.w],
                                   aBlendWeights.x, aBlendWeights.y, aBlendWeights.z, aBlendWeights.w);
    normalizeDualQuat(dCur);
    vec3 worldCur = applyWorldMatrix(dqTransformPoint(dCur, aPos), uCurrWorldRow0, uCurrWorldRow1, uCurrWorldRow2);

    BoneDualQuat dPrev = blendFour(prevBonesBuf.b[bi.x], prevBonesBuf.b[bi.y], prevBonesBuf.b[bi.z], prevBonesBuf.b[bi.w],
                                    aBlendWeights.x, aBlendWeights.y, aBlendWeights.z, aBlendWeights.w);
    normalizeDualQuat(dPrev);
    vec3 worldPrev = applyWorldMatrix(dqTransformPoint(dPrev, aPos), uPrevWorldRow0, uPrevWorldRow1, uPrevWorldRow2);

    gl_Position = uCamProj * vec4(worldCur, 1.0);
    vClipCurrUnjittered = uCurrUnjittered * vec4(worldCur, 1.0);
    vClipPrevUnjittered = uPrevUnjittered * vec4(worldPrev, 1.0);

    // normal
    vec3 qv = dCur.real.xyz;
    vec3 nObj = aNormal + 2.0 * cross(qv, cross(qv, aNormal) + dCur.real.w * aNormal);
    vec3 nWorld = vec3(dot(uCurrWorldRow0.xyz, nObj), dot(uCurrWorldRow1.xyz, nObj), dot(uCurrWorldRow2.xyz, nObj));
    vNormalView = mat3(uCam) * normalize(nWorld);
    vUv = aUv;
}
)GLSL";

const char* kAnimatedFragmentSrc = R"GLSL(
#version 430 core
in vec4 vClipCurrUnjittered;
in vec4 vClipPrevUnjittered;
in vec3 vNormalView;
in vec2 vUv;
layout(location = 0) out vec4 fragNormal;
layout(location = 1) out vec2 fragMotion;
layout(location = 2) out vec4 fragAlbedo;
layout(location = 3) out vec4 fragGlow;
uniform sampler2D uAlbedoTex;
// see AnimatedSkinBatch (RtaoPrepass.h)
uniform sampler2D uMasksTex;  // unit 1
uniform float uHasMasks;
uniform float uModelSpec;
uniform float uModelGlow;
uniform float uModelReflect;  // ptmodelreflect, masks.b gloss, same as the path tracer

// same mapping as the static prepass's materialRoughnessForDlss (ptmat)
uniform float uPtMat;
float modelRoughnessForDlss(float ks) {
    if (uPtMat < 0.5) return clamp(mix(0.9, 0.05, clamp(ks, 0.0, 1.0)), 0.05, 0.9);
    float a = clamp(0.1 / sqrt(max(ks, 1e-6)), 0.0025, 0.81);
    return sqrt(a);
}

void main() {
    // the same model material the path tracer shades with (see pathtrace_trace.comp's model material
    // block)
    vec3 albedo = pow(texture(uAlbedoTex, vUv).rgb, vec3(2.2));
    vec3 masks = uHasMasks > 0.5 ? texture(uMasksTex, vUv).rgb : vec3(1.0, 0.0, 0.0);
    float specIntensity = clamp(0.5 * uModelSpec * masks.r, 0.0, 1.0);
    if (uHasMasks > 0.5) specIntensity = max(specIntensity, masks.b);  // mirror-sharp wherever masked, like the path tracer
    float glowMask = 0.0;  // glow not used (matches the path tracer, it read as fullbright)
    float ksForRoughness = uPtMat > 0.5 && uHasMasks > 0.5 ? max(0.5 * uModelSpec * masks.r, masks.b * 1600.0) : specIntensity;  // 1600 = MAT_MIRROR_KS (pathtrace_trace.comp)
    fragNormal = vec4(normalize(vNormalView), modelRoughnessForDlss(ksForRoughness));
    fragAlbedo = vec4(albedo * (1.0 - glowMask), specIntensity);
    float envRefl = uHasMasks > 0.5 ? clamp(masks.b * uModelReflect, 0.0, 1.0) : 0.0;
    fragGlow = vec4(albedo * uModelGlow * glowMask, envRefl > 0.004 ? 0.51 + 0.49 * envRefl : 0.0);
    vec2 currUv = vClipCurrUnjittered.xy / vClipCurrUnjittered.w * 0.5 + 0.5;
    if (vClipPrevUnjittered.w > 1e-4) {
        vec2 prevUv = vClipPrevUnjittered.xy / vClipPrevUnjittered.w * 0.5 + 0.5;
        fragMotion = prevUv - currUv;
    } else {
        fragMotion = vec2(1000.0, 1000.0);
    }
}
)GLSL";

bool compile(GLenum type, const char* src, GLuint& outShader, std::string* error) {
    outShader = glCreateShader_(type);
    glShaderSource_(outShader, 1, &src, nullptr);
    glCompileShader_(outShader);
    GLint ok = 0;
    glGetShaderiv_(outShader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        GLsizei len = 0;
        glGetShaderInfoLog_(outShader, sizeof(log), &len, log);
        if (error) *error = std::string("RtaoPrepass: shader compile failed: ") + std::string(log, len);
        return false;
    }
    return true;
}

}  // namespace

RtaoPrepass::~RtaoPrepass() {
    if (!initialized_) return;  // never initialized, nothing was ever created
    if (fbo_) glDeleteFramebuffers_(1, &fbo_);
    if (ebo_) glDeleteBuffers_(1, &ebo_);
    if (vbo_) glDeleteBuffers_(1, &vbo_);
    if (vao_) localDeleteVertexArrays(1, &vao_);
    if (program_) glDeleteProgram_(program_);
    if (skyProgram_) { glDeleteProgram_(skyProgram_); skyProgram_ = 0; }
    if (fallbackTex_) glDeleteTextures(1, &fallbackTex_);  // pbr materials

    // animated model motion vectors
    if (animProgram_) glDeleteProgram_(animProgram_);
    if (animCurrBoneSsbo_) glDeleteBuffers_(1, &animCurrBoneSsbo_);
    if (animPrevBoneSsbo_) glDeleteBuffers_(1, &animPrevBoneSsbo_);
    // particle motion vectors
    if (partProgram_) glDeleteProgram_(partProgram_);
    if (partVbo_) glDeleteBuffers_(1, &partVbo_);
    if (partVao_) localDeleteVertexArrays(1, &partVao_);
    if (partFbo_) glDeleteFramebuffers_(1, &partFbo_);
    for (auto& kv : animSkinSources_) {
        if (kv.second.ebo) glDeleteBuffers_(1, &kv.second.ebo);
        if (kv.second.vbo) glDeleteBuffers_(1, &kv.second.vbo);
        if (kv.second.vao) localDeleteVertexArrays(1, &kv.second.vao);
    }
}

// the passes above only draw geometry, so every sky pixel kept the cleared motion of (0,0)
const char* kSkyMotionVertexSrc = R"GLSL(
#version 330 core
uniform mat4 uCurrUnjittered;
noperspective out vec4 vNear;
noperspective out vec4 vFar;
void main() {
    vec2 ndc = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
    mat4 inv = inverse(uCurrUnjittered);
    vNear = inv * vec4(ndc, -1.0, 1.0);
    vFar = inv * vec4(ndc, 1.0, 1.0);
    gl_Position = vec4(ndc, 1.0, 1.0);
}
)GLSL";

const char* kSkyMotionFragmentSrc = R"GLSL(
#version 330 core
noperspective in vec4 vNear;
noperspective in vec4 vFar;
uniform mat4 uCurrUnjittered;
uniform mat4 uPrevUnjittered;
layout(location = 0) out vec4 fragNormal;
layout(location = 1) out vec2 fragMotion;
layout(location = 2) out vec4 fragAlbedo;
layout(location = 3) out vec4 fragGlow;
void main() {
    vec3 dir = vFar.xyz / vFar.w - vNear.xyz / vNear.w;
    vec4 c = uCurrUnjittered * vec4(dir, 0.0);
    vec4 pv = uPrevUnjittered * vec4(dir, 0.0);
    fragNormal = vec4(0.0);
    fragAlbedo = vec4(0.0);
    fragGlow = vec4(0.0);
    if (c.w > 1e-6 && pv.w > 1e-6) {
        fragMotion = (pv.xy / pv.w * 0.5 + 0.5) - (c.xy / c.w * 0.5 + 0.5);
    } else {
        fragMotion = vec2(1000.0, 1000.0);  // same "no valid reprojection" sentinel as fragMotion above
    }
}
)GLSL";

bool RtaoPrepass::initialize(std::string* error) {
    if (!resolveLocalVaoFunctions(error)) return false;

    GLuint vs = 0, fs = 0;
    if (!compile(GL_VERTEX_SHADER, kVertexSrc, vs, error)) return false;
    if (!compile(GL_FRAGMENT_SHADER, kFragmentSrc, fs, error)) { glDeleteShader_(vs); return false; }

    program_ = glCreateProgram_();
    glAttachShader_(program_, vs);
    glAttachShader_(program_, fs);
    glLinkProgram_(program_);
    glDeleteShader_(vs);
    glDeleteShader_(fs);

    GLint linked = 0;
    glGetProgramiv_(program_, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[2048];
        GLsizei len = 0;
        glGetProgramInfoLog_(program_, sizeof(log), &len, log);
        if (error) *error = std::string("RtaoPrepass: program link failed: ") + std::string(log, len);
        return false;
    }

    {
        GLuint svs = 0, sfs = 0;
        std::string skyErr;
        if (compile(GL_VERTEX_SHADER, kSkyMotionVertexSrc, svs, &skyErr)) {
            if (compile(GL_FRAGMENT_SHADER, kSkyMotionFragmentSrc, sfs, &skyErr)) {
                skyProgram_ = glCreateProgram_();
                glAttachShader_(skyProgram_, svs);
                glAttachShader_(skyProgram_, sfs);
                glLinkProgram_(skyProgram_);
                glDeleteShader_(sfs);
                GLint skyLinked = 0;
                glGetProgramiv_(skyProgram_, GL_LINK_STATUS, &skyLinked);
                if (!skyLinked) { glDeleteProgram_(skyProgram_); skyProgram_ = 0; }
            }
            glDeleteShader_(svs);
        }
        if (skyProgram_) {
            skyUCurrUnjitteredLoc_ = glGetUniformLocation_(skyProgram_, "uCurrUnjittered");
            skyUPrevUnjitteredLoc_ = glGetUniformLocation_(skyProgram_, "uPrevUnjittered");
        }
    }

    uCamProjLoc_ = glGetUniformLocation_(program_, "uCamProj");
    uCamLoc_ = glGetUniformLocation_(program_, "uCam");
    uCurrUnjitteredLoc_ = glGetUniformLocation_(program_, "uCurrUnjittered");
    uPrevUnjitteredLoc_ = glGetUniformLocation_(program_, "uPrevUnjittered");
    uAlbedoTexLoc_ = glGetUniformLocation_(program_, "uAlbedoTex");
    // new per-batch uniforms
    uNormalTexLoc_ = glGetUniformLocation_(program_, "uNormalTex");
    uGlowTexLoc_ = glGetUniformLocation_(program_, "uGlowTex");
    uHasNormalLoc_ = glGetUniformLocation_(program_, "uHasNormal");
    uHasSpecLoc_ = glGetUniformLocation_(program_, "uHasSpec");
    uHasGlowLoc_ = glGetUniformLocation_(program_, "uHasGlow");
    uColorScaleLoc_ = glGetUniformLocation_(program_, "uColorScale");
    uGlowColorLoc_ = glGetUniformLocation_(program_, "uGlowColor");
    // pbr light transport on uMetalness
    uMetalnessLoc_ = glGetUniformLocation_(program_, "uMetalness");
    // on uSpecScale
    uSpecScaleLoc_ = glGetUniformLocation_(program_, "uSpecScale");
    uPtMatLoc_ = glGetUniformLocation_(program_, "uPtMat");
    uRoughOverrideLoc_ = glGetUniformLocation_(program_, "uRoughOverride");
    uAlphaTestLoc_ = glGetUniformLocation_(program_, "uAlphaTest");
    uUvOffsetLoc_ = glGetUniformLocation_(program_, "uUvOffset");
    uUvStepLoc_ = glGetUniformLocation_(program_, "uUvStep");

    localGenVertexArrays(1, &vao_);
    glGenBuffers_(1, &vbo_);
    glGenBuffers_(1, &ebo_);

    GLint prevVaoInit = 0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVaoInit);
    localBindVertexArray(vao_);
    glBindBuffer_(GL_ARRAY_BUFFER, vbo_);
    glBindBuffer_(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    const GLsizei stride = 12 * sizeof(float);
    glVertexAttribPointer_(0, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<const void*>(0));
    glEnableVertexAttribArray_(0);
    glVertexAttribPointer_(1, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<const void*>(3 * sizeof(float)));
    glEnableVertexAttribArray_(1);
    glVertexAttribPointer_(3, 4, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<const void*>(6 * sizeof(float)));
    glEnableVertexAttribArray_(3);
    glVertexAttribPointer_(2, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<const void*>(10 * sizeof(float)));
    glEnableVertexAttribArray_(2);
    localBindVertexArray(static_cast<GLuint>(prevVaoInit));
    glBindBuffer_(GL_ARRAY_BUFFER, 0);  // global (non-VAO) state, restore to unbound explicitly

    glGenTextures(1, &fallbackTex_);
    glBindTexture(GL_TEXTURE_2D, fallbackTex_);
    const unsigned char neutral[4] = { 128, 128, 255, 255 };
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, neutral);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenFramebuffers_(1, &fbo_);
    initialized_ = true;

    // same hard-failure discipline as everything above it in this function
    if (!initAnimatedProgram(error)) return false;
    return true;
}

bool RtaoPrepass::uploadMesh(const float* interleavedPosNormalUv, uint32_t vertexCount,
                              const uint32_t* indices, uint32_t indexCount,
                              const PrepassBatch* batches, uint32_t batchCount, std::string* error) {
    if (!initialized_) { if (error) *error = "RtaoPrepass::uploadMesh: not initialized."; return false; }
    GLint prevVao = 0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
    localBindVertexArray(vao_);
    glBindBuffer_(GL_ARRAY_BUFFER, vbo_);
    glBufferData_(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(float)) * 12 * vertexCount,
                  interleavedPosNormalUv, GL_STATIC_DRAW);
    glBindBuffer_(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData_(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(uint32_t)) * indexCount,
                  indices, GL_STATIC_DRAW);
    localBindVertexArray(static_cast<GLuint>(prevVao));
    indexCount_ = indexCount;

    // copy the batch list
    batches_.assign(batches, batches + batchCount);
    return true;
}

bool RtaoPrepass::attachTargets(uint32_t depthGlTexture, uint32_t normalGlTexture, uint32_t motionGlTexture,
                                 uint32_t albedoGlTexture, uint32_t glowGlTexture, uint32_t /*width*/, uint32_t /*height*/, std::string* error) {
    if (!initialized_) { if (error) *error = "RtaoPrepass::attachTargets: not initialized."; return false; }
    glBindFramebuffer_(GL_FRAMEBUFFER, fbo_);
    glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthGlTexture, 0);
    glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, normalGlTexture, 0);
    glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, motionGlTexture, 0);
    // albedo as a third color attachment, matching kFragmentSrc's layout(location = 2) out vec4
    // fragAlbedo
    glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, albedoGlTexture, 0);
    // glow as a fourth color attachment, matching kFragmentSrc's layout(location = 3) out vec4
    // fragGlow
    glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, glowGlTexture, 0);
    const GLenum drawBufs[4] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
    glDrawBuffers_(4, drawBufs);
    const GLenum status = glCheckFramebufferStatus_(GL_FRAMEBUFFER);
    glBindFramebuffer_(GL_FRAMEBUFFER, 0);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        if (error) *error = "RtaoPrepass::attachTargets: framebuffer incomplete (status 0x" +
                             std::to_string(status) + ") -- check depth/normal/motion/albedo/glow SharedTexture formats/sizes match.";
        return false;
    }
    haveTargets_ = true;
    return true;
}

bool RtaoPrepass::render(uint32_t width, uint32_t height, const float* camProjFlat16, const float* camFlat16,
                          const float* currentVpUnjitteredFlat16, const float* prevVpUnjitteredFlat16) {
    if (!initialized_ || !haveTargets_ || indexCount_ == 0) return false;

    GLint prevFbo = 0, prevViewport[4]{};
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
    glGetIntegerv(GL_VIEWPORT, prevViewport);
    GLboolean prevDepthMask = GL_TRUE;
    glGetBooleanv(GL_DEPTH_WRITEMASK, &prevDepthMask);
    GLboolean prevCull = glIsEnabled(GL_CULL_FACE);
    GLboolean prevDepthTest = glIsEnabled(GL_DEPTH_TEST);
    // these three were being set below but never saved/restored
    GLint prevDepthFunc = GL_LESS;
    glGetIntegerv(GL_DEPTH_FUNC, &prevDepthFunc);
    GLint prevCullFaceMode = GL_BACK;
    glGetIntegerv(GL_CULL_FACE_MODE, &prevCullFaceMode);
    GLint prevFrontFace = GL_CCW;
    glGetIntegerv(GL_FRONT_FACE, &prevFrontFace);
    GLint prevVao = 0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
    GLfloat prevClearColor[4]{};
    glGetFloatv(GL_COLOR_CLEAR_VALUE, prevClearColor);
    GLfloat prevClearDepth = 1.0f;
    glGetFloatv(GL_DEPTH_CLEAR_VALUE, &prevClearDepth);
    GLint prevActiveTexture = GL_TEXTURE0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTexture);
    glActiveTexture_(GL_TEXTURE0);
    GLint prevBoundTexture0 = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevBoundTexture0);
    glActiveTexture_(GL_TEXTURE1);
    GLint prevBoundTexture1 = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevBoundTexture1);
    glActiveTexture_(GL_TEXTURE2);
    GLint prevBoundTexture2 = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevBoundTexture2);
    glActiveTexture_(GL_TEXTURE0);

    glBindFramebuffer_(GL_FRAMEBUFFER, fbo_);
    glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CW);
    // (wrong) assumption that this draw's CCW-outward-wound geometry needed GL_CCW here

    glClearDepth(1.0);
    glClearColor(0, 0, 0, 0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glUseProgram_(program_);
    if (uPtMatLoc_ >= 0) glUniform1f_(uPtMatLoc_, ptMat_ ? 1.0f : 0.0f);
    if (uCamProjLoc_ >= 0) glUniformMatrix4fv_(uCamProjLoc_, 1, GL_FALSE, camProjFlat16);
    if (uCamLoc_ >= 0) glUniformMatrix4fv_(uCamLoc_, 1, GL_FALSE, camFlat16);
    if (uCurrUnjitteredLoc_ >= 0) glUniformMatrix4fv_(uCurrUnjitteredLoc_, 1, GL_FALSE, currentVpUnjitteredFlat16);
    if (uPrevUnjitteredLoc_ >= 0) glUniformMatrix4fv_(uPrevUnjitteredLoc_, 1, GL_FALSE, prevVpUnjitteredFlat16);
    if (uAlbedoTexLoc_ >= 0) glUniform1i_(uAlbedoTexLoc_, 0);  // always unit 0
    if (uNormalTexLoc_ >= 0) glUniform1i_(uNormalTexLoc_, 1);  // always unit 1
    if (uGlowTexLoc_ >= 0) glUniform1i_(uGlowTexLoc_, 2);  // always unit 2

    localBindVertexArray(vao_);
    if (!batches_.empty()) {
        for (const PrepassBatch& b : batches_) {
            if (b.indexCount == 0) continue;
            glActiveTexture_(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(b.textureGlId));
            // per-batch normal/glow texture binds and has*/tint uniforms
            glActiveTexture_(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, b.hasNormal && b.normalTexGlId ? static_cast<GLuint>(b.normalTexGlId) : fallbackTex_);
            glActiveTexture_(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, b.hasGlow && b.glowTexGlId ? static_cast<GLuint>(b.glowTexGlId) : fallbackTex_);
            if (uHasNormalLoc_ >= 0) glUniform1f_(uHasNormalLoc_, (b.hasNormal && b.normalTexGlId) ? 1.0f : 0.0f);
            if (uHasSpecLoc_ >= 0) glUniform1f_(uHasSpecLoc_, b.hasSpec ? 1.0f : 0.0f);
            if (uHasGlowLoc_ >= 0) glUniform1f_(uHasGlowLoc_, (b.hasGlow && b.glowTexGlId) ? 1.0f : 0.0f);
            if (uColorScaleLoc_ >= 0) glUniform3f_(uColorScaleLoc_, b.colorscale[0], b.colorscale[1], b.colorscale[2]);
            if (uGlowColorLoc_ >= 0) glUniform3f_(uGlowColorLoc_, b.glowcolor[0], b.glowcolor[1], b.glowcolor[2]);
            // pbr light transport on uMetalness
            if (uMetalnessLoc_ >= 0) glUniform1f_(uMetalnessLoc_, b.metalness);
            // on uSpecScale
            if (uSpecScaleLoc_ >= 0) glUniform1f_(uSpecScaleLoc_, b.specScale);
            if (uRoughOverrideLoc_ >= 0) glUniform1f_(uRoughOverrideLoc_, b.roughness);
            if (uAlphaTestLoc_ >= 0) glUniform1f_(uAlphaTestLoc_, b.alphaTestCutoff);
            // see PrepassBatch::uvScroll
            if (uUvOffsetLoc_ >= 0) glUniform2f_(uUvOffsetLoc_, b.uvScroll[0] * scrollTime_, b.uvScroll[1] * scrollTime_);
            if (uUvStepLoc_ >= 0) glUniform2f_(uUvStepLoc_, b.uvScroll[0] * scrollDt_, b.uvScroll[1] * scrollDt_);
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(b.indexCount), GL_UNSIGNED_INT,
                           reinterpret_cast<const void*>(static_cast<uintptr_t>(b.indexOffset) * sizeof(uint32_t)));
        }
    } else {
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indexCount_), GL_UNSIGNED_INT, nullptr);
    }
    // see kSkyMotionFragmentSrc
    if (skyProgram_) {
        glUseProgram_(skyProgram_);
        if (skyUCurrUnjitteredLoc_ >= 0) glUniformMatrix4fv_(skyUCurrUnjitteredLoc_, 1, GL_FALSE, currentVpUnjitteredFlat16);
        if (skyUPrevUnjitteredLoc_ >= 0) glUniformMatrix4fv_(skyUPrevUnjitteredLoc_, 1, GL_FALSE, prevVpUnjitteredFlat16);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_FALSE);
        glDisable(GL_CULL_FACE);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glEnable(GL_CULL_FACE);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);
    }
    localBindVertexArray(static_cast<GLuint>(prevVao));
    glUseProgram_(0);

    // restore, this pass must never leak state into Sauerbraten's subsequent rendering this frame
    glBindFramebuffer_(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo));
    glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
    glClearColor(prevClearColor[0], prevClearColor[1], prevClearColor[2], prevClearColor[3]);
    glClearDepth(prevClearDepth);
    glDepthMask(prevDepthMask);
    glDepthFunc(static_cast<GLenum>(prevDepthFunc));
    glCullFace(static_cast<GLenum>(prevCullFaceMode));
    glFrontFace(static_cast<GLenum>(prevFrontFace));
    if (!prevCull) glDisable(GL_CULL_FACE);
    if (!prevDepthTest) glDisable(GL_DEPTH_TEST);
    glActiveTexture_(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevBoundTexture2));  // pbr materials
    glActiveTexture_(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevBoundTexture1));  // pbr materials
    glActiveTexture_(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevBoundTexture0));
    glActiveTexture_(static_cast<GLenum>(prevActiveTexture));
    return true;
}


bool RtaoPrepass::initAnimatedProgram(std::string* error) {
    if (!resolveLocalBindBufferBase(error)) return false;

    GLuint vs = 0, fs = 0;
    if (!compile(GL_VERTEX_SHADER, kAnimatedVertexSrc, vs, error)) return false;
    if (!compile(GL_FRAGMENT_SHADER, kAnimatedFragmentSrc, fs, error)) { glDeleteShader_(vs); return false; }

    animProgram_ = glCreateProgram_();
    glAttachShader_(animProgram_, vs);
    glAttachShader_(animProgram_, fs);
    glLinkProgram_(animProgram_);
    glDeleteShader_(vs);
    glDeleteShader_(fs);

    GLint linked = 0;
    glGetProgramiv_(animProgram_, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[2048];
        GLsizei len = 0;
        glGetProgramInfoLog_(animProgram_, sizeof(log), &len, log);
        if (error) *error = std::string("RtaoPrepass: animated program link failed: ") + std::string(log, len);
        return false;
    }

    animUCamProjLoc_ = glGetUniformLocation_(animProgram_, "uCamProj");
    animUCurrUnjitteredLoc_ = glGetUniformLocation_(animProgram_, "uCurrUnjittered");
    animUPrevUnjitteredLoc_ = glGetUniformLocation_(animProgram_, "uPrevUnjittered");
    animUCamLoc_ = glGetUniformLocation_(animProgram_, "uCam");
    animUAlbedoTexLoc_ = glGetUniformLocation_(animProgram_, "uAlbedoTex");
    animUMasksTexLoc_ = glGetUniformLocation_(animProgram_, "uMasksTex");
    animUHasMasksLoc_ = glGetUniformLocation_(animProgram_, "uHasMasks");
    animUModelSpecLoc_ = glGetUniformLocation_(animProgram_, "uModelSpec");
    animUModelGlowLoc_ = glGetUniformLocation_(animProgram_, "uModelGlow");
    animUModelReflectLoc_ = glGetUniformLocation_(animProgram_, "uModelReflect");
    animUPtMatLoc_ = glGetUniformLocation_(animProgram_, "uPtMat");
    animUCurrWorldRow0Loc_ = glGetUniformLocation_(animProgram_, "uCurrWorldRow0");
    animUCurrWorldRow1Loc_ = glGetUniformLocation_(animProgram_, "uCurrWorldRow1");
    animUCurrWorldRow2Loc_ = glGetUniformLocation_(animProgram_, "uCurrWorldRow2");
    animUPrevWorldRow0Loc_ = glGetUniformLocation_(animProgram_, "uPrevWorldRow0");
    animUPrevWorldRow1Loc_ = glGetUniformLocation_(animProgram_, "uPrevWorldRow1");
    animUPrevWorldRow2Loc_ = glGetUniformLocation_(animProgram_, "uPrevWorldRow2");

    glGenBuffers_(1, &animCurrBoneSsbo_);
    glGenBuffers_(1, &animPrevBoneSsbo_);

    animInitialized_ = true;
    return true;
}

bool RtaoPrepass::hasAnimatedSkinSource(uint64_t modelKey) const {
    return animSkinSources_.find(modelKey) != animSkinSources_.end();
}

bool RtaoPrepass::uploadAnimatedSkinSource(uint64_t modelKey, const void* vertexData, uint32_t vertexStride,
                                            uint32_t vertexCount, const uint32_t* indices, uint32_t indexCount,
                                            const AnimatedSkinBatch* batches, uint32_t batchCount,
                                            std::string* error) {
    if (!animInitialized_) { if (error) *error = "RtaoPrepass::uploadAnimatedSkinSource: animated program not initialized."; return false; }
    if (animSkinSources_.count(modelKey)) return true;

    AnimatedSkinGlSource src;
    localGenVertexArrays(1, &src.vao);
    glGenBuffers_(1, &src.vbo);
    glGenBuffers_(1, &src.ebo);

    GLint prevVao = 0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
    localBindVertexArray(src.vao);
    glBindBuffer_(GL_ARRAY_BUFFER, src.vbo);
    glBufferData_(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertexStride) * vertexCount, vertexData, GL_STATIC_DRAW);
    glBindBuffer_(GL_ELEMENT_ARRAY_BUFFER, src.ebo);
    glBufferData_(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(uint32_t)) * indexCount, indices, GL_STATIC_DRAW);

    glVertexAttribPointer_(0, 3, GL_FLOAT, GL_FALSE, static_cast<GLsizei>(vertexStride), reinterpret_cast<const void*>(0));
    glEnableVertexAttribArray_(0);
    glVertexAttribPointer_(1, 4, GL_UNSIGNED_INT, GL_FALSE, static_cast<GLsizei>(vertexStride), reinterpret_cast<const void*>(static_cast<uintptr_t>(48)));
    glEnableVertexAttribArray_(1);
    glVertexAttribPointer_(2, 4, GL_FLOAT, GL_FALSE, static_cast<GLsizei>(vertexStride), reinterpret_cast<const void*>(static_cast<uintptr_t>(64)));
    glEnableVertexAttribArray_(2);
    // SkinSourceVertexData
    glVertexAttribPointer_(3, 3, GL_FLOAT, GL_FALSE, static_cast<GLsizei>(vertexStride), reinterpret_cast<const void*>(static_cast<uintptr_t>(12)));
    glEnableVertexAttribArray_(3);
    glVertexAttribPointer_(4, 2, GL_FLOAT, GL_FALSE, static_cast<GLsizei>(vertexStride), reinterpret_cast<const void*>(static_cast<uintptr_t>(40)));
    glEnableVertexAttribArray_(4);

    localBindVertexArray(static_cast<GLuint>(prevVao));
    glBindBuffer_(GL_ARRAY_BUFFER, 0);

    src.indexCount = indexCount;
    if (batches && batchCount) src.batches.assign(batches, batches + batchCount);
    else src.batches.push_back(AnimatedSkinBatch{0u, 0u, indexCount});
    animSkinSources_[modelKey] = src;
    return true;
}

bool RtaoPrepass::renderAnimated(uint32_t width, uint32_t height, const AnimatedInstance* instances, uint32_t instanceCount,
                                   const float* camProjFlat16, const float* camFlat16, const float* currentVpUnjitteredFlat16,
                                   const float* prevVpUnjitteredFlat16) {
    if (!initialized_ || !animInitialized_ || !haveTargets_ || instanceCount == 0) return true;

    GLint prevFbo = 0, prevViewport[4]{};
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
    glGetIntegerv(GL_VIEWPORT, prevViewport);
    GLboolean prevDepthMask = GL_TRUE;
    glGetBooleanv(GL_DEPTH_WRITEMASK, &prevDepthMask);
    GLboolean prevCull = glIsEnabled(GL_CULL_FACE);
    GLboolean prevDepthTest = glIsEnabled(GL_DEPTH_TEST);
    GLint prevDepthFunc = GL_LESS;
    glGetIntegerv(GL_DEPTH_FUNC, &prevDepthFunc);
    GLint prevCullFaceMode = GL_BACK;
    glGetIntegerv(GL_CULL_FACE_MODE, &prevCullFaceMode);
    GLint prevFrontFace = GL_CCW;
    glGetIntegerv(GL_FRONT_FACE, &prevFrontFace);
    GLint prevVao = 0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);

    glBindFramebuffer_(GL_FRAMEBUFFER, fbo_);
    glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));

    // spec-safe way to leave normalTex_/albedoTex_/glowTex_ untouched at these pixels
    const GLenum animDrawBufs[4] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
    glDrawBuffers_(4, animDrawBufs);
    GLint prevActiveTexture = GL_TEXTURE0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTexture);
    glActiveTexture_(GL_TEXTURE0);
    GLint prevBoundTexture0 = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevBoundTexture0);
    glActiveTexture_(GL_TEXTURE1);  // masks (model material), saved/restored like unit 0
    GLint prevBoundTexture1 = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevBoundTexture1);
    glActiveTexture_(GL_TEXTURE0);

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CW);


    glUseProgram_(animProgram_);
    if (animUCamProjLoc_ >= 0) glUniformMatrix4fv_(animUCamProjLoc_, 1, GL_FALSE, camProjFlat16);
    if (animUCurrUnjitteredLoc_ >= 0) glUniformMatrix4fv_(animUCurrUnjitteredLoc_, 1, GL_FALSE, currentVpUnjitteredFlat16);
    if (animUPrevUnjitteredLoc_ >= 0) glUniformMatrix4fv_(animUPrevUnjitteredLoc_, 1, GL_FALSE, prevVpUnjitteredFlat16);
    if (animUCamLoc_ >= 0) glUniformMatrix4fv_(animUCamLoc_, 1, GL_FALSE, camFlat16);
    if (animUAlbedoTexLoc_ >= 0) glUniform1i_(animUAlbedoTexLoc_, 0);
    if (animUMasksTexLoc_ >= 0) glUniform1i_(animUMasksTexLoc_, 1);
    if (animUModelReflectLoc_ >= 0) glUniform1f_(animUModelReflectLoc_, modelReflect_);
    if (animUPtMatLoc_ >= 0) glUniform1f_(animUPtMatLoc_, ptMat_ ? 1.0f : 0.0f);

    localBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, animCurrBoneSsbo_);
    localBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, animPrevBoneSsbo_);

    for (uint32_t i = 0; i < instanceCount; ++i) {
        const AnimatedInstance& inst = instances[i];
        auto it = animSkinSources_.find(inst.modelKey);
        if (it == animSkinSources_.end() || it->second.indexCount == 0) continue;
        const AnimatedSkinGlSource& src = it->second;

        const float* prevBones = inst.prevBones ? inst.prevBones : inst.currentBones;
        const float* prevWorldMatrix = inst.prevBones ? inst.prevWorldMatrix : inst.currentWorldMatrix;

        glBindBuffer_(GL_SHADER_STORAGE_BUFFER, animCurrBoneSsbo_);
        glBufferData_(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(sizeof(float)) * 8 * inst.boneCount, inst.currentBones, GL_STREAM_DRAW);
        glBindBuffer_(GL_SHADER_STORAGE_BUFFER, animPrevBoneSsbo_);
        glBufferData_(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(sizeof(float)) * 8 * inst.boneCount, prevBones, GL_STREAM_DRAW);

        if (animUCurrWorldRow0Loc_ >= 0) glUniform4f_(animUCurrWorldRow0Loc_, inst.currentWorldMatrix[0], inst.currentWorldMatrix[1], inst.currentWorldMatrix[2], inst.currentWorldMatrix[3]);
        if (animUCurrWorldRow1Loc_ >= 0) glUniform4f_(animUCurrWorldRow1Loc_, inst.currentWorldMatrix[4], inst.currentWorldMatrix[5], inst.currentWorldMatrix[6], inst.currentWorldMatrix[7]);
        if (animUCurrWorldRow2Loc_ >= 0) glUniform4f_(animUCurrWorldRow2Loc_, inst.currentWorldMatrix[8], inst.currentWorldMatrix[9], inst.currentWorldMatrix[10], inst.currentWorldMatrix[11]);
        if (animUPrevWorldRow0Loc_ >= 0) glUniform4f_(animUPrevWorldRow0Loc_, prevWorldMatrix[0], prevWorldMatrix[1], prevWorldMatrix[2], prevWorldMatrix[3]);
        if (animUPrevWorldRow1Loc_ >= 0) glUniform4f_(animUPrevWorldRow1Loc_, prevWorldMatrix[4], prevWorldMatrix[5], prevWorldMatrix[6], prevWorldMatrix[7]);
        if (animUPrevWorldRow2Loc_ >= 0) glUniform4f_(animUPrevWorldRow2Loc_, prevWorldMatrix[8], prevWorldMatrix[9], prevWorldMatrix[10], prevWorldMatrix[11]);

        localBindVertexArray(src.vao);
        for (const AnimatedSkinBatch& b : src.batches) {
            if (b.indexCount == 0) continue;
            glActiveTexture_(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, b.masksGlTexture ? static_cast<GLuint>(b.masksGlTexture) : fallbackTex_);
            glActiveTexture_(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, b.glTexture ? static_cast<GLuint>(b.glTexture) : fallbackTex_);
            if (animUHasMasksLoc_ >= 0) glUniform1f_(animUHasMasksLoc_, b.masksGlTexture ? 1.0f : 0.0f);
            if (animUModelSpecLoc_ >= 0) glUniform1f_(animUModelSpecLoc_, b.spec);
            if (animUModelGlowLoc_ >= 0) glUniform1f_(animUModelGlowLoc_, b.glow);
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(b.indexCount), GL_UNSIGNED_INT,
                           reinterpret_cast<const void*>(static_cast<uintptr_t>(b.indexOffset) * sizeof(uint32_t)));
        }
    }

    localBindVertexArray(static_cast<GLuint>(prevVao));
    glUseProgram_(0);

    glActiveTexture_(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevBoundTexture1));
    glActiveTexture_(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevBoundTexture0));
    glActiveTexture_(static_cast<GLenum>(prevActiveTexture));

    glBindFramebuffer_(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo));
    glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
    glDepthMask(prevDepthMask);
    glDepthFunc(static_cast<GLenum>(prevDepthFunc));
    glCullFace(static_cast<GLenum>(prevCullFaceMode));
    glFrontFace(static_cast<GLenum>(prevFrontFace));
    if (!prevCull) glDisable(GL_CULL_FACE);
    if (!prevDepthTest) glDisable(GL_DEPTH_TEST);
    return true;
}

// Each particle vertex carries where it is now and where it was last frame
namespace {
const char* kParticleMotionVs = R"GLSL(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aPrev;
layout(location = 2) in vec2 aUv;
layout(location = 3) in vec4 aColor;
layout(location = 4) in vec2 aMode;
uniform mat4 uCamProj;
uniform mat4 uCurrUnjittered;
uniform mat4 uPrevUnjittered;
out vec4 vCurr;
out vec4 vPrev;
out vec2 vUv;
out float vAlpha;
flat out vec2 vMode;
void main() {
    gl_Position = uCamProj * vec4(aPos, 1.0);
    vCurr = uCurrUnjittered * vec4(aPos, 1.0);
    vPrev = uPrevUnjittered * vec4(aPrev, 1.0);
    vUv = aUv;
    vAlpha = aColor.a;
    vMode = aMode;
}
)GLSL";

const char* kParticleMotionFs = R"GLSL(
#version 330 core
in vec4 vCurr;
in vec4 vPrev;
in vec2 vUv;
in float vAlpha;
flat in vec2 vMode;
uniform sampler2D uTex;
uniform float uMinCoverage;
layout(location = 0) out vec2 fragMotion;
void main() {
    vec4 t = texture(uTex, vUv);
    if (vMode.y > 0.5) t.a *= 1.0 - smoothstep(0.25, 0.5, length(vUv - 0.5));
    float cover = (vMode.x > 0.5 ? max(t.r, max(t.g, t.b)) * t.a : t.a) * vAlpha;
    if (cover < uMinCoverage) discard;
    vec2 currUv = vCurr.xy / vCurr.w * 0.5 + 0.5;
    fragMotion = vPrev.w > 1e-4 ? (vPrev.xy / vPrev.w * 0.5 + 0.5) - currUv : vec2(1000.0, 1000.0);
}
)GLSL";
}  // namespace

bool RtaoPrepass::initParticleMotion() {
    if (partState_) return partState_ > 0;
    partState_ = -1;
    if (!initialized_) return false;
    GLuint vs = 0, fs = 0;
    std::string err;
    if (!compile(GL_VERTEX_SHADER, kParticleMotionVs, vs, &err)) return false;
    if (!compile(GL_FRAGMENT_SHADER, kParticleMotionFs, fs, &err)) { glDeleteShader_(vs); return false; }
    partProgram_ = glCreateProgram_();
    glAttachShader_(partProgram_, vs);
    glAttachShader_(partProgram_, fs);
    glLinkProgram_(partProgram_);
    glDeleteShader_(vs);
    glDeleteShader_(fs);
    GLint linked = 0;
    glGetProgramiv_(partProgram_, GL_LINK_STATUS, &linked);
    if (!linked) { glDeleteProgram_(partProgram_); partProgram_ = 0; return false; }
    partUCamProjLoc_ = glGetUniformLocation_(partProgram_, "uCamProj");
    partUCurrLoc_ = glGetUniformLocation_(partProgram_, "uCurrUnjittered");
    partUPrevLoc_ = glGetUniformLocation_(partProgram_, "uPrevUnjittered");
    partUTexLoc_ = glGetUniformLocation_(partProgram_, "uTex");
    partUMinLoc_ = glGetUniformLocation_(partProgram_, "uMinCoverage");

    GLint prevVao = 0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
    localGenVertexArrays(1, &partVao_);
    glGenBuffers_(1, &partVbo_);
    localBindVertexArray(partVao_);
    glBindBuffer_(GL_ARRAY_BUFFER, partVbo_);
    const GLsizei stride = 14 * sizeof(float);
    const int sizes[5] = { 3, 3, 2, 4, 2 };
    for (int a = 0, off = 0; a < 5; off += sizes[a], ++a) {
        glVertexAttribPointer_(a, sizes[a], GL_FLOAT, GL_FALSE, stride, reinterpret_cast<const void*>(static_cast<uintptr_t>(off) * sizeof(float)));
        glEnableVertexAttribArray_(a);
    }
    localBindVertexArray(static_cast<GLuint>(prevVao));
    glBindBuffer_(GL_ARRAY_BUFFER, 0);
    glGenFramebuffers_(1, &partFbo_);
    partState_ = 1;
    return true;
}

bool RtaoPrepass::renderParticleMotion(uint32_t width, uint32_t height, uint32_t motionGlTexture, uint32_t sceneDepthGlTexture,
                                        const float* verts, uint32_t vertexCount, const ParticleRun* runs, uint32_t runCount,
                                        const float* camProjFlat16, const float* currentVpUnjitteredFlat16,
                                        const float* prevVpUnjitteredFlat16, float minCoverage) {
    if (vertexCount == 0 || runCount == 0 || !initParticleMotion()) return false;

    GLint prevFbo = 0, prevViewport[4]{}, prevVao = 0, prevDepthFunc = GL_LESS, prevActive = GL_TEXTURE0, prevTex0 = 0, prevProgram = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
    glGetIntegerv(GL_VIEWPORT, prevViewport);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
    glGetIntegerv(GL_DEPTH_FUNC, &prevDepthFunc);
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
    GLboolean prevDepthMask = GL_TRUE;
    glGetBooleanv(GL_DEPTH_WRITEMASK, &prevDepthMask);
    const GLboolean prevDepthTest = glIsEnabled(GL_DEPTH_TEST), prevBlend = glIsEnabled(GL_BLEND), prevCull = glIsEnabled(GL_CULL_FACE);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActive);
    glActiveTexture_(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);

    glBindFramebuffer_(GL_FRAMEBUFFER, partFbo_);
    glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, motionGlTexture, 0);
    glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, sceneDepthGlTexture, 0);
    const GLenum buf = GL_COLOR_ATTACHMENT0;
    glDrawBuffers_(1, &buf);
    bool drew = false;
    if (glCheckFramebufferStatus_(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
        glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_FALSE);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glUseProgram_(partProgram_);
        glUniformMatrix4fv_(partUCamProjLoc_, 1, GL_FALSE, camProjFlat16);
        glUniformMatrix4fv_(partUCurrLoc_, 1, GL_FALSE, currentVpUnjitteredFlat16);
        glUniformMatrix4fv_(partUPrevLoc_, 1, GL_FALSE, prevVpUnjitteredFlat16);
        glUniform1i_(partUTexLoc_, 0);
        glUniform1f_(partUMinLoc_, minCoverage);
        localBindVertexArray(partVao_);
        glBindBuffer_(GL_ARRAY_BUFFER, partVbo_);
        glBufferData_(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(float)) * 14 * vertexCount, verts, GL_STREAM_DRAW);
        glBindBuffer_(GL_ARRAY_BUFFER, 0);
        for (uint32_t i = 0; i < runCount; ++i) {
            glBindTexture(GL_TEXTURE_2D, runs[i].glTexture);
            glDrawArrays(GL_TRIANGLES, static_cast<GLint>(runs[i].first), static_cast<GLsizei>(runs[i].count));
        }
        drew = true;
    }
    // detach the scene depth (it's the scene target's attachment too)
    glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

    localBindVertexArray(static_cast<GLuint>(prevVao));
    glUseProgram_(static_cast<GLuint>(prevProgram));
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex0));
    glActiveTexture_(static_cast<GLenum>(prevActive));
    glBindFramebuffer_(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo));
    glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
    glDepthMask(prevDepthMask);
    glDepthFunc(static_cast<GLenum>(prevDepthFunc));
    if (!prevDepthTest) glDisable(GL_DEPTH_TEST);
    if (prevBlend) glEnable(GL_BLEND);
    if (prevCull) glEnable(GL_CULL_FACE);
    return drew;
}

}  // namespace sauerinterop
