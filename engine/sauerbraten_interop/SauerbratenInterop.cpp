#ifndef NTDDI_VERSION
#define NTDDI_VERSION 0x0A000000  // NTDDI_WIN10
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00  // _WIN32_WINNT_WIN10
#endif
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <new>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <unordered_set>

// void-and-cluster blue noise (Ulichney 1993)
static std::vector<uint32_t> generateBlueNoiseRanks(int size, uint32_t seed) {
    const int n = size * size;
    const float sigma = 1.5f;
    std::vector<float> lut(n);
    for (int y = 0; y < size; ++y) for (int x = 0; x < size; ++x) {
        const int dx = x <= size / 2 ? x : size - x, dy = y <= size / 2 ? y : size - y;
        lut[y * size + x] = std::exp(-float(dx * dx + dy * dy) / (2.0f * sigma * sigma));
    }
    std::vector<uint8_t> bits(n, 0);
    std::vector<float> energy(n, 0.0f);
    auto splat = [&](int idx, float sgn) {
        const int px = idx % size, py = idx / size;
        for (int y = 0; y < size; ++y) {
            const float* row = &lut[((y - py + size) % size) * size];
            for (int x = 0; x < size; ++x) energy[y * size + x] += sgn * row[(x - px + size) % size];
        }
    };
    auto extreme = [&](uint8_t want, bool pickMax) {
        int best = -1; float bestE = 0.0f;
        for (int i = 0; i < n; ++i) {
            if (bits[i] != want) continue;
            if (best < 0 || (pickMax ? energy[i] > bestE : energy[i] < bestE)) { best = i; bestE = energy[i]; }
        }
        return best;
    };
    uint32_t rng = seed ? seed : 1u;
    auto next = [&]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
    const int initial = n / 10;
    for (int placed = 0; placed < initial;) {
        const int i = int(next() % uint32_t(n));
        if (!bits[i]) { bits[i] = 1; splat(i, 1.0f); ++placed; }
    }
    for (int iter = 0; iter < n; ++iter) {  // relax the initial pattern
        const int c = extreme(1, true);
        bits[c] = 0; splat(c, -1.0f);
        const int v = extreme(0, false);
        bits[v] = 1; splat(v, 1.0f);
        if (v == c) break;
    }
    std::vector<uint32_t> rank(n, 0);
    const std::vector<uint8_t> protoBits = bits;
    const std::vector<float> protoEnergy = energy;
    for (int r = initial - 1; r >= 0; --r) {  // phase 1
        const int c = extreme(1, true);
        bits[c] = 0; splat(c, -1.0f);
        rank[c] = uint32_t(r);
    }
    bits = protoBits; energy = protoEnergy;
    for (int r = initial; r < n; ++r) {  // phases 2+3
        const int v = extreme(0, false);
        bits[v] = 1; splat(v, 1.0f);
        rank[v] = uint32_t(r);
    }
    return rank;
}
#define SAUERINTEROP_STL_PLACEMENT_NEW

#include "SauerbratenInterop.h"
#include "engine.h"  // Sauerbraten
                     // worldsize, VAR/FVAR/command, and (transitively, via shared/cube.h) both the
                     // full GL constant set

#include "WorldGeometryExtract.h"
#include "MapModelExtract.h"
#include "GlassExtract.h"
#include "WaterExtract.h"
#include "AnimatedModelExtract.h"
#include "RtaoPrepass.h"
extern int ptlitparticlecount();
extern vec camright, camup;

extern vec sunlightdir;

extern bvec sunlightcolor;
extern float sunlightscale;
extern bvec skylightcolor;
extern int sunlight;
extern bvec ambientcolor;
// rendersky.cpp's globals
extern Texture *sky[6];
extern float spinsky;
extern int yawsky;

extern float nearplane;
extern int farplane;
extern int fsaa;

#include "InteropContext.h"
#include "InteropFrame.h"
#include "RayTracingScene.h"
#include "SharedTexture.h"
#include "SharedTexture3D.h"
#include "SharedSemaphore.h"
#include "gl_ext_loader.h"
#include <dbghelp.h>  // GL error capture
#pragma comment(lib, "dbghelp.lib")

#include <cstring>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

extern int framestats;  // main.cpp, frame stats
namespace sauerinterop {

namespace {
constexpr uint32_t kPomHeightGlIdBit = 0x40000000u;

std::vector<unsigned char> buildHeightMaxMipChain(const std::vector<unsigned char>& rgba, uint32_t w, uint32_t h)
{
    std::vector<unsigned char> cur(size_t(w) * h), out;
    for (size_t i = 0; i < cur.size(); ++i) cur[i] = rgba[i * 4 + 3];
    uint32_t cw = w, ch = h;
    const uint32_t levels = 1u + static_cast<uint32_t>(std::floor(std::log2(static_cast<double>(std::max(w, h)))));
    for (uint32_t l = 0; l < levels; ++l) {
        for (unsigned char v : cur) { out.push_back(v); out.push_back(v); out.push_back(v); out.push_back(v); }
        if (l + 1 == levels) break;
        const uint32_t nw = std::max(1u, cw >> 1), nh = std::max(1u, ch >> 1);
        std::vector<unsigned char> next(size_t(nw) * nh);
        for (uint32_t y = 0; y < nh; ++y)
            for (uint32_t x = 0; x < nw; ++x) {
                // source footprint
                const uint32_t x0 = x * cw / nw, x1 = std::max(x0 + 1, ((x + 1) * cw + nw - 1) / nw);
                const uint32_t y0 = y * ch / nh, y1 = std::max(y0 + 1, ((y + 1) * ch + nh - 1) / nh);
                unsigned char m = 0;
                for (uint32_t sy = y0; sy < y1 && sy < ch; ++sy)
                    for (uint32_t sx = x0; sx < x1 && sx < cw; ++sx) m = std::max(m, cur[size_t(sy) * cw + sx]);
                next[size_t(y) * nw + x] = m;
            }
        cur.swap(next); cw = nw; ch = nh;
    }
    return out;
}

bool readTextureForRayTracing(GLuint id, std::vector<unsigned char>& rgba, uint32_t& width, uint32_t& height)
{
    width = height = 0;
    rgba.clear();
    if(!id) return false;
    GLint prev = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev);
    glBindTexture(GL_TEXTURE_2D, id);
    GLint w = 0, h = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);
    if(w <= 0 || h <= 0 || w > 8192 || h > 8192)
    {
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prev));
        return false;
    }
    const size_t pixelCount = static_cast<size_t>(w) * static_cast<size_t>(h);
    const size_t rowBytes = static_cast<size_t>(w) * 4u;
    rgba.resize(rowBytes * static_cast<size_t>(h));

    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());

    if(hasTSW)
    {
        GLint swizzle[4] = { GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA };
        glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzle);

        const bool identity =
            swizzle[0] == GL_RED   && swizzle[1] == GL_GREEN &&
            swizzle[2] == GL_BLUE  && swizzle[3] == GL_ALPHA;

        if(!identity)
        {
            GLint greenBits = 0, blueBits = 0, alphaBits = 0;
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_GREEN_SIZE, &greenBits);
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_BLUE_SIZE, &blueBits);
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_ALPHA_SIZE, &alphaBits);

            const std::vector<unsigned char> source = rgba;

            auto component = [&](const unsigned char *p, GLint selector) -> unsigned char
            {
                switch(selector)
                {
                    case GL_RED:   return p[0];
                    case GL_GREEN:
                        // for GL_RG, the second stored component is returned in the alpha slot of an
                        // RGBA readback
                        return (greenBits > 0 && blueBits == 0 && alphaBits == 0) ? p[3] : p[1];
                    case GL_BLUE:  return p[2];
                    case GL_ALPHA: return p[3];
                    case GL_ZERO:  return 0;
                    case GL_ONE:   return 255;
                    default:       return 0;
                }
            };

            for(size_t i = 0; i < pixelCount; ++i)
            {
                const unsigned char *src = &source[i * 4u];
                unsigned char *dst = &rgba[i * 4u];
                dst[0] = component(src, swizzle[0]);
                dst[1] = component(src, swizzle[1]);
                dst[2] = component(src, swizzle[2]);
                dst[3] = component(src, swizzle[3]);
            }
        }
    }
    // do not vertically flip this readback
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prev));
    width = static_cast<uint32_t>(w);
    height = static_cast<uint32_t>(h);
    return true;
}
}


    namespace {

        VAR(rtaodebug, 0, 0, 3);
        // gates the "DIAG lastRigidDiag changed" logmsg block (below, near scene_->build() in the
        // path-trace frame)
        VAR(rtrigiddiag, 0, 0, 1);
        FVAR(rtaoradius, 0.1f, 16.0f, 256.0f);
        FVAR(rtaointensity, 0.0f, 1.0f, 4.0f);
        FVAR(rtaobias, 0.0001f, 0.8f, 4.0f);
        VAR(rtaorays, 1, 8, 32);
        VAR(rtaoblend, 0, 1, 1);

        VAR(rtaojitter, 0, 1, 1);  // 0 = no camera jitter ; 1 = per-frame Halton(2,3) sub-pixel jitter
        VAR(rtaojitterphases, 2, 8, 64);  // Halton cycle length before it repeats, 8 is a common TAA/DLSS default
        FVAR(rtaojitterscale, 0.0f, 1.0f, 4.0f);
        FVAR(pathtracejittermagnitude, 0.0f, 1.0f, 1.0f);
        // sky light
        FVAR(pathtraceskylight, 0.0f, 1.0f, 8.0f);
        // without DLSS: path-trace sub-pixel jitter width (anti-aliasing filter width)
        FVAR(pathtraceaajitter, 0.0f, 0.5f, 1.0f);
        // first-person weapon/hands in the path tracer
        FVAR(pathtraceviewmodelscale, 0.05f, 0.5f, 1.0f);
        // diagnostic
        VAR(pathtracenorefit, 0, 0, 1);
        VAR(pathtracepom, 0, 1, 1);
        VAR(pathtracepomsteps, 4, 64, 256);  // max height-field march steps (Remix default 64)
        VAR(pathtracepommode, 0, 1, 1);
        // glass-volume (MAT_GLASS) tint strength toward the glass colour's hue
        VAR(ptglass, 0, 1, 1);
        FVAR(pathtraceglasstint, 0.0f, 0.6f, 1.0f);
        // path-traced dynamic lights (muzzle flashes, rockets, explosions)
        FVAR(ptdynlightradius, 0.0f, 1.0f, 16.0f);
        FVAR(ptdynlightbrightness, 0.0f, 1.0f, 16.0f);
        // model environment reflection
        FVAR(ptmodelreflect, 0.0f, 0.5f, 1.0f);  // = reflectance
        // caustics water caustics, hybrid bidirectional
        VAR(ptcaustics, 0, 1, 1);
        FVAR(ptcausticradius, 0.1f, 0.6f, 16.0f);
        VAR(ptcausticphotons, 8, 512, 2048);
        FVAR(ptcausticrange, 16.0f, 256.0f, 1024.0f);
        FVAR(ptcausticripple, 0.0f, 1.0f, 4.0f);
        // alpha-geometry (textured) glass tint strength toward its texture colour
        FVAR(pathtracealphaglasstint, 0.0f, 1.0f, 1.0f);
        FVAR(pathtracepomdepth, 0.0f, 1.0f, 8.0f);  // multiplier on every texture's parallax depth
        // path-traced samples (full light paths) per pixel per frame
        VAR(pathtracespp, 1, 1, 16);
        VAR(pathtracelinearlights, 0, 0, 1);
        VAR(pathtracelightclamp, 0, 0, 1);
        // specular strength (0..1) for plain textures
        FVAR(pathtraceplainspec, 0.0f, 0.0f, 1.0f);
        VAR(pathtracebluenoise, 0, 1, 1);
        FVAR(pathtracerigidinflate, 0.0f, 0.0f, 2.0f);
        VAR(rtaotemporal, 0, 1, 1);

        // ray-traced sun shadow ---- Toggles the shadow-ray-tracing half of rtao.comp's existing
        // dispatch there
        VAR(rtaoshadow, 0, 0, 1);
        // angular radius of the sun disk, tangent-of-half-angle units
        FVAR(rtaoshadowradius, 0.0f, 0.04f, 0.5f);
        // how strongly the ray-traced directional shadow darkens the existing scene
        FVAR(rtaoshadowintensity, 0.0f, 1.0f, 4.0f);
        VAR(rtaoshadowrays, 1, 1, 8);

        // path tracing (simple mode separate from RTAO/shadow above) ---- Master on/off
        VAR(pathtrace, 0, 1, 1);
        // indirect bounces past the primary (camera-visible) hit
        VAR(pathtracebounces, 0, 2, 8);
        // angular radius of the sun disk for path-traced NEE shadow rays
        FVAR(pathtracesunangle, 0.0f, 0.02f, 0.5f);
        FVAR(pathtraceglassthinness, 0.0f, 1.0f, 1.0f);
        VAR(pathtracemaxlights, 1, 2000, 8192);
        // caps the sample-count-based accumulation weight
        VAR(pathtracemaxhistory, 1, 32, 256);
        FVAR(pathtraceexposure, 0.05f, 1.0f, 20.0f);
        // switches which tonemap operator kPathTraceCompositeFs uses
        VAR(pathtracehdr, 0, 0, 2);
        // hdr display pipeline (SDR monitors)
        VAR(pthdr, 0, 0, 1);
        VAR(pthdrcurve, 0, 1, 1);
        FVAR(pthdrev, -4.0f, 0.0f, 4.0f);  // exposure compensation, EV (stops)
        FVAR(pthdrkey, 0.02f, 0.18f, 1.0f);
        FVAR(pthdradapt, 0.0f, 0.6f, 1.0f);
        FVAR(pthdrspeed, 0.1f, 1.5f, 10.0f);  // adaptation speed (1/seconds)
        FVAR(pthdrsky, 0.25f, 2.5f, 8.0f);  // visible-sky brightness (the path tracer shows it at 0.35x its texture)
        FVAR(pathtracecontrast, 0.1f, 1.0f, 3.0f);
        FVAR(pathtracesaturation, 0.0f, 1.0f, 3.0f);
        VAR(pathtracelut, 0, 0, 3);
        FVAR(pathtraceglowscale, 0.0f, 4.0f, 64.0f);

        VAR(pathtracebloom, 0, 1, 1);
        FVAR(pathtracebloomthreshold, 0.0f, 1.0f, 16.0f);
        // additive strength of the blurred bloom result composited back onto the scene
        FVAR(pathtracebloomintensity, 0.0f, 0.6f, 4.0f);

        VAR(volfog, 0, 0, 1);
        VAR(volfogdebug, 0, 0, 10);
        // base atmospheric extinction/scattering coefficient at fogBaseHeight, in world-units^-1
        FVAR(volfogdensity, 0.0f, 0.05f, 1.0f);
        FVAR(volfogsun, 0.0f, 1.0f, 8.0f);
        // map lights, dynamic lights and glowing surfaces in the fog
        FVAR(volfoglights, 0.0f, 1.0f, 16.0f);
        VAR(volfoglightexact, 0, 2, 4);
        VAR(volfoglightsamples, 1, 1, 4);
        // 1 = glowing surfaces (glow maps, lava) light the fog with multiple importance sampling
        VAR(volfogglowmis, 0, 1, 1);
        VAR(ptstats, 0, 0, 1);
        // + a per-section split of the trace pass's GPU time (shader clock, needs VK_KHR_shader_clock)
        VAR(ptprofile, 0, 0, 1);
        // 1 = the ray-tracing interop's messages also go to the in-game console
        VARP(ptconsole, 0, 0, 1);
        // fire/flame particle entities light the scene (0 = off), their falloff radius scale, and
        // flicker amount
        FVAR(ptflamelight, 0.0f, 1.0f, 8.0f);
        FVAR(ptflameradius, 0.1f, 1.0f, 8.0f);
        FVAR(ptflameflicker, 0.0f, 0.15f, 1.0f);
        VAR(ptpartlit, 0, 1, 2);
        FVAR(ptpartreflect, 0.0f, 1.0f, 8.0f);
        FVAR(ptpartlightgain, 0.0f, 1.0f, 8.0f);
        // particle motion vectors for DLSS (4.5 SR/RR and 5)
        VAR(partmv, 0, 1, 1);
        // Sauerbraten's grass in the path tracer
        VAR(ptgrass, 0, 1, 1);
        FVAR(ptgrasscutoff, 0.05f, 0.35f, 0.95f);
        FVAR(partmvmin, 0.0f, 0.2f, 1.0f);
        // 1 = bounce hits trace one light picked by its share of the light there (unbiased), not every
        // light in range
        VAR(ptlightsample, 0, 1, 1);
        VAR(ptlightgrid, 0, 1, 1);
        void rebuildGeometry(bool engineJustRebuilt);
        VARF(ptwater, 0, 1, 1, rebuildGeometry(false));
        FVAR(ptwaterclarity, 0.05f, 2.0f, 50.0f);
        // wave amplitude (geometry + normals, everywhere)
        FVAR(ptwaterwaves, 0.0f, 1.0f, 8.0f);
        // path-traced waves all on the GPU (rt_water.glsl)
        FVAR(ptwind, 0.5f, 5.0f, 30.0f);  // wind speed, m/s, bigger, longer, rougher waves
        FVAR(ptwinddir, 0.0f, 30.0f, 360.0f);  // wind direction, degrees
        FVAR(ptfetch, 10.0f, 4000.0f, 100000.0f);
        FVAR(ptspread, 0.5f, 6.0f, 64.0f);  // directional spread
        FVAR(ptchop, 0.0f, 0.7f, 2.0f);  // choppiness
        FVAR(ptwavesize, 4.0f, 50.0f, 400.0f);  // largest wave tile, metres (the others are 1/3.7, 1/13.9 of it)
        FVAR(ptwavespeed, 0.0f, 1.0f, 4.0f);  // time scale
        FVAR(ptwavedetail, 0.0f, 1.0f, 4.0f);  // finest cascade (small ripples, caustic networks)
        VAR(ptwavedepth, 0, 1, 1);  // 1 = depth + fetch aware (pools / shallows get small waves only)
        VAR(ptwavestats, 0, 0, 1);  // 1 = log each wave-map bake's depths and wave weights (stderr)
        // wave geometry around the camera, generated + refit entirely on the GPU every frame
        // (water_patch.comp)
        VAR(ptwaves, 0, 1, 1);  // 0 = flat water geometry (normals only)
        FVAR(ptwaveheight, 0.0f, 1.0f, 8.0f);  // patch displacement scale (geometry only)
        FVAR(ptwaverange, 32.0f, 128.0f, 512.0f);  // patch half-size around the camera, world units
        FVAR(ptwavegrid, 1.0f, 2.0f, 8.0f);  // grid spacing, world units (rounded down to 1/2/4/8)
        FVAR(ptwavefade, 1.0f, 24.0f, 128.0f);  // fade to flat at the patch edge, world units
        FVAR(ptwaterscatter, 0.0f, 0.5f, 8.0f);
        // single scattering of the sun and the lights in the water
        FVAR(ptwaterrays, 0.0f, 0.002f, 0.1f);  // scattering per world unit, ray strength (0 = off)
        FVAR(ptwaterraysg, 0.0f, 0.7f, 0.95f);
        FVAR(ptwaterfocus, 0.0f, 1.0f, 2.0f);  // wave focusing, the bright/dark streaks (0 = smooth shafts only)
        VAR(ptwaterhistory, 1, 6, 256);
        // colour-codes the camera path's first underwater segment (pathtrace_trace.comp waterDebug)
        VAR(ptwaterdebug, 0, 0, 1);
        // radiance cache (multi-bounce GI)
        VAR(ptcache, 0, 1, 1);
        FVAR(ptcachecell, 0.5f, 4.0f, 64.0f);  // base cell size, doubles with distance
        VAR(ptcacheframes, 1, 32, 256);
        VAR(ptcacheupdate, 1, 16, 256);  // 1 in N pixels skip the cache at the first bounce
        VAR(ptcachefallback, 0, 1, 1);
        // pick the cache's update paths (1 in ptcacheupdate) per 8x4 pixel block
        VAR(ptcachecoherent, 0, 1, 1);
        FVAR(ptcachefloor, 0.0f, 0.35f, 1.0f);
        VAR(ptsky, 0, 1, 1);
        VAR(ptmat, 0, 0, 1);
        VAR(ptalbedo, 0, 1, 1);
        FVAR(ptalbedogamma, 1.0f, 1.6f, 2.2f);
        VARF(ptlava, 0, 1, 1, rebuildGeometry(false));
        // lava glow strength (times pathtraceglowscale)
        FVARF(ptlavaglow, 0.0f, 1.0f, 32.0f, rebuildGeometry(false));
        // lava flows (its texture scrolls, like the raster's)
        VARF(ptlavascroll, 0, 1, 1, rebuildGeometry(false));
        VARF(ptlavaprepass, 0, 0, 1, rebuildGeometry(false));
        // 1 = blackbody lava
        VARF(ptlavamode, 0, 1, 1, rebuildGeometry(false));
        VARF(ptlavadisplace, 0, 1, 1, rebuildGeometry(false));
        FVAR(ptlavasize, 4.0f, 24.0f, 256.0f);  // crust plate size, world units
        FVAR(ptlavaflow, 0.0f, 1.0f, 16.0f);  // flow speed, world units/s (the raster lava's is 1)
        FVAR(ptlavaspeed, 0.0f, 1.0f, 8.0f);  // animation speed (plate drift, churn)
        FVAR(ptlavarelief, 0.0f, 2.0f, 8.0f);  // crust relief, world units
        FVAR(ptlavacrack, 0.01f, 0.08f, 0.5f);  // crack width (fraction of a plate)
        FVAR(ptlavatemp, 800.0f, 1100.0f, 2500.0f);
        FVAR(ptlavacrust, 300.0f, 650.0f, 1100.0f);  // crust temperature, K (higher = crust glows dull red)
        FVAR(ptlavacrustamt, 0.0f, 0.45f, 1.0f);  // how much of the surface is crust (0 = all melt, 1 = all crust)
        void lavaTintFromMap(float& hue, float& sat)
        {
            auto yiq = [](const bvec& c, float& y, float& i, float& q)
            {
                const float r = powf(c.x / 255.0f, 2.2f), g = powf(c.y / 255.0f, 2.2f), b = powf(c.z / 255.0f, 2.2f);  // linear, like the blackbody
                y = 0.299f * r + 0.587f * g + 0.114f * b;
                i = 0.596f * r - 0.274f * g - 0.322f * b;
                q = 0.211f * r - 0.523f * g + 0.312f * b;
            };
            float y0, i0, q0, y1, i1, q1;
            yiq(bvec(0xFF, 0x40, 0x00), y0, i0, q0);
            yiq(getlavacolor(MAT_LAVA), y1, i1, q1);
            const float c0 = sqrtf(i0 * i0 + q0 * q0), c1 = sqrtf(i1 * i1 + q1 * q1);
            hue = c1 > 1e-5f ? atan2f(q1, i1) - atan2f(q0, i0) : 0.0f;  // a grey lavacolour
            sat = std::min((c1 / std::max(y1, 1e-4f)) / (c0 / y0), 1.5f);
        }
        // multiplier on the light glowing models cast (teleporters, mdlptemissive's light), 0 = off
        FVAR(ptmodellight, 0.0f, 1.0f, 64.0f);
        VAR(ptlightexact, 0, 4, 8);
        VAR(ptlightrandom, 0, 0, 15);
        // ris light sampling optional, for slower GPUs
        VAR(ptris, 0, 2, 2);
        VAR(ptriscandidates, 1, 8, 63);
        VAR(ptrissamples, 1, 4, 4);
        VAR(ptrestir, 0, 2, 2);
        VAR(ptrestirmin, 0, 16, 256);
        // 0 compute trace, 1 (default) the same shader from a ray-tracing pipeline
        VAR(ptraygen, 0, 1, 2);
        VAR(ptraygenstack, 0, 0, 65536);  // diagnostic
        VAR(ptrestircandidates, 1, 4, 32);
        VAR(ptrestirspatial, 0, 2, 4);
        // RIS at bounce hits its budget (ptrissamples/ptriscandidates are the camera hit's)
        VAR(ptrisbouncesamples, 1, 1, 4);
        VAR(ptrisbouncecandidates, 1, 4, 63);
        VAR(ptrisgrid, 0, 1, 1);
        // truncate the trace pass
        VAR(ptcut, 0, 0, 7);
        VAR(ptcutshadow, 0, 0, 1);  // skip the camera hit's point-light shadow rays
        FVAR(volfogambientfloor, 0.0f, 0.15f, 1.0f);
        VAR(volfogreproject, 0, 1, 1);
        FVAR(volfogheight, -2048.0f, 0.0f, 2048.0f);
        // exponential height-fog falloff rate
        FVAR(volfogfalloff, 0.0f, 0.001f, 1.0f);
        // Henyey-Greenstein phase function asymmetry `g`
        FVAR(volfogphase, -0.99f, 0.35f, 0.99f);
        VAR(volfognoise, 0, 1, 1);
        // interleaved ray-traced sun-shadow refresh period
        VAR(volfogshadowfreq, 1, 4, 16);
        FVAR(volfogdist, 64.0f, 1024.0f, 16384.0f);
        // how far volumetric fog is visible (world units), sky included
        FVAR(volfogfar, 0.0f, 8192.0f, 65536.0f);
        // quality preset
        VAR(volfogres, 0, 1, 2);
        // skip froxels behind whatever the camera can see (a cheap per-column depth pass first)
        VAR(volfogcull, 0, 1, 1);
        // light only 1 in N froxels per frame (rotating), the rest reuse last frame's
        VAR(volfogrelight, 1, 4, 8);

        VAR(dlssquality, 0, 1, 4);
        VAR(dlssdebug, 0, 0, 1);
        // overlays the motion vectors DLSS and the temporal resolve read
        VAR(mvdebug, 0, 0, 3);
        VAR(ptsurfacemotion, 0, 1, 1);
        // 1 = DLSS-RR gets the path-traced water's own normal/roughness/albedo guides
        // (runRrWaterGuides)
        VAR(rrwaterguides, 0, 1, 1);
        FVAR(mvdebugscale, 0.25f, 16.0f, 1000.0f);  // on-screen pixels of motion per frame that reach full colour
        VAR(mvdebuggrid, 0, 32, 256);  // arrow grid spacing in screen pixels, 0 = no arrows
        FVAR(mvdebuggain, 0.1f, 4.0f, 100.0f);
        FVAR(mvdebugopacity, 0.0f, 0.85f, 1.0f);  // colour field over the scene, 0 = arrows only

        VAR(dlssscene, 0, 1, 1);
        // DLSS-on-final-frame, render resolution as a fraction of output resolution
        FVAR(dlssscenescale, 0.1f, 1.0f, 1.0f);

        VAR(dlssrr, 0, 1, 1);

        // Streamline 2.12.0 DLSSMode ordinals
        const char* kDlssModeLabels[5] = { "DLAA", "Quality", "Balanced", "Performance", "Ultra Performance" };
        const int kDlssModeOrdinals[5] = { 6, 3, 2, 1, 4 };

        void logmsg(const std::string& msg);
        // chatty per-frame / per-model messages, only with ptverbose 1
        VAR(ptverbose, 0, 0, 1);
        void vlogmsg(const std::string& msg) { if (ptverbose) logmsg(msg); }


        std::unordered_set<dynent*> rtFreshDynents_;

        struct RtMaskRect { float minX, minY, maxX, maxY; };
        std::vector<RtMaskRect> rtMaskRectsNdc_;

        struct AnimatedPoseSlot {
            uint64_t modelKey = 0;
            std::vector<float> bones[2];  // boneCount*8 floats each, ping-pong
            float worldMatrix[2][12] = {};
            int current = 0;
            bool hasPrev = false;
            uint64_t lastUpdatedFrame = 0;
        };
        std::unordered_map<uint64_t, AnimatedPoseSlot> animPoseHistoryDynent_;
        // animated ET_MAPMODEL entities have no dynent at all (currentDynent_ is nullptr for them)
        std::vector<AnimatedPoseSlot> animPoseHistoryMapmodel_;
        std::unordered_map<uint64_t, AnimatedPoseSlot> animPoseHistoryViewModel_;  // by modelKey, gun and hands are separate parts
        // monotonic, this-file-frame counter
        uint64_t animPoseFrameCounter_ = 0;
        // per-frame ordinal
        int animMapmodelOrdinal_ = 0;
        // this frame's list of animated instances to draw for motion vectors
        struct AnimatedDrawInstance {
            uint64_t modelKey = 0;
            std::vector<float> currentBones;
            float currentWorldMatrix[12] = {};
            std::vector<float> prevBones;
            float prevWorldMatrix[12] = {};
        };
        std::vector<AnimatedDrawInstance> animDrawList_;
        // per-frame-drained queue of GL-side bind-pose uploads for
        // RtaoPrepass::uploadAnimatedSkinSource()
        struct PendingAnimSkinUpload {
            uint64_t modelKey = 0;
            std::vector<uint8_t> vertexData;  // raw bytes, vertexStride * vertexCount
            uint32_t vertexStride = 0;
            uint32_t vertexCount = 0;
            std::vector<uint32_t> indices;
            std::unordered_map<uint32_t, uint32_t> glIdForIndex;
        };
        std::vector<PendingAnimSkinUpload> animPendingSkinUploads_;
        void captureSkinUploadGlIds(PendingAnimSkinUpload& pending, const SkinSourceVertexData* verts, uint32_t count,
                                    interop::RayTracingScene* sc) {
            if (!sc) return;
            auto add = [&](uint32_t idx) {
                if (idx == 0u || pending.glIdForIndex.count(idx)) return;
                const uint32_t gl = sc->pathTraceTextureGlId(idx);
                pending.glIdForIndex[idx] = gl < 0x40000000u ? gl : 0u;
            };
            for (uint32_t vi = 0; vi < count; ++vi) {
                add(verts[vi].textureIndex);
                const uint32_t m = verts[vi].materialIndex;
                if ((m & 0x80000000u) != 0u && m != 0xFFFFFFFFu) add(m & 0xFFFFu);
            }
        }
        struct RigidPrevTransform {
            float worldMatrix[12] = {};
            uint64_t frame = 0;
        };
        std::unordered_map<uint64_t, RigidPrevTransform> rigidPrevTransforms_;

        struct RayTracingSkinSink : SkinExtractSink {
            interop::RayTracingScene* scene;
            interop::InteropContext* ctx;
            RayTracingSkinSink(interop::RayTracingScene* s, interop::InteropContext* c) : scene(s), ctx(c) {}

            std::vector<RtMaskRect>* maskRects = nullptr;

            dynent* currentDynent_ = nullptr;
            bool lastRigidAccepted_ = false;
            bool lastPoseAccepted_ = false;
            void beginInstance(dynent* d) override { currentDynent_ = d; lastRigidAccepted_ = false; lastPoseAccepted_ = false; }
            bool lastRigidInstanceAccepted() override { return lastRigidAccepted_; }
            bool lastPoseAccepted() override { return lastPoseAccepted_; }

            bool hasSkinSource(uint64_t modelKey, uint32_t boneCount) override {
                return scene->hasSkinSource(modelKey, boneCount);
            }
            void submitSkinSource(uint64_t modelKey, const SkinSourceVertexData* vertices, uint32_t vertexCount,
                                   const uint32_t* indices, uint32_t indexCount, uint32_t boneCount) override {
                static_assert(sizeof(SkinSourceVertexData) == sizeof(interop::RayTracingScene::SkinSourceVertex),
                              "SkinSourceVertexData (model.h) and RayTracingScene::SkinSourceVertex must stay layout-identical");
                const auto* nativeVerts = reinterpret_cast<const interop::RayTracingScene::SkinSourceVertex*>(vertices);

                // tee the same already-built bind-pose vertex/index arrays into
                // animPendingSkinUploads_
                {
                    PendingAnimSkinUpload pending;
                    pending.modelKey = modelKey;
                    pending.vertexStride = static_cast<uint32_t>(sizeof(SkinSourceVertexData));
                    pending.vertexCount = vertexCount;
                    pending.vertexData.assign(reinterpret_cast<const uint8_t*>(vertices),
                                               reinterpret_cast<const uint8_t*>(vertices) + static_cast<size_t>(sizeof(SkinSourceVertexData)) * vertexCount);
                    pending.indices.assign(indices, indices + indexCount);
                    captureSkinUploadGlIds(pending, vertices, vertexCount, scene);
                    animPendingSkinUploads_.push_back(std::move(pending));
                }

                std::string err;
                if (!scene->registerSkinSource(modelKey, nativeVerts, vertexCount, indices, indexCount, boneCount, &err)) {
                    logmsg("RayTracingScene::registerSkinSource failed: " + err);
                } else {
                    vlogmsg("RayTracingScene::registerSkinSource OK (modelKey=" + std::to_string(modelKey) +
                        ", boneCount=" + std::to_string(boneCount) + ", vertexCount=" + std::to_string(vertexCount) + ").");
                }
            }
            uint32_t registerTexture(uint32_t textureGlId) override {
                if (textureGlId == 0) return 0;
                uint32_t idx = 0;
                if (scene->isPathTraceTextureRegistered(textureGlId, &idx)) return idx;
                // A texture that already failed (table full on a very big map, or unreadable) isn't
                // retried every frame
                static std::vector<uint32_t> failed;
                static uint64_t failedEpoch = ~0ull;
                if (failedEpoch != scene->geometryEpoch()) { failed.clear(); failedEpoch = scene->geometryEpoch(); }
                if (std::find(failed.begin(), failed.end(), textureGlId) != failed.end()) return 0;
                std::vector<unsigned char> rgba;
                uint32_t tw = 0, th = 0;
                if (!readTextureForRayTracing(static_cast<GLuint>(textureGlId), rgba, tw, th)) {
                    logmsg("AnimatedModelExtract: failed to read GL texture " + std::to_string(textureGlId) + " for path tracing.");
                    failed.push_back(textureGlId);
                    return 0;
                }
                if (!scene->registerPathTraceTexture(textureGlId, tw, th, rgba.data(), rgba.size(), 0, &idx)) {
                    logmsg("AnimatedModelExtract: failed to register GL texture " + std::to_string(textureGlId) + ": " + ctx->lastError());
                    failed.push_back(textureGlId);
                    return 0;
                }
                return idx;
            }
            void submitPose(uint64_t modelKey, const float* boneDualQuats, uint32_t boneCount,
                            const float worldMatrixIn[12]) override {
                // RTX Remix's rtx.viewModel perspective correction
                float worldMatrix[12];
                std::memcpy(worldMatrix, worldMatrixIn, sizeof(worldMatrix));
                const bool isViewModel = rtHudModelPass;
                if (isViewModel) {
                    const float k = tanf(fovy * 0.5f * RAD) / tanf(curavatarfov * 0.5f * RAD);
                    const float s = pathtraceviewmodelscale;
                    matrix4 w;
                    w.a = vec4(worldMatrixIn[0], worldMatrixIn[4], worldMatrixIn[8], 0.0f);
                    w.b = vec4(worldMatrixIn[1], worldMatrixIn[5], worldMatrixIn[9], 0.0f);
                    w.c = vec4(worldMatrixIn[2], worldMatrixIn[6], worldMatrixIn[10], 0.0f);
                    w.d = vec4(worldMatrixIn[3], worldMatrixIn[7], worldMatrixIn[11], 1.0f);
                    matrix4 dm; dm.identity();
                    dm.a.x = k * s; dm.b.y = k * s; dm.c.z = s;
                    matrix4 camW; camW.mul(cammatrix, w);
                    matrix4 dcamW; dcamW.mul(dm, camW);
                    matrix4 corrected; corrected.mul(invcammatrix, dcamW);
                    const float out[12] = {
                        corrected.a.x, corrected.b.x, corrected.c.x, corrected.d.x,
                        corrected.a.y, corrected.b.y, corrected.c.y, corrected.d.y,
                        corrected.a.z, corrected.b.z, corrected.c.z, corrected.d.z };
                    std::memcpy(worldMatrix, out, sizeof(worldMatrix));
                }
                if (maskRects && !currentDynent_ && !isViewModel) {
                    const float cx = worldMatrix[3], cy = worldMatrix[7], cz = worldMatrix[11];
                    const float kHalfExtent = 24.0f;
                    float minX = 1e30f, minY = 1e30f, maxX = -1e30f, maxY = -1e30f;
                    bool any = false;
                    for (int ix = 0; ix < 2; ++ix) for (int iy = 0; iy < 2; ++iy) for (int iz = 0; iz < 2; ++iz) {
                        vec corner(cx + (ix ? kHalfExtent : -kHalfExtent),
                                   cy + (iy ? kHalfExtent : -kHalfExtent),
                                   cz + (iz ? kHalfExtent : -kHalfExtent));
                        vec4 clip;
                        camprojmatrix.transform(corner, clip);
                        if (clip.w <= 0.01f) continue;  // behind the camera, not a screen-space point
                        const float ndcX = clip.x / clip.w, ndcY = clip.y / clip.w;
                        minX = min(minX, ndcX); maxX = max(maxX, ndcX);
                        minY = min(minY, ndcY); maxY = max(maxY, ndcY);
                        any = true;
                    }
                    if (any) maskRects->push_back(RtMaskRect{minX, minY, maxX, maxY});
                }
                {
                    AnimatedPoseSlot* slot = nullptr;
                    if (isViewModel) {
                        slot = &animPoseHistoryViewModel_[modelKey];
                    } else if (currentDynent_) {
                        slot = &animPoseHistoryDynent_[reinterpret_cast<uint64_t>(currentDynent_) ^ (modelKey * 0x9E3779B97F4A7C15ull)];
                    } else {
                        if (static_cast<size_t>(animMapmodelOrdinal_) >= animPoseHistoryMapmodel_.size()) {
                            animPoseHistoryMapmodel_.resize(animMapmodelOrdinal_ + 1);
                        }
                        slot = &animPoseHistoryMapmodel_[animMapmodelOrdinal_];
                        ++animMapmodelOrdinal_;
                    }

                    AnimatedDrawInstance draw;
                    draw.modelKey = modelKey;
                    draw.currentBones.assign(boneDualQuats, boneDualQuats + static_cast<size_t>(boneCount) * 8);
                    std::memcpy(draw.currentWorldMatrix, worldMatrix, sizeof(float) * 12);

                    const bool prevValid = slot->hasPrev && slot->modelKey == modelKey &&
                        slot->lastUpdatedFrame + 1 == animPoseFrameCounter_ &&
                        slot->bones[slot->current].size() == draw.currentBones.size();
                    if (prevValid) {
                        draw.prevBones = slot->bones[slot->current];
                        std::memcpy(draw.prevWorldMatrix, slot->worldMatrix[slot->current], sizeof(float) * 12);
                    }

                    const int nextSlot = 1 - slot->current;
                    slot->bones[nextSlot] = draw.currentBones;
                    std::memcpy(slot->worldMatrix[nextSlot], worldMatrix, sizeof(float) * 12);
                    slot->current = nextSlot;
                    slot->modelKey = modelKey;
                    slot->hasPrev = true;
                    slot->lastUpdatedFrame = animPoseFrameCounter_;

                    animDrawList_.push_back(std::move(draw));
                }

                const uint32_t kAnimatedMeshMaterialSlot = 0x00FFFFFEu;
                if (scene->addSkinnedDynamicMesh(modelKey, boneDualQuats, boneCount, worldMatrix, kAnimatedMeshMaterialSlot,
                                                 isViewModel ? 0x02u : 0xFFu)) {
                    lastPoseAccepted_ = true;
                    if (currentDynent_) rtFreshDynents_.insert(currentDynent_);
                    if (currentDynent_) {
                        static std::unordered_map<uint64_t, int> lastLogMillis;
                        const uint64_t key = reinterpret_cast<uint64_t>(currentDynent_) ^ (modelKey * 0x9E3779B97F4A7C15ULL);
                        int& last = lastLogMillis[key];
                        if (lastmillis - last > 1000) {
                            last = lastmillis;
                            logmsg("addSkinnedDynamicMesh OK for dynent " + std::to_string(reinterpret_cast<uintptr_t>(currentDynent_)) +
                                " (modelKey=" + std::to_string(modelKey) + ", boneCount=" + std::to_string(boneCount) + ").");
                        }
                    }
                } else {
                    vlogmsg("RayTracingScene::addSkinnedDynamicMesh failed: " + ctx->lastError());
                }
            }

            // override
            void submitDynamicMesh(uint64_t modelKey, const SkinSourceVertexData* vertices, uint32_t vertexCount,
                                    const uint32_t* indices, uint32_t indexCount) override {
                if (!vertices || !indices || vertexCount == 0 || indexCount == 0) return;
                static_assert(sizeof(SkinSourceVertexData) == sizeof(interop::RayTracingScene::SkinSourceVertex),
                              "SkinSourceVertexData (model.h) and RayTracingScene::SkinSourceVertex must stay layout-identical");
                const auto* nativeVerts = reinterpret_cast<const interop::RayTracingScene::SkinSourceVertex*>(vertices);
                if (scene->addTexturedDynamicMesh(nativeVerts, vertexCount, indices, indexCount)) {
                    static std::unordered_map<uint64_t, int> lastLogMillis;
                    int& last = lastLogMillis[modelKey];
                    if (lastmillis - last > 1000) {
                        last = lastmillis;
                        vlogmsg("addTexturedDynamicMesh OK for modelKey=" + std::to_string(modelKey) +
                            " (vertexCount=" + std::to_string(vertexCount) + ").");
                    }
                } else {
                    logmsg("RayTracingScene::addTexturedDynamicMesh failed: " + ctx->lastError());
                }
            }

            // perf blas-sharing instancing for rigid (non- deforming) content
            bool hasRigidSource(uint64_t modelKey) override { return scene->hasRigidSource(modelKey); }
            void registerRigidSource(uint64_t modelKey, const SkinSourceVertexData* vertices, uint32_t vertexCount,
                                      const uint32_t* indices, uint32_t indexCount) override {
                if (!vertices || !indices || vertexCount == 0 || indexCount == 0) return;
                static_assert(sizeof(SkinSourceVertexData) == sizeof(interop::RayTracingScene::SkinSourceVertex),
                              "SkinSourceVertexData (model.h) and RayTracingScene::SkinSourceVertex must stay layout-identical");
                std::vector<SkinSourceVertexData> inflated;
                const SkinSourceVertexData* toRegister = vertices;
                if (pathtracerigidinflate > 0.0f && vertexCount > 0) {
                    float minX = vertices[0].px, minY = vertices[0].py, minZ = vertices[0].pz;
                    float maxX = minX, maxY = minY, maxZ = minZ;
                    for (uint32_t i = 1; i < vertexCount; ++i) {
                        const SkinSourceVertexData& v = vertices[i];
                        minX = std::min(minX, v.px); maxX = std::max(maxX, v.px);
                        minY = std::min(minY, v.py); maxY = std::max(maxY, v.py);
                        minZ = std::min(minZ, v.pz); maxZ = std::max(maxZ, v.pz);
                    }
                    const float cx = (minX + maxX) * 0.5f, cy = (minY + maxY) * 0.5f, cz = (minZ + maxZ) * 0.5f;
                    const float dx = (maxX - minX) * 0.5f, dy = (maxY - minY) * 0.5f, dz = (maxZ - minZ) * 0.5f;
                    const float radius = sqrtf(dx * dx + dy * dy + dz * dz);
                    if (radius > 1e-4f) {
                        const float scale = (radius + pathtracerigidinflate) / radius;
                        inflated.assign(vertices, vertices + vertexCount);
                        for (auto& v : inflated) {
                            v.px = cx + (v.px - cx) * scale;
                            v.py = cy + (v.py - cy) * scale;
                            v.pz = cz + (v.pz - cz) * scale;
                        }
                        toRegister = inflated.data();
                    }
                }
                const auto* nativeVerts = reinterpret_cast<const interop::RayTracingScene::SkinSourceVertex*>(toRegister);
                if (scene->registerRigidSource(modelKey, nativeVerts, vertexCount, indices, indexCount)) {
                    vlogmsg("RayTracingScene::registerRigidSource OK (modelKey=" + std::to_string(modelKey) +
                        ", vertexCount=" + std::to_string(vertexCount) + ").");
                    PendingAnimSkinUpload pending;
                    pending.modelKey = modelKey;
                    pending.vertexStride = static_cast<uint32_t>(sizeof(SkinSourceVertexData));
                    pending.vertexCount = vertexCount;
                    pending.vertexData.assign(reinterpret_cast<const uint8_t*>(toRegister),
                        reinterpret_cast<const uint8_t*>(toRegister) + static_cast<size_t>(sizeof(SkinSourceVertexData)) * vertexCount);
                    pending.indices.assign(indices, indices + indexCount);
                    captureSkinUploadGlIds(pending, toRegister, vertexCount, scene);
                    animPendingSkinUploads_.push_back(std::move(pending));
                } else {
                    logmsg("RayTracingScene::registerRigidSource failed: " + ctx->lastError());
                }
            }
            // zero-latency per-frame instance transform
            void submitRigidInstance(uint64_t modelKey, const float worldMatrix[12]) override {
                lastRigidAccepted_ = scene->addRigidInstance(modelKey, worldMatrix);
                // motion vectors
                if (lastRigidAccepted_ && rtPathTraceActive) {
                    const int64_t qx = static_cast<int64_t>(std::floor(worldMatrix[3] + 0.5f));
                    const int64_t qy = static_cast<int64_t>(std::floor(worldMatrix[7] + 0.5f));
                    const uint64_t key = modelKey ^ (static_cast<uint64_t>(qx) * 0x9E3779B97F4A7C15ull) ^ (static_cast<uint64_t>(qy) * 0xC2B2AE3D27D4EB4Full);
                    RigidPrevTransform& hist = rigidPrevTransforms_[key];
                    static const float kIdentityBone[8] = { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                    AnimatedDrawInstance draw;
                    draw.modelKey = modelKey;
                    draw.currentBones.assign(kIdentityBone, kIdentityBone + 8);
                    std::memcpy(draw.currentWorldMatrix, worldMatrix, sizeof(float) * 12);
                    if (hist.frame != 0 && hist.frame + 1 == animPoseFrameCounter_) {
                        draw.prevBones = draw.currentBones;
                        std::memcpy(draw.prevWorldMatrix, hist.worldMatrix, sizeof(float) * 12);
                    }
                    std::memcpy(hist.worldMatrix, worldMatrix, sizeof(float) * 12);
                    hist.frame = animPoseFrameCounter_;
                    animDrawList_.push_back(std::move(draw));
                }
                if (!lastRigidAccepted_) {
                    static std::unordered_map<uint64_t, int> lastLog;
                    int& last = lastLog[modelKey];
                    if (lastmillis - last > 1000) {
                        last = lastmillis;
                        vlogmsg("DIAG addRigidInstance FAILED (modelKey=" + std::to_string(modelKey) + "): " + ctx->lastError());
                    }
                }
            }
        };

        bool attempted_ = false;  // an init attempt has happened at least once
        bool ready_ = false;  // init succeeded and RTAO is currently usable
        bool hasGeometry_ = false;
        // load_world() finished a new map, the next frame rebuilds the ray-tracing geometry (see
        // onMapLoaded())
        bool mapLoadPending_ = false;
        // this frame's volfog dispatch wrote the light volume
        bool partLightThisFrame_ = false;
        // vrough/vmetal edited a material, rebuild next frame (onMaterialsEdited())
        bool materialsEditPending_ = false;
        // clear before the next dispatch (map change, ptcacheclear, or ptcache switched back on)
        bool cacheClearPending_ = false;
        int lastPtCache_ = -1;
        // the yaw each spinning item (teleporter) was last rendered at, keyed by its ground position
        struct ItemYaw { float yaw = 0, rate = 0; int millis = -1; };
        std::unordered_map<uint64_t, ItemYaw> itemYaws_;
        uint64_t itemYawKey(float x, float y) { return (uint64_t(uint32_t(int(floorf(x * 4)))) << 32) | uint32_t(int(floorf(y * 4))); }
        // every entity whose model casts light (mdlptemissive with a light radius)
        struct EmitterLight { int ent; vec offset; float radius, gap; vec color; vec localNormal; float yawOffset, staticYaw; bool spins; };
        std::vector<EmitterLight> emitterLights_;
        // this, not scene_->isBuilt(), is onFrame()'s gate for attempting a build
        bool hasPtWater_ = false;
        bool hasPtLava_ = false;  // the current geometry has a lava chunk
        // the 6 skybox face GL textures (rendersky.cpp's own `Texture *sky[6]`)
        uint32_t skyFaceTexIndices_[6] = {0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
        bool skyValid_ = false;
        uint32_t blueNoiseTexIndex_ = 0xFFFFFFFFu;
        std::unique_ptr<RayTracingSkinSink> pathTraceSkinSink_;
        uint32_t curWidth_ = 0, curHeight_ = 0;
        uint64_t nativeResourceEpoch_ = 0;
        uint64_t frameCounter_ = 0;

        bool historyValid_ = false;
        // path tracing separate tracked validity flag
        bool pathTraceHistoryValid_ = false;
        uint32_t jitterIndex_ = 0;
        uint32_t pathTraceJitterIndex_ = 0;
        float jitterNdcX_ = 0.0f, jitterNdcY_ = 0.0f;
        // same per-frame offset as jitterNdcX_/jitterNdcY_ above, in pixel units instead of NDC
        float jitterPixelX_ = 0.0f, jitterPixelY_ = 0.0f;
        matrix4 projMatrixUnjittered_;
        matrix4 prevCamProjMatrix_;
        matrix4 prevCamProjMatrixUnjittered_;
        matrix4 mvCurrVp_, mvPrevVp_;
        bool mvFresh_ = false;
        matrix4 ptInvCamProjFrame_;
        std::vector<float> partMvVerts_;
        std::vector<RtaoPrepass::ParticleRun> partMvRuns_;
        bool grassSubmitted_ = false;
        std::vector<int> pendingGrassTex_;

        std::unique_ptr<interop::InteropContext> ctx_;
        std::unique_ptr<interop::RayTracingScene> scene_;
        std::unique_ptr<interop::InteropFrame> frame_;

        std::unique_ptr<interop::InteropContext> dlssCtx_;
        std::unique_ptr<interop::InteropFrame> dlssFrame_;

        int volFogMat_ = MAT_AIR;
        float cameraLiquidSurfaceZ_ = 0.0f;  // the camera's water surface height (setFogMaterialState)
        float volFogBlend_ = 1.0f;
        int volFogAboveMat_ = MAT_AIR;
        bool volFogHistoryValid_ = false;
        matrix4 volFogInvViewProjUsed_;
        // far-fog tail parameters for this frame (see volfogfar), filled alongside the froxel
        // constants in onFrameLate()
        struct VolFogTail {
            float camPos[3] = {0, 0, 0};
            float density[4] = {0, 0, 0, 0};  // baseDensity, baseHeight, falloff, far distance
            float sun[4] = {0, 0, 1, 0};  // toward-sun dir, phase g
            float sunColor[3] = {0, 0, 0};
            float ambient[3] = {0, 0, 0};  // skyAmbient * 0.25 (the inject pass's unshadowed ambient term)
            float base[3] = {1, 1, 1};  // medium colour (water/lava) or white
        } volFogTail_;
        vec volFogCamPosUsed_ = vec(0, 0, 0);
        vec volFogCamForwardUsed_ = vec(0, 0, 1);

        sauerinterop::AnimatedInstanceCollector animatedCollector_;
        uint64_t lastSeenGeometryEpoch_ = 0;

        std::unique_ptr<interop::SharedTexture> depthTex_, normalTex_, motionTex_;
        std::unique_ptr<interop::SharedTexture> albedoTex_;
        std::unique_ptr<interop::SharedTexture> glowTex_;
        std::unique_ptr<interop::SharedTexture> aoOutput_, aoHistoryPrev_, aoHistoryNext_;
        std::unique_ptr<interop::SharedTexture> shadowOutput_, shadowHistoryPrev_, shadowHistoryNext_;
        std::unique_ptr<interop::SharedTexture> momentsHistoryPrev_, momentsHistoryNext_;
        std::unique_ptr<interop::SharedTexture> pathTraceOutput_, pathTraceHistoryPrev_, pathTraceHistoryNext_;
        std::unique_ptr<interop::SharedTexture> pathTraceHistoryMetaPrev_, pathTraceHistoryMetaNext_;
        // trace/resolve split's intermediate buffer
        std::unique_ptr<interop::SharedTexture> pathTraceRawRadiance_;

        std::unique_ptr<interop::SharedTexture> pathTracePrimaryNormal_;

        // uncompressed linear distance (world units) from the camera to this pixel's primary hit
        std::unique_ptr<interop::SharedTexture> pathTraceLinearDepth_;

        RtaoPrepass prepass_;


        GLuint debugProgram_ = 0, debugVao_ = 0;
        // see mvdebug
        GLuint mvDebugProgram_ = 0;
        GLint mvDebugUMvLoc_ = -1, mvDebugUMvRefLoc_ = -1, mvDebugUModeLoc_ = -1, mvDebugUMvSizeLoc_ = -1, mvDebugUVpSizeLoc_ = -1,
              mvDebugUScaleLoc_ = -1, mvDebugUCellLoc_ = -1, mvDebugUGainLoc_ = -1, mvDebugUOpacityLoc_ = -1;
        GLint debugUAOLoc_ = -1;
        GLint debugUShadowLoc_ = -1;
        GLint debugUModeLoc_ = -1;

        GLuint compositeProgram_ = 0;
        GLint compositeUAOLoc_ = -1;
        GLint compositeUShadowLoc_ = -1;
        GLint compositeUShadowIntensityLoc_ = -1;

        // path tracing separate composite program
        GLuint pathTraceCompositeProgram_ = 0;
        GLint pathTraceCompositeUOutputLoc_ = -1;
        GLint pathTraceCompositeUAOLoc_ = -1;
        GLint pathTraceCompositeUAOEnabledLoc_ = -1;
        GLint pathTraceCompositeUExposureLoc_ = -1;
        // post-tonemap contrast/saturation
        GLint pathTraceCompositeUContrastLoc_ = -1;
        GLint pathTraceCompositeUSaturationLoc_ = -1;
        GLint pathTraceCompositeURrModeLoc_ = -1;
        GLint pathTraceCompositeUHdrLoc_ = -1;
        GLint pathTraceCompositeULutLoc_ = -1;
        GLint pathTraceCompositeUHasValidDepthLoc_ = -1;
        // (DLSS unification)
        GLint pathTraceCompositeUDepthSourceLoc_ = -1;
        GLint pathTraceCompositeUOutputSizeInvLoc_ = -1;
        GLint pathTraceCompositeUNearLoc_ = -1;
        GLint pathTraceCompositeUFarLoc_ = -1;
        GLint pathTraceCompositeUMapFogEnabledLoc_ = -1, pathTraceCompositeUMapFogColorLoc_ = -1, pathTraceCompositeUMapFogParamsLoc_ = -1;
        // volumetric fog
        GLint pathTraceCompositeUVolFogVolumeLoc_ = -1;
        GLint pathTraceCompositeUVolFogEnabledLoc_ = -1;
        GLint pathTraceCompositeULinearDepthLoc_ = -1;
        GLint pathTraceCompositeUTanHalfFovLoc_ = -1;
        GLint pathTraceCompositeUVolFogNearFarLoc_ = -1;
        GLint pathTraceCompositeUVolFogInvViewProjLoc_ = -1, pathTraceCompositeUVolFogCamPosLoc_ = -1,
              pathTraceCompositeUVolFogTailDensityLoc_ = -1, pathTraceCompositeUVolFogTailSunLoc_ = -1,
              pathTraceCompositeUVolFogTailSunColorLoc_ = -1, pathTraceCompositeUVolFogTailAmbientLoc_ = -1,
              pathTraceCompositeUVolFogTailBaseLoc_ = -1;
        GLint pathTraceCompositeUVolFogGridZLoc_ = -1;
        GLint pathTraceCompositeUVolFogDebugModeLoc_ = -1;
        GLuint pathTraceDepthProgram_ = 0;
        GLint pathTraceDepthUOutputLoc_ = -1;
        GLint pathTraceDepthUOutputSizeInvLoc_ = -1;
        GLint pathTraceDepthURtMaskLoc_ = -1;
        GLint pathTraceDepthURtMaskEnabledLoc_ = -1;

        GLuint rtMaskFbo_ = 0;
        GLuint rtMaskTex_ = 0;
        uint32_t rtMaskW_ = 0, rtMaskH_ = 0;
        GLuint rtMaskProgram_ = 0;
        GLint rtMaskURectLoc_ = -1;
        GLint pathTraceCompositeURtMaskLoc_ = -1;
        GLint pathTraceCompositeURtMaskEnabledLoc_ = -1;

        std::unique_ptr<interop::SharedTexture> sceneColor_;
        std::unique_ptr<interop::SharedTexture> sceneDepthTex_;
        GLuint sceneFbo_ = 0;
        uint32_t sceneTargetW_ = 0, sceneTargetH_ = 0;
        bool sceneTargetActive_ = false;  // this frame's answer, set by beginSceneTarget(), read by presentScene()
        bool loggedFsaaRefusal_ = false;  // log-once guard

        GLuint presentSceneProgram_ = 0;
        GLint presentSceneUSceneLoc_ = -1;
        // increment 4
        GLint presentSceneUOutputSizeInvLoc_ = -1;
        // HDR bloom sample + additive composite, right at this same final blit
        GLint presentSceneUBloomLoc_ = -1, presentSceneUBloomIntensityLoc_ = -1;
        // depth-only reconstruction pass
        GLuint sceneDepthRestoreProgram_ = 0;
        GLint sceneDepthRestoreUDepthLoc_ = -1;
        GLint sceneDepthRestoreUOutputSizeInvLoc_ = -1;

        // dedicated HDR bloom for the path-traced pipeline
        GLuint bloomTexA_ = 0, bloomTexB_ = 0;
        GLuint bloomFboA_ = 0, bloomFboB_ = 0;
        uint32_t bloomW_ = 0, bloomH_ = 0;
        GLuint bloomThresholdProgram_ = 0;
        GLint bloomThresholdUSceneLoc_ = -1, bloomThresholdUOutputSizeInvLoc_ = -1, bloomThresholdUThresholdLoc_ = -1;
        GLuint bloomBlurProgram_ = 0;
        GLint bloomBlurUSrcLoc_ = -1, bloomBlurUOutputSizeInvLoc_ = -1, bloomBlurUDirectionLoc_ = -1;
        // hdr display pipeline auto exposure (see pthdr)
        static constexpr int kAeMeterSize = 64;
        GLuint aeMeterTex_ = 0, aeMeterFbo_ = 0;
        GLuint aeTex_[2] = {0, 0}, aeFbo_[2] = {0, 0};
        int aeCur_ = 0;  // aeTex_[aeCur_] holds the current exposure
        bool aeHistory_ = false;  // false = next adapt snaps straight to its target
        int aeLastMillis_ = 0;
        GLuint aeMeterProgram_ = 0, aeAdaptProgram_ = 0;
        GLint aeMeterUSrcLoc_ = -1, aeMeterUSrcSizeLoc_ = -1;
        GLint aeAdaptUMeterLoc_ = -1, aeAdaptUPrevLoc_ = -1, aeAdaptUKeyLoc_ = -1, aeAdaptUAdaptLoc_ = -1,
              aeAdaptUEvLoc_ = -1, aeAdaptUBlendLoc_ = -1, aeAdaptUResetLoc_ = -1;
        GLint pathTraceCompositeUAeTexLoc_ = -1, pathTraceCompositeUAeEnabledLoc_ = -1, pathTraceCompositeUSkyGainLoc_ = -1;
        // the tone curve the composite and the RR tone pass use this frame
        int displayCurveMode() { return pthdr ? 3 + pthdrcurve : pathtracehdr; }

        bool sceneTargetWanted_ = false;
        int sceneTargetRenderW_ = 0, sceneTargetRenderH_ = 0;
        bool dlssSceneRealDlssActive_ = false;

        bool dlssSceneConfigured_ = false;
        uint32_t dlssSceneConfiguredOutputWidth_ = 0, dlssSceneConfiguredOutputHeight_ = 0;
        int dlssSceneConfiguredModeOrdinal_ = -1;
        bool dlssSceneConfiguredUsedRr_ = false;
        static constexpr uint32_t kMaxDlssRrConsecutiveFailures = 3;
        uint32_t dlssSceneRrConsecutiveFailures_ = 0;
        int dlssSceneLastRawRr_ = -1;
        uint32_t dlssSceneRenderWidth_ = 0, dlssSceneRenderHeight_ = 0;
        bool dlssSceneHistoryValid_ = false;
        // DLSS's upscaled output for this pipeline
        std::unique_ptr<interop::SharedTexture> dlssSceneColorOutput_;

        std::unique_ptr<interop::SharedTexture> dlssRrDiffuseAlbedo_, dlssRrSpecularAlbedo_;
        std::unique_ptr<interop::SharedTexture> dlssRrColorInput_, dlssRrDepthInput_, dlssRrMotionInput_,
            dlssRrNormalRoughnessInput_, dlssRrSpecularMotionInput_;
        // the split pass's GL program/FBO
        GLuint dlssSplitProgram_ = 0, dlssSplitFbo_ = 0;
        GLuint rrToneProgram_ = 0, rrToneFbo_ = 0;
        GLint rrToneUSrcLoc_ = -1, rrToneUDirectionLoc_ = -1, rrToneUHdrLoc_ = -1,
              rrToneUContrastLoc_ = -1, rrToneUSaturationLoc_ = -1, rrToneULutLoc_ = -1;
        std::unique_ptr<interop::SharedTexture> rrDisplayOutput_;
        bool rrDisplayValid_ = false;
        GLint dlssSplitUAlbedoLoc_ = -1, dlssSplitUGlowLoc_ = -1;
        // DLSS-RR specular motion vectors, see runRrSpecMotion()
        GLuint rrSpecMotionProgram_ = 0, rrSpecMotionFbo_ = 0;
        // see runPtSurfaceMotion()
        GLuint ptSurfMotionProgram_ = 0, ptSurfMotionFbo_ = 0;
        // see runRrWaterGuides()
        GLuint rrWaterProgram_ = 0, rrWaterFbo_ = 0;
        GLint rrWaterUPtNormalLoc_ = -1, rrWaterUPtRawLoc_ = -1, rrWaterUAlbedoLoc_ = -1, rrWaterUPtInvVpLoc_ = -1,
              rrWaterUCamLoc_ = -1, rrWaterUCamPosLoc_ = -1, rrWaterUSizeInvLoc_ = -1;
        GLint ptSurfMotionUPtRawLoc_ = -1, ptSurfMotionUPrepassDepthLoc_ = -1, ptSurfMotionUPtInvVpLoc_ = -1, ptSurfMotionUInvVpLoc_ = -1,
              ptSurfMotionUCurrVpLoc_ = -1, ptSurfMotionUPrevVpLoc_ = -1, ptSurfMotionUCamPosLoc_ = -1, ptSurfMotionUSizeInvLoc_ = -1,
              ptSurfMotionUPtNormalLoc_ = -1;
        GLint rrSpecMotionULinearDepthLoc_ = -1, rrSpecMotionUMotionLoc_ = -1, rrSpecMotionUInvCurrVpLoc_ = -1,
              rrSpecMotionUPrevVpLoc_ = -1, rrSpecMotionUCamPosLoc_ = -1, rrSpecMotionUSizeInvLoc_ = -1;

        bool dlssPathTraceConfigured_ = false;
        bool dlssPathTraceActive_ = false;
        uint32_t dlssPathTraceConfiguredOutputWidth_ = 0, dlssPathTraceConfiguredOutputHeight_ = 0;
        int dlssPathTraceConfiguredModeOrdinal_ = -1;
        bool dlssPathTraceConfiguredUsedRr_ = false;
        uint32_t dlssPathTraceRrConsecutiveFailures_ = 0;
        int dlssPathTraceLastRawRr_ = -1;
        uint32_t dlssPathTraceRenderWidth_ = 0, dlssPathTraceRenderHeight_ = 0;
        bool dlssPathTraceHistoryValid_ = false;
        uint32_t dlssPathTraceJitterIndex_ = 0;
        int dlssPathTraceJitterPhaseCount_ = 8;
        // ("obvious directional crawl" in the stochastic noise at pathtracejittermagnitude 1)
        matrix4 prevPtCamProj_;
        std::unique_ptr<interop::SharedTexture> dlssPathTraceColorOutput_;

        const char* kShaderDirectory = "gl_vk_interop_v2/shaders";

        const uint32_t kInteropFramePoolSize = 8;  // 2-3 execute*() calls a frame (path trace, fog, ...)

        const wchar_t* kStreamlinePluginsDirectory = L"gl_vk_interop_v2/streamline/bin";
        // DLSS (Streamline)
        std::wstring streamlinePluginDirectory() {
            wchar_t buf[MAX_PATH];
            const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
            std::wstring exe(buf, (n > 0 && n < MAX_PATH) ? n : 0);
            const size_t slash = exe.find_last_of(L"\\/");
            return (slash == std::wstring::npos ? std::wstring(L".") : exe.substr(0, slash)) + L"\\streamline";
        }
        const wchar_t* kStreamlineLogDirectory = L"gl_vk_interop_v2/streamline/log";

        void logmsg(const std::string& msg) {
            // log.txt always
            if (ptconsole) conoutf("[sauerinterop] %s", msg.c_str());
            else logoutf("[sauerinterop] %s", msg.c_str());
            std::fprintf(stdout, "[sauerinterop] %s\n", msg.c_str());
            std::fflush(stdout);
        }

        float haltonSequence(uint32_t index, uint32_t base) {
            float result = 0.0f, f = 1.0f;
            while (index > 0) {
                f /= static_cast<float>(base);
                result += f * static_cast<float>(index % base);
                index /= base;
            }
            return result;
        }

        static float ptLightChannel(float v) {
            return pathtracelinearlights ? powf(max(v, 0.0f), 2.2f) : v;
        }
        int lastStaticLightCount_ = 0;  // static lights at the front of the last gather
        void gatherPathTraceLights(std::vector<float>& packed) {
            packed.clear();
            vector<extentity *> &ents = entities::getents();
            int emitted = 0;
            int noFalloffCount = 0, dynlightCount = 0;
            static vec dynO[256], dynColor[256];
            static float dynRadius[256];
            const int dynAvail = getpathtracedynlights(dynO, dynRadius, dynColor, min(256, pathtracemaxlights));
            const int staticCap = pathtracemaxlights - dynAvail;
            loopv(ents) {
                if (emitted >= staticCap) break;
                extentity &e = *ents[i];
                if (e.type != ET_LIGHT) continue;
                if (e.attr1 < 0) continue;
                packed.push_back(e.o.x);
                packed.push_back(e.o.y);
                packed.push_back(e.o.z);
                packed.push_back(static_cast<float>(e.attr1));
                packed.push_back(ptLightChannel(clamp(e.attr2 / 255.0f, 0.0f, 4.0f)));
                packed.push_back(ptLightChannel(clamp(e.attr3 / 255.0f, 0.0f, 4.0f)));
                packed.push_back(ptLightChannel(clamp(e.attr4 / 255.0f, 0.0f, 4.0f)));
                const bool noFalloff = e.attr1 == 0;
                if (noFalloff) ++noFalloffCount;
                // flag bits packed into one float
                vec capA, capB;
                const bool capsule = lightcapsule(e, capA, capB);
                const float sourceRadius = lightsourceradius(e);
                const bool shaped = capsule || sourceRadius > 0;
                const vec halfAxis = capsule ? vec(capB).sub(e.o) : vec(0, 0, 0);
                packed.push_back((noFalloff ? 2.0f : 0.0f) + (shaped ? 4.0f : 0.0f));
                packed.push_back(shaped ? sourceRadius : 0.0f);
                packed.push_back(halfAxis.x);
                packed.push_back(halfAxis.y);
                packed.push_back(halfAxis.z);
                ++emitted;
            }

            if (ptflamelight > 0) {
                loopv(ents) {
                    if (emitted >= staticCap) break;
                    const extentity &e = *ents[i];
                    if (e.type != ET_PARTICLES || (e.attr1 != 0 && e.attr1 != 11)) continue;
                    // the same radius/height/colour defaults makeparticles() uses
                    float radius, height; int color;
                    if (e.attr1 == 0) {
                        radius = e.attr2 ? e.attr2 / 100.0f : 1.5f;
                        height = e.attr3 ? e.attr3 / 100.0f : radius / 3;
                        color = e.attr4 ? ((e.attr4 & 0xF00) << 12) | ((e.attr4 & 0x0F0) << 8) | ((e.attr4 & 0x00F) << 4) | 0x0F0F0F : 0x903020;
                    } else {
                        radius = e.attr2 / 100.0f; height = e.attr3 / 100.0f;
                        color = ((e.attr4 & 0xF00) << 12) | ((e.attr4 & 0x0F0) << 8) | ((e.attr4 & 0x00F) << 4) | 0x0F0F0F;
                    }
                    if (radius <= 0 || height <= 0) continue;
                    vec c((color >> 16) / 255.0f, ((color >> 8) & 0xFF) / 255.0f, (color & 0xFF) / 255.0f);
                    const float cmax = max(c.x, max(c.y, c.z));
                    if (cmax <= 0) continue;
                    c.div(cmax);
                    // flames rise ~24*height units over their life (regularflame())
                    const vec o = vec(e.o).add(vec(0, 0, 4.0f + 10.0f * height));
                    const float t = float(lastmillis);
                    const float flicker = 1.0f + ptflameflicker * (0.6f * sinf(t * 0.013f + i * 1.7f) + 0.4f * sinf(t * 0.031f + i * 4.1f));
                    const float bright = ptflamelight * max(flicker, 0.0f);
                    packed.push_back(o.x); packed.push_back(o.y); packed.push_back(o.z);
                    packed.push_back((48.0f + 40.0f * max(radius, height)) * ptflameradius);
                    packed.push_back(ptLightChannel(c.x) * bright);
                    packed.push_back(ptLightChannel(c.y) * bright);
                    packed.push_back(ptLightChannel(c.z) * bright);
                    packed.push_back(0.0f); packed.push_back(0.0f); packed.push_back(0.0f); packed.push_back(0.0f); packed.push_back(0.0f);
                    ++emitted;
                }
            }

            lastStaticLightCount_ = emitted;

            for (int n = 0; n < dynAvail; ++n) {
                const vec& o = dynO[n];
                const vec& color = dynColor[n];
                const float radius = dynRadius[n] * ptdynlightradius;
                packed.push_back(o.x);
                packed.push_back(o.y);
                packed.push_back(o.z);
                packed.push_back(radius);
                packed.push_back(ptLightChannel(max(color.x, 0.0f)) * ptdynlightbrightness);
                packed.push_back(ptLightChannel(max(color.y, 0.0f)) * ptdynlightbrightness);
                packed.push_back(ptLightChannel(max(color.z, 0.0f)) * ptdynlightbrightness);
                packed.push_back(1.0f);
                packed.push_back(0.0f);
                packed.push_back(0.0f);
                packed.push_back(0.0f);
                packed.push_back(0.0f);
                ++emitted;
                ++dynlightCount;
            }

            for (const EmitterLight& el : emitterLights_) {
                if (!ents.inrange(el.ent) || ptmodellight <= 0) continue;
                const vec o = vec(ents[el.ent]->o).add(el.offset);
                // the disc's facing, turned with the model (reserved.yzw)
                vec n(0, 0, 0);
                if (!el.localNormal.iszero()) {
                    float yaw = el.staticYaw;
                    if (el.spins) {
                        auto yt = itemYaws_.find(itemYawKey(ents[el.ent]->o.x, ents[el.ent]->o.y));
                        if (yt != itemYaws_.end() && yt->second.millis >= 0)
                            yaw = yt->second.yaw + yt->second.rate * float(lastmillis - yt->second.millis);
                    }
                    n = vec(el.localNormal).rotate_around_z((yaw + el.yawOffset) * RAD);
                }
                packed.push_back(o.x); packed.push_back(o.y); packed.push_back(o.z);
                packed.push_back(el.radius);
                packed.push_back(ptLightChannel(el.color.x) * ptmodellight);
                packed.push_back(ptLightChannel(el.color.y) * ptmodellight);
                packed.push_back(ptLightChannel(el.color.z) * ptmodellight);
                packed.push_back(1.0f);
                packed.push_back(el.gap);
                packed.push_back(n.x); packed.push_back(n.y); packed.push_back(n.z);
                ++emitted;
            }

            static int lastLoggedNoFalloff = -1, lastLoggedDyn = -1, lastLoggedTotal = -1;
            if (noFalloffCount != lastLoggedNoFalloff || dynlightCount != lastLoggedDyn || emitted != lastLoggedTotal) {
                vlogmsg("gatherPathTraceLights: " + std::to_string(emitted) + " lights uploaded (" +
                    std::to_string(dynlightCount) + " dynlights), " +
                    std::to_string(noFalloffCount) + " flagged noFalloff (attr1==0). Spotlights are not modeled.");
                lastLoggedNoFalloff = noFalloffCount; lastLoggedDyn = dynlightCount; lastLoggedTotal = emitted;
            }
        }

        void buildEmitterLights() {
            emitterLights_.clear();
            std::unordered_map<model*, vec> colorOf;
            const vector<extentity *> &ents = entities::getents();
            loopv(ents) {
                const extentity &e = *ents[i];
                const char *name = NULL;
                float yaw = 0;
                if (e.type == ET_MAPMODEL) { name = mapmodelname(e.attr2); yaw = float(e.attr1); }
                else name = entities::entmodel(e);  // the game's own choice (teleporters, items)
                if (!name) continue;
                model *m = loadmodel(name);
                float radius = 0, intensity = 0;
                Texture *tex = NULL, *masks = NULL;
                if (!m || !m->ptemissivelight(radius, intensity, tex, masks) || !tex || !masks) continue;
                auto it = colorOf.find(m);
                if (it == colorOf.end()) {
                    vec c(1, 1, 1);
                    std::vector<unsigned char> skin, mask;
                    uint32_t sw = 0, sh = 0, mw = 0, mh = 0;
                    if (readTextureForRayTracing(static_cast<GLuint>(tex->id), skin, sw, sh) &&
                        readTextureForRayTracing(static_cast<GLuint>(masks->id), mask, mw, mh) && sw && sh && mw && mh) {
                        double sum[3] = {0, 0, 0}, wsum = 0;
                        for (uint32_t y = 0; y < sh; ++y) for (uint32_t x = 0; x < sw; ++x) {
                            const uint32_t mx = x * mw / sw, my = y * mh / sh;
                            const double g = mask[(static_cast<size_t>(my) * mw + mx) * 4 + 1] / 255.0;
                            if (g < 0.08) continue;
                            const unsigned char *px = &skin[(static_cast<size_t>(y) * sw + x) * 4];
                            loopk(3) sum[k] += pow(px[k] / 255.0, 2.2) * g;
                            wsum += g;
                        }
                        if (wsum > 0) {
                            const double mx = max(sum[0], max(sum[1], sum[2]));
                            if (mx > 0) c = vec(float(sum[0] / mx), float(sum[1] / mx), float(sum[2] / mx));
                        }
                    }
                    it = colorOf.emplace(m, c).first;
                }
                vec center, rad;
                m->boundbox(center, rad);
                EmitterLight el;
                el.ent = i;
                // teleporters/items spin about their origin, keep only the height
                if (e.type == ET_MAPMODEL) el.offset = vec(center).rotate_around_z(yaw * RAD);
                else el.offset = vec(0, 0, center.z + 1);
                el.localNormal = vec(0, 0, 0);
                const float thin = min(rad.x, rad.y), wide = max(rad.x, rad.y);
                if (wide > 0 && thin < wide * 0.5f) el.localNormal = rad.x < rad.y ? vec(1, 0, 0) : vec(0, 1, 0);
                el.yawOffset = m->offsetyaw;
                el.staticYaw = e.type == ET_MAPMODEL ? yaw : 0.0f;
                el.spins = e.type != ET_MAPMODEL;
                el.radius = radius;
                el.gap = max(rad.x, max(rad.y, rad.z)) * 1.1f;
                el.color = vec(it->second).mul(intensity);
                emitterLights_.push_back(el);
            }
            if (!emitterLights_.empty()) logmsg("emissive model lights: " + std::to_string(emitterLights_.size()));
        }

        // light grid layout in rt_lightgrid.glsl
        std::vector<float> lightGridBuiltFor_;
        const void* lightGridScene_ = nullptr;
        static bool lightNoFalloff(const float* L) { return (int(L[7] + 0.5f) & 2) != 0; }
        // half a capsule light's axis (zero otherwise)
        static vec lightHalfAxis(const float* L) { return (int(L[7] + 0.5f) & 4) ? vec(L[9], L[10], L[11]) : vec(0, 0, 0); }
        // nearest point of light L's segment to p
        static vec lightNearest(const float* L, const vec& p) {
            const vec c(L[0], L[1], L[2]), h = lightHalfAxis(L);
            const float l2 = h.squaredlen();
            if (l2 < 1e-8f) return c;
            return vec(h).mul(clamp(vec(p).sub(c).dot(h) / l2, -1.0f, 1.0f)).add(c);
        }
        // distance from light L's segment to the box [bmin, bmin + size], conservative
        static float lightBoxDist(const float* L, const vec& bmin, float size, float& slack) {
            const vec c(L[0], L[1], L[2]), h = lightHalfAxis(L);
            const float len = 2.0f * h.magnitude();
            const int n = len > 0 ? int(ceilf(len / (size * 0.5f))) + 1 : 1;
            slack = n > 1 ? 0.5f * len / float(n - 1) : 0.0f;
            float best = 1e30f;
            for (int j = 0; j < n; ++j) {
                const float t = n > 1 ? -1.0f + 2.0f * float(j) / float(n - 1) : 0.0f;
                const vec q = vec(h).mul(t).add(c);
                const vec b(clamp(q.x, bmin.x, bmin.x + size), clamp(q.y, bmin.y, bmin.y + size), clamp(q.z, bmin.z, bmin.z + size));
                best = min(best, b.dist(q));
            }
            return best;
        }
        int lightGridBuiltEnabled_ = -1;
        void updateLightGrid(const std::vector<float>& packed, int staticCount) {
            if (!scene_) return;
            // only position, radius and flags decide the cells, colours (flickering flame lights)
            // change without a rebuild
            std::vector<float> gridKey;
            gridKey.reserve(static_cast<size_t>(staticCount) * 5);
            for (int i = 0; i < staticCount; ++i) {
                const float* L = &packed[static_cast<size_t>(i) * 12];
                gridKey.push_back(L[0]); gridKey.push_back(L[1]); gridKey.push_back(L[2]); gridKey.push_back(L[3]); gridKey.push_back(L[7]);
                const vec h = lightHalfAxis(L);  // a capsule's axis (light shapes), not an emitter's spinning disc facing
                gridKey.push_back(h.x); gridKey.push_back(h.y); gridKey.push_back(h.z);
            }
            if (lightGridScene_ == scene_.get() && lightGridBuiltEnabled_ == ptlightgrid && lightGridBuiltFor_ == gridKey) return;
            const int t0 = SDL_GetTicks();
            std::vector<uint32_t> grid(16, 0u);
            std::vector<uint32_t> globalList;
            vec lo(1e16f, 1e16f, 1e16f), hi(-1e16f, -1e16f, -1e16f);
            int finite = 0;
            for (int i = 0; i < staticCount; ++i) {
                const float* L = &packed[static_cast<size_t>(i) * 12];
                if (lightNoFalloff(L)) { globalList.push_back(static_cast<uint32_t>(i)); continue; }  // noFalloff (flag bit 1)
                const vec c(L[0], L[1], L[2]), h = lightHalfAxis(L); const float r = L[3];
                lo.min(vec(c).sub(h).sub(r)); lo.min(vec(c).add(h).sub(r));
                hi.max(vec(c).sub(h).add(r)); hi.max(vec(c).add(h).add(r));
                ++finite;
            }
            uint32_t dims[3] = {0, 0, 0};
            float cell = 64.0f;
            std::vector<std::vector<uint32_t>> cells;
            size_t entries = 0;
            if (finite > 0) {
                const vec ext = vec(hi).sub(lo);
                cell = max(64.0f, max(ext.x, max(ext.y, ext.z)) / 48.0f);
                for (;;) {
                    loopk(3) dims[k] = static_cast<uint32_t>(max(1, int(ceilf(ext[k] / cell))));
                    // estimate before filling
                    double est = 0;
                    for (int i = 0; i < staticCount; ++i) {
                        const float* L = &packed[static_cast<size_t>(i) * 12];
                        if (lightNoFalloff(L)) continue;
                        const vec h = lightHalfAxis(L);
                        double span3 = 1.0;
                        loopk(3) span3 *= (2.0 * L[3] + 2.0 * fabs(h[k])) / cell + 2.0;
                        est += span3;
                    }
                    if (est <= 8.0e6 || cell >= 4096.0f) break;
                    cell *= 2.0f;
                }
                cells.assign(static_cast<size_t>(dims[0]) * dims[1] * dims[2], std::vector<uint32_t>());
                for (int i = 0; i < staticCount; ++i) {
                    const float* L = &packed[static_cast<size_t>(i) * 12];
                    if (lightNoFalloff(L)) continue;
                    const vec c(L[0], L[1], L[2]), h = lightHalfAxis(L); const float r = L[3];
                    int c0[3], c1[3];
                    loopk(3) {
                        const float ext = fabs(h[k]) + r;
                        c0[k] = clamp(int(floorf((c[k] - ext - lo[k]) / cell)), 0, int(dims[k]) - 1);
                        c1[k] = clamp(int(floorf((c[k] + ext - lo[k]) / cell)), 0, int(dims[k]) - 1);
                    }
                    for (int z = c0[2]; z <= c1[2]; ++z)
                    for (int y = c0[1]; y <= c1[1]; ++y)
                    for (int x = c0[0]; x <= c1[0]; ++x) {
                        const vec bmin(lo.x + x * cell, lo.y + y * cell, lo.z + z * cell);
                        float slack;
                        if (lightBoxDist(L, bmin, cell, slack) >= r + slack) continue;
                        cells[(static_cast<size_t>(z) * dims[1] + y) * dims[0] + x].push_back(static_cast<uint32_t>(i));
                        ++entries;
                    }
                }
            }
            const size_t cellCount = cells.size();
            grid.resize(16 + cellCount * 2, 0u);
            size_t maxList = 0;
            for (size_t ci = 0; ci < cellCount; ++ci) {
                grid[16 + ci * 2] = static_cast<uint32_t>(grid.size());
                grid[17 + ci * 2] = static_cast<uint32_t>(cells[ci].size());
                maxList = max(maxList, cells[ci].size());
                grid.insert(grid.end(), cells[ci].begin(), cells[ci].end());
            }
            auto fbits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
            grid[0] = dims[0]; grid[1] = dims[1]; grid[2] = dims[2];
            grid[3] = fbits(cell);
            grid[4] = fbits(lo.x); grid[5] = fbits(lo.y); grid[6] = fbits(lo.z);
            grid[7] = static_cast<uint32_t>(grid.size());
            grid[8] = static_cast<uint32_t>(globalList.size());
            grid.insert(grid.end(), globalList.begin(), globalList.end());
            // per cell, an alias table over its list weighted by each light's brightness x falloff at
            // the cell
            {
                const uint32_t listBase = static_cast<uint32_t>(16 + cellCount * 2);
                const uint32_t tableBase = static_cast<uint32_t>(grid.size());
                grid.resize(grid.size() + entries * 3, 0u);
                std::vector<float> w, prob;
                std::vector<uint32_t> alias, under, over;
                for (size_t ci = 0; ci < cellCount; ++ci) {
                    const std::vector<uint32_t>& list = cells[ci];
                    const size_t n = list.size();
                    if (!n) continue;
                    const size_t x = ci % dims[0], y = (ci / dims[0]) % dims[1], z = ci / (static_cast<size_t>(dims[0]) * dims[1]);
                    const vec bmin(lo.x + x * cell, lo.y + y * cell, lo.z + z * cell);
                    const vec centre = vec(bmin).add(cell * 0.5f);
                    w.assign(n, 0.0f);
                    double sum = 0;
                    for (size_t k = 0; k < n; ++k) {
                        const float* L = &packed[static_cast<size_t>(list[k]) * 12];
                        const float r = max(L[3], 1e-3f);
                        float slack;
                        const float nearDist = lightBoxDist(L, bmin, cell, slack);
                        const float atten = max(max(0.0f, 1.0f - centre.dist(lightNearest(L, centre)) / r), 0.25f * max(0.0f, 1.0f - nearDist / r));
                        const float lum = 0.2126f * max(L[4], 0.0f) + 0.7152f * max(L[5], 0.0f) + 0.0722f * max(L[6], 0.0f);
                        w[k] = lum * atten;
                        sum += w[k];
                    }
                    prob.assign(n, 0.0f); alias.assign(n, 0u); under.clear(); over.clear();
                    std::vector<float> pdf(n);
                    for (size_t k = 0; k < n; ++k) {
                        pdf[k] = sum > 0 ? float(0.8 * w[k] / sum + 0.2 / double(n)) : 1.0f / float(n);
                        prob[k] = pdf[k] * float(n);
                        (prob[k] < 1.0f ? under : over).push_back(static_cast<uint32_t>(k));
                    }
                    for (size_t k = 0; k < n; ++k) alias[k] = static_cast<uint32_t>(k);
                    while (!under.empty() && !over.empty()) {
                        const uint32_t s = under.back(); under.pop_back();
                        const uint32_t l = over.back();
                        alias[s] = l;
                        prob[l] -= 1.0f - prob[s];
                        if (prob[l] < 1.0f) { over.pop_back(); under.push_back(l); }
                    }
                    for (uint32_t k : over) prob[k] = 1.0f;
                    for (uint32_t k : under) prob[k] = 1.0f;  // float leftovers
                    const size_t e0 = tableBase + size_t(3) * (grid[16 + ci * 2] - listBase);
                    for (size_t k = 0; k < n; ++k) {
                        std::memcpy(&grid[e0 + 3 * k], &prob[k], 4);
                        grid[e0 + 3 * k + 1] = alias[k];
                        std::memcpy(&grid[e0 + 3 * k + 2], &pdf[k], 4);
                    }
                }
                grid[11] = tableBase;
                grid[12] = listBase;
                grid[13] = 1u;
            }
            grid[9] = static_cast<uint32_t>(staticCount);
            grid[10] = (staticCount > 0 && ptlightgrid) ? 1u : 0u;
            std::string err;
            if (!scene_->setLightGrid(grid.data(), grid.size(), &err)) {
                logmsg("RayTracingScene::setLightGrid failed: " + err);
                return;
            }
            lightGridBuiltFor_ = gridKey;
            lightGridScene_ = scene_.get();
            lightGridBuiltEnabled_ = ptlightgrid;
            char buf[256];
            snprintf(buf, sizeof(buf), "light grid: %d static lights (%d no-falloff), %ux%ux%u cells of %.0f units, %zu entries (longest list %zu), %zu KB, %d ms",
                     staticCount, int(globalList.size()), dims[0], dims[1], dims[2], cell, entries, maxList, grid.size() * 4 / 1024, int(SDL_GetTicks() - t0));
            logmsg(buf);
        }

        bool makeTexture(std::unique_ptr<interop::SharedTexture>& out, interop::PixelFormat fmt, uint32_t w, uint32_t h, const char* name) {
            out = std::make_unique<interop::SharedTexture>(*ctx_);
            if (!out->create(w, h, fmt)) {
                logmsg(std::string("SharedTexture::create failed for ") + name + ": " + ctx_->lastError());
                out.reset();
                return false;
            }
            return true;
        }

        // same as makeTexture() above, but bound to dlssCtx_'s separate VkDevice
        bool makeDlssTexture(std::unique_ptr<interop::SharedTexture>& out, interop::PixelFormat fmt, uint32_t w, uint32_t h, const char* name) {
            if (!dlssCtx_) { out.reset(); return false; }
            out = std::make_unique<interop::SharedTexture>(*dlssCtx_);
            if (!out->create(w, h, fmt)) {
                logmsg(std::string("SharedTexture::create failed for ") + name + " (dlssCtx_): " + dlssCtx_->lastError());
                out.reset();
                return false;
            }
            return true;
        }

        void destroySceneTarget() {
            if (sceneFbo_) { glDeleteFramebuffers_(1, &sceneFbo_); sceneFbo_ = 0; }
            sceneDepthTex_.reset();
            sceneColor_.reset();
            sceneTargetW_ = sceneTargetH_ = 0;
        }

        void destroyRtMask() {
            if (rtMaskFbo_) { glDeleteFramebuffers_(1, &rtMaskFbo_); rtMaskFbo_ = 0; }
            if (rtMaskTex_) { glDeleteTextures(1, &rtMaskTex_); rtMaskTex_ = 0; }
            rtMaskW_ = rtMaskH_ = 0;
        }

        bool ensureRtMask(uint32_t w, uint32_t h, std::string* error) {
            if (rtMaskFbo_ && rtMaskW_ == w && rtMaskH_ == h) return true;
            destroyRtMask();
            if (w == 0 || h == 0) {
                if (error) *error = "ensureRtMask: zero size.";
                return false;
            }
            glGenTextures(1, &rtMaskTex_);
            glBindTexture(GL_TEXTURE_2D, rtMaskTex_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, static_cast<GLsizei>(w), static_cast<GLsizei>(h), 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);

            glGenFramebuffers_(1, &rtMaskFbo_);
            glBindFramebuffer_(GL_FRAMEBUFFER, rtMaskFbo_);
            glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rtMaskTex_, 0);
            const GLenum drawBufs[1] = { GL_COLOR_ATTACHMENT0 };
            glDrawBuffers_(1, drawBufs);
            const GLenum status = glCheckFramebufferStatus_(GL_FRAMEBUFFER);
            glBindFramebuffer_(GL_FRAMEBUFFER, 0);
            if (status != GL_FRAMEBUFFER_COMPLETE) {
                if (error) *error = "ensureRtMask: framebuffer incomplete (status 0x" + std::to_string(status) + ").";
                destroyRtMask();
                return false;
            }
            rtMaskW_ = w;
            rtMaskH_ = h;
            return true;
        }

        bool initRtMaskProgram(std::string* error) {
            const char* kRtMaskVs = R"GLSL(
#version 330 core
uniform vec4 uRectNdc;  // minX, minY, maxX, maxY
void main() {
    float x = (gl_VertexID & 1) != 0 ? uRectNdc.z : uRectNdc.x;
    float y = (gl_VertexID & 2) != 0 ? uRectNdc.w : uRectNdc.y;
    gl_Position = vec4(x, y, 0.0, 1.0);
}
)GLSL";
            const char* kRtMaskFs = R"GLSL(
#version 330 core
out float fragMask;
void main() { fragMask = 1.0; }
)GLSL";
            auto compile = [&](GLenum type, const char* src, GLuint& outShader) -> bool {
                outShader = glCreateShader_(type);
                glShaderSource_(outShader, 1, &src, nullptr);
                glCompileShader_(outShader);
                GLint ok = 0;
                glGetShaderiv_(outShader, GL_COMPILE_STATUS, &ok);
                if (!ok) {
                    char buf[2048]; GLsizei len = 0;
                    glGetShaderInfoLog_(outShader, sizeof(buf), &len, buf);
                    if (error) *error = std::string("rt mask shader compile failed: ") + std::string(buf, len);
                    return false;
                }
                return true;
                };
            GLuint vs = 0, fs = 0;
            if (!compile(GL_VERTEX_SHADER, kRtMaskVs, vs)) return false;
            if (!compile(GL_FRAGMENT_SHADER, kRtMaskFs, fs)) return false;
            rtMaskProgram_ = glCreateProgram_();
            glAttachShader_(rtMaskProgram_, vs);
            glAttachShader_(rtMaskProgram_, fs);
            glLinkProgram_(rtMaskProgram_);
            glDeleteShader_(vs);
            glDeleteShader_(fs);
            GLint linked = 0;
            glGetProgramiv_(rtMaskProgram_, GL_LINK_STATUS, &linked);
            if (!linked) {
                char buf[2048]; GLsizei len = 0;
                glGetProgramInfoLog_(rtMaskProgram_, sizeof(buf), &len, buf);
                if (error) *error = std::string("rt mask program link failed: ") + std::string(buf, len);
                return false;
            }
            rtMaskURectLoc_ = glGetUniformLocation_(rtMaskProgram_, "uRectNdc");
            return true;
        }

        void renderRtMask() {
            if (!rtMaskFbo_ || !rtMaskProgram_ || !debugVao_) return;
            GLint prevFbo = 0;
            glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
            GLint prevVao = 0;
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
            GLint prevViewport[4];
            glGetIntegerv(GL_VIEWPORT, prevViewport);
            GLboolean prevDepthTest = glIsEnabled(GL_DEPTH_TEST);
            GLboolean prevScissorTest = glIsEnabled(GL_SCISSOR_TEST);

            glBindFramebuffer_(GL_FRAMEBUFFER, rtMaskFbo_);
            glViewport(0, 0, static_cast<GLsizei>(rtMaskW_), static_cast<GLsizei>(rtMaskH_));
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_SCISSOR_TEST);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            if (!rtMaskRectsNdc_.empty()) {
                glUseProgram_(rtMaskProgram_);
                glBindVertexArray_(debugVao_);
                for (const RtMaskRect& r : rtMaskRectsNdc_) {
                    if (rtMaskURectLoc_ >= 0) glUniform4f_(rtMaskURectLoc_, r.minX, r.minY, r.maxX, r.maxY);
                    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
                }
            }

            glBindFramebuffer_(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo));
            glBindVertexArray_(static_cast<GLuint>(prevVao));
            glUseProgram_(0);
            glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
            if (prevDepthTest) glEnable(GL_DEPTH_TEST);
            if (prevScissorTest) glEnable(GL_SCISSOR_TEST);
        }

        bool ensureSceneTarget(uint32_t w, uint32_t h, std::string* error) {
            if (sceneFbo_ && sceneTargetW_ == w && sceneTargetH_ == h) return true;
            destroySceneTarget();
            if (w == 0 || h == 0) {
                if (error) *error = "ensureSceneTarget: zero size.";
                return false;
            }

            sceneColor_ = std::make_unique<interop::SharedTexture>(*ctx_);
            if (!sceneColor_->create(w, h, interop::PixelFormat::Rgba16Float)) {
                if (error) *error = std::string("SharedTexture::create failed for sceneColor_: ") + ctx_->lastError();
                sceneColor_.reset();
                return false;
            }
            glBindTexture(GL_TEXTURE_2D, sceneColor_->glTextureId());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);

            if (!makeTexture(sceneDepthTex_, interop::PixelFormat::Depth32Float, w, h, "sceneDepthTex_")) {
                if (error) *error = std::string("SharedTexture::create failed for sceneDepthTex_: ") + ctx_->lastError();
                sceneColor_.reset();
                return false;
            }
            glBindTexture(GL_TEXTURE_2D, sceneDepthTex_->glTextureId());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);

            glGenFramebuffers_(1, &sceneFbo_);
            glBindFramebuffer_(GL_FRAMEBUFFER, sceneFbo_);
            glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sceneColor_->glTextureId(), 0);
            glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, sceneDepthTex_->glTextureId(), 0);
            const GLenum drawBufs[1] = { GL_COLOR_ATTACHMENT0 };
            glDrawBuffers_(1, drawBufs);
            const GLenum status = glCheckFramebufferStatus_(GL_FRAMEBUFFER);
            glBindFramebuffer_(GL_FRAMEBUFFER, 0);
            if (status != GL_FRAMEBUFFER_COMPLETE) {
                if (error) *error = "ensureSceneTarget: framebuffer incomplete (status 0x" + std::to_string(status) + ").";
                destroySceneTarget();
                return false;
            }

            sceneTargetW_ = w;
            sceneTargetH_ = h;
            return true;
        }

        // dedicated HDR bloom for the path-traced pipeline
        void destroyBloomTarget() {
            if (bloomFboA_) { glDeleteFramebuffers_(1, &bloomFboA_); bloomFboA_ = 0; }
            if (bloomFboB_) { glDeleteFramebuffers_(1, &bloomFboB_); bloomFboB_ = 0; }
            if (bloomTexA_) { glDeleteTextures(1, &bloomTexA_); bloomTexA_ = 0; }
            if (bloomTexB_) { glDeleteTextures(1, &bloomTexB_); bloomTexB_ = 0; }
            bloomW_ = bloomH_ = 0;
        }

        // auto exposure targets (see aeMeterTex_)
        bool ensureAutoExposureTargets() {
            if (aeMeterFbo_ && aeFbo_[0] && aeFbo_[1]) return true;
            auto makeOne = [&](GLuint& tex, GLuint& fbo, int size, GLenum fmt) -> bool {
                glGenTextures(1, &tex);
                glBindTexture(GL_TEXTURE_2D, tex);
                glTexImage2D(GL_TEXTURE_2D, 0, fmt, size, size, 0, GL_RED, GL_FLOAT, nullptr);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glBindTexture(GL_TEXTURE_2D, 0);
                glGenFramebuffers_(1, &fbo);
                glBindFramebuffer_(GL_FRAMEBUFFER, fbo);
                glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
                const GLenum drawBuf = GL_COLOR_ATTACHMENT0;
                glDrawBuffers_(1, &drawBuf);
                return glCheckFramebufferStatus_(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
            };
            GLint prevFbo = 0; glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
            // the adapted value's .g carries the metered average (debugging)
            const bool ok = makeOne(aeMeterTex_, aeMeterFbo_, kAeMeterSize, GL_R16F) &&
                            makeOne(aeTex_[0], aeFbo_[0], 1, GL_RG32F) && makeOne(aeTex_[1], aeFbo_[1], 1, GL_RG32F);
            glBindFramebuffer_(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo));
            aeHistory_ = false;
            if (!ok) logmsg("auto exposure: framebuffer setup failed -- pthdr runs without auto exposure.");
            return ok;
        }
        // Meters `src` (linear radiance) and adapts the exposure one step
        bool updateAutoExposure(interop::SharedTexture* src) {
            if (!src || !aeMeterProgram_ || !aeAdaptProgram_ || !debugVao_ || !ensureAutoExposureTargets()) return false;
            GLint prevVao = 0, prevFbo = 0, prevProgram = 0, prevActiveTex = 0;
            GLint prevViewport[4] = {0, 0, 0, 0};
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
            glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
            glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
            glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
            glGetIntegerv(GL_VIEWPORT, prevViewport);
            const GLboolean prevDepthTest = glIsEnabled(GL_DEPTH_TEST), prevBlend = glIsEnabled(GL_BLEND), prevScissor = glIsEnabled(GL_SCISSOR_TEST);
            glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND); glDisable(GL_SCISSOR_TEST);
            glBindVertexArray_(debugVao_);
            glActiveTexture_(GL_TEXTURE0);
            GLint prevTex0 = 0; glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);

            glBindFramebuffer_(GL_FRAMEBUFFER, aeMeterFbo_);
            glViewport(0, 0, kAeMeterSize, kAeMeterSize);
            glUseProgram_(aeMeterProgram_);
            glBindTexture(GL_TEXTURE_2D, src->glTextureId());
            if (aeMeterUSrcLoc_ >= 0) glUniform1i_(aeMeterUSrcLoc_, 0);
            if (aeMeterUSrcSizeLoc_ >= 0) glUniform2f_(aeMeterUSrcSizeLoc_, static_cast<float>(src->width()), static_cast<float>(src->height()));
            glDrawArrays(GL_TRIANGLES, 0, 3);

            const int next = aeCur_ ^ 1;
            glBindFramebuffer_(GL_FRAMEBUFFER, aeFbo_[next]);
            glViewport(0, 0, 1, 1);
            glUseProgram_(aeAdaptProgram_);
            glBindTexture(GL_TEXTURE_2D, aeMeterTex_);
            glActiveTexture_(GL_TEXTURE1);
            GLint prevTex1 = 0; glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex1);
            glBindTexture(GL_TEXTURE_2D, aeTex_[aeCur_]);
            if (aeAdaptUMeterLoc_ >= 0) glUniform1i_(aeAdaptUMeterLoc_, 0);
            if (aeAdaptUPrevLoc_ >= 0) glUniform1i_(aeAdaptUPrevLoc_, 1);
            if (aeAdaptUKeyLoc_ >= 0) glUniform1f_(aeAdaptUKeyLoc_, pthdrkey);
            if (aeAdaptUAdaptLoc_ >= 0) glUniform1f_(aeAdaptUAdaptLoc_, pthdradapt);
            if (aeAdaptUEvLoc_ >= 0) glUniform1f_(aeAdaptUEvLoc_, pthdrev);
            const int dtMs = aeLastMillis_ ? std::clamp(totalmillis - aeLastMillis_, 0, 1000) : 0;
            aeLastMillis_ = totalmillis;
            const float blend = 1.0f - std::exp(-static_cast<float>(dtMs) * 0.001f * pthdrspeed);
            if (aeAdaptUBlendLoc_ >= 0) glUniform1f_(aeAdaptUBlendLoc_, blend);
            if (aeAdaptUResetLoc_ >= 0) glUniform1f_(aeAdaptUResetLoc_, aeHistory_ ? 0.0f : 1.0f);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex1));
            glActiveTexture_(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex0));
            aeCur_ = next;
            aeHistory_ = true;

            glBindFramebuffer_(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo));
            glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
            glUseProgram_(static_cast<GLuint>(prevProgram));
            glBindVertexArray_(static_cast<GLuint>(prevVao));
            glActiveTexture_(static_cast<GLenum>(prevActiveTex));
            if (prevDepthTest) glEnable(GL_DEPTH_TEST);
            if (prevBlend) glEnable(GL_BLEND);
            if (prevScissor) glEnable(GL_SCISSOR_TEST);
            return true;
        }

        bool ensureBloomTarget(uint32_t w, uint32_t h, std::string* error) {
            if (bloomFboA_ && bloomFboB_ && bloomW_ == w && bloomH_ == h) return true;
            destroyBloomTarget();
            if (w == 0 || h == 0) {
                if (error) *error = "ensureBloomTarget: zero size.";
                return false;
            }
            auto makeOne = [&](GLuint& tex, GLuint& fbo, const char* label) -> bool {
                glGenTextures(1, &tex);
                glBindTexture(GL_TEXTURE_2D, tex);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, static_cast<GLsizei>(w), static_cast<GLsizei>(h), 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glBindTexture(GL_TEXTURE_2D, 0);
                glGenFramebuffers_(1, &fbo);
                glBindFramebuffer_(GL_FRAMEBUFFER, fbo);
                glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
                const GLenum drawBufs[1] = { GL_COLOR_ATTACHMENT0 };
                glDrawBuffers_(1, drawBufs);
                const GLenum status = glCheckFramebufferStatus_(GL_FRAMEBUFFER);
                glBindFramebuffer_(GL_FRAMEBUFFER, 0);
                if (status != GL_FRAMEBUFFER_COMPLETE) {
                    if (error) *error = std::string("ensureBloomTarget: ") + label + " framebuffer incomplete (status 0x" + std::to_string(status) + ").";
                    return false;
                }
                return true;
            };
            if (!makeOne(bloomTexA_, bloomFboA_, "bloomA") || !makeOne(bloomTexB_, bloomFboB_, "bloomB")) {
                destroyBloomTarget();
                return false;
            }
            bloomW_ = w;
            bloomH_ = h;
            return true;
        }


        void destroyResources() {
            depthTex_.reset(); normalTex_.reset(); motionTex_.reset();
            albedoTex_.reset();  // path tracing
            glowTex_.reset();  // pbr materials
            dlssRrDiffuseAlbedo_.reset(); dlssRrSpecularAlbedo_.reset();  // dlss ray reconstruction
            dlssRrColorInput_.reset(); dlssRrDepthInput_.reset(); dlssRrMotionInput_.reset();  // dlss ray reconstruction
            dlssRrNormalRoughnessInput_.reset(); dlssRrSpecularMotionInput_.reset();  // dlss ray reconstruction
            aoOutput_.reset(); aoHistoryPrev_.reset(); aoHistoryNext_.reset();
            shadowOutput_.reset(); shadowHistoryPrev_.reset(); shadowHistoryNext_.reset();
            momentsHistoryPrev_.reset(); momentsHistoryNext_.reset();  // SVGF-style
            pathTraceOutput_.reset(); pathTraceHistoryPrev_.reset(); pathTraceHistoryNext_.reset();  // path tracing
            pathTraceHistoryMetaPrev_.reset(); pathTraceHistoryMetaNext_.reset();
            pathTraceRawRadiance_.reset();
            pathTracePrimaryNormal_.reset();
            pathTraceLinearDepth_.reset();
            // dlssPathTraceColorOutput_ must not be reset here
        }

        void attachDlssSplitTargets() {
            if (!dlssSplitFbo_ || !dlssRrDiffuseAlbedo_ || !dlssRrSpecularAlbedo_) return;
            glBindFramebuffer_(GL_FRAMEBUFFER, dlssSplitFbo_);
            glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dlssRrDiffuseAlbedo_->glTextureId(), 0);
            glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, dlssRrSpecularAlbedo_->glTextureId(), 0);
            const GLenum drawBufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
            glDrawBuffers_(2, drawBufs);
            glBindFramebuffer_(GL_FRAMEBUFFER, 0);
        }

        bool createResources(uint32_t w, uint32_t h) {
            using interop::PixelFormat;
            if (!makeTexture(depthTex_, PixelFormat::Depth32Float, w, h, "depthTex_")) return false;
            if (!makeTexture(normalTex_, PixelFormat::Rgba16Float, w, h, "normalTex_")) return false;
            if (!makeTexture(motionTex_, PixelFormat::Rg16Float, w, h, "motionTex_")) return false;
            if (!makeTexture(aoOutput_, PixelFormat::R8Unorm, w, h, "aoOutput_")) return false;
            glBindTexture(GL_TEXTURE_2D, aoOutput_->glTextureId());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);
            if (!makeTexture(aoHistoryPrev_, PixelFormat::Rgba16Float, w, h, "aoHistoryPrev_")) return false;
            if (!makeTexture(aoHistoryNext_, PixelFormat::Rgba16Float, w, h, "aoHistoryNext_")) return false;
            if (!makeTexture(shadowOutput_, PixelFormat::R8Unorm, w, h, "shadowOutput_")) return false;
            if (!makeTexture(shadowHistoryPrev_, PixelFormat::Rgba16Float, w, h, "shadowHistoryPrev_")) return false;
            if (!makeTexture(shadowHistoryNext_, PixelFormat::Rgba16Float, w, h, "shadowHistoryNext_")) return false;
            if (!makeTexture(momentsHistoryPrev_, PixelFormat::Rgba16Float, w, h, "momentsHistoryPrev_")) return false;
            if (!makeTexture(momentsHistoryNext_, PixelFormat::Rgba16Float, w, h, "momentsHistoryNext_")) return false;

            if (!makeTexture(albedoTex_, PixelFormat::Rgba8Unorm, w, h, "albedoTex_")) return false;
            // pbr materials per-pixel glow (RtaoPrepass's new 5th render target)
            if (!makeTexture(glowTex_, PixelFormat::Rgba8Unorm, w, h, "glowTex_")) return false;
            if (dlssCtx_) {
                if (!makeDlssTexture(dlssRrDiffuseAlbedo_, PixelFormat::Rgba8Unorm, w, h, "dlssRrDiffuseAlbedo_")) {
                    logmsg("DLSS-RR: dlssRrDiffuseAlbedo_ creation failed -- Ray Reconstruction unavailable this session.");
                }
                if (!makeDlssTexture(dlssRrSpecularAlbedo_, PixelFormat::Rgba8Unorm, w, h, "dlssRrSpecularAlbedo_")) {
                    logmsg("DLSS-RR: dlssRrSpecularAlbedo_ creation failed -- Ray Reconstruction unavailable this session.");
                }
                attachDlssSplitTargets();
                if (!makeDlssTexture(dlssRrColorInput_, PixelFormat::Rgba16Float, w, h, "dlssRrColorInput_")) {
                    logmsg("DLSS-RR: dlssRrColorInput_ creation failed -- Ray Reconstruction unavailable this session.");
                }
                if (!makeDlssTexture(dlssRrDepthInput_, PixelFormat::Depth32Float, w, h, "dlssRrDepthInput_")) {
                    logmsg("DLSS-RR: dlssRrDepthInput_ creation failed -- Ray Reconstruction unavailable this session.");
                }
                if (!makeDlssTexture(dlssRrMotionInput_, PixelFormat::Rg16Float, w, h, "dlssRrMotionInput_")) {
                    logmsg("DLSS-RR: dlssRrMotionInput_ creation failed -- Ray Reconstruction unavailable this session.");
                }
                if (!makeDlssTexture(dlssRrNormalRoughnessInput_, PixelFormat::Rgba16Float, w, h, "dlssRrNormalRoughnessInput_")) {
                    logmsg("DLSS-RR: dlssRrNormalRoughnessInput_ creation failed -- Ray Reconstruction unavailable this session.");
                }
                if (!makeDlssTexture(dlssRrSpecularMotionInput_, PixelFormat::Rg16Float, w, h, "dlssRrSpecularMotionInput_")) {
                    logmsg("DLSS-RR: dlssRrSpecularMotionInput_ creation failed -- Ray Reconstruction unavailable this session.");
                }
            } else {
                dlssRrDiffuseAlbedo_.reset(); dlssRrSpecularAlbedo_.reset();
                dlssRrColorInput_.reset(); dlssRrDepthInput_.reset(); dlssRrMotionInput_.reset();
                dlssRrNormalRoughnessInput_.reset(); dlssRrSpecularMotionInput_.reset();
            }
            if (!makeTexture(pathTraceOutput_, PixelFormat::Rgba32Float, w, h, "pathTraceOutput_")) return false;
            glBindTexture(GL_TEXTURE_2D, pathTraceOutput_->glTextureId());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);
            if (!makeTexture(pathTraceHistoryPrev_, PixelFormat::Rgba32Float, w, h, "pathTraceHistoryPrev_")) return false;
            if (!makeTexture(pathTraceHistoryNext_, PixelFormat::Rgba32Float, w, h, "pathTraceHistoryNext_")) return false;
            // second, small ping-ponged pair
            if (!makeTexture(pathTraceHistoryMetaPrev_, PixelFormat::Rgba16Float, w, h, "pathTraceHistoryMetaPrev_")) return false;
            if (!makeTexture(pathTraceHistoryMetaNext_, PixelFormat::Rgba16Float, w, h, "pathTraceHistoryMetaNext_")) return false;
            // trace/resolve split's intermediate buffer
            if (!makeTexture(pathTraceRawRadiance_, PixelFormat::Rgba32Float, w, h, "pathTraceRawRadiance_")) return false;
            if (!makeTexture(pathTracePrimaryNormal_, PixelFormat::Rgba16Float, w, h, "pathTracePrimaryNormal_")) return false;
            if (!makeTexture(pathTraceLinearDepth_, PixelFormat::Rg16Float, w, h, "pathTraceLinearDepth_")) return false;

            std::string err;
            if (!prepass_.attachTargets(depthTex_->glTextureId(), normalTex_->glTextureId(), motionTex_->glTextureId(),
                                         albedoTex_->glTextureId(), glowTex_->glTextureId(), w, h, &err)) {
                logmsg("RtaoPrepass::attachTargets failed: " + err);
                return false;
            }
            return true;
        }

        void rebuildGeometry(bool engineJustRebuilt = false) {
            if (!ready_) return;
            // where a rebuild's time goes (logged at the end)
            using RbClock = std::chrono::steady_clock;
            const RbClock::time_point rbStart = RbClock::now();
            auto rbMs = [](RbClock::time_point a, RbClock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            double rbBlasMs = 0.0;
            if (!engineJustRebuilt) allchanged();
            historyValid_ = false;
            pathTraceHistoryValid_ = false;

            cacheClearPending_ = true;
            const RbClock::time_point rbEngineDone = RbClock::now();
            std::vector<ChunkMeshData> chunks;
            std::vector<sauerinterop::RtMaterialDesc> materials;
            std::vector<sauerinterop::EmissiveTriangleDesc> emissiveTriangles;
            extractSolidCubeChunks(chunks, materials, emissiveTriangles);

            // path-traced water every water surface as one static, non-opaque chunk (WaterExtract.h)
            hasPtLava_ = false;
            std::vector<float> waterTops, lavaTops;  // displaced water / lava patch tops (8 floats each)
            if (ptlava) {
                ChunkMeshData lavaChunk;
                sauerinterop::extractLavaVolumes(materials, lavaChunk, emissiveTriangles, ptlavaglow, ptlavascroll != 0, ptlavamode != 0,
                    (ptlavamode && ptlavadisplace) ? &lavaTops : nullptr);  // displaced lava
                if (!lavaChunk.rtIndices.empty()) {
                    if (!ptlavaprepass) { lavaChunk.glVertexData.clear(); lavaChunk.glIndices.clear(); lavaChunk.glBatches.clear(); }
                    uint32_t lastId = 0;
                    for (uint32_t id : lavaChunk.rtTextureGlIds) {
                        if (id == 0 || id == lastId) continue;
                        lastId = id;
                        if (scene_->isPathTraceTextureRegistered(id, nullptr)) continue;
                        std::vector<unsigned char> rgba;
                        uint32_t tw = 0, th = 0;
                        if (!readTextureForRayTracing(static_cast<GLuint>(id), rgba, tw, th) ||
                            !scene_->registerPathTraceTexture(id, tw, th, rgba.data(), rgba.size(), 0u, nullptr))
                            logmsg("rebuildGeometry: lava texture " + std::to_string(id) + " not registered.");
                    }
                    logmsg("rebuildGeometry: path-traced lava chunk, " +
                        std::to_string(lavaChunk.rtIndices.size() / 3) + " triangles.");
                    chunks.push_back(std::move(lavaChunk));
                    hasPtLava_ = true;
                }
            }
            hasPtWater_ = false;
            if (ptwater) {
                ChunkMeshData waterChunk;
                sauerinterop::extractWaterVolumes(materials, waterChunk, &waterTops);
                if (!waterChunk.rtIndices.empty()) {
                    logmsg("rebuildGeometry: path-traced water chunk, " +
                        std::to_string(waterChunk.rtIndices.size() / 3) + " triangles.");
                    chunks.push_back(std::move(waterChunk));
                    hasPtWater_ = true;
                }
            }
            {
                std::vector<float> tops(waterTops);
                tops.insert(tops.end(), lavaTops.begin(), lavaTops.end());
                scene_->setWaterSurfaces(tops.empty() ? nullptr : tops.data(), static_cast<uint32_t>(tops.size() / 8),
                    sauerinterop::kWaterChunkId, static_cast<uint32_t>(waterTops.size() / 8) * 2u,
                    sauerinterop::kLavaChunkId, static_cast<uint32_t>(lavaTops.size() / 8) * 2u);
            }

            {
                std::vector<ChunkMeshData> mapmodelChunks;
                sauerinterop::extractStaticMapModelChunks(mapmodelChunks);
                if (!mapmodelChunks.empty()) {
                    logmsg("rebuildGeometry: " + std::to_string(mapmodelChunks.size()) + " static map-model chunks.");
                    for (auto& mc : mapmodelChunks) chunks.push_back(std::move(mc));
                }
            }

            {
                ChunkMeshData glassChunk;
                if (ptglass) sauerinterop::extractGlassVolumes(materials, glassChunk);
                if (!glassChunk.rtIndices.empty()) {
                    logmsg("rebuildGeometry: real glass volume chunk, " +
                        std::to_string(glassChunk.rtIndices.size() / 3) + " triangles.");
                    chunks.push_back(std::move(glassChunk));
                }
            }

            if (chunks.empty()) {
                logmsg("rebuildGeometry: no world geometry found (empty map, or nothing uploaded to the GPU yet).");
                hasGeometry_ = false;
                return;
            }
            const RbClock::time_point rbExtractDone = RbClock::now();
            scene_->resetGeometry();
            const RbClock::time_point rbResetDone = RbClock::now();

            {
                bool allFacesOk = true;
                loopi(6) {
                    Texture *face = sky[i];
                    if (!face || !face->id || face == notexture) { skyFaceTexIndices_[i] = 0xFFFFFFFFu; allFacesOk = false; continue; }
                    uint32_t idx = 0xFFFFFFFFu;
                    if (scene_->isPathTraceTextureRegistered(face->id, &idx)) {
                        skyFaceTexIndices_[i] = idx;
                        continue;
                    }
                    std::vector<unsigned char> rgba;
                    uint32_t tw = 0, th = 0;
                    if (!readTextureForRayTracing(face->id, rgba, tw, th) ||
                        // face->clamp (rendersky.cpp loads skies with clamp 3 = clamp-to-edge both
                        // axes)
                        !scene_->registerPathTraceTexture(face->id, tw, th, rgba.data(), rgba.size(), static_cast<uint32_t>(face->clamp), &idx)) {
                        logmsg("rebuildGeometry: failed to register skybox face " + std::to_string(i) + ": " + ctx_->lastError());
                        skyFaceTexIndices_[i] = 0xFFFFFFFFu;
                        allFacesOk = false;
                        continue;
                    }
                    skyFaceTexIndices_[i] = idx;
                }
                skyValid_ = allFacesOk;
                logmsg("rebuildGeometry: skybox skyValid_=" + std::to_string(skyValid_) +
                    " indices=" + std::to_string(skyFaceTexIndices_[0]) + "," + std::to_string(skyFaceTexIndices_[1]) + "," +
                    std::to_string(skyFaceTexIndices_[2]) + "," + std::to_string(skyFaceTexIndices_[3]) + "," +
                    std::to_string(skyFaceTexIndices_[4]) + "," + std::to_string(skyFaceTexIndices_[5]) +
                    " sky[0..5] glids=" + std::to_string(sky[0]?sky[0]->id:0) + "," + std::to_string(sky[1]?sky[1]->id:0) + "," +
                    std::to_string(sky[2]?sky[2]->id:0) + "," + std::to_string(sky[3]?sky[3]->id:0) + "," +
                    std::to_string(sky[4]?sky[4]->id:0) + "," + std::to_string(sky[5]?sky[5]->id:0));
            }

            {
                static std::vector<unsigned char> blueNoiseRgba;
                const int kBn = 64;
                if (blueNoiseRgba.empty()) {
                    const std::vector<uint32_t> r0 = generateBlueNoiseRanks(kBn, 0x1234567u);
                    const std::vector<uint32_t> r1 = generateBlueNoiseRanks(kBn, 0x89ABCDEu);
                    blueNoiseRgba.resize(static_cast<size_t>(kBn) * kBn * 4);
                    for (int i = 0; i < kBn * kBn; ++i) {
                        blueNoiseRgba[i * 4 + 0] = static_cast<unsigned char>(r0[i] * 256u / uint32_t(kBn * kBn));
                        blueNoiseRgba[i * 4 + 1] = static_cast<unsigned char>(r1[i] * 256u / uint32_t(kBn * kBn));
                        blueNoiseRgba[i * 4 + 2] = 0;
                        blueNoiseRgba[i * 4 + 3] = 255;
                    }
                }
                const uint32_t kBlueNoiseFakeGlId = 0x7FFFB1E0u;
                uint32_t idx = 0xFFFFFFFFu;
                if (!scene_->isPathTraceTextureRegistered(kBlueNoiseFakeGlId, &idx) &&
                    !scene_->registerPathTraceTexture(kBlueNoiseFakeGlId, kBn, kBn, blueNoiseRgba.data(), blueNoiseRgba.size(), 0, &idx)) {
                    logmsg("rebuildGeometry: blue-noise texture registration failed: " + ctx_->lastError());
                    idx = 0xFFFFFFFFu;
                }
                blueNoiseTexIndex_ = idx;
            }

            // the world blend map as one texture
            {
                const int res = std::max(1, std::min(worldsize / 2, 2048));
                const float texel = float(worldsize) / float(res);
                std::vector<unsigned char> bm(static_cast<size_t>(res) * res * 4);
                BlendMapCache *bmCache = newblendmapcache();
                setblendmaporigin(bmCache, ivec(0, 0, 0), worldsize);
                for (int y = 0; y < res; ++y)
                    for (int x = 0; x < res; ++x) {
                        const uchar v = lookupblendmap(bmCache, vec((x + 0.5f) * texel, (y + 0.5f) * texel, 0.0f));
                        unsigned char* px = &bm[(static_cast<size_t>(y) * res + x) * 4];
                        px[0] = px[1] = px[2] = px[3] = v;
                    }
                freeblendmapcache(bmCache);
                uint32_t bmIdx = 0;
                if (!scene_->isPathTraceTextureRegistered(sauerinterop::kWorldBlendmapGlId, &bmIdx) &&
                    !scene_->registerPathTraceTexture(sauerinterop::kWorldBlendmapGlId, static_cast<uint32_t>(res), static_cast<uint32_t>(res),
                                                      bm.data(), bm.size(), 3u, &bmIdx)) {
                    logmsg("rebuildGeometry: world blend map texture registration failed: " + ctx_->lastError());
                }
            }

            logmsg("rebuildGeometry: extracted " + std::to_string(chunks.size()) +
                " chunks, adding each to RayTracingScene (one real, synchronous Vulkan build per chunk)...");

            uint32_t totalVerts = 0, totalTris = 0, chunksAdded = 0;
            // textures that couldn't be read or registered this rebuild (the table is full on a very
            // big map)
            std::unordered_set<uint32_t> failedTextures;
            std::vector<float> combinedGlVertexData;
            std::vector<uint32_t> combinedGlIndices;
            // combined per-texture batch list
            std::vector<sauerinterop::PrepassBatch> combinedGlBatches;

            // register every material's normal/glow texture (if any) before any chunk/batch/material-
            // table upload below
            bool materialTexturesOk = true;
            for (const sauerinterop::RtMaterialDesc& mat : materials) {
                auto registerOne = [&](uint32_t glId) -> bool {
                    if (glId == 0) return true;
                    if (scene_->isPathTraceTextureRegistered(glId, nullptr)) return true;
                    if (failedTextures.count(glId)) return false;
                    std::vector<unsigned char> rgba;
                    uint32_t tw = 0, th = 0;
                    if (!readTextureForRayTracing(static_cast<GLuint>(glId), rgba, tw, th) ||
                        !scene_->registerPathTraceTexture(glId, tw, th, rgba.data(), rgba.size(), 0, nullptr)) {
                        failedTextures.insert(glId);
                        return false;
                    }
                    return true;
                };
                if (mat.hasNormal && !registerOne(mat.normalTexGlId)) materialTexturesOk = false;
                if (mat.hasGlow && !registerOne(mat.glowTexGlId)) materialTexturesOk = false;
                if (mat.parallax && mat.normalTexGlId != 0) {
                    const uint32_t heightId = mat.normalTexGlId | kPomHeightGlIdBit;
                    if (!scene_->isPathTraceTextureRegistered(heightId, nullptr) && !failedTextures.count(heightId)) {
                        std::vector<unsigned char> rgba;
                        uint32_t tw = 0, th = 0;
                        bool ok = false;
                        if (readTextureForRayTracing(static_cast<GLuint>(mat.normalTexGlId), rgba, tw, th)) {
                            std::vector<unsigned char> chain = buildHeightMaxMipChain(rgba, tw, th);
                            ok = scene_->registerPathTraceTexture(heightId, tw, th, chain.data(), chain.size(), 0, nullptr, true);
                        }
                        if (!ok) failedTextures.insert(heightId);  // counted in the summary below
                    }
                }
                if (mat.diffuseTexGlId != 0 && !registerOne(mat.diffuseTexGlId)) materialTexturesOk = false;
            }
            (void)materialTexturesOk;  // failures are counted in failedTextures and logged once below

            for (auto& chunk : chunks) {
                if (chunksAdded % 20 == 0) {
                    vlogmsg("rebuildGeometry: chunk " + std::to_string(chunksAdded) + "/" + std::to_string(chunks.size()) + "...");
                }
                const uint32_t vertexCount = static_cast<uint32_t>(chunk.rtPositions.size() / 3);

                for (const sauerinterop::ChunkGlBatch& batch : chunk.glBatches) {
                    if (batch.textureGlId == 0) continue;
                    if (scene_->isPathTraceTextureRegistered(batch.textureGlId, nullptr)) continue;
                    if (failedTextures.count(batch.textureGlId)) continue;
                    std::vector<unsigned char> rgba;
                    uint32_t tw = 0, th = 0;
                    if (!readTextureForRayTracing(static_cast<GLuint>(batch.textureGlId), rgba, tw, th) ||
                        !scene_->registerPathTraceTexture(batch.textureGlId, tw, th, rgba.data(), rgba.size(), batch.textureClamp, nullptr)) {
                        failedTextures.insert(batch.textureGlId);
                    }
                }

                {
                    uint32_t lastLm = 0;
                    for (uint32_t lmId : chunk.rtBlendLmTexGlIds) {
                        if (lmId == 0 || lmId == lastLm) continue;
                        lastLm = lmId;
                        if (scene_->isPathTraceTextureRegistered(lmId, nullptr)) continue;
                        std::vector<unsigned char> rgba;
                        uint32_t tw = 0, th = 0;
                        if (!readTextureForRayTracing(static_cast<GLuint>(lmId), rgba, tw, th) ||
                            !scene_->registerPathTraceTexture(lmId, tw, th, rgba.data(), rgba.size(), 3u, nullptr)) {
                            logmsg("rebuildGeometry: blend lightmap " + std::to_string(lmId) + " not registered -- per-vertex blend fallback.");
                        }
                    }
                }

                // materialSlot is this chunk's dense, sequential index (chunksAdded, before the
                // increment below)
                const RbClock::time_point rbBlasStart = RbClock::now();
                const bool rbAdded = scene_->addTexturedChunkMesh(chunk.chunkId, chunk.rtPositions.data(),
                    chunk.rtUvs.data(), chunk.rtTextureGlIds.data(), vertexCount,
                    chunk.rtIndices.data(), static_cast<uint32_t>(chunk.rtIndices.size()),
                    chunksAdded, chunk.avgAlbedo,
                    chunk.rtNormals.data(), chunk.rtTangents.data(), chunk.rtMaterialIndices.data(),
                    chunk.opaqueIndexCount,
                    // texture blending
                    chunk.rtBlendMaterialIndices.data(), chunk.rtBlendAlphas.data(), chunk.rtBlendUvs.data(),
                    // mapmodel alpha-test
                    chunk.rtAlphaTestCutoffs.data(),
                    // see ChunkMeshData::rtBlendLmUvs
                    chunk.rtBlendLmUvs.size() == static_cast<size_t>(vertexCount) * 2 ? chunk.rtBlendLmUvs.data() : nullptr,
                    chunk.rtBlendLmTexGlIds.size() == static_cast<size_t>(vertexCount) ? chunk.rtBlendLmTexGlIds.data() : nullptr);
                rbBlasMs += rbMs(rbBlasStart, RbClock::now());
                if (!rbAdded) {
                    char idBuf[24];
                    std::snprintf(idBuf, sizeof(idBuf), "0x%llx", static_cast<unsigned long long>(chunk.chunkId));
                    logmsg(std::string("RayTracingScene::addChunkMesh failed for chunk ") + idBuf + ": " + ctx_->lastError());
                    continue;
                }
                const uint32_t vertexBase = static_cast<uint32_t>(combinedGlVertexData.size() / 12);
                const uint32_t indexBase = static_cast<uint32_t>(combinedGlIndices.size());
                combinedGlVertexData.insert(combinedGlVertexData.end(), chunk.glVertexData.begin(), chunk.glVertexData.end());
                for (uint32_t idx : chunk.glIndices) {
                    combinedGlIndices.push_back(idx + vertexBase);
                }
                for (const sauerinterop::ChunkGlBatch& srcBatch : chunk.glBatches) {
                    sauerinterop::PrepassBatch b;
                    b.textureGlId = srcBatch.textureGlId;
                    b.indexOffset = indexBase + srcBatch.indexOffset;
                    b.indexCount = srcBatch.indexCount;
                    b.alphaTestCutoff = srcBatch.alphaTestCutoff;  // see PrepassBatch::alphaTestCutoff
                    if (srcBatch.materialIndex < materials.size()) {
                        const sauerinterop::RtMaterialDesc& mat = materials[srcBatch.materialIndex];
                        b.normalTexGlId = mat.normalTexGlId;
                        b.glowTexGlId = mat.glowTexGlId;
                        b.hasNormal = mat.hasNormal;
                        b.hasSpec = mat.hasSpec;
                        b.hasGlow = mat.hasGlow;
                        b.colorscale[0]=mat.colorscale[0]; b.colorscale[1]=mat.colorscale[1]; b.colorscale[2]=mat.colorscale[2];
                        b.glowcolor[0]=mat.glowcolor[0]; b.glowcolor[1]=mat.glowcolor[1]; b.glowcolor[2]=mat.glowcolor[2];
                        // pbr light transport
                        b.metalness = mat.metalness;
                        b.specScale = mat.specScale;
                        b.roughness = mat.roughness;
                        b.uvScroll[0] = mat.scroll[0]; b.uvScroll[1] = mat.scroll[1];  // scrolling textures
                    }
                    if (b.indexCount > 0) combinedGlBatches.push_back(b);
                }
                totalVerts += vertexCount;
                totalTris += static_cast<uint32_t>(chunk.rtIndices.size() / 3);
                ++chunksAdded;
            }

            if (chunksAdded == 0) {
                logmsg("rebuildGeometry: every chunk failed to add -- see the RayTracingScene::addChunkMesh errors above.");
                hasGeometry_ = false;
                return;
            }

            {
                const RbClock::time_point rbFlushStart = RbClock::now();
                if (!scene_->flushChunkBuilds()) logmsg("RayTracingScene::flushChunkBuilds: some chunks failed: " + ctx_->lastError());
                rbBlasMs += rbMs(rbFlushStart, RbClock::now());
                const auto fs = scene_->lastFlushStats();
                char fb[256];
                std::snprintf(fb, sizeof(fb), "blas flush: %u chunks in %u batches | create %.0f ms, build %.0f ms, compact %.0f ms | chunk staging (CPU) %.0f ms",
                              fs.chunks, fs.batches, fs.createMs, fs.buildMs, fs.compactMs, rbBlasMs - rbMs(rbFlushStart, RbClock::now()));
                logmsg(fb);
            }
            hasGeometry_ = true;
            const RbClock::time_point rbChunksDone = RbClock::now();
            std::string err;
            if (!prepass_.uploadMesh(combinedGlVertexData.data(), static_cast<uint32_t>(combinedGlVertexData.size() / 12),
                combinedGlIndices.data(), static_cast<uint32_t>(combinedGlIndices.size()),
                combinedGlBatches.data(), static_cast<uint32_t>(combinedGlBatches.size()), &err)) {
                logmsg("RtaoPrepass::uploadMesh failed: " + err);
            }
            if (!scene_->buildPathTraceMaterialTable(&err)) {
                logmsg("RayTracingScene::buildPathTraceMaterialTable failed: " + err);
            }
            buildEmitterLights();  // lights of glowing models

            // uploads the Sauerbraten-VSlot-indexed material table
            {
                std::vector<interop::RayTracingScene::PathTraceMaterialInput> materialInputs(materials.size());
                for (size_t i = 0; i < materials.size(); ++i) {
                    const sauerinterop::RtMaterialDesc& src = materials[i];
                    interop::RayTracingScene::PathTraceMaterialInput& dst = materialInputs[i];
                    dst.diffuseTexGlId = src.diffuseTexGlId;
                    dst.normalTexGlId = src.normalTexGlId;
                    dst.glowTexGlId = src.glowTexGlId;
                    dst.hasNormal = src.hasNormal;
                    dst.hasSpec = src.hasSpec;
                    dst.hasGlow = src.hasGlow;
                    dst.uniformSpec = src.uniformSpec;
                    dst.parallax = src.parallax;
                    dst.parallaxScale = src.parallaxScale;
                    dst.parallaxBias = src.parallaxBias;
                    dst.heightTexGlId = src.parallax ? (src.normalTexGlId | kPomHeightGlIdBit) : 0u;
                    dst.colorscale[0]=src.colorscale[0]; dst.colorscale[1]=src.colorscale[1]; dst.colorscale[2]=src.colorscale[2];
                    dst.glowcolor[0]=src.glowcolor[0]; dst.glowcolor[1]=src.glowcolor[1]; dst.glowcolor[2]=src.glowcolor[2];
                    // pbr light transport
                    dst.metalness = src.metalness;
                    dst.specScale = src.specScale;
                    dst.roughness = src.roughness;  // roughness override
                    dst.isGlass = src.isGlass;
                    dst.glassAlpha = src.glassAlpha;
                    // animated water
                    dst.isWater = src.isWater;
                    dst.isLava = src.isLava;  // path-traced lava
                    dst.glowSelectLum = src.glowSelectLum;  // fog glow MIS
                    dst.waterReferenceThickness = src.waterReferenceThickness;
                    dst.scroll[0] = src.scroll[0]; dst.scroll[1] = src.scroll[1];  // scrolling textures
                }
                {
                    int scrolling = 0;
                    for (const auto& m : materials) if (m.scroll[0] != 0.0f || m.scroll[1] != 0.0f) ++scrolling;
                    if (scrolling) logmsg("scrolling textures: " + std::to_string(scrolling) + " materials.");
                }
                if (!scene_->setPathTraceMaterials(materialInputs.data(), static_cast<uint32_t>(materialInputs.size()), &err)) {
                    logmsg("RayTracingScene::setPathTraceMaterials failed: " + err);
                }
            }

            // uploads the emissive-triangle list
            {
                std::vector<float> emissivePacked(emissiveTriangles.size() * 13);
                for (size_t i = 0; i < emissiveTriangles.size(); ++i) {
                    const sauerinterop::EmissiveTriangleDesc& t = emissiveTriangles[i];
                    float* dst = emissivePacked.data() + i * 13;
                    dst[0]=t.v0[0]; dst[1]=t.v0[1]; dst[2]=t.v0[2];
                    dst[3]=t.v1[0]; dst[4]=t.v1[1]; dst[5]=t.v1[2];
                    dst[6]=t.v2[0]; dst[7]=t.v2[1]; dst[8]=t.v2[2];
                    dst[12]=t.procedural;  // path-traced lava
                    dst[9]=t.radiance[0]; dst[10]=t.radiance[1]; dst[11]=t.radiance[2];
                }
                if (!scene_->setPathTraceEmissiveTriangles(emissivePacked.empty() ? nullptr : emissivePacked.data(),
                    static_cast<uint32_t>(emissiveTriangles.size()), &err)) {
                    logmsg("RayTracingScene::setPathTraceEmissiveTriangles failed: " + err);
                }
            }

            if (!failedTextures.empty())
                logmsg("rebuildGeometry: " + std::to_string(failedTextures.size()) + " texture(s) couldn't be loaded for path tracing (" +
                       ctx_->lastError() + ") -- shown as grey, geometry kept.");
            logmsg("staged geometry: " + std::to_string(chunksAdded) + "/" + std::to_string(chunks.size()) + " vtxarrays, " +
                std::to_string(totalVerts) + " verts, " + std::to_string(totalTris) + " tris, " +
                std::to_string(combinedGlBatches.size()) + " GL texture batches, " +
                std::to_string(materials.size()) + " materials, " +
                std::to_string(emissiveTriangles.size()) + " emissive triangles.");
            const RbClock::time_point rbEnd = RbClock::now();
            char rbBuf[512];
            std::snprintf(rbBuf, sizeof(rbBuf),
                "rebuild timing: total %.0f ms | engine allchanged %.0f | extract %.0f | reset %.0f | chunk loop %.0f (BLAS %.0f, textures etc %.0f) | tables/lights/upload %.0f",
                rbMs(rbStart, rbEnd), rbMs(rbStart, rbEngineDone), rbMs(rbEngineDone, rbExtractDone), rbMs(rbExtractDone, rbResetDone),
                rbMs(rbResetDone, rbChunksDone), rbBlasMs, rbMs(rbResetDone, rbChunksDone) - rbBlasMs, rbMs(rbChunksDone, rbEnd));
            logmsg(rbBuf);
        }

        void rtaorebuild() { rebuildGeometry(); }
        COMMAND(rtaorebuild, "");
        // empties it (it also clears on map load and when ptcache is switched back on)
        void ptcacheclear() { cacheClearPending_ = true; }
        COMMAND(ptcacheclear, "");


        const char* kBlitVs = R"GLSL(
#version 330 core
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

        const char* kBlitFs = R"GLSL(
#version 330 core
out vec4 fragColor;
uniform sampler2D uAO;
uniform sampler2D uShadow;
uniform int uMode;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (uMode == 1) {
        float ao = texelFetch(uAO, p, 0).r;
        float shadow = texelFetch(uShadow, p, 0).r;
        fragColor = vec4(vec3(ao * shadow), 1.0);
    } else {
        fragColor = vec4(vec3(texelFetch(uAO, p, 0).r), 1.0);
    }
}
)GLSL";

        const char* kCompositeFs = R"GLSL(
#version 330 core
out vec4 fragColor;
uniform sampler2D uAO;
uniform sampler2D uShadow;
uniform float uShadowIntensity;
void main() {
    float shadow = texelFetch(uShadow, ivec2(gl_FragCoord.xy), 0).r;
    shadow = mix(1.0, shadow, clamp(uShadowIntensity, 0.0, 1.0));

    float ao = texelFetch(uAO, ivec2(gl_FragCoord.xy), 0).r;
    // shadowOutput_ always holds a valid, meaningful value once executeRtao() has run at least once
    fragColor = vec4(vec3(ao * shadow), 1.0);
}
)GLSL";

        const char* kPathTraceCompositeFs = R"GLSL(
#version 330 core
out vec4 fragColor;
uniform sampler2D uPathTrace;
uniform sampler2D uAO;
uniform float uAOEnabled;
uniform float uExposure;
uniform float uContrast;
uniform float uSaturation;
uniform float uHdr;
uniform int uLut;
// uAeTex (1x1) = the adapted auto exposure, multiplied into uExposure when uAeEnabled
uniform sampler2D uAeTex;
uniform float uAeEnabled;
uniform float uSkyGain;
uniform float uRrMode;
uniform float uHasValidDepth;
uniform sampler2D uDepthSource;
uniform vec2 uOutputSizeInv;
uniform sampler2D uRtMask;
uniform float uRtMaskEnabled;
uniform float uNear;
uniform float uFar;
// Sauerbraten map fog
uniform float uMapFogEnabled;
uniform vec3 uMapFogColor;
uniform vec2 uMapFogParams;
uniform sampler3D uVolFogVolume;
uniform float uVolFogEnabled;
uniform sampler2D uLinearDepth;
uniform vec2 uTanHalfFov;  // tan(fovx/2), tan(fovy/2), map fog's distance -> view depth
uniform vec2 uVolFogNearFar;
uniform float uVolFogGridZ;
uniform float uVolFogDebugMode;
// far-fog tail beyond the froxel grid (see the volfogfar cvar)
uniform mat4 uVolFogInvViewProj;
uniform vec3 uVolFogCamPos;
uniform vec4 uVolFogTailDensity;  // baseDensity, baseHeight, falloff, far distance
uniform vec4 uVolFogTailSun;  // toward-sun dir, phase g
uniform vec3 uVolFogTailSunColor;
uniform vec3 uVolFogTailAmbient;
uniform vec3 uVolFogTailBase;
vec4 volFogTail(vec2 screenUv, float dist) {
    float s0 = uVolFogNearFar.y;
    float s1 = min(dist, uVolFogTailDensity.w);
    if (s1 <= s0 || uVolFogTailDensity.x <= 0.0) return vec4(0.0, 0.0, 0.0, 1.0);
    vec4 wf = uVolFogInvViewProj * vec4(screenUv * 2.0 - 1.0, 1.0, 1.0);
    vec3 dir = normalize(wf.xyz / wf.w - uVolFogCamPos);
    float L = s1 - s0;
    float k = uVolFogTailDensity.z;
    float a = max(uVolFogCamPos.z + dir.z * s0 - uVolFogTailDensity.y, 0.0);
    float b = max(uVolFogCamPos.z + dir.z * s1 - uVolFogTailDensity.y, 0.0);
    float tau = (abs(k * (b - a)) < 1e-4)
        ? uVolFogTailDensity.x * L * exp(-k * 0.5 * (a + b))
        : uVolFogTailDensity.x * L * (exp(-k * a) - exp(-k * b)) / (k * (b - a));
    float T = exp(-tau);
    float g = uVolFogTailSun.w;
    float cosT = dot(dir, normalize(uVolFogTailSun.xyz));
    float denom = max(1.0 + g * g - 2.0 * g * cosT, 1e-4);
    float phase = (1.0 - g * g) / (4.0 * 3.14159265359 * denom * sqrt(denom));
    vec3 Lin = uVolFogTailBase * (uVolFogTailSunColor * phase + uVolFogTailAmbient);
    return vec4(Lin * (1.0 - T), T);
}

float interleavedGradientNoise(vec2 fragCoord) {
    return fract(52.9829189 * fract(dot(fragCoord, vec2(0.06711056, 0.00583715))));
}

vec4 sampleVolFogBilateral(vec2 screenUv, ivec2 centerTexel, float centerDist) {
    ivec2 depthDims = textureSize(uLinearDepth, 0);
    ivec3 volDims = textureSize(uVolFogVolume, 0);
    ivec2 texStep = max(depthDims / max(volDims.xy, ivec2(1)), ivec2(1));
    ivec2 offsets[5] = ivec2[](ivec2(0, 0), ivec2(texStep.x, 0), ivec2(-texStep.x, 0), ivec2(0, texStep.y), ivec2(0, -texStep.y));
    vec4 sum = vec4(0.0);
    float wsum = 0.0;
    for (int i = 0; i < 5; ++i) {
        ivec2 sTexel = clamp(centerTexel + offsets[i], ivec2(0), depthDims - ivec2(1));
        float sDist = texelFetch(uLinearDepth, sTexel, 0).r;
        vec2 sUv = (vec2(sTexel) + 0.5) / vec2(depthDims);
        float wCoordS = clamp(log(max(sDist, uVolFogNearFar.x) / uVolFogNearFar.x) / log(uVolFogNearFar.y / uVolFogNearFar.x), 0.0, 1.0);
        vec4 s = texture(uVolFogVolume, vec3(sUv.x, 1.0 - sUv.y, wCoordS));
        float depthWeight = 1.0 / (1.0 + abs(sDist - centerDist) / max(centerDist * 0.1, 1.0));
        sum += s * depthWeight;
        wsum += depthWeight;
    }
    return sum / max(wsum, 1e-4);
}

// Narkowicz's compact analytic fit to the ACES reference filmic tonemap curve
vec3 acesFilmicTonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
// neutral display curve (pathtracehdr 0)
const float NEUTRAL_SHOULDER = 0.8;
float neutralShoulder(float v) {
    if (v <= NEUTRAL_SHOULDER) return v;
    float r = 1.0 - NEUTRAL_SHOULDER;
    return NEUTRAL_SHOULDER + r * (1.0 - exp(-(v - NEUTRAL_SHOULDER) / r));
}
float neutralShoulderInv(float y) {
    if (y <= NEUTRAL_SHOULDER) return y;
    float r = 1.0 - NEUTRAL_SHOULDER;
    return NEUTRAL_SHOULDER - r * log(max(1.0 - (y - NEUTRAL_SHOULDER) / r, 1e-6));
}
vec3 neutralTonemap(vec3 x) {
    x = max(x, vec3(0.0));
    float m = max(max(x.r, x.g), x.b);
    return m <= NEUTRAL_SHOULDER ? x : x * (neutralShoulder(m) / m);
}

// pathtracehdr's alternate operator
vec3 uncharted2TonemapPartial(vec3 x) {
    const float A = 0.15;  // shoulder strength
    const float B = 0.50;  // linear strength
    const float C = 0.10;  // linear angle
    const float D = 0.20;  // toe strength
    const float E = 0.02;  // toe numerator
    const float F = 0.30;  // toe denominator
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}
vec3 uncharted2Filmic(vec3 x) {
    const float kHdrWhite = 11.2;
    vec3 curr = uncharted2TonemapPartial(x);
    vec3 whiteScale = vec3(1.0) / uncharted2TonemapPartial(vec3(kHdrWhite));
    return clamp(curr * whiteScale, 0.0, 1.0);
}
// hdr display pipeline (pthdr) curves
vec3 agxContrastApprox(vec3 x) {
    vec3 x2 = x * x, x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
vec3 agxPunchyTonemap(vec3 x) {
    const mat3 agxMat = mat3(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                             0.0784335999999992, 0.878468636469772, 0.0784336,
                             0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    const mat3 agxMatInv = mat3(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                                -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                                -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    const float minEv = -12.47393, maxEv = 4.026069;
    vec3 v = agxMat * max(x, vec3(1e-10));
    v = clamp(log2(v), minEv, maxEv);
    v = (v - minEv) / (maxEv - minEv);
    v = agxContrastApprox(v);
    // punchy look (slope 1, power 1.35, saturation 1.4), on the encoded value
    float luma = dot(v, vec3(0.2126, 0.7152, 0.0722));
    v = pow(max(v, vec3(0.0)), vec3(1.35));
    luma = dot(v, vec3(0.2126, 0.7152, 0.0722));
    v = luma + 1.4 * (v - luma);
    v = agxMatInv * v;
    return clamp(pow(max(v, vec3(0.0)), vec3(2.2)), 0.0, 1.0);  // encoded (2.2) -> linear display
}
vec3 pbrNeutralTonemap(vec3 color) {
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;
    color = max(color, vec3(0.0));
    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;
    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return clamp(color, 0.0, 1.0);
    const float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;
    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return clamp(mix(color, vec3(newPeak), g), 0.0, 1.0);
}
vec3 displayTonemap(vec3 x, float mode) {
    if (mode > 3.5) return pbrNeutralTonemap(x);
    if (mode > 2.5) return agxPunchyTonemap(x);
    if (mode > 1.5) return neutralTonemap(x);
    if (mode > 0.5) return uncharted2Filmic(x);
    return acesFilmicTonemap(x);
}

// asc cdl (Slope/Offset/Power) color grading
vec3 gradeCdl(vec3 c, vec3 slope, vec3 offset, vec3 power) {
    return pow(clamp(c * slope + offset, 0.0, 1.0), power);
}

vec3 filmicSCurveLuma(vec3 c, float amount) {
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float shaped = mix(luma, smoothstep(0.0, 1.0, luma), amount);
    return c * (shaped / max(luma, 1e-4));
}

// pathtracelut's filmic grading presets
vec3 applyLut(vec3 c, int lut) {
    if (lut == 0) return c;

    float maxc = max(c.r, max(c.g, c.b));
    float minc = min(c.r, min(c.g, c.b));
    float sat = maxc - minc;
    float tintStrength = 1.0 - smoothstep(0.15, 0.55, sat);

    vec3 slope = vec3(1.0), offset = vec3(0.0), power = vec3(1.0);
    float curveAmount = 0.0, satBoost = 1.0;
    if (lut == 1) {
        // a quiet, barely-there cinematic polish, a hair of warmth, a hair of extra punch
        slope = vec3(1.012, 1.0, 0.99); offset = vec3(0.0012, 0.0, 0.0018); power = vec3(0.99, 1.0, 1.008);
        curveAmount = 0.05; satBoost = 1.015;
    } else if (lut == 2) {
        // warm blockbuster teal-orange, cool-leaning shadows, warm-leaning highlights, a gentle punch
        slope = vec3(1.035, 1.012, 0.965); offset = vec3(-0.006, -0.002, 0.008); power = vec3(0.975, 1.0, 1.03);
        curveAmount = 0.12; satBoost = 1.05;
    } else if (lut == 3) {
        // moody cyberpunk, teal-leaning shadows, magenta-leaning highlights
        slope = vec3(1.03, 0.985, 1.03); offset = vec3(-0.004, 0.006, 0.008); power = vec3(1.012, 0.99, 1.0);
        curveAmount = 0.14; satBoost = 1.06;
    }

    vec3 graded = gradeCdl(c, mix(vec3(1.0), slope, tintStrength), offset * tintStrength, mix(vec3(1.0), power, tintStrength));
    graded = filmicSCurveLuma(graded, curveAmount);
    float gLuma = dot(graded, vec3(0.2126, 0.7152, 0.0722));
    graded = mix(vec3(gLuma), graded, satBoost);
    return clamp(graded, 0.0, 1.0);
}

float applyWorldSpaceBias(float rawDepthIn, float worldBias) {
    float ndc = rawDepthIn * 2.0 - 1.0;
    float denom = uFar + uNear - ndc * (uFar - uNear);
    if (denom <= 1e-6) return rawDepthIn;
    float linearZ = (2.0 * uNear * uFar) / denom;
    float biasedZ = max(linearZ - worldBias, uNear);  // never push past the near plane itself
    float biasedNdc = (uFar + uNear - (2.0 * uNear * uFar) / biasedZ) / (uFar - uNear);
    return clamp(biasedNdc * 0.5 + 0.5, 0.0, 1.0);
}

void main() {
    vec2 pixel = gl_FragCoord.xy;

    if (uRtMaskEnabled > 0.5 && texelFetch(uRtMask, ivec2(pixel), 0).r > 0.5) discard;

    vec3 hdr = texelFetch(uPathTrace, ivec2(pixel), 0).rgb;

    ivec2 depthTexel = ivec2(pixel * uOutputSizeInv * vec2(textureSize(uDepthSource, 0)));
    float encoded = texelFetch(uDepthSource, depthTexel, 0).a;
    bool isDynamicHit = encoded < 0.0;  // sign-encoded dynamic-hit flag, see pathtrace_trace.comp
    float rawDepth = abs(encoded);
    // the visible sky at pthdrsky x (see uSkyGain)
    if (rawDepth >= 1.0) hdr *= uSkyGain;

    // rtaoblend is the same independent switch in both rendering modes
    if (uAOEnabled > 0.5 && !isDynamicHit) {
        float ao = texture(uAO, pixel * uOutputSizeInv).r;
        hdr *= ao;
    }

    if (uVolFogEnabled > 0.5) {
        vec2 screenUv = pixel * uOutputSizeInv;
        float dist = texelFetch(uLinearDepth, depthTexel, 0).r;
        float wCoordRaw = clamp(log(max(dist, uVolFogNearFar.x) / uVolFogNearFar.x) / log(uVolFogNearFar.y / uVolFogNearFar.x), 0.0, 1.0);
        if (uVolFogDebugMode > 0.5 && uVolFogDebugMode < 1.5) { fragColor = vec4(vec3(wCoordRaw), 1.0); return; }
        vec4 fog;
        if (uVolFogDebugMode > 6.5) {
            fog = texture(uVolFogVolume, vec3(screenUv.x, 1.0 - screenUv.y, wCoordRaw));
        } else if (uVolFogDebugMode > 3.5 && uVolFogDebugMode < 4.5) {
            fog = vec4(0.6, 0.6, 0.65, 0.85);
        } else {
            fog = sampleVolFogBilateral(screenUv, depthTexel, dist);
        }
        if (uVolFogDebugMode > 1.5 && uVolFogDebugMode < 2.5) { fragColor = vec4(fog.rgb, 1.0); return; }
        if (uVolFogDebugMode > 2.5 && uVolFogDebugMode < 3.5) { fragColor = vec4(vec3(fog.a), 1.0); return; }
        if (uVolFogDebugMode > 4.5) { fragColor = vec4(fog.rgb, 1.0); return; }
        vec4 tail = volFogTail(screenUv, dist);
        hdr = hdr * (fog.a * tail.a) + fog.rgb + fog.a * tail.rgb;
    }

    // auto exposure on top of pathtraceexposure
    float exposure = uExposure * (uAeEnabled > 0.5 ? texelFetch(uAeTex, ivec2(0), 0).r : 1.0);
    vec3 mapped = displayTonemap(hdr * exposure, (uRrMode > 0.5 && uHdr > 2.5) ? 2.0 : uHdr);

    if (uRrMode < 0.5) mapped = clamp((mapped - 0.5) * uContrast + 0.5, 0.0, 1.0);

    if (uRrMode < 0.5) {
        float luma = dot(mapped, vec3(0.2126, 0.7152, 0.0722));
        mapped = clamp(mix(vec3(luma), mapped, uSaturation), 0.0, 1.0);
        mapped = applyLut(mapped, uLut);
    }

    mapped = pow(mapped, vec3(1.0 / 2.2));
    // Sauerbraten's map fog (fog/fogcolour, or water/lava fog when the camera is in them)
    float mapFogLinear = texelFetch(uLinearDepth, depthTexel, 0).r;
    bool mapFogSky = rawDepth >= 1.0 || !(mapFogLinear < 60000.0);
    if (uMapFogEnabled > 0.5 && !mapFogSky) {
        float fogNdc = rawDepth * 2.0 - 1.0;
        float fogViewZ = (2.0 * uNear * uFar) / max(uFar + uNear - fogNdc * (uFar - uNear), 1e-6);
        vec2 fogNdcXY = pixel * uOutputSizeInv * 2.0 - 1.0;
        float rayLenPerViewZ = length(vec3(fogNdcXY * uTanHalfFov, 1.0));
        fogViewZ = max(fogViewZ, mapFogLinear / rayLenPerViewZ);
        float fogKeep = clamp(uMapFogParams.y - fogViewZ * uMapFogParams.x, 0.0, 1.0);
        mapped = mix(uMapFogColor, mapped, fogKeep);
    }
    if (uRrMode < 0.5) {
        float finalDither = interleavedGradientNoise(pixel) - 0.5;
        mapped = clamp(mapped + finalDither / 255.0, 0.0, 1.0);
    }
    fragColor = vec4(mapped, 1.0);

    if (uHasValidDepth > 0.5) {
        if (isDynamicHit) {
            gl_FragDepth = applyWorldSpaceBias(rawDepth, 0.05);
        } else {
            const float kDepthBias = 0.0003;
            gl_FragDepth = clamp(rawDepth - kDepthBias, 0.0, 1.0);
        }
    }
}
)GLSL";

        const char* kPathTraceDepthFs = R"GLSL(
#version 330 core
uniform sampler2D uPathTrace;
uniform vec2 uOutputSizeInv;
uniform sampler2D uRtMask;
uniform float uRtMaskEnabled;
void main() {
    vec2 pixel = gl_FragCoord.xy;
    if (uRtMaskEnabled > 0.5 && texelFetch(uRtMask, ivec2(pixel), 0).r > 0.5) discard;
    ivec2 depthTexel = ivec2(pixel * uOutputSizeInv * vec2(textureSize(uPathTrace, 0)));
    float encoded = texelFetch(uPathTrace, depthTexel, 0).a;
    if (encoded >= 0.0) discard;
    gl_FragDepth = -encoded;
}
)GLSL";

        const char* kBloomThresholdFs = R"GLSL(
#version 330 core
out vec4 fragColor;
uniform sampler2D uScene;
uniform vec2 uOutputSizeInv;
uniform float uThreshold;
void main() {
    vec2 uv = gl_FragCoord.xy * uOutputSizeInv;
    vec3 color = texture(uScene, uv).rgb;
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float weight = clamp((luma - uThreshold) / max(luma, 1e-4), 0.0, 1.0);
    fragColor = vec4(color * weight, 1.0);
}
)GLSL";

        const char* kBloomBlurFs = R"GLSL(
#version 330 core
out vec4 fragColor;
uniform sampler2D uSrc;
uniform vec2 uOutputSizeInv;
uniform vec2 uDirection;
void main() {
    vec2 uv = gl_FragCoord.xy * uOutputSizeInv;
    float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec3 result = texture(uSrc, uv).rgb * weights[0];
    for (int i = 1; i < 5; ++i) {
        vec2 offset = uDirection * uOutputSizeInv * float(i) * 1.5;
        result += texture(uSrc, uv + offset).rgb * weights[i];
        result += texture(uSrc, uv - offset).rgb * weights[i];
    }
    fragColor = vec4(result, 1.0);
}
)GLSL";

        // auto exposure, pass 1
        const char* kAeMeterFs = R"GLSL(
#version 330 core
out vec4 fragColor;
uniform sampler2D uSrc;
uniform vec2 uSrcSize;
void main() {
    vec2 cell = uSrcSize / 64.0;
    float acc = 0.0;
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
            vec2 p = (floor(gl_FragCoord.xy) + (vec2(i, j) + 0.5) * 0.5) * cell;
            vec3 c = texelFetch(uSrc, ivec2(clamp(p, vec2(0.0), uSrcSize - 1.0)), 0).rgb;
            float l = dot(max(c, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
            acc += log2(clamp(l, 1e-4, 64.0));
        }
    fragColor = vec4(acc * 0.25, 0.0, 0.0, 1.0);
}
)GLSL";
        // pass 2 (1x1)
        const char* kAeAdaptFs = R"GLSL(
#version 330 core
out vec4 fragColor;
uniform sampler2D uMeter;
uniform sampler2D uPrev;
uniform float uKey, uAdapt, uEv, uBlend, uReset;
void main() {
    float sum = 0.0, wsum = 0.0;
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x) {
            vec2 d = (vec2(x, y) + 0.5) / 64.0 - 0.5;
            float w = 1.0 + 2.0 * max(0.0, 1.0 - 2.5 * length(d));  // centre-weighted
            sum += texelFetch(uMeter, ivec2(x, y), 0).r * w;
            wsum += w;
        }
    float avgLum = exp2(sum / wsum);
    float target = clamp(pow(uKey / max(avgLum, 1e-4), uAdapt) * exp2(uEv), 0.25, 8.0);
    float prev = texelFetch(uPrev, ivec2(0), 0).r;
    float e = (uReset > 0.5 || !(prev > 0.0)) ? target : exp2(mix(log2(prev), log2(target), uBlend));
    fragColor = vec4(e, avgLum, 0.0, 1.0);
}
)GLSL";

        const char* kPresentSceneFs = R"GLSL(
#version 330 core
out vec4 fragColor;
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uBloomIntensity;
uniform vec2 uOutputSizeInv;
void main() {
    vec2 uv = gl_FragCoord.xy * uOutputSizeInv;
    vec4 scene = texture(uScene, uv);
    vec3 bloom = texture(uBloom, uv).rgb * uBloomIntensity;
    vec3 sceneWithBloom = scene.rgb + bloom;
    float presentDither = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))) - 0.5;
    fragColor = vec4(clamp(sceneWithBloom + presentDither / 255.0, 0.0, 1.0), scene.a);
}
)GLSL";

        const char* kSceneDepthRestoreFs = R"GLSL(
#version 330 core
uniform sampler2D uDepth;
uniform vec2 uOutputSizeInv;
void main() {
    vec2 uv = gl_FragCoord.xy * uOutputSizeInv;
    gl_FragDepth = texture(uDepth, uv).r;
}
)GLSL";

        const char* kDlssSplitFs = R"GLSL(
#version 330 core
layout(location = 0) out vec4 fragDiffuse;
layout(location = 1) out vec4 fragSpecular;
uniform sampler2D uAlbedo;  // albedoTex_, rgb = combined linear albedo*colorscale
uniform sampler2D uGlow;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 albedo = texelFetch(uAlbedo, p, 0).rgb;
    float packedA = texelFetch(uGlow, p, 0).a;
    float metalness = packedA < 0.5 ? clamp(packedA / 0.49, 0.0, 1.0) : 0.0;
    float envRefl = packedA >= 0.5 ? clamp((packedA - 0.51) / 0.49, 0.0, 1.0) : 0.0;
    vec3 F0 = mix(vec3(0.04), albedo, metalness);
    F0 = mix(F0, vec3(1.0), envRefl);  // same as pathtrace_trace.comp's curF0
    fragDiffuse = vec4(albedo * (1.0 - metalness) * (1.0 - envRefl), 1.0);
    fragSpecular = vec4(F0, 1.0);
}
)GLSL";

        // dlss-rr specular motion vectors (Streamline RR guide 4.1.8)
        const char* kRrSpecMotionFs = R"GLSL(
#version 330 core
layout(location = 0) out vec2 fragMotion;
uniform sampler2D uLinearDepth;  // r = primary distance, g = specular virtual distance (0 = none)
uniform sampler2D uMotion;  // surface motion (prevUv - currUv)
uniform mat4 uInvCurrVp;  // this frame's unjittered view-projection, inverted
uniform mat4 uPrevVp;  // last frame's unjittered view-projection
uniform vec3 uCamPos;
uniform vec2 uSizeInv;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 surface = texelFetch(uMotion, p, 0).rg;
    float vd = texelFetch(uLinearDepth, p, 0).g;
    if (!(vd > 0.0)) { fragMotion = surface; return; }
    vec2 uv = gl_FragCoord.xy * uSizeInv;
    vec4 far4 = uInvCurrVp * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 dir = normalize(far4.xyz / far4.w - uCamPos);
    vec4 prevClip = vd >= 59000.0 ? uPrevVp * vec4(dir, 0.0) : uPrevVp * vec4(uCamPos + dir * vd, 1.0);
    if (prevClip.w <= 1e-4) { fragMotion = surface; return; }
    fragMotion = (prevClip.xy / prevClip.w * 0.5 + 0.5) - uv;
}
)GLSL";

        const char* kPtSurfaceMotionFs = R"GLSL(
#version 330 core
layout(location = 0) out vec2 fragMotion;
uniform sampler2D uPtRaw;  // path tracer raw radiance
uniform sampler2D uPrepassDepth;  // prepass hardware depth
uniform mat4 uPtInvVp;  // path tracer's (jittered) inverse view-projection
uniform mat4 uInvVp;  // prepass's (jittered) inverse view-projection
uniform mat4 uCurrVp, uPrevVp;  // unjittered, this frame / last frame
uniform sampler2D uPtNormal;  // path tracer primary normal
uniform vec3 uCamPos;
uniform vec2 uSizeInv;
vec3 unproject(mat4 inv, vec2 uv, float d) { vec4 w = inv * vec4(uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0); return w.xyz / w.w; }
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float d = texelFetch(uPtRaw, p, 0).a;
    if (!(d > 0.0 && d < 1.0)) discard;  // dynamic hit (own motion), sky, or nothing
    vec2 uv = gl_FragCoord.xy * uSizeInv;
    vec3 P = unproject(uPtInvVp, uv, d);
    vec4 n = texelFetch(uPtNormal, p, 0);
    vec3 Pprev = P;
    if (n.r > 2.0) {
        Pprev = P + vec3(n.ba, 0.0);
    } else {
        float ptDist = length(P - uCamPos);
        float pd = texelFetch(uPrepassDepth, p, 0).r;
        float preDist = pd >= 1.0 ? 1e30 : length(unproject(uInvVp, uv, pd) - uCamPos);
        if (ptDist >= preDist - max(0.25, 0.01 * ptDist)) discard;  // the prepass already has this surface
    }
    vec4 c = uCurrVp * vec4(P, 1.0), pr = uPrevVp * vec4(Pprev, 1.0);
    if (pr.w <= 1e-4 || c.w <= 1e-4) { fragMotion = vec2(1000.0); return; }
    fragMotion = (pr.xy / pr.w * 0.5 + 0.5) - (c.xy / c.w * 0.5 + 0.5);
}
)GLSL";

        const char* kRrWaterGuidesFs = R"GLSL(
#version 330 core
layout(location = 0) out vec4 fragNormalRoughness;
layout(location = 1) out vec4 fragDiffuse;
layout(location = 2) out vec4 fragSpecular;
uniform sampler2D uPtNormal;
uniform sampler2D uPtRaw;
uniform sampler2D uAlbedo;
uniform mat4 uPtInvVp;
uniform mat4 uCam;  // world -> view (the prepass's normal space)
uniform vec3 uCamPos;
uniform vec2 uSizeInv;
vec3 octDecode(vec2 f) {
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = max(-n.z, 0.0);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 s = texelFetch(uPtNormal, p, 0);
    if (!(s.r > 2.0)) discard;
    vec3 N = octDecode(vec2(s.r - 4.0, s.g));
    float d = texelFetch(uPtRaw, p, 0).a;
    vec4 w = uPtInvVp * vec4(gl_FragCoord.xy * uSizeInv * 2.0 - 1.0, clamp(d, 0.0, 1.0) * 2.0 - 1.0, 1.0);
    vec3 V = normalize(uCamPos - w.xyz / w.w);
    if (dot(N, V) < 0.0) N = -N;  // seen from under the water
    float c = clamp(dot(N, V), 0.0, 1.0);
    float F = 0.02 + 0.98 * pow(1.0 - c, 5.0);
    fragNormalRoughness = vec4(normalize(mat3(uCam) * N), 0.02);
    fragDiffuse = vec4(max(texelFetch(uAlbedo, p, 0).rgb, vec3(0.05)) * (1.0 - F), 1.0);
    fragSpecular = vec4(vec3(F), 1.0);
}
)GLSL";

        // drawn over the final frame at window resolution
        const char* kMvDebugFs = R"GLSL(
#version 330 core
out vec4 fragColor;
uniform sampler2D uMv;
uniform sampler2D uMvRef;  // mode 3
uniform int uMode;
uniform vec2 uMvSize, uVpSize;
uniform float uScale, uCell, uGain, uOpacity;
const float TAU = 6.28318530718;
vec3 hue(float h) { return clamp(abs(fract(h + vec3(0.0, 2.0, 1.0) / 3.0) * 6.0 - 3.0) - 1.0, 0.0, 1.0); }
// direction of travel (curr - prev) in screen pixels at a screen position
vec2 travelAt(vec2 frag, out int state) {
    ivec2 t = clamp(ivec2(frag * uMvSize / uVpSize), ivec2(0), ivec2(uMvSize) - 1);
    vec2 m = texelFetch(uMv, t, 0).rg;
    state = 0;
    if (any(isnan(m)) || any(isinf(m))) { state = 2; return vec2(0.0); }
    if (any(greaterThan(abs(m), vec2(100.0)))) { state = 1; return vec2(0.0); }
    if (uMode == 3) {
        vec2 r = texelFetch(uMvRef, t, 0).rg;
        if (any(isnan(r)) || any(isinf(r))) { state = 2; return vec2(0.0); }
        if (any(greaterThan(abs(r), vec2(100.0)))) { state = 1; return vec2(0.0); }
        m -= r;
    }
    return -m * uVpSize;
}
vec3 flowColour(vec2 v) {
    float len = length(v);
    if (len < 1e-6) return vec3(0.0);
    return hue(atan(v.y, v.x) / TAU) * clamp(len / uScale, 0.0, 1.0);
}
float segDist(vec2 q, vec2 a, vec2 b) {
    vec2 ab = b - a; float t = clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    return length(q - a - ab * t);
}
void main() {
    vec2 frag = gl_FragCoord.xy;
    int state;
    vec2 v = travelAt(frag, state);
    vec3 rgb = flowColour(v);
    if (state == 1) rgb = vec3(((int(frag.x) / 8 + int(frag.y) / 8) & 1) == 0 ? 0.35 : 0.55);
    if (state == 2) rgb = (int(frag.x + frag.y) / 6 & 1) == 0 ? vec3(1.0, 0.0, 1.0) : vec3(1.0);
    float alpha = state == 0 ? uOpacity : 1.0;

    // arrows
    if (uCell > 0.0) {
        float line = 1e9, dot0 = 1e9;
        ivec2 cell = ivec2(floor(frag / uCell));
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            vec2 c = (vec2(cell + ivec2(dx, dy)) + 0.5) * uCell;
            int s;
            vec2 a = travelAt(c, s) * uGain;
            if (s != 0) continue;
            float len = length(a);
            dot0 = min(dot0, length(frag - c));
            if (len < 0.5) continue;
            a *= min(len, 1.5 * uCell) / len;
            vec2 tip = c + a, dir = normalize(a), side = vec2(-dir.y, dir.x);
            float head = min(6.0, 0.4 * length(a));
            line = min(line, segDist(frag, c, tip));
            line = min(line, segDist(frag, tip, tip - dir * head + side * head * 0.6));
            line = min(line, segDist(frag, tip, tip - dir * head - side * head * 0.6));
        }
        float outline = smoothstep(2.6, 1.6, min(line, dot0 - 0.5));
        float core = smoothstep(1.4, 0.6, min(line, dot0 - 1.0));
        rgb = mix(rgb, vec3(0.0), outline * (1.0 - core));
        rgb = mix(rgb, vec3(1.0), core);
        alpha = max(alpha, outline);
    }

    // legend
    vec2 lc = vec2(uVpSize.x - 72.0, 72.0);
    float lr = length(frag - lc);
    if (lr < 56.0) {
        vec2 lv = (frag - lc) / 48.0 * uScale;
        rgb = lr < 48.0 ? flowColour(lv) : vec3(0.08);
        if (abs(lr - 48.0) < 1.0 || (lr < 48.0 && min(abs(frag.x - lc.x), abs(frag.y - lc.y)) < 0.5)) rgb = vec3(0.6);
        alpha = 1.0;
    }
    fragColor = vec4(rgb, alpha);
}
)GLSL";

        bool initRrTone(std::string* error) {
            const std::string composite = kPathTraceCompositeFs;
            const size_t fnBegin = composite.find("vec3 acesFilmicTonemap(");
            const std::string endMarker = "    return clamp(graded, 0.0, 1.0);\n}\n";
            const size_t fnEnd = composite.find(endMarker, fnBegin);
            if (fnBegin == std::string::npos || fnEnd == std::string::npos) {
                if (error) *error = "rr tone: could not locate tonemap functions in kPathTraceCompositeFs";
                return false;
            }
            const std::string functions = composite.substr(fnBegin, fnEnd + endMarker.size() - fnBegin);
            const std::string fsSrc = std::string(R"GLSL(#version 330 core
out vec4 fragColor;
uniform sampler2D uSrc;
uniform int uDirection;  // 0 = display -> linear (before RR), 1 = linear -> display (after RR)
uniform float uHdr;
uniform float uContrast;
uniform float uSaturation;
uniform int uLut;
)GLSL") + functions + R"GLSL(
vec3 invAces(vec3 y) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    vec3 A = a - c * y, B = b - d * y, C = -e * y;
    return (-B + sqrt(B * B - 4.0 * A * C)) / (2.0 * A);
}
vec3 invU2(vec3 y) {
    const float A = 0.15, B = 0.50, C = 0.10, D = 0.20, E = 0.02, F = 0.30;
    vec3 ws = vec3(1.0) / uncharted2TonemapPartial(vec3(11.2));
    vec3 z = y / ws + E / F;
    vec3 qa = A * (z - 1.0), qb = B * (z - C), qc = D * (F * z - E);
    return (-qb - sqrt(qb * qb - 4.0 * qa * qc)) / (2.0 * qa);
}
void main() {
    vec3 c = texelFetch(uSrc, ivec2(gl_FragCoord.xy), 0).rgb;
    if (uDirection == 0) {
        vec3 m = min(pow(clamp(c, 0.0, 1.0), vec3(2.2)), vec3(0.999));
        vec3 lin;
        if (uHdr > 1.5) {
            float my = max(max(m.r, m.g), m.b);
            lin = my <= NEUTRAL_SHOULDER ? m : m * (neutralShoulderInv(my) / max(my, 1e-6));
        }
        else if (uHdr > 0.5) lin = invU2(m);
        else lin = invAces(m);
        fragColor = vec4(max(lin, vec3(0.0)), 1.0);
    } else {
        vec3 x = max(c, vec3(0.0));
        vec3 mapped = displayTonemap(x, uHdr);
        mapped = clamp((mapped - 0.5) * uContrast + 0.5, 0.0, 1.0);
        float luma = dot(mapped, vec3(0.2126, 0.7152, 0.0722));
        mapped = clamp(mix(vec3(luma), mapped, uSaturation), 0.0, 1.0);
        mapped = applyLut(mapped, uLut);
        fragColor = vec4(pow(mapped, vec3(1.0 / 2.2)), 1.0);
    }
}
)GLSL";
            auto compile = [&](GLenum type, const char* src, GLuint& outShader) -> bool {
                outShader = glCreateShader_(type);
                glShaderSource_(outShader, 1, &src, nullptr);
                glCompileShader_(outShader);
                GLint ok = 0;
                glGetShaderiv_(outShader, GL_COMPILE_STATUS, &ok);
                if (!ok) {
                    char buf[2048]; GLsizei len = 0;
                    glGetShaderInfoLog_(outShader, sizeof(buf), &len, buf);
                    if (error) *error = std::string("rr tone shader compile failed: ") + std::string(buf, len);
                    return false;
                }
                return true;
            };
            GLuint vs = 0, fs = 0;
            if (!compile(GL_VERTEX_SHADER, kBlitVs, vs)) return false;
            if (!compile(GL_FRAGMENT_SHADER, fsSrc.c_str(), fs)) return false;
            rrToneProgram_ = glCreateProgram_();
            glAttachShader_(rrToneProgram_, vs);
            glAttachShader_(rrToneProgram_, fs);
            glLinkProgram_(rrToneProgram_);
            glDeleteShader_(vs);
            glDeleteShader_(fs);
            GLint linked = 0;
            glGetProgramiv_(rrToneProgram_, GL_LINK_STATUS, &linked);
            if (!linked) {
                char buf[2048]; GLsizei len = 0;
                glGetProgramInfoLog_(rrToneProgram_, sizeof(buf), &len, buf);
                if (error) *error = std::string("rr tone program link failed: ") + std::string(buf, len);
                glDeleteProgram_(rrToneProgram_);
                rrToneProgram_ = 0;
                return false;
            }
            rrToneUSrcLoc_ = glGetUniformLocation_(rrToneProgram_, "uSrc");
            rrToneUDirectionLoc_ = glGetUniformLocation_(rrToneProgram_, "uDirection");
            rrToneUHdrLoc_ = glGetUniformLocation_(rrToneProgram_, "uHdr");
            rrToneUContrastLoc_ = glGetUniformLocation_(rrToneProgram_, "uContrast");
            rrToneUSaturationLoc_ = glGetUniformLocation_(rrToneProgram_, "uSaturation");
            rrToneULutLoc_ = glGetUniformLocation_(rrToneProgram_, "uLut");
            glGenFramebuffers_(1, &rrToneFbo_);
            return true;
        }

        bool runRrTone(GLuint srcTex, interop::SharedTexture* dst, int direction) {
            if (!rrToneProgram_ || !rrToneFbo_ || !debugVao_ || !srcTex || !dst) return false;
            GLint prevVao = 0, prevFbo = 0, prevProgram = 0, prevActiveTex = 0, prevTex0 = 0;
            GLint prevViewport[4] = {0, 0, 0, 0};
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
            glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
            glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
            glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
            glGetIntegerv(GL_VIEWPORT, prevViewport);
            const GLboolean prevDepthTest = glIsEnabled(GL_DEPTH_TEST);
            const GLboolean prevBlend = glIsEnabled(GL_BLEND);
            const GLboolean prevScissor = glIsEnabled(GL_SCISSOR_TEST);
            glActiveTexture_(GL_TEXTURE0);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);

            glBindFramebuffer_(GL_FRAMEBUFFER, rrToneFbo_);
            glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst->glTextureId(), 0);
            const GLenum drawBuf = GL_COLOR_ATTACHMENT0;
            glDrawBuffers_(1, &drawBuf);
            glViewport(0, 0, static_cast<GLsizei>(dst->width()), static_cast<GLsizei>(dst->height()));
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            glDisable(GL_SCISSOR_TEST);
            glUseProgram_(rrToneProgram_);
            glBindTexture(GL_TEXTURE_2D, srcTex);
            if (rrToneUSrcLoc_ >= 0) glUniform1i_(rrToneUSrcLoc_, 0);
            if (rrToneUDirectionLoc_ >= 0) glUniform1i_(rrToneUDirectionLoc_, direction);
            if (rrToneUHdrLoc_ >= 0) glUniform1f_(rrToneUHdrLoc_, static_cast<float>(displayCurveMode()));
            if (rrToneUContrastLoc_ >= 0) glUniform1f_(rrToneUContrastLoc_, pathtracecontrast);
            if (rrToneUSaturationLoc_ >= 0) glUniform1f_(rrToneUSaturationLoc_, pathtracesaturation);
            if (rrToneULutLoc_ >= 0) glUniform1i_(rrToneULutLoc_, static_cast<int>(pathtracelut));
            glBindVertexArray_(debugVao_);
            glDrawArrays(GL_TRIANGLES, 0, 3);

            glBindVertexArray_(static_cast<GLuint>(prevVao));
            glUseProgram_(static_cast<GLuint>(prevProgram));
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex0));
            glActiveTexture_(static_cast<GLenum>(prevActiveTex));
            glBindFramebuffer_(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo));
            glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
            if (prevDepthTest) glEnable(GL_DEPTH_TEST);
            if (prevBlend) glEnable(GL_BLEND);
            if (prevScissor) glEnable(GL_SCISSOR_TEST);
            return true;
        }

        bool initDlssSplit(std::string* error) {
            auto compile = [&](GLenum type, const char* src, GLuint& outShader) -> bool {
                outShader = glCreateShader_(type);
                glShaderSource_(outShader, 1, &src, nullptr);
                glCompileShader_(outShader);
                GLint ok = 0;
                glGetShaderiv_(outShader, GL_COMPILE_STATUS, &ok);
                if (!ok) {
                    char buf[2048]; GLsizei len = 0;
                    glGetShaderInfoLog_(outShader, sizeof(buf), &len, buf);
                    if (error) *error = std::string("dlss split shader compile failed: ") + std::string(buf, len);
                    return false;
                }
                return true;
                };
            GLuint vs = 0, fs = 0;
            if (!compile(GL_VERTEX_SHADER, kBlitVs, vs)) return false;
            if (!compile(GL_FRAGMENT_SHADER, kDlssSplitFs, fs)) return false;
            dlssSplitProgram_ = glCreateProgram_();
            glAttachShader_(dlssSplitProgram_, vs);
            glAttachShader_(dlssSplitProgram_, fs);
            glLinkProgram_(dlssSplitProgram_);
            glDeleteShader_(vs);
            glDeleteShader_(fs);
            GLint linked = 0;
            glGetProgramiv_(dlssSplitProgram_, GL_LINK_STATUS, &linked);
            if (!linked) {
                char buf[2048]; GLsizei len = 0;
                glGetProgramInfoLog_(dlssSplitProgram_, sizeof(buf), &len, buf);
                if (error) *error = std::string("dlss split program link failed: ") + std::string(buf, len);
                return false;
            }
            dlssSplitUAlbedoLoc_ = glGetUniformLocation_(dlssSplitProgram_, "uAlbedo");
            dlssSplitUGlowLoc_ = glGetUniformLocation_(dlssSplitProgram_, "uGlow");
            glGenFramebuffers_(1, &dlssSplitFbo_);
            {
                GLuint svs = 0, sfs = 0;
                std::string specErr;
                if (compile(GL_VERTEX_SHADER, kBlitVs, svs) && compile(GL_FRAGMENT_SHADER, kRrSpecMotionFs, sfs)) {
                    rrSpecMotionProgram_ = glCreateProgram_();
                    glAttachShader_(rrSpecMotionProgram_, svs);
                    glAttachShader_(rrSpecMotionProgram_, sfs);
                    glLinkProgram_(rrSpecMotionProgram_);
                    GLint specLinked = 0;
                    glGetProgramiv_(rrSpecMotionProgram_, GL_LINK_STATUS, &specLinked);
                    if (!specLinked) {
                        glDeleteProgram_(rrSpecMotionProgram_);
                        rrSpecMotionProgram_ = 0;
                        logmsg("DLSS-RR: specular motion program link failed -- using surface motion.");
                    }
                } else {
                    logmsg(std::string("DLSS-RR: specular motion shader compile failed -- using surface motion. ") + (error ? *error : std::string()));
                    if (error) error->clear();
                }
                if (svs) glDeleteShader_(svs);
                if (sfs) glDeleteShader_(sfs);
                if (rrSpecMotionProgram_) {
                    rrSpecMotionULinearDepthLoc_ = glGetUniformLocation_(rrSpecMotionProgram_, "uLinearDepth");
                    rrSpecMotionUMotionLoc_ = glGetUniformLocation_(rrSpecMotionProgram_, "uMotion");
                    rrSpecMotionUInvCurrVpLoc_ = glGetUniformLocation_(rrSpecMotionProgram_, "uInvCurrVp");
                    rrSpecMotionUPrevVpLoc_ = glGetUniformLocation_(rrSpecMotionProgram_, "uPrevVp");
                    rrSpecMotionUCamPosLoc_ = glGetUniformLocation_(rrSpecMotionProgram_, "uCamPos");
                    rrSpecMotionUSizeInvLoc_ = glGetUniformLocation_(rrSpecMotionProgram_, "uSizeInv");
                    glGenFramebuffers_(1, &rrSpecMotionFbo_);
                }
            }
            return true;
        }

        bool rrCompositeMode() {
            return pathtrace && dlssSceneRealDlssActive_ && dlssSceneConfiguredUsedRr_;
        }

        void runDlssSplit() {
            if (!dlssSplitFbo_ || !dlssSplitProgram_ || !debugVao_ || !albedoTex_ || !glowTex_) return;
            GLint prevVao = 0, prevFbo = 0, prevProgram = 0, prevActiveTex = 0, prevTex0 = 0, prevTex1 = 0;
            GLint prevViewport[4] = {0, 0, 0, 0};
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
            glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
            glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
            glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
            glGetIntegerv(GL_VIEWPORT, prevViewport);
            const GLboolean prevDepthTest = glIsEnabled(GL_DEPTH_TEST);
            const GLboolean prevBlend = glIsEnabled(GL_BLEND);
            glActiveTexture_(GL_TEXTURE0);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);
            glActiveTexture_(GL_TEXTURE1);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex1);

            glBindFramebuffer_(GL_FRAMEBUFFER, dlssSplitFbo_);
            glViewport(0, 0, static_cast<GLsizei>(curWidth_), static_cast<GLsizei>(curHeight_));
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            glUseProgram_(dlssSplitProgram_);
            glActiveTexture_(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, albedoTex_->glTextureId());
            if (dlssSplitUAlbedoLoc_ >= 0) glUniform1i_(dlssSplitUAlbedoLoc_, 0);
            glActiveTexture_(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, glowTex_->glTextureId());
            if (dlssSplitUGlowLoc_ >= 0) glUniform1i_(dlssSplitUGlowLoc_, 1);
            glBindVertexArray_(debugVao_);
            glDrawArrays(GL_TRIANGLES, 0, 3);

            glBindVertexArray_(static_cast<GLuint>(prevVao));
            glUseProgram_(static_cast<GLuint>(prevProgram));
            glBindFramebuffer_(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo));
            glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
            if (prevDepthTest) glEnable(GL_DEPTH_TEST);
            if (prevBlend) glEnable(GL_BLEND);
            glActiveTexture_(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex1));
            glActiveTexture_(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex0));
            glActiveTexture_(static_cast<GLenum>(prevActiveTex));
        }

        // DLSS-RR specular motion vectors into dlssRrSpecularMotionInput_
        bool runRrSpecMotion(const matrix4& invCurrVp, const matrix4& prevVp) {
            if (!rrSpecMotionProgram_ || !rrSpecMotionFbo_ || !debugVao_ || !pathTraceLinearDepth_ || !motionTex_ || !dlssRrSpecularMotionInput_) return false;
            GLint prevVao = 0, prevFbo = 0, prevProgram = 0, prevActiveTex = 0, prevTex0 = 0, prevTex1 = 0;
            GLint prevViewport[4] = {0, 0, 0, 0};
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
            glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
            glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
            glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
            glGetIntegerv(GL_VIEWPORT, prevViewport);
            const GLboolean prevDepthTest = glIsEnabled(GL_DEPTH_TEST);
            const GLboolean prevBlend = glIsEnabled(GL_BLEND);
            glActiveTexture_(GL_TEXTURE0);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);
            glActiveTexture_(GL_TEXTURE1);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex1);

            glBindFramebuffer_(GL_FRAMEBUFFER, rrSpecMotionFbo_);
            glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dlssRrSpecularMotionInput_->glTextureId(), 0);
            const GLenum drawBuf = GL_COLOR_ATTACHMENT0;
            glDrawBuffers_(1, &drawBuf);
            glViewport(0, 0, static_cast<GLsizei>(curWidth_), static_cast<GLsizei>(curHeight_));
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            glUseProgram_(rrSpecMotionProgram_);
            glActiveTexture_(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, pathTraceLinearDepth_->glTextureId());
            if (rrSpecMotionULinearDepthLoc_ >= 0) glUniform1i_(rrSpecMotionULinearDepthLoc_, 0);
            glActiveTexture_(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, motionTex_->glTextureId());
            if (rrSpecMotionUMotionLoc_ >= 0) glUniform1i_(rrSpecMotionUMotionLoc_, 1);
            if (rrSpecMotionUInvCurrVpLoc_ >= 0) glUniformMatrix4fv_(rrSpecMotionUInvCurrVpLoc_, 1, GL_FALSE, invCurrVp.a.v);
            if (rrSpecMotionUPrevVpLoc_ >= 0) glUniformMatrix4fv_(rrSpecMotionUPrevVpLoc_, 1, GL_FALSE, prevVp.a.v);
            if (rrSpecMotionUCamPosLoc_ >= 0) glUniform3f_(rrSpecMotionUCamPosLoc_, invcammatrix.d.x, invcammatrix.d.y, invcammatrix.d.z);
            if (rrSpecMotionUSizeInvLoc_ >= 0) glUniform2f_(rrSpecMotionUSizeInvLoc_, 1.0f / static_cast<float>(curWidth_), 1.0f / static_cast<float>(curHeight_));
            glBindVertexArray_(debugVao_);
            glDrawArrays(GL_TRIANGLES, 0, 3);

            glBindVertexArray_(static_cast<GLuint>(prevVao));
            glUseProgram_(static_cast<GLuint>(prevProgram));
            glBindFramebuffer_(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo));
            glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
            if (prevDepthTest) glEnable(GL_DEPTH_TEST);
            if (prevBlend) glEnable(GL_BLEND);
            glActiveTexture_(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex1));
            glActiveTexture_(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex0));
            glActiveTexture_(static_cast<GLenum>(prevActiveTex));
            return true;
        }

        // path-traced surface motion into motionTex_, see kPtSurfaceMotionFs
        void runPtSurfaceMotion() {
            if (!ptsurfacemotion || !ptSurfMotionProgram_ || !ptSurfMotionFbo_ || !debugVao_ || !pathTraceRawRadiance_ || !pathTracePrimaryNormal_ || !depthTex_ || !motionTex_ || !mvFresh_) return;
            if (pathTraceRawRadiance_->width() != motionTex_->width() || pathTraceRawRadiance_->height() != motionTex_->height() ||
                pathTracePrimaryNormal_->width() != motionTex_->width() || pathTracePrimaryNormal_->height() != motionTex_->height()) return;
            GLint prevVao = 0, prevFbo = 0, prevProgram = 0, prevActiveTex = 0, prevTex0 = 0, prevTex1 = 0, prevTex2 = 0;
            GLint prevViewport[4] = {0, 0, 0, 0};
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
            glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
            glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
            glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
            glGetIntegerv(GL_VIEWPORT, prevViewport);
            const GLboolean prevDepthTest = glIsEnabled(GL_DEPTH_TEST);
            const GLboolean prevBlend = glIsEnabled(GL_BLEND);
            const GLboolean prevScissor = glIsEnabled(GL_SCISSOR_TEST);
            const GLboolean prevCull = glIsEnabled(GL_CULL_FACE);
            glActiveTexture_(GL_TEXTURE0);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);
            glActiveTexture_(GL_TEXTURE1);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex1);
            glActiveTexture_(GL_TEXTURE2);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex2);

            glBindFramebuffer_(GL_FRAMEBUFFER, ptSurfMotionFbo_);
            glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, motionTex_->glTextureId(), 0);
            const GLenum drawBuf = GL_COLOR_ATTACHMENT0;
            glDrawBuffers_(1, &drawBuf);
            glViewport(0, 0, static_cast<GLsizei>(motionTex_->width()), static_cast<GLsizei>(motionTex_->height()));
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            glDisable(GL_SCISSOR_TEST);
            glDisable(GL_CULL_FACE);
            glUseProgram_(ptSurfMotionProgram_);
            glActiveTexture_(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, pathTraceRawRadiance_->glTextureId());
            if (ptSurfMotionUPtRawLoc_ >= 0) glUniform1i_(ptSurfMotionUPtRawLoc_, 0);
            glActiveTexture_(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, depthTex_->glTextureId());
            if (ptSurfMotionUPrepassDepthLoc_ >= 0) glUniform1i_(ptSurfMotionUPrepassDepthLoc_, 1);
            glActiveTexture_(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, pathTracePrimaryNormal_->glTextureId());
            if (ptSurfMotionUPtNormalLoc_ >= 0) glUniform1i_(ptSurfMotionUPtNormalLoc_, 2);
            if (ptSurfMotionUPtInvVpLoc_ >= 0) glUniformMatrix4fv_(ptSurfMotionUPtInvVpLoc_, 1, GL_FALSE, ptInvCamProjFrame_.a.v);
            if (ptSurfMotionUInvVpLoc_ >= 0) glUniformMatrix4fv_(ptSurfMotionUInvVpLoc_, 1, GL_FALSE, invcamprojmatrix.a.v);
            if (ptSurfMotionUCurrVpLoc_ >= 0) glUniformMatrix4fv_(ptSurfMotionUCurrVpLoc_, 1, GL_FALSE, mvCurrVp_.a.v);
            if (ptSurfMotionUPrevVpLoc_ >= 0) glUniformMatrix4fv_(ptSurfMotionUPrevVpLoc_, 1, GL_FALSE, mvPrevVp_.a.v);
            if (ptSurfMotionUCamPosLoc_ >= 0) glUniform3f_(ptSurfMotionUCamPosLoc_, invcammatrix.d.x, invcammatrix.d.y, invcammatrix.d.z);
            if (ptSurfMotionUSizeInvLoc_ >= 0) glUniform2f_(ptSurfMotionUSizeInvLoc_, 1.0f / static_cast<float>(motionTex_->width()), 1.0f / static_cast<float>(motionTex_->height()));
            glBindVertexArray_(debugVao_);
            glDrawArrays(GL_TRIANGLES, 0, 3);

            glBindVertexArray_(static_cast<GLuint>(prevVao));
            glUseProgram_(static_cast<GLuint>(prevProgram));
            glBindFramebuffer_(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo));
            glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
            if (prevDepthTest) glEnable(GL_DEPTH_TEST);
            if (prevBlend) glEnable(GL_BLEND);
            if (prevScissor) glEnable(GL_SCISSOR_TEST);
            if (prevCull) glEnable(GL_CULL_FACE);
            glActiveTexture_(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex2));
            glActiveTexture_(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex1));
            glActiveTexture_(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex0));
            glActiveTexture_(static_cast<GLenum>(prevActiveTex));
        }

        // water RR guides into the RR-only guide copies, see kRrWaterGuidesFs
        void runRrWaterGuides() {
            if (!rrwaterguides || !rrWaterProgram_ || !rrWaterFbo_ || !debugVao_ || !pathTracePrimaryNormal_ || !pathTraceRawRadiance_ || !albedoTex_ ||
                !dlssRrNormalRoughnessInput_ || !dlssRrDiffuseAlbedo_ || !dlssRrSpecularAlbedo_) return;
            const uint32_t w = dlssRrNormalRoughnessInput_->width(), h = dlssRrNormalRoughnessInput_->height();
            if (pathTracePrimaryNormal_->width() != w || pathTracePrimaryNormal_->height() != h ||
                pathTraceRawRadiance_->width() != w || pathTraceRawRadiance_->height() != h ||
                dlssRrDiffuseAlbedo_->width() != w || dlssRrSpecularAlbedo_->width() != w || albedoTex_->width() != w) return;
            GLint prevVao = 0, prevFbo = 0, prevProgram = 0, prevActiveTex = 0, prevTex[3] = {0, 0, 0};
            GLint prevViewport[4] = {0, 0, 0, 0};
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
            glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
            glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
            glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
            glGetIntegerv(GL_VIEWPORT, prevViewport);
            const GLboolean prevDepthTest = glIsEnabled(GL_DEPTH_TEST);
            const GLboolean prevBlend = glIsEnabled(GL_BLEND);
            const GLboolean prevScissor = glIsEnabled(GL_SCISSOR_TEST);
            const GLboolean prevCull = glIsEnabled(GL_CULL_FACE);
            for (int u = 0; u < 3; ++u) { glActiveTexture_(GL_TEXTURE0 + u); glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex[u]); }

            glBindFramebuffer_(GL_FRAMEBUFFER, rrWaterFbo_);
            glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dlssRrNormalRoughnessInput_->glTextureId(), 0);
            glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, dlssRrDiffuseAlbedo_->glTextureId(), 0);
            glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, dlssRrSpecularAlbedo_->glTextureId(), 0);
            const GLenum drawBufs[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
            glDrawBuffers_(3, drawBufs);
            glViewport(0, 0, static_cast<GLsizei>(w), static_cast<GLsizei>(h));
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            glDisable(GL_SCISSOR_TEST);
            glDisable(GL_CULL_FACE);
            glUseProgram_(rrWaterProgram_);
            const GLuint texIds[3] = { pathTracePrimaryNormal_->glTextureId(), pathTraceRawRadiance_->glTextureId(), albedoTex_->glTextureId() };
            for (int u = 0; u < 3; ++u) { glActiveTexture_(GL_TEXTURE0 + u); glBindTexture(GL_TEXTURE_2D, texIds[u]); }
            if (rrWaterUPtNormalLoc_ >= 0) glUniform1i_(rrWaterUPtNormalLoc_, 0);
            if (rrWaterUPtRawLoc_ >= 0) glUniform1i_(rrWaterUPtRawLoc_, 1);
            if (rrWaterUAlbedoLoc_ >= 0) glUniform1i_(rrWaterUAlbedoLoc_, 2);
            if (rrWaterUPtInvVpLoc_ >= 0) glUniformMatrix4fv_(rrWaterUPtInvVpLoc_, 1, GL_FALSE, ptInvCamProjFrame_.a.v);
            if (rrWaterUCamLoc_ >= 0) glUniformMatrix4fv_(rrWaterUCamLoc_, 1, GL_FALSE, cammatrix.a.v);
            if (rrWaterUCamPosLoc_ >= 0) glUniform3f_(rrWaterUCamPosLoc_, invcammatrix.d.x, invcammatrix.d.y, invcammatrix.d.z);
            if (rrWaterUSizeInvLoc_ >= 0) glUniform2f_(rrWaterUSizeInvLoc_, 1.0f / static_cast<float>(w), 1.0f / static_cast<float>(h));
            glBindVertexArray_(debugVao_);
            glDrawArrays(GL_TRIANGLES, 0, 3);

            glBindVertexArray_(static_cast<GLuint>(prevVao));
            glUseProgram_(static_cast<GLuint>(prevProgram));
            glBindFramebuffer_(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo));
            glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
            if (prevDepthTest) glEnable(GL_DEPTH_TEST);
            if (prevBlend) glEnable(GL_BLEND);
            if (prevScissor) glEnable(GL_SCISSOR_TEST);
            if (prevCull) glEnable(GL_CULL_FACE);
            for (int u = 2; u >= 0; --u) { glActiveTexture_(GL_TEXTURE0 + u); glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex[u])); }
            glActiveTexture_(static_cast<GLenum>(prevActiveTex));
        }

        void copyDlssRrInputs(interop::SharedTexture* color, interop::SharedTexture* depth) {
            if (!glCopyImageSubData_ || !color || !depth || !motionTex_ || !normalTex_ ||
                !dlssRrColorInput_ || !dlssRrDepthInput_ || !dlssRrMotionInput_ ||
                !dlssRrNormalRoughnessInput_ || !dlssRrSpecularMotionInput_) return;
            const GLsizei w = static_cast<GLsizei>(curWidth_), h = static_cast<GLsizei>(curHeight_);
            glCopyImageSubData_(color->glTextureId(), GL_TEXTURE_2D, 0, 0, 0, 0,
                dlssRrColorInput_->glTextureId(), GL_TEXTURE_2D, 0, 0, 0, 0, w, h, 1);
            glCopyImageSubData_(depth->glTextureId(), GL_TEXTURE_2D, 0, 0, 0, 0,
                dlssRrDepthInput_->glTextureId(), GL_TEXTURE_2D, 0, 0, 0, 0, w, h, 1);
            glCopyImageSubData_(motionTex_->glTextureId(), GL_TEXTURE_2D, 0, 0, 0, 0,
                dlssRrMotionInput_->glTextureId(), GL_TEXTURE_2D, 0, 0, 0, 0, w, h, 1);
            glCopyImageSubData_(normalTex_->glTextureId(), GL_TEXTURE_2D, 0, 0, 0, 0,
                dlssRrNormalRoughnessInput_->glTextureId(), GL_TEXTURE_2D, 0, 0, 0, 0, w, h, 1);
            // specular motion vectors
            glCopyImageSubData_(motionTex_->glTextureId(), GL_TEXTURE_2D, 0, 0, 0, 0,
                dlssRrSpecularMotionInput_->glTextureId(), GL_TEXTURE_2D, 0, 0, 0, 0, w, h, 1);
            {
                GLenum e; std::string glErr;
                while ((e = glGetError()) != GL_NO_ERROR) { if (!glErr.empty()) glErr += ", "; glErr += std::to_string(static_cast<unsigned>(e)); }
                if (!glErr.empty()) {
                    logmsg("copyDlssRrInputs: glCopyImageSubData_ GL error(s): " + glErr +
                        " (copy region " + std::to_string(w) + "x" + std::to_string(h) +
                        ", color src " + std::to_string(color->width()) + "x" + std::to_string(color->height()) +
                        " -> dlssRrColorInput_ " + std::to_string(dlssRrColorInput_->width()) + "x" + std::to_string(dlssRrColorInput_->height()) +
                        ", depth src " + std::to_string(depth->width()) + "x" + std::to_string(depth->height()) +
                        " -> dlssRrDepthInput_ " + std::to_string(dlssRrDepthInput_->width()) + "x" + std::to_string(dlssRrDepthInput_->height()) +
                        ", motion src " + std::to_string(motionTex_->width()) + "x" + std::to_string(motionTex_->height()) +
                        " -> dlssRrMotionInput_ " + std::to_string(dlssRrMotionInput_->width()) + "x" + std::to_string(dlssRrMotionInput_->height()) +
                        ", normal src " + std::to_string(normalTex_->width()) + "x" + std::to_string(normalTex_->height()) +
                        " -> dlssRrNormalRoughnessInput_ " + std::to_string(dlssRrNormalRoughnessInput_->width()) + "x" + std::to_string(dlssRrNormalRoughnessInput_->height()) + ")");
                }
            }
        }

        bool initDebugBlit(std::string* error) {
            auto compile = [&](GLenum type, const char* src, GLuint& outShader) -> bool {
                outShader = glCreateShader_(type);
                glShaderSource_(outShader, 1, &src, nullptr);
                glCompileShader_(outShader);
                GLint ok = 0;
                glGetShaderiv_(outShader, GL_COMPILE_STATUS, &ok);
                if (!ok) {
                    char buf[2048]; GLsizei len = 0;
                    glGetShaderInfoLog_(outShader, sizeof(buf), &len, buf);
                    if (error) *error = std::string("debug blit shader compile failed: ") + std::string(buf, len);
                    return false;
                }
                return true;
                };
            GLuint vs = 0, fs = 0;
            if (!compile(GL_VERTEX_SHADER, kBlitVs, vs)) return false;
            if (!compile(GL_FRAGMENT_SHADER, kBlitFs, fs)) return false;
            debugProgram_ = glCreateProgram_();
            glAttachShader_(debugProgram_, vs);
            glAttachShader_(debugProgram_, fs);
            glLinkProgram_(debugProgram_);
            glDeleteShader_(vs);
            glDeleteShader_(fs);
            GLint linked = 0;
            glGetProgramiv_(debugProgram_, GL_LINK_STATUS, &linked);
            if (!linked) {
                char buf[2048]; GLsizei len = 0;
                glGetProgramInfoLog_(debugProgram_, sizeof(buf), &len, buf);
                if (error) *error = std::string("debug blit program link failed: ") + std::string(buf, len);
                return false;
            }
            debugUAOLoc_ = glGetUniformLocation_(debugProgram_, "uAO");
            debugUShadowLoc_ = glGetUniformLocation_(debugProgram_, "uShadow");
            debugUModeLoc_ = glGetUniformLocation_(debugProgram_, "uMode");
            glGenVertexArrays_(1, &debugVao_);
            {
                GLuint mvs = 0, mfs = 0;
                std::string mvErr;
                std::string* keep = error;
                error = &mvErr;
                if (compile(GL_VERTEX_SHADER, kBlitVs, mvs) && compile(GL_FRAGMENT_SHADER, kMvDebugFs, mfs)) {
                    mvDebugProgram_ = glCreateProgram_();
                    glAttachShader_(mvDebugProgram_, mvs);
                    glAttachShader_(mvDebugProgram_, mfs);
                    glLinkProgram_(mvDebugProgram_);
                    GLint mvLinked = 0;
                    glGetProgramiv_(mvDebugProgram_, GL_LINK_STATUS, &mvLinked);
                    if (!mvLinked) {
                        char buf[2048]; GLsizei len = 0;
                        glGetProgramInfoLog_(mvDebugProgram_, sizeof(buf), &len, buf);
                        logmsg("mvdebug: program link failed -- " + std::string(buf, len));
                        glDeleteProgram_(mvDebugProgram_);
                        mvDebugProgram_ = 0;
                    }
                } else {
                    logmsg("mvdebug: " + mvErr);
                }
                error = keep;
                if (mvs) glDeleteShader_(mvs);
                if (mfs) glDeleteShader_(mfs);
                if (mvDebugProgram_) {
                    mvDebugUMvLoc_ = glGetUniformLocation_(mvDebugProgram_, "uMv");
                    mvDebugUMvRefLoc_ = glGetUniformLocation_(mvDebugProgram_, "uMvRef");
                    mvDebugUModeLoc_ = glGetUniformLocation_(mvDebugProgram_, "uMode");
                    mvDebugUMvSizeLoc_ = glGetUniformLocation_(mvDebugProgram_, "uMvSize");
                    mvDebugUVpSizeLoc_ = glGetUniformLocation_(mvDebugProgram_, "uVpSize");
                    mvDebugUScaleLoc_ = glGetUniformLocation_(mvDebugProgram_, "uScale");
                    mvDebugUCellLoc_ = glGetUniformLocation_(mvDebugProgram_, "uCell");
                    mvDebugUGainLoc_ = glGetUniformLocation_(mvDebugProgram_, "uGain");
                    mvDebugUOpacityLoc_ = glGetUniformLocation_(mvDebugProgram_, "uOpacity");
                }
            }
            // non-fatal
            {
                GLuint svs = 0, sfs = 0;
                std::string psErr;
                std::string* keep = error;
                error = &psErr;
                if (compile(GL_VERTEX_SHADER, kBlitVs, svs) && compile(GL_FRAGMENT_SHADER, kPtSurfaceMotionFs, sfs)) {
                    ptSurfMotionProgram_ = glCreateProgram_();
                    glAttachShader_(ptSurfMotionProgram_, svs);
                    glAttachShader_(ptSurfMotionProgram_, sfs);
                    glLinkProgram_(ptSurfMotionProgram_);
                    GLint psLinked = 0;
                    glGetProgramiv_(ptSurfMotionProgram_, GL_LINK_STATUS, &psLinked);
                    if (!psLinked) {
                        char buf[2048]; GLsizei len = 0;
                        glGetProgramInfoLog_(ptSurfMotionProgram_, sizeof(buf), &len, buf);
                        logmsg("path-traced surface motion: program link failed -- " + std::string(buf, len));
                        glDeleteProgram_(ptSurfMotionProgram_);
                        ptSurfMotionProgram_ = 0;
                    }
                } else {
                    logmsg("path-traced surface motion: " + psErr);
                }
                error = keep;
                if (svs) glDeleteShader_(svs);
                if (sfs) glDeleteShader_(sfs);
                if (ptSurfMotionProgram_) {
                    ptSurfMotionUPtRawLoc_ = glGetUniformLocation_(ptSurfMotionProgram_, "uPtRaw");
                    ptSurfMotionUPrepassDepthLoc_ = glGetUniformLocation_(ptSurfMotionProgram_, "uPrepassDepth");
                    ptSurfMotionUPtInvVpLoc_ = glGetUniformLocation_(ptSurfMotionProgram_, "uPtInvVp");
                    ptSurfMotionUInvVpLoc_ = glGetUniformLocation_(ptSurfMotionProgram_, "uInvVp");
                    ptSurfMotionUCurrVpLoc_ = glGetUniformLocation_(ptSurfMotionProgram_, "uCurrVp");
                    ptSurfMotionUPrevVpLoc_ = glGetUniformLocation_(ptSurfMotionProgram_, "uPrevVp");
                    ptSurfMotionUCamPosLoc_ = glGetUniformLocation_(ptSurfMotionProgram_, "uCamPos");
                    ptSurfMotionUSizeInvLoc_ = glGetUniformLocation_(ptSurfMotionProgram_, "uSizeInv");
                    ptSurfMotionUPtNormalLoc_ = glGetUniformLocation_(ptSurfMotionProgram_, "uPtNormal");
                    glGenFramebuffers_(1, &ptSurfMotionFbo_);
                }
            }
            // non-fatal
            {
                GLuint wvs = 0, wfs = 0;
                std::string wErr;
                std::string* keep = error;
                error = &wErr;
                if (compile(GL_VERTEX_SHADER, kBlitVs, wvs) && compile(GL_FRAGMENT_SHADER, kRrWaterGuidesFs, wfs)) {
                    rrWaterProgram_ = glCreateProgram_();
                    glAttachShader_(rrWaterProgram_, wvs);
                    glAttachShader_(rrWaterProgram_, wfs);
                    glLinkProgram_(rrWaterProgram_);
                    GLint wLinked = 0;
                    glGetProgramiv_(rrWaterProgram_, GL_LINK_STATUS, &wLinked);
                    if (!wLinked) {
                        char buf[2048]; GLsizei len = 0;
                        glGetProgramInfoLog_(rrWaterProgram_, sizeof(buf), &len, buf);
                        logmsg("water RR guides: program link failed -- " + std::string(buf, len));
                        glDeleteProgram_(rrWaterProgram_);
                        rrWaterProgram_ = 0;
                    }
                } else {
                    logmsg("water RR guides: " + wErr);
                }
                error = keep;
                if (wvs) glDeleteShader_(wvs);
                if (wfs) glDeleteShader_(wfs);
                if (rrWaterProgram_) {
                    rrWaterUPtNormalLoc_ = glGetUniformLocation_(rrWaterProgram_, "uPtNormal");
                    rrWaterUPtRawLoc_ = glGetUniformLocation_(rrWaterProgram_, "uPtRaw");
                    rrWaterUAlbedoLoc_ = glGetUniformLocation_(rrWaterProgram_, "uAlbedo");
                    rrWaterUPtInvVpLoc_ = glGetUniformLocation_(rrWaterProgram_, "uPtInvVp");
                    rrWaterUCamLoc_ = glGetUniformLocation_(rrWaterProgram_, "uCam");
                    rrWaterUCamPosLoc_ = glGetUniformLocation_(rrWaterProgram_, "uCamPos");
                    rrWaterUSizeInvLoc_ = glGetUniformLocation_(rrWaterProgram_, "uSizeInv");
                    glGenFramebuffers_(1, &rrWaterFbo_);
                }
            }
            return true;
        }

        bool initComposite(std::string* error) {
            auto compile = [&](GLenum type, const char* src, GLuint& outShader) -> bool {
                outShader = glCreateShader_(type);
                glShaderSource_(outShader, 1, &src, nullptr);
                glCompileShader_(outShader);
                GLint ok = 0;
                glGetShaderiv_(outShader, GL_COMPILE_STATUS, &ok);
                if (!ok) {
                    char buf[2048]; GLsizei len = 0;
                    glGetShaderInfoLog_(outShader, sizeof(buf), &len, buf);
                    if (error) *error = std::string("composite shader compile failed: ") + std::string(buf, len);
                    return false;
                }
                return true;
                };
            GLuint vs = 0, fs = 0;
            if (!compile(GL_VERTEX_SHADER, kBlitVs, vs)) return false;  // reuses the same fullscreen-triangle vertex shader as the debug blit
            if (!compile(GL_FRAGMENT_SHADER, kCompositeFs, fs)) return false;
            compositeProgram_ = glCreateProgram_();
            glAttachShader_(compositeProgram_, vs);
            glAttachShader_(compositeProgram_, fs);
            glLinkProgram_(compositeProgram_);
            glDeleteShader_(vs);
            glDeleteShader_(fs);
            GLint linked = 0;
            glGetProgramiv_(compositeProgram_, GL_LINK_STATUS, &linked);
            if (!linked) {
                char buf[2048]; GLsizei len = 0;
                glGetProgramInfoLog_(compositeProgram_, sizeof(buf), &len, buf);
                if (error) *error = std::string("composite program link failed: ") + std::string(buf, len);
                return false;
            }
            compositeUAOLoc_ = glGetUniformLocation_(compositeProgram_, "uAO");
            compositeUShadowLoc_ = glGetUniformLocation_(compositeProgram_, "uShadow");
            compositeUShadowIntensityLoc_ = glGetUniformLocation_(compositeProgram_, "uShadowIntensity");
            return true;
        }

        bool initPathTraceComposite(std::string* error) {
            auto compile = [&](GLenum type, const char* src, GLuint& outShader) -> bool {
                outShader = glCreateShader_(type);
                glShaderSource_(outShader, 1, &src, nullptr);
                glCompileShader_(outShader);
                GLint ok = 0;
                glGetShaderiv_(outShader, GL_COMPILE_STATUS, &ok);
                if (!ok) {
                    char buf[2048]; GLsizei len = 0;
                    glGetShaderInfoLog_(outShader, sizeof(buf), &len, buf);
                    if (error) *error = std::string("path-trace composite shader compile failed: ") + std::string(buf, len);
                    return false;
                }
                return true;
                };
            GLuint vs = 0, fs = 0;
            if (!compile(GL_VERTEX_SHADER, kBlitVs, vs)) return false;
            if (!compile(GL_FRAGMENT_SHADER, kPathTraceCompositeFs, fs)) return false;
            pathTraceCompositeProgram_ = glCreateProgram_();
            glAttachShader_(pathTraceCompositeProgram_, vs);
            glAttachShader_(pathTraceCompositeProgram_, fs);
            glLinkProgram_(pathTraceCompositeProgram_);
            glDeleteShader_(vs);
            glDeleteShader_(fs);
            GLint linked = 0;
            glGetProgramiv_(pathTraceCompositeProgram_, GL_LINK_STATUS, &linked);
            if (!linked) {
                char buf[2048]; GLsizei len = 0;
                glGetProgramInfoLog_(pathTraceCompositeProgram_, sizeof(buf), &len, buf);
                if (error) *error = std::string("path-trace composite program link failed: ") + std::string(buf, len);
                return false;
            }
            pathTraceCompositeUOutputLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uPathTrace");
            pathTraceCompositeUAOLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uAO");
            pathTraceCompositeUAOEnabledLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uAOEnabled");
            pathTraceCompositeUExposureLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uExposure");
            pathTraceCompositeUContrastLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uContrast");
            pathTraceCompositeUSaturationLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uSaturation");
            pathTraceCompositeURrModeLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uRrMode");
            pathTraceCompositeUHdrLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uHdr");
            pathTraceCompositeUAeTexLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uAeTex");
            pathTraceCompositeUAeEnabledLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uAeEnabled");
            pathTraceCompositeUSkyGainLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uSkyGain");
            pathTraceCompositeULutLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uLut");
            pathTraceCompositeUHasValidDepthLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uHasValidDepth");
            pathTraceCompositeUDepthSourceLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uDepthSource");
            pathTraceCompositeUOutputSizeInvLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uOutputSizeInv");
            pathTraceCompositeUNearLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uNear");
            pathTraceCompositeUFarLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uFar");
            pathTraceCompositeUMapFogEnabledLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uMapFogEnabled");
            pathTraceCompositeUMapFogColorLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uMapFogColor");
            pathTraceCompositeUMapFogParamsLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uMapFogParams");
            pathTraceCompositeURtMaskLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uRtMask");
            pathTraceCompositeURtMaskEnabledLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uRtMaskEnabled");
            pathTraceCompositeUVolFogVolumeLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uVolFogVolume");
            pathTraceCompositeUVolFogEnabledLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uVolFogEnabled");
            pathTraceCompositeULinearDepthLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uLinearDepth");
            pathTraceCompositeUTanHalfFovLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uTanHalfFov");
            pathTraceCompositeUVolFogNearFarLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uVolFogNearFar");
            pathTraceCompositeUVolFogInvViewProjLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uVolFogInvViewProj");
            pathTraceCompositeUVolFogCamPosLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uVolFogCamPos");
            pathTraceCompositeUVolFogTailDensityLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uVolFogTailDensity");
            pathTraceCompositeUVolFogTailSunLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uVolFogTailSun");
            pathTraceCompositeUVolFogTailSunColorLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uVolFogTailSunColor");
            pathTraceCompositeUVolFogTailAmbientLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uVolFogTailAmbient");
            pathTraceCompositeUVolFogTailBaseLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uVolFogTailBase");
            pathTraceCompositeUVolFogGridZLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uVolFogGridZ");
            pathTraceCompositeUVolFogDebugModeLoc_ = glGetUniformLocation_(pathTraceCompositeProgram_, "uVolFogDebugMode");
            return true;
        }

        bool initPathTraceDepth(std::string* error) {
            auto compile = [&](GLenum type, const char* src, GLuint& outShader) -> bool {
                outShader = glCreateShader_(type);
                glShaderSource_(outShader, 1, &src, nullptr);
                glCompileShader_(outShader);
                GLint ok = 0;
                glGetShaderiv_(outShader, GL_COMPILE_STATUS, &ok);
                if (!ok) {
                    char buf[2048]; GLsizei len = 0;
                    glGetShaderInfoLog_(outShader, sizeof(buf), &len, buf);
                    if (error) *error = std::string("path-trace depth shader compile failed: ") + std::string(buf, len);
                    return false;
                }
                return true;
                };
            GLuint vs = 0, fs = 0;
            if (!compile(GL_VERTEX_SHADER, kBlitVs, vs)) return false;
            if (!compile(GL_FRAGMENT_SHADER, kPathTraceDepthFs, fs)) return false;
            pathTraceDepthProgram_ = glCreateProgram_();
            glAttachShader_(pathTraceDepthProgram_, vs);
            glAttachShader_(pathTraceDepthProgram_, fs);
            glLinkProgram_(pathTraceDepthProgram_);
            glDeleteShader_(vs);
            glDeleteShader_(fs);
            GLint linked = 0;
            glGetProgramiv_(pathTraceDepthProgram_, GL_LINK_STATUS, &linked);
            if (!linked) {
                char buf[2048]; GLsizei len = 0;
                glGetProgramInfoLog_(pathTraceDepthProgram_, sizeof(buf), &len, buf);
                if (error) *error = std::string("path-trace depth program link failed: ") + std::string(buf, len);
                return false;
            }
            pathTraceDepthUOutputLoc_ = glGetUniformLocation_(pathTraceDepthProgram_, "uPathTrace");
            pathTraceDepthUOutputSizeInvLoc_ = glGetUniformLocation_(pathTraceDepthProgram_, "uOutputSizeInv");
            pathTraceDepthURtMaskLoc_ = glGetUniformLocation_(pathTraceDepthProgram_, "uRtMask");
            pathTraceDepthURtMaskEnabledLoc_ = glGetUniformLocation_(pathTraceDepthProgram_, "uRtMaskEnabled");
            return true;
        }

        bool initPresentScene(std::string* error) {
            auto compile = [&](GLenum type, const char* src, GLuint& outShader) -> bool {
                outShader = glCreateShader_(type);
                glShaderSource_(outShader, 1, &src, nullptr);
                glCompileShader_(outShader);
                GLint ok = 0;
                glGetShaderiv_(outShader, GL_COMPILE_STATUS, &ok);
                if (!ok) {
                    char buf[2048]; GLsizei len = 0;
                    glGetShaderInfoLog_(outShader, sizeof(buf), &len, buf);
                    if (error) *error = std::string("present-scene shader compile failed: ") + std::string(buf, len);
                    return false;
                }
                return true;
                };
            GLuint vs = 0, fs = 0;
            if (!compile(GL_VERTEX_SHADER, kBlitVs, vs)) return false;
            if (!compile(GL_FRAGMENT_SHADER, kPresentSceneFs, fs)) return false;
            presentSceneProgram_ = glCreateProgram_();
            glAttachShader_(presentSceneProgram_, vs);
            glAttachShader_(presentSceneProgram_, fs);
            glLinkProgram_(presentSceneProgram_);
            glDeleteShader_(vs);
            glDeleteShader_(fs);
            GLint linked = 0;
            glGetProgramiv_(presentSceneProgram_, GL_LINK_STATUS, &linked);
            if (!linked) {
                char buf[2048]; GLsizei len = 0;
                glGetProgramInfoLog_(presentSceneProgram_, sizeof(buf), &len, buf);
                if (error) *error = std::string("present-scene program link failed: ") + std::string(buf, len);
                return false;
            }
            presentSceneUSceneLoc_ = glGetUniformLocation_(presentSceneProgram_, "uScene");
            presentSceneUOutputSizeInvLoc_ = glGetUniformLocation_(presentSceneProgram_, "uOutputSizeInv");
            presentSceneUBloomLoc_ = glGetUniformLocation_(presentSceneProgram_, "uBloom");
            presentSceneUBloomIntensityLoc_ = glGetUniformLocation_(presentSceneProgram_, "uBloomIntensity");

            GLuint depthVs = 0, depthFs = 0;
            if (!compile(GL_VERTEX_SHADER, kBlitVs, depthVs)) return false;
            if (!compile(GL_FRAGMENT_SHADER, kSceneDepthRestoreFs, depthFs)) return false;
            sceneDepthRestoreProgram_ = glCreateProgram_();
            glAttachShader_(sceneDepthRestoreProgram_, depthVs);
            glAttachShader_(sceneDepthRestoreProgram_, depthFs);
            glLinkProgram_(sceneDepthRestoreProgram_);
            glDeleteShader_(depthVs);
            glDeleteShader_(depthFs);
            linked = 0;
            glGetProgramiv_(sceneDepthRestoreProgram_, GL_LINK_STATUS, &linked);
            if (!linked) {
                char buf[2048]; GLsizei len = 0;
                glGetProgramInfoLog_(sceneDepthRestoreProgram_, sizeof(buf), &len, buf);
                if (error) *error = std::string("scene-depth-restore program link failed: ") + std::string(buf, len);
                return false;
            }
            sceneDepthRestoreUDepthLoc_ = glGetUniformLocation_(sceneDepthRestoreProgram_, "uDepth");
            sceneDepthRestoreUOutputSizeInvLoc_ = glGetUniformLocation_(sceneDepthRestoreProgram_, "uOutputSizeInv");

            // bloom threshold + blur programs
            GLuint bloomThreshVs = 0, bloomThreshFs = 0;
            if (!compile(GL_VERTEX_SHADER, kBlitVs, bloomThreshVs)) return false;
            if (!compile(GL_FRAGMENT_SHADER, kBloomThresholdFs, bloomThreshFs)) return false;
            bloomThresholdProgram_ = glCreateProgram_();
            glAttachShader_(bloomThresholdProgram_, bloomThreshVs);
            glAttachShader_(bloomThresholdProgram_, bloomThreshFs);
            glLinkProgram_(bloomThresholdProgram_);
            glDeleteShader_(bloomThreshVs);
            glDeleteShader_(bloomThreshFs);
            linked = 0;
            glGetProgramiv_(bloomThresholdProgram_, GL_LINK_STATUS, &linked);
            if (!linked) {
                char buf[2048]; GLsizei len = 0;
                glGetProgramInfoLog_(bloomThresholdProgram_, sizeof(buf), &len, buf);
                if (error) *error = std::string("bloom-threshold program link failed: ") + std::string(buf, len);
                return false;
            }
            bloomThresholdUSceneLoc_ = glGetUniformLocation_(bloomThresholdProgram_, "uScene");
            bloomThresholdUOutputSizeInvLoc_ = glGetUniformLocation_(bloomThresholdProgram_, "uOutputSizeInv");
            bloomThresholdUThresholdLoc_ = glGetUniformLocation_(bloomThresholdProgram_, "uThreshold");

            {
                auto build = [&](const char* fsSrc, GLuint& prog, const char* label) -> bool {
                    GLuint v = 0, f = 0;
                    if (!compile(GL_VERTEX_SHADER, kBlitVs, v) || !compile(GL_FRAGMENT_SHADER, fsSrc, f)) return false;
                    prog = glCreateProgram_();
                    glAttachShader_(prog, v); glAttachShader_(prog, f);
                    glLinkProgram_(prog);
                    glDeleteShader_(v); glDeleteShader_(f);
                    GLint ok = 0; glGetProgramiv_(prog, GL_LINK_STATUS, &ok);
                    if (!ok) { logmsg(std::string("auto exposure: ") + label + " program link failed"); prog = 0; return false; }
                    return true;
                };
                if (build(kAeMeterFs, aeMeterProgram_, "meter")) {
                    aeMeterUSrcLoc_ = glGetUniformLocation_(aeMeterProgram_, "uSrc");
                    aeMeterUSrcSizeLoc_ = glGetUniformLocation_(aeMeterProgram_, "uSrcSize");
                }
                if (build(kAeAdaptFs, aeAdaptProgram_, "adapt")) {
                    aeAdaptUMeterLoc_ = glGetUniformLocation_(aeAdaptProgram_, "uMeter");
                    aeAdaptUPrevLoc_ = glGetUniformLocation_(aeAdaptProgram_, "uPrev");
                    aeAdaptUKeyLoc_ = glGetUniformLocation_(aeAdaptProgram_, "uKey");
                    aeAdaptUAdaptLoc_ = glGetUniformLocation_(aeAdaptProgram_, "uAdapt");
                    aeAdaptUEvLoc_ = glGetUniformLocation_(aeAdaptProgram_, "uEv");
                    aeAdaptUBlendLoc_ = glGetUniformLocation_(aeAdaptProgram_, "uBlend");
                    aeAdaptUResetLoc_ = glGetUniformLocation_(aeAdaptProgram_, "uReset");
                }
            }

            GLuint bloomBlurVs = 0, bloomBlurFs = 0;
            if (!compile(GL_VERTEX_SHADER, kBlitVs, bloomBlurVs)) return false;
            if (!compile(GL_FRAGMENT_SHADER, kBloomBlurFs, bloomBlurFs)) return false;
            bloomBlurProgram_ = glCreateProgram_();
            glAttachShader_(bloomBlurProgram_, bloomBlurVs);
            glAttachShader_(bloomBlurProgram_, bloomBlurFs);
            glLinkProgram_(bloomBlurProgram_);
            glDeleteShader_(bloomBlurVs);
            glDeleteShader_(bloomBlurFs);
            linked = 0;
            glGetProgramiv_(bloomBlurProgram_, GL_LINK_STATUS, &linked);
            if (!linked) {
                char buf[2048]; GLsizei len = 0;
                glGetProgramInfoLog_(bloomBlurProgram_, sizeof(buf), &len, buf);
                if (error) *error = std::string("bloom-blur program link failed: ") + std::string(buf, len);
                return false;
            }
            bloomBlurUSrcLoc_ = glGetUniformLocation_(bloomBlurProgram_, "uSrc");
            bloomBlurUOutputSizeInvLoc_ = glGetUniformLocation_(bloomBlurProgram_, "uOutputSizeInv");
            bloomBlurUDirectionLoc_ = glGetUniformLocation_(bloomBlurProgram_, "uDirection");
            return true;
        }


        void drawDebugBlit() {
            static uint32_t callCount = 0;
            const bool logThisCall = (callCount++ % 60) == 0;
            interop::SharedTexture* srcTex = aoOutput_.get();
            int blitMode = 0;
            if (rtaodebug == 2) { srcTex = shadowOutput_.get(); }
            else if (rtaodebug == 3) { blitMode = 1; }
            if (logThisCall) {
                GLint vp[4] = { 0,0,0,0 };
                glGetIntegerv(GL_VIEWPORT, vp);
                vlogmsg("drawDebugBlit: debugProgram_=" + std::to_string(debugProgram_) +
                    " mode=" + std::to_string(rtaodebug) +
                    " srcTex=" + (srcTex ? std::to_string(srcTex->glTextureId()) : std::string("null")) +
                    " debugVao_=" + std::to_string(debugVao_) +
                    " debugUAOLoc_=" + std::to_string(debugUAOLoc_) +
                    " viewport=(" + std::to_string(vp[0]) + "," + std::to_string(vp[1]) + "," + std::to_string(vp[2]) + "," + std::to_string(vp[3]) + ")");
            }
            if (!debugProgram_ || !srcTex || !shadowOutput_) {
                if (logThisCall) vlogmsg("drawDebugBlit: bailing out early -- debugProgram_, srcTex, or shadowOutput_ is null/zero.");
                return;
            }
            GLint prevVao = 0;
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
            GLboolean prevDepthTest = glIsEnabled(GL_DEPTH_TEST);
            GLboolean prevBlend = glIsEnabled(GL_BLEND);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);

            glUseProgram_(debugProgram_);
            glActiveTexture_(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, srcTex->glTextureId());
            if (debugUAOLoc_ >= 0) glUniform1i_(debugUAOLoc_, 0);
            glActiveTexture_(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, shadowOutput_->glTextureId());
            if (debugUShadowLoc_ >= 0) glUniform1i_(debugUShadowLoc_, 1);
            glActiveTexture_(GL_TEXTURE0);
            if (debugUModeLoc_ >= 0) glUniform1i_(debugUModeLoc_, blitMode);
            glBindVertexArray_(debugVao_);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            GLenum drawErr = glGetError();
            if (drawErr != GL_NO_ERROR) {
                vlogmsg("drawDebugBlit: glGetError() after glDrawArrays = 0x" + [&] { char b[16]; std::snprintf(b, sizeof(b), "%x", drawErr); return std::string(b); }());
            }
            glBindVertexArray_(static_cast<GLuint>(prevVao));
            glUseProgram_(0);

            if (prevDepthTest) glEnable(GL_DEPTH_TEST);
            if (prevBlend) glEnable(GL_BLEND);
        }

        // see mvdebug
        void drawMotionDebug() {
            if (!mvDebugProgram_ || !debugVao_ || !motionTex_) return;
            interop::SharedTexture* src = motionTex_.get();
            if (mvdebug >= 2) {
                if (!dlssRrSpecularMotionInput_ || dlssRrSpecularMotionInput_->width() != motionTex_->width() ||
                    dlssRrSpecularMotionInput_->height() != motionTex_->height()) {
                    static bool warned = false;
                    if (!warned) { warned = true; logmsg("mvdebug 2/3: no DLSS-RR specular motion buffer this session (needs dlssscene 1 + dlssrr 1) -- showing surface motion"); }
                } else src = dlssRrSpecularMotionInput_.get();
            }
            GLint prevVao = 0, prevProgram = 0, prevActiveTex = 0, prevTex0 = 0, prevTex1 = 0, prevBlendSrc = 0, prevBlendDst = 0;
            GLint vp[4] = {0, 0, 0, 0};
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
            glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
            glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
            glGetIntegerv(GL_BLEND_SRC_RGB, &prevBlendSrc);
            glGetIntegerv(GL_BLEND_DST_RGB, &prevBlendDst);
            glGetIntegerv(GL_VIEWPORT, vp);
            const GLboolean prevDepthTest = glIsEnabled(GL_DEPTH_TEST);
            const GLboolean prevBlend = glIsEnabled(GL_BLEND);
            const GLboolean prevScissor = glIsEnabled(GL_SCISSOR_TEST);
            const GLboolean prevCull = glIsEnabled(GL_CULL_FACE);
            glActiveTexture_(GL_TEXTURE0);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);
            glActiveTexture_(GL_TEXTURE1);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex1);

            glDisable(GL_DEPTH_TEST);
            glDisable(GL_SCISSOR_TEST);
            glDisable(GL_CULL_FACE);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glUseProgram_(mvDebugProgram_);
            glActiveTexture_(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, src->glTextureId());
            glActiveTexture_(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, motionTex_->glTextureId());
            if (mvDebugUMvLoc_ >= 0) glUniform1i_(mvDebugUMvLoc_, 0);
            if (mvDebugUMvRefLoc_ >= 0) glUniform1i_(mvDebugUMvRefLoc_, 1);
            if (mvDebugUModeLoc_ >= 0) glUniform1i_(mvDebugUModeLoc_, src == motionTex_.get() ? 1 : mvdebug);
            if (mvDebugUMvSizeLoc_ >= 0) glUniform2f_(mvDebugUMvSizeLoc_, static_cast<float>(src->width()), static_cast<float>(src->height()));
            if (mvDebugUVpSizeLoc_ >= 0) glUniform2f_(mvDebugUVpSizeLoc_, static_cast<float>(vp[2]), static_cast<float>(vp[3]));
            if (mvDebugUScaleLoc_ >= 0) glUniform1f_(mvDebugUScaleLoc_, mvdebugscale);
            if (mvDebugUCellLoc_ >= 0) glUniform1f_(mvDebugUCellLoc_, static_cast<float>(mvdebuggrid));
            if (mvDebugUGainLoc_ >= 0) glUniform1f_(mvDebugUGainLoc_, mvdebuggain);
            if (mvDebugUOpacityLoc_ >= 0) glUniform1f_(mvDebugUOpacityLoc_, mvdebugopacity);
            glBindVertexArray_(debugVao_);
            glDrawArrays(GL_TRIANGLES, 0, 3);

            glBindVertexArray_(static_cast<GLuint>(prevVao));
            glUseProgram_(static_cast<GLuint>(prevProgram));
            glBlendFunc(static_cast<GLenum>(prevBlendSrc), static_cast<GLenum>(prevBlendDst));
            if (!prevBlend) glDisable(GL_BLEND);
            if (prevDepthTest) glEnable(GL_DEPTH_TEST);
            if (prevScissor) glEnable(GL_SCISSOR_TEST);
            if (prevCull) glEnable(GL_CULL_FACE);
            glActiveTexture_(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex1));
            glActiveTexture_(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex0));
            glActiveTexture_(static_cast<GLenum>(prevActiveTex));
        }

        void compositeAo() {
            static uint32_t callCount = 0;
            const bool logThisCall = (callCount++ % 60) == 0;
            if (logThisCall) {
                GLint vp[4] = { 0,0,0,0 };
                glGetIntegerv(GL_VIEWPORT, vp);
                GLint curFbo = 0;
                glGetIntegerv(GL_FRAMEBUFFER_BINDING, &curFbo);
                GLboolean cm[4] = { GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE };
                glGetBooleanv(GL_COLOR_WRITEMASK, cm);
                GLboolean scissorEnabled = glIsEnabled(GL_SCISSOR_TEST);
                GLint sb[4] = { 0,0,0,0 };
                glGetIntegerv(GL_SCISSOR_BOX, sb);
                GLboolean cullEnabled = glIsEnabled(GL_CULL_FACE);
                GLint cullFaceMode = 0, frontFaceMode = 0;
                glGetIntegerv(GL_CULL_FACE_MODE, &cullFaceMode);
                glGetIntegerv(GL_FRONT_FACE, &frontFaceMode);
                vlogmsg("compositeAo: aoOutput_=" + (aoOutput_ ? std::to_string(aoOutput_->glTextureId()) : std::string("null")) +
                    " compositeProgram_=" + std::to_string(compositeProgram_) +
                    " debugVao_=" + std::to_string(debugVao_) +
                    " compositeUAOLoc_=" + std::to_string(compositeUAOLoc_) +
                    " rtaointensity=" + std::to_string(rtaointensity) +
                    " fbo=" + std::to_string(curFbo) +
                    " colorMaskBeforeForce=(" + std::to_string((int)cm[0]) + "," + std::to_string((int)cm[1]) + "," + std::to_string((int)cm[2]) + "," + std::to_string((int)cm[3]) + ")" +
                    " scissorTestBeforeDisable=" + std::to_string((int)scissorEnabled) +
                    " scissorBox=(" + std::to_string(sb[0]) + "," + std::to_string(sb[1]) + "," + std::to_string(sb[2]) + "," + std::to_string(sb[3]) + ")" +
                    " cullFaceBeforeDisable=" + std::to_string((int)cullEnabled) +
                    " cullFaceMode=0x" + [&] { char b[16]; std::snprintf(b, sizeof(b), "%x", (unsigned)cullFaceMode); return std::string(b); }() +
                    " frontFaceMode=0x" + [&] { char b[16]; std::snprintf(b, sizeof(b), "%x", (unsigned)frontFaceMode); return std::string(b); }() +
                    " viewport=(" + std::to_string(vp[0]) + "," + std::to_string(vp[1]) + "," + std::to_string(vp[2]) + "," + std::to_string(vp[3]) + ")");
            }
            if (!aoOutput_ || !shadowOutput_ || !compositeProgram_ || !debugVao_) {
                if (logThisCall) vlogmsg("compositeAo: bailing out early -- aoOutput_, compositeProgram_, or debugVao_ is null/zero.");
                return;
            }
            if (rtaointensity <= 0.0f && rtaoshadowintensity <= 0.0f) {
                if (logThisCall) vlogmsg("compositeAo: bailing out early -- rtaointensity <= 0.");
                return;  // cheap no-op
            }

            GLint prevVao = 0;
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
            GLboolean prevDepthTest = glIsEnabled(GL_DEPTH_TEST);
            GLboolean prevDepthMask = GL_TRUE;
            glGetBooleanv(GL_DEPTH_WRITEMASK, &prevDepthMask);
            GLboolean prevBlend = glIsEnabled(GL_BLEND);
            GLint prevBlendSrcRgb = 0, prevBlendDstRgb = 0, prevBlendSrcAlpha = 0, prevBlendDstAlpha = 0;
            glGetIntegerv(GL_BLEND_SRC_RGB, &prevBlendSrcRgb);
            glGetIntegerv(GL_BLEND_DST_RGB, &prevBlendDstRgb);
            glGetIntegerv(GL_BLEND_SRC_ALPHA, &prevBlendSrcAlpha);
            glGetIntegerv(GL_BLEND_DST_ALPHA, &prevBlendDstAlpha);
            GLboolean prevColorMask[4] = { GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE };
            glGetBooleanv(GL_COLOR_WRITEMASK, prevColorMask);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            GLboolean prevScissorTest = glIsEnabled(GL_SCISSOR_TEST);
            glDisable(GL_SCISSOR_TEST);
            GLboolean prevCullFace = glIsEnabled(GL_CULL_FACE);
            glDisable(GL_CULL_FACE);

            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glEnable(GL_BLEND);
            // standard "multiply blend" result = srcColor * dstColor + dstColor * 0 = srcColor *
            // dstColor
            glBlendFunc(GL_DST_COLOR, GL_ZERO);

            glUseProgram_(compositeProgram_);
            glActiveTexture_(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, aoOutput_->glTextureId());
            if (compositeUAOLoc_ >= 0) glUniform1i_(compositeUAOLoc_, 0);
            glActiveTexture_(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, shadowOutput_->glTextureId());
            if (compositeUShadowLoc_ >= 0) glUniform1i_(compositeUShadowLoc_, 1);
            if (compositeUShadowIntensityLoc_ >= 0) glUniform1f_(compositeUShadowIntensityLoc_, rtaoshadowintensity);
            glActiveTexture_(GL_TEXTURE0);
            glBindVertexArray_(debugVao_);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            GLenum drawErr = glGetError();
            if (drawErr != GL_NO_ERROR) {
                vlogmsg("compositeAo: glGetError() after glDrawArrays = 0x" + [&] { char b[16]; std::snprintf(b, sizeof(b), "%x", drawErr); return std::string(b); }());
            }
            glBindVertexArray_(static_cast<GLuint>(prevVao));
            glUseProgram_(0);

            if (prevCullFace) glEnable(GL_CULL_FACE);
            if (prevScissorTest) glEnable(GL_SCISSOR_TEST);
            glColorMask(prevColorMask[0], prevColorMask[1], prevColorMask[2], prevColorMask[3]);
            glBlendFuncSeparate_(prevBlendSrcRgb, prevBlendDstRgb, prevBlendSrcAlpha, prevBlendDstAlpha);
            if (!prevBlend) glDisable(GL_BLEND);
            if (prevDepthTest) glEnable(GL_DEPTH_TEST);
            glDepthMask(prevDepthMask);
        }

        void compositePathTrace() {
            static uint32_t callCount = 0;
            const bool logThisCall = (callCount++ % 60) == 0;
            interop::SharedTexture* pathTraceSource = (dlssPathTraceActive_ && dlssPathTraceColorOutput_)
                ? dlssPathTraceColorOutput_.get()
                : (rrCompositeMode() && pathTraceRawRadiance_ ? pathTraceRawRadiance_.get() : pathTraceOutput_.get());
            const bool hasValidDepth = (pathTraceOutput_.get() != nullptr);
            if (!pathTraceSource || !pathTraceCompositeProgram_ || !debugVao_) {
                if (logThisCall) logmsg("compositePathTrace: bailing out early -- pathTraceSource, pathTraceCompositeProgram_, or debugVao_ is null/zero.");
                return;
            }
            if (rtaoblend && !aoOutput_) {
                if (logThisCall) logmsg("compositePathTrace: rtaoblend is enabled but aoOutput_ is null; leaving path-trace result untouched.");
                return;
            }

            GLint prevVao = 0;
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
            GLboolean prevDepthTest = glIsEnabled(GL_DEPTH_TEST);
            GLboolean prevDepthMask = GL_TRUE;
            glGetBooleanv(GL_DEPTH_WRITEMASK, &prevDepthMask);
            GLint prevDepthFunc = GL_LESS;
            glGetIntegerv(GL_DEPTH_FUNC, &prevDepthFunc);
            GLboolean prevBlend = glIsEnabled(GL_BLEND);
            GLint prevBlendSrcRgb = 0, prevBlendDstRgb = 0, prevBlendSrcAlpha = 0, prevBlendDstAlpha = 0;
            glGetIntegerv(GL_BLEND_SRC_RGB, &prevBlendSrcRgb);
            glGetIntegerv(GL_BLEND_DST_RGB, &prevBlendDstRgb);
            glGetIntegerv(GL_BLEND_SRC_ALPHA, &prevBlendSrcAlpha);
            glGetIntegerv(GL_BLEND_DST_ALPHA, &prevBlendDstAlpha);
            GLboolean prevColorMask[4] = { GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE };
            glGetBooleanv(GL_COLOR_WRITEMASK, prevColorMask);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            GLboolean prevScissorTest = glIsEnabled(GL_SCISSOR_TEST);
            glDisable(GL_SCISSOR_TEST);
            GLboolean prevCullFace = glIsEnabled(GL_CULL_FACE);
            glDisable(GL_CULL_FACE);

            if (hasValidDepth) {
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(GL_LEQUAL);
            } else {
                glDisable(GL_DEPTH_TEST);
            }
            glDepthMask(GL_FALSE);
            glDisable(GL_BLEND);

            // this composite's draw-target resolution
            const uint32_t compositeTargetW = sceneTargetActive_ ? static_cast<uint32_t>(sceneTargetRenderW_) : curWidth_;
            const uint32_t compositeTargetH = sceneTargetActive_ ? static_cast<uint32_t>(sceneTargetRenderH_) : curHeight_;
            const float outputSizeInvX = compositeTargetW > 0 ? 1.0f / static_cast<float>(compositeTargetW) : 0.0f;
            const float outputSizeInvY = compositeTargetH > 0 ? 1.0f / static_cast<float>(compositeTargetH) : 0.0f;

            // meter + adapt this frame's exposure first
            const bool aeReady = pthdr && updateAutoExposure(pathTraceSource);

            glUseProgram_(pathTraceCompositeProgram_);
            glActiveTexture_(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, pathTraceSource->glTextureId());
            if (pathTraceCompositeUOutputLoc_ >= 0) glUniform1i_(pathTraceCompositeUOutputLoc_, 0);
            if (pathTraceCompositeUOutputSizeInvLoc_ >= 0) glUniform2f_(pathTraceCompositeUOutputSizeInvLoc_, outputSizeInvX, outputSizeInvY);
            if (pathTraceCompositeUNearLoc_ >= 0) glUniform1f_(pathTraceCompositeUNearLoc_, nearplane);
            if (pathTraceCompositeUFarLoc_ >= 0) glUniform1f_(pathTraceCompositeUFarLoc_, static_cast<float>(farplane));
            {
                const bool mapFog = !volfog && curfogend > curfogstart && curfogend < 1000000.0f;
                if (pathTraceCompositeUMapFogEnabledLoc_ >= 0) glUniform1f_(pathTraceCompositeUMapFogEnabledLoc_, mapFog ? 1.0f : 0.0f);
                if (pathTraceCompositeUMapFogColorLoc_ >= 0) glUniform3f_(pathTraceCompositeUMapFogColorLoc_, curfogcolor.x, curfogcolor.y, curfogcolor.z);
                const float fogRange = max(curfogend - curfogstart, 1e-3f);
                if (pathTraceCompositeUMapFogParamsLoc_ >= 0) glUniform2f_(pathTraceCompositeUMapFogParamsLoc_, 1.0f / fogRange, curfogend / fogRange);
            }

            // RTAO is produced every frame by the Vulkan prepass, including path-traced frames
            glActiveTexture_(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, aoOutput_->glTextureId());
            if (pathTraceCompositeUAOLoc_ >= 0) glUniform1i_(pathTraceCompositeUAOLoc_, 1);
            const bool aoMultiplyEnabled = (rtaoblend != 0);
            if (pathTraceCompositeUAOEnabledLoc_ >= 0) glUniform1f_(pathTraceCompositeUAOEnabledLoc_, aoMultiplyEnabled ? 1.0f : 0.0f);
            if (pathTraceCompositeUHasValidDepthLoc_ >= 0) glUniform1f_(pathTraceCompositeUHasValidDepthLoc_, hasValidDepth ? 1.0f : 0.0f);
            glActiveTexture_(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, hasValidDepth ? pathTraceOutput_->glTextureId() : 0);
            if (pathTraceCompositeUDepthSourceLoc_ >= 0) glUniform1i_(pathTraceCompositeUDepthSourceLoc_, 3);
            glActiveTexture_(GL_TEXTURE0);

            const bool rtMaskEnabled = (rtMaskTex_ != 0);
            glActiveTexture_(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, rtMaskEnabled ? rtMaskTex_ : 0);
            if (pathTraceCompositeURtMaskLoc_ >= 0) glUniform1i_(pathTraceCompositeURtMaskLoc_, 2);
            if (pathTraceCompositeURtMaskEnabledLoc_ >= 0) glUniform1f_(pathTraceCompositeURtMaskEnabledLoc_, rtMaskEnabled ? 1.0f : 0.0f);
            glActiveTexture_(GL_TEXTURE0);

            // tunable exposure feeding the ACES filmic curve
            if (pathTraceCompositeUExposureLoc_ >= 0) glUniform1f_(pathTraceCompositeUExposureLoc_, pathtraceexposure);
            // post-tonemap contrast/saturation
            if (pathTraceCompositeUContrastLoc_ >= 0) glUniform1f_(pathTraceCompositeUContrastLoc_, pathtracecontrast);
            if (pathTraceCompositeUSaturationLoc_ >= 0) glUniform1f_(pathTraceCompositeUSaturationLoc_, pathtracesaturation);
            if (pathTraceCompositeURrModeLoc_ >= 0) glUniform1f_(pathTraceCompositeURrModeLoc_, rrCompositeMode() ? 1.0f : 0.0f);
            // pathtracehdr tonemap-operator toggle + pathtracelut color-grade preset select
            if (pathTraceCompositeUHdrLoc_ >= 0) glUniform1f_(pathTraceCompositeUHdrLoc_, static_cast<float>(displayCurveMode()));
            // texture unit 6 (0-4 are taken above/below)
            glActiveTexture_(GL_TEXTURE6);
            glBindTexture(GL_TEXTURE_2D, aeReady ? aeTex_[aeCur_] : 0);
            if (pathTraceCompositeUAeTexLoc_ >= 0) glUniform1i_(pathTraceCompositeUAeTexLoc_, 6);
            glActiveTexture_(GL_TEXTURE0);
            if (pathTraceCompositeUAeEnabledLoc_ >= 0) glUniform1f_(pathTraceCompositeUAeEnabledLoc_, aeReady ? 1.0f : 0.0f);
            if (pathTraceCompositeUSkyGainLoc_ >= 0) glUniform1f_(pathTraceCompositeUSkyGainLoc_, pthdr ? pthdrsky : 1.0f);
            if (pathTraceCompositeULutLoc_ >= 0) glUniform1i_(pathTraceCompositeULutLoc_, pathtracelut);

            const bool volFogReady = volfog != 0 && scene_ && scene_->volFogGridX() != 0;
            glActiveTexture_(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_3D, volFogReady ? scene_->volFogIntegrated().glTextureId() : 0);
            if (volFogReady) {
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
            }
            if (pathTraceCompositeUVolFogVolumeLoc_ >= 0) glUniform1i_(pathTraceCompositeUVolFogVolumeLoc_, 4);
            glActiveTexture_(GL_TEXTURE0);
            if (pathTraceCompositeUVolFogEnabledLoc_ >= 0) glUniform1f_(pathTraceCompositeUVolFogEnabledLoc_, volFogReady ? 1.0f : 0.0f);
            // linear distance is read by map fog too (glass pixels), bound regardless of volfog
            glActiveTexture_(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, pathTraceLinearDepth_ ? pathTraceLinearDepth_->glTextureId() : 0);
            glActiveTexture_(GL_TEXTURE0);
            if (pathTraceCompositeULinearDepthLoc_ >= 0) glUniform1i_(pathTraceCompositeULinearDepthLoc_, 5);
            if (pathTraceCompositeUTanHalfFovLoc_ >= 0) {
                const float tanY = tanf(fovy * 0.5f * RAD);
                glUniform2f_(pathTraceCompositeUTanHalfFovLoc_, tanY * aspect, tanY);
            }
            if (volFogReady) {
                glActiveTexture_(GL_TEXTURE5);
                glBindTexture(GL_TEXTURE_2D, pathTraceLinearDepth_ ? pathTraceLinearDepth_->glTextureId() : 0);
                glActiveTexture_(GL_TEXTURE0);
                if (pathTraceCompositeULinearDepthLoc_ >= 0) glUniform1i_(pathTraceCompositeULinearDepthLoc_, 5);
                if (pathTraceCompositeUVolFogNearFarLoc_ >= 0) glUniform2f_(pathTraceCompositeUVolFogNearFarLoc_, nearplane, volfogdist);
                {
                    const VolFogTail& t = volFogTail_;
                    if (pathTraceCompositeUVolFogInvViewProjLoc_ >= 0) glUniformMatrix4fv_(pathTraceCompositeUVolFogInvViewProjLoc_, 1, GL_FALSE, volFogInvViewProjUsed_.a.v);
                    if (pathTraceCompositeUVolFogCamPosLoc_ >= 0) glUniform3f_(pathTraceCompositeUVolFogCamPosLoc_, t.camPos[0], t.camPos[1], t.camPos[2]);
                    if (pathTraceCompositeUVolFogTailDensityLoc_ >= 0) glUniform4f_(pathTraceCompositeUVolFogTailDensityLoc_, t.density[0], t.density[1], t.density[2], t.density[3]);
                    if (pathTraceCompositeUVolFogTailSunLoc_ >= 0) glUniform4f_(pathTraceCompositeUVolFogTailSunLoc_, t.sun[0], t.sun[1], t.sun[2], t.sun[3]);
                    if (pathTraceCompositeUVolFogTailSunColorLoc_ >= 0) glUniform3f_(pathTraceCompositeUVolFogTailSunColorLoc_, t.sunColor[0], t.sunColor[1], t.sunColor[2]);
                    if (pathTraceCompositeUVolFogTailAmbientLoc_ >= 0) glUniform3f_(pathTraceCompositeUVolFogTailAmbientLoc_, t.ambient[0], t.ambient[1], t.ambient[2]);
                    if (pathTraceCompositeUVolFogTailBaseLoc_ >= 0) glUniform3f_(pathTraceCompositeUVolFogTailBaseLoc_, t.base[0], t.base[1], t.base[2]);
                }
                if (pathTraceCompositeUVolFogGridZLoc_ >= 0) glUniform1f_(pathTraceCompositeUVolFogGridZLoc_, static_cast<float>(scene_->volFogGridZ()));
                if (pathTraceCompositeUVolFogDebugModeLoc_ >= 0) glUniform1f_(pathTraceCompositeUVolFogDebugModeLoc_, static_cast<float>(volfogdebug));
            }

            glBindVertexArray_(debugVao_);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            GLenum drawErr = glGetError();
            if (drawErr != GL_NO_ERROR) {
                logmsg("compositePathTrace: glGetError() after glDrawArrays = 0x" + [&] { char b[16]; std::snprintf(b, sizeof(b), "%x", drawErr); return std::string(b); }());
            }

            if (pathTraceDepthProgram_ && hasValidDepth) {
                glDepthMask(GL_TRUE);
                glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
                glUseProgram_(pathTraceDepthProgram_);
                glActiveTexture_(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, pathTraceOutput_->glTextureId());
                if (pathTraceDepthUOutputLoc_ >= 0) glUniform1i_(pathTraceDepthUOutputLoc_, 0);
                if (pathTraceDepthUOutputSizeInvLoc_ >= 0) glUniform2f_(pathTraceDepthUOutputSizeInvLoc_, outputSizeInvX, outputSizeInvY);
                glActiveTexture_(GL_TEXTURE2);
                glBindTexture(GL_TEXTURE_2D, rtMaskEnabled ? rtMaskTex_ : 0);
                if (pathTraceDepthURtMaskLoc_ >= 0) glUniform1i_(pathTraceDepthURtMaskLoc_, 2);
                if (pathTraceDepthURtMaskEnabledLoc_ >= 0) glUniform1f_(pathTraceDepthURtMaskEnabledLoc_, rtMaskEnabled ? 1.0f : 0.0f);
                glActiveTexture_(GL_TEXTURE0);
                glDrawArrays(GL_TRIANGLES, 0, 3);
                GLenum depthDrawErr = glGetError();
                if (depthDrawErr != GL_NO_ERROR) {
                    logmsg("compositePathTrace: glGetError() after depth-only glDrawArrays = 0x" + [&] { char b[16]; std::snprintf(b, sizeof(b), "%x", depthDrawErr); return std::string(b); }());
                }
            }

            glBindVertexArray_(static_cast<GLuint>(prevVao));
            glUseProgram_(0);

            if (prevCullFace) glEnable(GL_CULL_FACE);
            if (prevScissorTest) glEnable(GL_SCISSOR_TEST);
            glColorMask(prevColorMask[0], prevColorMask[1], prevColorMask[2], prevColorMask[3]);
            glBlendFuncSeparate_(prevBlendSrcRgb, prevBlendDstRgb, prevBlendSrcAlpha, prevBlendDstAlpha);
            if (prevBlend) glEnable(GL_BLEND);
            if (prevDepthTest) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
            glDepthFunc(prevDepthFunc);
            glDepthMask(prevDepthMask);
        }

    }  // namespace

    void applyCameraJitter() {
        projMatrixUnjittered_ = projmatrix;

        if (!rtaojitter || !dlssSceneRealDlssActive_ || screenw <= 0 || screenh <= 0) {
            jitterNdcX_ = jitterNdcY_ = 0.0f;
            jitterPixelX_ = jitterPixelY_ = 0.0f;
            return;
        }

        const float renderW = dlssSceneRenderWidth_ > 0 ? static_cast<float>(dlssSceneRenderWidth_) : static_cast<float>(screenw);
        const float renderH = dlssSceneRenderHeight_ > 0 ? static_cast<float>(dlssSceneRenderHeight_) : static_cast<float>(screenh);
        const float upscale = std::max(static_cast<float>(screenw) / renderW, static_cast<float>(screenh) / renderH);
        const uint32_t basePhases = static_cast<uint32_t>(rtaojitterphases < 2 ? 2 : rtaojitterphases);
        const uint32_t phases = std::min(256u, std::max(basePhases, static_cast<uint32_t>(std::ceil(float(basePhases) * upscale * upscale))));
        const uint32_t sampleIndex = (jitterIndex_ % phases) + 1;
        ++jitterIndex_;

        const float haltonX = haltonSequence(sampleIndex, 2) - 0.5f;
        const float haltonY = haltonSequence(sampleIndex, 3) - 0.5f;

        jitterPixelX_ = haltonX * static_cast<float>(rtaojitterscale);
        jitterPixelY_ = haltonY * static_cast<float>(rtaojitterscale);

        jitterNdcX_ = 2.0f * jitterPixelX_ / renderW;
        jitterNdcY_ = 2.0f * jitterPixelY_ / renderH;

        projmatrix.jitter(jitterNdcX_, jitterNdcY_);
    }

        bool ensureDlssPathTraceConfigured(uint32_t outputWidth, uint32_t outputHeight) {
            interop::StreamlineContext* sl = dlssCtx_ ? dlssCtx_->streamline() : nullptr;
            if (!sl || !sl->enabled()) {
                static bool loggedOnce = false;
                if (!loggedOnce) {
                    logmsg("DLSS-for-pathtrace: Streamline is not available, dlssscene falls back to native resolution.");
                    loggedOnce = true;
                }
                dlssPathTraceConfigured_ = false;
                return false;
            }
            if (outputWidth == 0 || outputHeight == 0) {
                dlssPathTraceConfigured_ = false;
                return false;
            }
            int modeIndex = static_cast<int>(dlssquality);
            if (modeIndex < 0) modeIndex = 0;
            if (modeIndex > 4) modeIndex = 4;
            const int modeOrdinal = kDlssModeOrdinals[modeIndex];

            if (dlssrr != dlssPathTraceLastRawRr_) {
                dlssPathTraceRrConsecutiveFailures_ = 0;
                dlssPathTraceLastRawRr_ = dlssrr;
            }
            const bool wantRr = dlssrr != 0 && dlssPathTraceRrConsecutiveFailures_ < kMaxDlssRrConsecutiveFailures;

            const bool needsReconfigure = !dlssPathTraceConfigured_ ||
                dlssPathTraceConfiguredOutputWidth_ != outputWidth || dlssPathTraceConfiguredOutputHeight_ != outputHeight ||
                dlssPathTraceConfiguredModeOrdinal_ != modeOrdinal || dlssPathTraceConfiguredUsedRr_ != wantRr;
            if (!needsReconfigure) return true;

            std::string err;
            if (!dlssFrame_->prepareForResourceRecreation(&err)) {
                logmsg("DLSS-for-pathtrace: prepareForResourceRecreation failed before reconfigure: " + err);
            }
            while (glGetError() != GL_NO_ERROR) {}
            uint32_t newRenderWidth = 0, newRenderHeight = 0;
            bool actualRr = wantRr;
            bool configureOk = actualRr
                ? dlssCtx_->configureDlssRr(outputWidth, outputHeight, static_cast<sl::DLSSMode>(modeOrdinal), &newRenderWidth, &newRenderHeight)
                : dlssCtx_->configureDlss(outputWidth, outputHeight, static_cast<sl::DLSSMode>(modeOrdinal), &newRenderWidth, &newRenderHeight);
            if (!configureOk && actualRr) {
                logmsg("DLSS-for-pathtrace: configureDlssRr failed: " + (sl ? sl->lastError() : ctx_->lastError()) +
                    " -- falling back to plain DLSS for this attempt.");
                ++dlssPathTraceRrConsecutiveFailures_;
                if (dlssPathTraceRrConsecutiveFailures_ == kMaxDlssRrConsecutiveFailures) {
                    logmsg("DLSS-for-pathtrace: configureDlssRr failed " + std::to_string(kMaxDlssRrConsecutiveFailures) +
                        " times in a row -- giving up on Ray Reconstruction this session (falling back to plain "
                        "DLSS/DLAA) until dlssrr is toggled off and back on.");
                }
                actualRr = false;
                configureOk = dlssCtx_->configureDlss(outputWidth, outputHeight, static_cast<sl::DLSSMode>(modeOrdinal), &newRenderWidth, &newRenderHeight);
            }
            if (!configureOk) {
                logmsg("DLSS-for-pathtrace: configureDlss failed: " + (sl ? sl->lastError() : ctx_->lastError()));
                dlssPathTraceConfigured_ = false;
                return false;
            }
            if (newRenderWidth == 0 || newRenderHeight == 0) {
                logmsg("DLSS-for-pathtrace: configureDlss returned an invalid render resolution (" +
                    std::to_string(newRenderWidth) + "x" + std::to_string(newRenderHeight) + ")");
                dlssPathTraceConfigured_ = false;
                return false;
            }
            if (actualRr) dlssPathTraceRrConsecutiveFailures_ = 0;

            dlssPathTraceRenderWidth_ = newRenderWidth;
            dlssPathTraceRenderHeight_ = newRenderHeight;
            dlssPathTraceConfiguredOutputWidth_ = outputWidth;
            dlssPathTraceConfiguredOutputHeight_ = outputHeight;
            dlssPathTraceConfiguredModeOrdinal_ = modeOrdinal;

            if (!dlssPathTraceColorOutput_ || dlssPathTraceColorOutput_->width() != outputWidth || dlssPathTraceColorOutput_->height() != outputHeight) {
                dlssPathTraceColorOutput_.reset();
                if (!makeDlssTexture(dlssPathTraceColorOutput_, interop::PixelFormat::Rgba16Float, outputWidth, outputHeight, "dlssPathTraceColorOutput_")) {
                    logmsg("DLSS-for-pathtrace: dlssPathTraceColorOutput_ creation failed -- this session falls back to the unified pipeline/native resolution.");
                    dlssPathTraceConfigured_ = false;
                    return false;
                }
            }

            const float scaleX = static_cast<float>(outputWidth) / static_cast<float>(dlssPathTraceRenderWidth_);
            const float scaleY = static_cast<float>(outputHeight) / static_cast<float>(dlssPathTraceRenderHeight_);
            const float scale = scaleX > scaleY ? scaleX : scaleY;
            dlssPathTraceJitterPhaseCount_ = static_cast<int>(scale * scale * 8.0f + 0.5f);
            if (dlssPathTraceJitterPhaseCount_ < 1) dlssPathTraceJitterPhaseCount_ = 1;

            dlssPathTraceHistoryValid_ = false;
            dlssPathTraceJitterIndex_ = 0;
            dlssPathTraceConfiguredUsedRr_ = actualRr;
            dlssPathTraceConfigured_ = true;

            if (dlssdebug) {
                logmsg("DLSS-for-pathtrace: configured " + std::string(kDlssModeLabels[modeIndex]) +
                    (actualRr ? " [Ray Reconstruction]" : "") + " -- render " +
                    std::to_string(dlssPathTraceRenderWidth_) + "x" + std::to_string(dlssPathTraceRenderHeight_) + " -> output " +
                    std::to_string(outputWidth) + "x" + std::to_string(outputHeight) +
                    " (jitter phases " + std::to_string(dlssPathTraceJitterPhaseCount_) + ")");
            }
            return true;
        }

        // ensureDlssConfigured()

        // DLSS-on-final-frame, (re)configures the Streamline session for this pipeline's purpose
        bool ensureDlssSceneConfigured() {
            interop::StreamlineContext* sl = dlssCtx_ ? dlssCtx_->streamline() : nullptr;
            if (!sl || !sl->enabled()) {
                static bool loggedOnce = false;
                if (!loggedOnce) {
                    logmsg("DLSS-scene: Streamline is not available (bin64/streamline missing, or slInit failed, "
                        "see the Streamline log). dlssscene falls back to a plain present.");
                    loggedOnce = true;
                }
                return false;
            }

            const uint32_t outputWidth = static_cast<uint32_t>(screenw), outputHeight = static_cast<uint32_t>(screenh);
            int modeIndex = static_cast<int>(dlssquality);
            if (modeIndex < 0) modeIndex = 0;
            if (modeIndex > 4) modeIndex = 4;
            const int modeOrdinal = kDlssModeOrdinals[modeIndex];

            if (dlssrr != 0 && pathtrace == 0) {
                static bool loggedRrNoPathtrace = false;
                if (!loggedRrNoPathtrace) {
                    logmsg("DLSS-scene: dlssrr is on but pathtrace is 0 -- Ray Reconstruction has no real noisy "
                        "signal to denoise from plain rasterized rendering, falling back to plain DLSS/DLAA "
                        "until pathtrace 1.");
                    loggedRrNoPathtrace = true;
                }
            }
            const int rrRequested = dlssrr != 0 ? 1 : 0;
            if (rrRequested != dlssSceneLastRawRr_) {
                dlssSceneRrConsecutiveFailures_ = 0;
                dlssSceneLastRawRr_ = rrRequested;
            }
            const bool wantRr = rrRequested != 0 && pathtrace != 0 && dlssSceneRrConsecutiveFailures_ < kMaxDlssRrConsecutiveFailures;

            const bool needsReconfigure = !dlssSceneConfigured_ ||
                dlssSceneConfiguredOutputWidth_ != outputWidth || dlssSceneConfiguredOutputHeight_ != outputHeight ||
                dlssSceneConfiguredModeOrdinal_ != modeOrdinal || dlssSceneConfiguredUsedRr_ != wantRr;
            if (!needsReconfigure) return true;

            std::string err;
            if (!dlssFrame_->prepareForResourceRecreation(&err)) {
                logmsg("DLSS-scene: prepareForResourceRecreation failed before reconfigure: " + err);
            }
            while (glGetError() != GL_NO_ERROR) {}
            uint32_t newRenderWidth = 0, newRenderHeight = 0;
            bool actualRr = wantRr;
            bool configureOk = actualRr
                ? dlssCtx_->configureDlssRr(outputWidth, outputHeight, static_cast<sl::DLSSMode>(modeOrdinal), &newRenderWidth, &newRenderHeight)
                : dlssCtx_->configureDlss(outputWidth, outputHeight, static_cast<sl::DLSSMode>(modeOrdinal), &newRenderWidth, &newRenderHeight);
            if (!configureOk && actualRr) {
                logmsg("DLSS-scene: configureDlssRr failed: " + (sl ? sl->lastError() : ctx_->lastError()) +
                    " -- falling back to plain DLSS for this attempt.");
                ++dlssSceneRrConsecutiveFailures_;
                if (dlssSceneRrConsecutiveFailures_ == kMaxDlssRrConsecutiveFailures) {
                    logmsg("DLSS-scene: configureDlssRr failed " + std::to_string(kMaxDlssRrConsecutiveFailures) +
                        " times in a row -- giving up on Ray Reconstruction this session (falling back to plain "
                        "DLSS/DLAA) until dlssrr is toggled off and back on.");
                }
                actualRr = false;
                configureOk = dlssCtx_->configureDlss(outputWidth, outputHeight, static_cast<sl::DLSSMode>(modeOrdinal), &newRenderWidth, &newRenderHeight);
            }
            if (!configureOk) {
                logmsg("DLSS-scene: configureDlss failed: " + (sl ? sl->lastError() : ctx_->lastError()));
                dlssSceneConfigured_ = false;
                return false;
            }
            if (newRenderWidth == 0 || newRenderHeight == 0) {
                logmsg("DLSS-scene: configureDlss returned an invalid render resolution (" +
                    std::to_string(newRenderWidth) + "x" + std::to_string(newRenderHeight) + ")");
                dlssSceneConfigured_ = false;
                return false;
            }
            if (actualRr) dlssSceneRrConsecutiveFailures_ = 0;

            if (!dlssSceneColorOutput_ || dlssSceneColorOutput_->width() != outputWidth || dlssSceneColorOutput_->height() != outputHeight) {
                dlssSceneColorOutput_.reset();
                if (!makeDlssTexture(dlssSceneColorOutput_, interop::PixelFormat::Rgba16Float, outputWidth, outputHeight, "dlssSceneColorOutput_")) {
                    logmsg("DLSS-scene: dlssSceneColorOutput_ creation failed -- dlssscene falls back to a plain bilinear present for this session.");
                    dlssSceneConfigured_ = false;
                    return false;
                }
                // presentScene() samples this via texture(), not texelFetch
                glBindTexture(GL_TEXTURE_2D, dlssSceneColorOutput_->glTextureId());
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glBindTexture(GL_TEXTURE_2D, 0);
            }

            dlssSceneRenderWidth_ = newRenderWidth;
            dlssSceneRenderHeight_ = newRenderHeight;
            dlssSceneConfiguredOutputWidth_ = outputWidth;
            dlssSceneConfiguredOutputHeight_ = outputHeight;
            dlssSceneConfiguredModeOrdinal_ = modeOrdinal;
            dlssSceneConfiguredUsedRr_ = actualRr;
            dlssSceneHistoryValid_ = false;
            dlssSceneConfigured_ = true;

            if (dlssdebug) {
                logmsg("DLSS-scene: configured " + std::string(kDlssModeLabels[modeIndex]) +
                    (actualRr ? " [Ray Reconstruction]" : "") + " -- render " +
                    std::to_string(dlssSceneRenderWidth_) + "x" + std::to_string(dlssSceneRenderHeight_) + " -> output " +
                    std::to_string(outputWidth) + "x" + std::to_string(outputHeight));
            }
            return true;
        }

        bool resolveSceneTarget() {
            sceneTargetWanted_ = false;
            dlssSceneRealDlssActive_ = false;
            if (!dlssscene) return false;
            if (screenw <= 0 || screenh <= 0) return false;
            if (!presentSceneProgram_ || !sceneDepthRestoreProgram_) return false;  // initPresentScene() never succeeded
            // MSAA on the default framebuffer is meaningless once the scene is rendered offscreen
            // anyway
            if (fsaa > 0) return false;

            if (ensureDlssSceneConfigured()) {
                sceneTargetRenderW_ = static_cast<int>(dlssSceneRenderWidth_);
                sceneTargetRenderH_ = static_cast<int>(dlssSceneRenderHeight_);
                dlssSceneRealDlssActive_ = true;
            } else {
                const float scale = std::min(1.0f, std::max(0.1f, dlssscenescale));
                sceneTargetRenderW_ = std::max(1, static_cast<int>(screenw * scale + 0.5f));
                sceneTargetRenderH_ = std::max(1, static_cast<int>(screenh * scale + 0.5f));
            }
            sceneTargetWanted_ = true;
            return true;
        }

    bool executeRtaoDispatch(const matrix4& camProjUnjittered);

    // per-frame driver for RtaoPrepass's new animated-instance motion-vector draw
    void runAnimatedMotionVectorPass(const float* camProjFlat16, const float* camFlat16, const float* currUnjitteredFlat16, const float* prevUnjitteredFlat16) {
        prepass_.setModelReflect(ptmodelreflect);  // DLSS-RR guide gloss matches the path tracer
        prepass_.setPtMat(ptmat != 0);
        for (auto& pending : animPendingSkinUploads_) {
            const uint32_t triCount = static_cast<uint32_t>(pending.indices.size() / 3);
            auto texOfVertex = [&](uint32_t v) -> uint32_t {
                if (v >= pending.vertexCount) return 0u;
                uint32_t idx = 0;
                std::memcpy(&idx, pending.vertexData.data() + static_cast<size_t>(v) * pending.vertexStride + 80, sizeof(uint32_t));
                return idx;
            };
            // SkinSourceVertexData::materialIndex (byte 84), skelmodel.h's rtModelMaterial encoding
            auto matOfVertex = [&](uint32_t v) -> uint32_t {
                if (v >= pending.vertexCount) return 0xFFFFFFFFu;
                uint32_t m = 0;
                std::memcpy(&m, pending.vertexData.data() + static_cast<size_t>(v) * pending.vertexStride + 84, sizeof(uint32_t));
                return m;
            };
            std::vector<uint32_t> order(triCount);
            for (uint32_t t = 0; t < triCount; ++t) order[t] = t;
            std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                const uint32_t ta = texOfVertex(pending.indices[a * 3]), tb = texOfVertex(pending.indices[b * 3]);
                if (ta != tb) return ta < tb;
                return matOfVertex(pending.indices[a * 3]) < matOfVertex(pending.indices[b * 3]);
            });
            std::vector<uint32_t> sortedIndices;
            sortedIndices.reserve(pending.indices.size());
            std::vector<sauerinterop::RtaoPrepass::AnimatedSkinBatch> batches;
            uint32_t curTex = 0xFFFFFFFFu, curMat = 0u;
            for (uint32_t t : order) {
                const uint32_t tex = texOfVertex(pending.indices[t * 3]);
                const uint32_t mat = matOfVertex(pending.indices[t * 3]);
                if (tex != curTex || mat != curMat || batches.empty()) {
                    sauerinterop::RtaoPrepass::AnimatedSkinBatch b;
                    // GL names resolved at registration, see PendingAnimSkinUpload::glIdForIndex
                    auto glIdOf = [&](uint32_t idx) -> uint32_t {
                        auto it = pending.glIdForIndex.find(idx);
                        return it != pending.glIdForIndex.end() ? it->second : 0u;
                    };
                    b.glTexture = glIdOf(tex);
                    if ((mat & 0x80000000u) != 0u && mat != 0xFFFFFFFFu) {
                        const uint32_t masksIdx = mat & 0xFFFFu;
                        b.masksGlTexture = glIdOf(masksIdx);
                        b.spec = static_cast<float>((mat >> 16) & 0xFFu) / 63.75f;
                        b.glow = static_cast<float>((mat >> 24) & 0x7Fu) / 4.0f;  // mdlptemissive (the prepass doesn't draw glow)
                    }
                    b.indexOffset = static_cast<uint32_t>(sortedIndices.size());
                    batches.push_back(b);
                    curTex = tex;
                    curMat = mat;
                }
                sortedIndices.push_back(pending.indices[t * 3 + 0]);
                sortedIndices.push_back(pending.indices[t * 3 + 1]);
                sortedIndices.push_back(pending.indices[t * 3 + 2]);
                batches.back().indexCount += 3;
            }
            std::string err;
            if (!prepass_.uploadAnimatedSkinSource(pending.modelKey, pending.vertexData.data(), pending.vertexStride,
                                                     pending.vertexCount, sortedIndices.data(),
                                                     static_cast<uint32_t>(sortedIndices.size()),
                                                     batches.data(), static_cast<uint32_t>(batches.size()), &err)) {
                logmsg("RtaoPrepass::uploadAnimatedSkinSource failed: " + err);
            }
        }
        animPendingSkinUploads_.clear();

        if (animDrawList_.empty()) return;
        std::vector<sauerinterop::RtaoPrepass::AnimatedInstance> instances;
        instances.reserve(animDrawList_.size());
        for (auto& draw : animDrawList_) {
            sauerinterop::RtaoPrepass::AnimatedInstance inst;
            inst.modelKey = draw.modelKey;
            inst.currentBones = draw.currentBones.data();
            inst.boneCount = static_cast<uint32_t>(draw.currentBones.size() / 8);
            std::memcpy(inst.currentWorldMatrix, draw.currentWorldMatrix, sizeof(float) * 12);
            if (!draw.prevBones.empty()) {
                inst.prevBones = draw.prevBones.data();
                std::memcpy(inst.prevWorldMatrix, draw.prevWorldMatrix, sizeof(float) * 12);
            }
            instances.push_back(inst);
        }
        prepass_.renderAnimated(curWidth_, curHeight_, instances.data(), static_cast<uint32_t>(instances.size()),
                                 camProjFlat16, camFlat16, currUnjitteredFlat16, prevUnjitteredFlat16);
    }

    bool volFogActive() { return volfog != 0; }

    // see itemYaws_
    void noteItemYaw(const vec &o, float yaw) {
        ItemYaw &y = itemYaws_[itemYawKey(o.x, o.y)];
        if (y.millis >= 0 && lastmillis > y.millis) {
            float d = yaw - y.yaw;
            d -= 360.0f * floorf((d + 180.0f) / 360.0f);
            y.rate = d / float(lastmillis - y.millis);
        }
        y.yaw = yaw;
        y.millis = lastmillis;
    }

    // scrolling textures / water waves
    float scrollClockSeconds() { return static_cast<float>(lastmillis % 3600000) / 1000.0f; }

    // see SauerbratenInterop.h
    void onMapLoaded() {
        if (ready_) mapLoadPending_ = true;
    }
    void onMaterialsEdited() {
        if (ready_) materialsEditPending_ = true;
    }

    bool ptWaterActive() { return pathtrace != 0 && hasPtWater_; }
    bool ptLavaActive() { return pathtrace != 0 && hasPtLava_; }

    void setFogMaterialState(int fogmat, float fogblend, int abovemat, float surfacez) {
        cameraLiquidSurfaceZ_ = surfacez;
        volFogMat_ = fogmat;
        volFogBlend_ = fogblend;
        volFogAboveMat_ = abovemat;
    }

    void particleMotion() {
        if (partMvRuns_.empty() || !mvFresh_ || !sceneTargetActive_ || !motionTex_ || !sceneDepthTex_) return;
        if (!dlssSceneRealDlssActive_) return;
        if (motionTex_->width() != sceneDepthTex_->width() || motionTex_->height() != sceneDepthTex_->height()) return;
        const bool drew = prepass_.renderParticleMotion(motionTex_->width(), motionTex_->height(), motionTex_->glTextureId(), sceneDepthTex_->glTextureId(),
            partMvVerts_.data(), static_cast<uint32_t>(partMvVerts_.size() / 14), partMvRuns_.data(), static_cast<uint32_t>(partMvRuns_.size()),
            camprojmatrix.a.v, mvCurrVp_.a.v, mvPrevVp_.a.v, partmvmin);
        static bool logged = false;
        if (!logged) {
            logged = true;
            GLenum e = glGetError();
            logmsg(std::string("particle motion vectors: ") + (drew ? "drawing" : "FAILED") + " (" +
                std::to_string(partMvVerts_.size() / 84) + " quads, " + std::to_string(partMvRuns_.size()) + " textures, GL error " + std::to_string(e) + ")");
        }
    }

    void bindVolFogTexture() {
        const bool ready = volFogActive() && scene_ && scene_->volFogGridX() != 0 && !pathtrace;
        GLOBALPARAMF(uVolFogActive, ready ? 1.0f : 0.0f, 0, 0, 0);
        // see kPartLightTexUnit
        interop::SharedTexture3D* partLight = (partLightThisFrame_ && pathtrace && scene_) ? scene_->volFogPartLightCurrent() : nullptr;
        GLOBALPARAMF(uPtPartLight, partLight ? 1.0f : 0.0f, ptpartlightgain * pathtraceexposure, 0, 0);
        if (partLight) {
            glActiveTexture_(GL_TEXTURE0 + 9);
            glBindTexture(GL_TEXTURE_3D, partLight->glTextureId());
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
            glActiveTexture_(GL_TEXTURE0);
        }
        if (partLight || ready) {
            GLOBALPARAM(uVolFogInvViewProj, volFogInvViewProjUsed_);
            GLOBALPARAMF(uVolFogCamPos, volFogCamPosUsed_.x, volFogCamPosUsed_.y, volFogCamPosUsed_.z, 0);
            GLOBALPARAMF(uVolFogCamForward, volFogCamForwardUsed_.x, volFogCamForwardUsed_.y, volFogCamForwardUsed_.z, 0);
            GLOBALPARAMF(uVolFogNearFar, nearplane, volfogdist, 0, 0);
            GLOBALPARAMF(uVolFogScreenSize, static_cast<float>(scenew), static_cast<float>(sceneh), 0, 0);
        }
        if (!ready) return;
        glActiveTexture_(GL_TEXTURE0 + 8);
        glBindTexture(GL_TEXTURE_3D, scene_->volFogIntegrated().glTextureId());
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glActiveTexture_(GL_TEXTURE0);
        GLOBALPARAM(uVolFogInvViewProj, volFogInvViewProjUsed_);
        GLOBALPARAMF(uVolFogCamPos, volFogCamPosUsed_.x, volFogCamPosUsed_.y, volFogCamPosUsed_.z, 0);
        GLOBALPARAMF(uVolFogCamForward, volFogCamForwardUsed_.x, volFogCamForwardUsed_.y, volFogCamForwardUsed_.z, 0);
        GLOBALPARAMF(uVolFogNearFar, nearplane, volfogdist, 0, 0);
        GLOBALPARAMF(uVolFogScreenSize, static_cast<float>(scenew), static_cast<float>(sceneh), 0, 0);
        GLOBALPARAMF(uVolFogGridZ, static_cast<float>(scene_->volFogGridZ()), 0, 0, 0);
    }

    std::string g_glDebugErrors;
    int g_glDebugErrorCount = 0;
    // synchronous debug output runs on the offending GL call's stack
    std::string glErrorCallStack() {
        static bool symInit = false;
        HANDLE proc = GetCurrentProcess();
        if (!symInit) { SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME); SymInitialize(proc, nullptr, TRUE); symInit = true; }
        void* frames[24];
        const USHORT n = CaptureStackBackTrace(1, 24, frames, nullptr);
        std::string out;
        int shown = 0;
        for (USHORT i = 0; i < n && shown < 8; ++i) {
            char buf[sizeof(SYMBOL_INFO) + 256] = {};
            SYMBOL_INFO* sym = reinterpret_cast<SYMBOL_INFO*>(buf);
            sym->SizeOfStruct = sizeof(SYMBOL_INFO); sym->MaxNameLen = 255;
            DWORD64 disp = 0;
            if (!SymFromAddr(proc, reinterpret_cast<DWORD64>(frames[i]), &disp, sym)) continue;  // driver frames without symbols
            std::string name = sym->Name;
            if (name.find("glErrorCallStack") != std::string::npos || name.find("sauerGlDebugCallback") != std::string::npos) continue;
            IMAGEHLP_LINE64 line{}; line.SizeOfStruct = sizeof(line); DWORD ldisp = 0;
            out += " <- " + name;
            if (SymGetLineFromAddr64(proc, reinterpret_cast<DWORD64>(frames[i]), &ldisp, &line)) {
                const char* f = std::strrchr(line.FileName, 92 /* backslash */);
                out += std::string(" (") + (f ? f + 1 : line.FileName) + ":" + std::to_string(line.LineNumber) + ")";
            }
            ++shown;
        }
        return out;
    }
    void APIENTRY sauerGlDebugCallback(GLenum /*source*/, GLenum type, GLuint id, GLenum /*severity*/, GLsizei /*length*/,
                                       const GLchar* message, const void* /*userParam*/) {
        if (type != 0x824C /*GL_DEBUG_TYPE_ERROR*/ || !message) return;
        ++g_glDebugErrorCount;
        static std::unordered_set<std::string> seenWithStack;
        if (g_glDebugErrors.size() < 4000 && g_glDebugErrors.find(message) == std::string::npos) {
            g_glDebugErrors += "[id " + std::to_string(id) + "] ";
            g_glDebugErrors += message;
            if (seenWithStack.insert(message).second) g_glDebugErrors += " CALLER:" + glErrorCallStack();
            g_glDebugErrors += " | ";
        }
    }
    void installGlDebugCapture() {
        typedef void (APIENTRY *PFN_DebugCallback)(GLenum, GLenum, GLuint, GLenum, GLsizei, const GLchar*, const void*);
        typedef void (APIENTRY *PFN_glDebugMessageCallback)(PFN_DebugCallback, const void*);
        auto setCallback = reinterpret_cast<PFN_glDebugMessageCallback>(SDL_GL_GetProcAddress("glDebugMessageCallback"));
        if (!setCallback) { logmsg("GL error capture: glDebugMessageCallback unavailable."); return; }
        // opt-in (env SAUER_GL_DEBUG=1)
        const char* glDebugEnv = std::getenv("SAUER_GL_DEBUG");
        if (!glDebugEnv || !*glDebugEnv || std::strcmp(glDebugEnv, "0") == 0) { logmsg("GL error capture: off (set SAUER_GL_DEBUG=1 to enable)."); return; }
        glEnable(0x92E0 /*GL_DEBUG_OUTPUT*/);
        glEnable(0x8242 /*GL_DEBUG_OUTPUT_SYNCHRONOUS*/);
        setCallback(sauerGlDebugCallback, nullptr);
        logmsg("GL error capture: driver debug output installed (errors are logged with their cause).");
    }

    float scrollClockSeconds();
    // CPU ms per frame inside onFrameEarly
    double earlyAcc_[6] = {0, 0, 0, 0, 0, 0};
    int earlyFrames_ = 0, earlyStart_ = 0;
    Uint64 earlyT_ = 0;
    void earlyLap(int section) {
        const Uint64 now = SDL_GetPerformanceCounter();
        if (section >= 0) earlyAcc_[section] += (now - earlyT_) * 1000.0 / double(SDL_GetPerformanceFrequency());
        earlyT_ = now;
    }
    void earlyReport() {
        if (!::framestats) { earlyFrames_ = 0; for (double& a : earlyAcc_) a = 0; return; }
        if (!earlyFrames_++) earlyStart_ = totalmillis;
        if (totalmillis - earlyStart_ < 2000) return;
        const double f = earlyFrames_;
        char buf[320];
        snprintf(buf, sizeof(buf), "framestats interop-early: dyn reset %.2f, sprites/setup %.2f, anim mapmodels %.2f, anim flush %.2f, mv setup %.2f, mv pass %.2f",
                 earlyAcc_[0] / f, earlyAcc_[1] / f, earlyAcc_[2] / f, earlyAcc_[3] / f, earlyAcc_[4] / f, earlyAcc_[5] / f);
        logmsg(buf);
        earlyFrames_ = 0;
        for (double& a : earlyAcc_) a = 0;
    }

    void onFrameEarly() {
        // nothing from last frame survives an early return
        grassSubmitted_ = false;  // set again by onFrameLate() when it adds this frame's
        partMvVerts_.clear();
        partMvRuns_.clear();
        mvFresh_ = false;
        // the prepass scrolls its albedo and its motion vectors follow the texture (this frame's time
        // and step)
        prepass_.setScrollClock(scrollClockSeconds(), static_cast<float>(curtime) / 1000.0f);
        rtPathTraceActive = false;

        if (worldsize <= 0) return;  // no map loaded yet, nothing to build/trace against

        if (!attempted_) {
            attempted_ = true;
            std::string err;

            ctx_ = std::make_unique<interop::InteropContext>();
            if (!ctx_->initialize(kShaderDirectory)) {
                logmsg("InteropContext::initialize failed: " + ctx_->lastError());
                ctx_.reset();
                return;
            }
            if (!ctx_->capabilities().hasRayQuery) {
                logmsg("this GPU/driver reports no ray query support -- RTAO cannot run this session.");
                ctx_.reset();
                return;
            }

            scene_ = std::make_unique<interop::RayTracingScene>(*ctx_, kShaderDirectory, 3, kInteropFramePoolSize);
            pathTraceSkinSink_ = std::make_unique<RayTracingSkinSink>(scene_.get(), ctx_.get());
            pathTraceSkinSink_->maskRects = &rtMaskRectsNdc_;
            frame_ = std::make_unique<interop::InteropFrame>(*ctx_, kInteropFramePoolSize);
            if (!frame_->initialize()) {
                logmsg("InteropFrame::initialize failed: " + ctx_->lastError());
                scene_.reset();
                frame_.reset();
                ctx_.reset();
                return;
            }

            dlssCtx_ = std::make_unique<interop::InteropContext>();
            if (!dlssCtx_->initialize(kShaderDirectory, streamlinePluginDirectory(), kStreamlineLogDirectory)) {
                logmsg("DLSS: dedicated InteropContext::initialize (with Streamline) failed: " + dlssCtx_->lastError() +
                    " -- DLSS/DLSS-RR unavailable this session. RTAO and path tracing are not affected by this.");
                dlssCtx_.reset();
            } else {
                dlssFrame_ = std::make_unique<interop::InteropFrame>(*dlssCtx_, 3);
                if (!dlssFrame_->initialize()) {
                    logmsg("DLSS: dedicated InteropFrame::initialize failed: " + dlssCtx_->lastError() +
                        " -- DLSS/DLSS-RR unavailable this session. RTAO and path tracing are not affected by this.");
                    dlssFrame_.reset();
                    dlssCtx_.reset();
                }
            }

            if (!prepass_.initialize(&err)) {
                logmsg("RtaoPrepass::initialize failed: " + err);
                scene_.reset();
                frame_.reset();
                ctx_.reset();
                return;
            }
            if (!initDebugBlit(&err)) {
                logmsg("debug blit init failed (non-fatal, rtaodebug just won't work): " + err);
            }
            if (!initComposite(&err)) {
                logmsg("AO composite init failed (non-fatal, RTAO just won't be visible in the normal game view): " + err);
            }
            if (!initPathTraceComposite(&err)) {
                logmsg("path-trace composite init failed (non-fatal, pathtrace just won't be visible in the normal game view): " + err);
            }
            if (!initPathTraceDepth(&err)) {
                logmsg("path-trace depth-only pass init failed (non-fatal, animated characters just won't get real depth-buffer presence): " + err);
            }
            if (!initRtMaskProgram(&err)) {
                logmsg("rt mask program init failed (non-fatal, pickup/teleport flicker fix just won't apply this session): " + err);
            }
            // dlss ray reconstruction . Must run before createResources() below
            if (!initDlssSplit(&err)) {
                logmsg("DLSS Ray Reconstruction split-pass init failed (non-fatal, dlssrr just won't work): " + err);
            }
            if (!initRrTone(&err)) {
                logmsg("DLSS Ray Reconstruction tone-pass init failed (non-fatal, RR will receive display-space color): " + err);
            }
            if (!initPresentScene(&err)) {
                logmsg("present-scene init failed (non-fatal, dlssscene just won't work): " + err);
            }

            installGlDebugCapture();
            ready_ = true;
        }
        if (!ready_) return;

        {
            std::string dummyPresentError;
            if (!ctx_->presentStreamlineDummyFrame(&dummyPresentError)) {
                static bool loggedDummyPresentFailure = false;
                if (!loggedDummyPresentFailure) {
                    logmsg("presentStreamlineDummyFrame failed (non-fatal, logged once): " + dummyPresentError);
                    loggedDummyPresentFailure = true;
                }
            }
        }

        int renderW = screenw, renderH = screenh;
        dlssPathTraceActive_ = false;
        if (resolveSceneTarget()) {
            renderW = sceneTargetRenderW_;
            renderH = sceneTargetRenderH_;
        } else if (pathtrace && screenw > 0 && screenh > 0 && presentSceneProgram_ && sceneDepthRestoreProgram_ && fsaa <= 0) {
            sceneTargetWanted_ = true;
            sceneTargetRenderW_ = screenw;
            sceneTargetRenderH_ = screenh;
            dlssSceneRealDlssActive_ = false;
            renderW = screenw;
            renderH = screenh;
        }

        if (static_cast<uint32_t>(renderW) != curWidth_ || static_cast<uint32_t>(renderH) != curHeight_) {
            if (curWidth_ != 0) {
                std::string err;
                if (!frame_->prepareForResourceRecreation(&err)) {
                    logmsg("prepareForResourceRecreation failed: " + err);
                }
            }
            destroyResources();
            curWidth_ = static_cast<uint32_t>(renderW);
            curHeight_ = static_cast<uint32_t>(renderH);
            if (!createResources(curWidth_, curHeight_)) {
                logmsg("resource (re)creation failed -- RTAO disabled for this session.");
                ready_ = false;
                return;
            }
            {
                std::string maskErr;
                if (!ensureRtMask(curWidth_, curHeight_, &maskErr)) {
                    logmsg("ensureRtMask failed (non-fatal, pickup/teleport flicker fix just won't apply this session): " + maskErr);
                }
            }
            // depthTex_/normalTex_/.../pathTraceOutput_/etc
            ++nativeResourceEpoch_;
            historyValid_ = false;
            pathTraceHistoryValid_ = false;
            if (!hasGeometry_) rebuildGeometry();
        }

        // map change a new map finished loading (onMapLoaded())
        if (mapLoadPending_) {
            mapLoadPending_ = false;
            materialsEditPending_ = false;
            logmsg("map loaded -- rebuilding ray-tracing geometry.");
            rebuildGeometry(true);
            cacheClearPending_ = true;
        }
        // vrough/vmetal
        if (materialsEditPending_) {
            materialsEditPending_ = false;
            rebuildGeometry(false);
            cacheClearPending_ = true;
        }

        if (!hasGeometry_) return;

        rtPathTraceActive = (pathtrace != 0);

        {
            const uint64_t currentEpoch = scene_->geometryEpoch();
            if (currentEpoch != lastSeenGeometryEpoch_) {
                animatedCollector_.invalidateKnownModels();
                lastSeenGeometryEpoch_ = currentEpoch;
            }
        }

        earlyLap(-1);
        scene_->resetDynamicGeometry();
        earlyLap(0);

        rtFreshDynents_.clear();
        // see sauerGlDebugCallback()
        if (g_glDebugErrorCount > 0) {
            static int lastGlErrorLogMillis = -100000;
            if (totalmillis - lastGlErrorLogMillis > 1000) {
                lastGlErrorLogMillis = totalmillis;
                logmsg("GL error x" + std::to_string(g_glDebugErrorCount) + " last frame (driver): " + g_glDebugErrors);
            }
            g_glDebugErrors.clear();
            g_glDebugErrorCount = 0;
        }
        rtMaskRectsNdc_.clear();
        ++animPoseFrameCounter_;
        animMapmodelOrdinal_ = 0;
        animDrawList_.clear();

        animatedCollector_.setDirectPoseTarget(pathtrace ? pathTraceSkinSink_.get() : nullptr);

        RayTracingSkinSink skinSink(scene_.get(), ctx_.get());
        for (int texid : pendingGrassTex_) skinSink.registerTexture(static_cast<uint32_t>(texid));
        pendingGrassTex_.clear();
        const bool wantSprites = pathtrace && ptpartreflect > 0;
        const bool wantMotion = partmv && dlssscene;
        if (wantSprites || wantMotion) {
            static vector<ptglowsprite> sprites;
            sprites.setsize(0);
            ptcollectglowsprites(sprites, wantMotion);
            if (wantMotion) {
                // two triangles per quad, 14 floats per vertex (RtaoPrepass::renderParticleMotion)
                static const int tri[6] = { 0, 1, 2, 0, 2, 3 };
                partMvVerts_.reserve(static_cast<size_t>(sprites.length()) * 6 * 14);
                for (int n = 0; n < sprites.length(); ++n) {
                    const ptglowsprite &s = sprites[n];
                    const uint32_t first = static_cast<uint32_t>(partMvVerts_.size() / 14);
                    if (partMvRuns_.empty() || partMvRuns_.back().glTexture != static_cast<uint32_t>(s.texid))
                        partMvRuns_.push_back({ static_cast<uint32_t>(s.texid), first, 0 });
                    for (int k : tri) {
                        const float v[14] = { s.c[k].x, s.c[k].y, s.c[k].z, s.p[k].x, s.p[k].y, s.p[k].z, s.u[k], s.v[k],
                                              s.color.x, s.color.y, s.color.z, s.alpha, s.lum ? 1.0f : 0.0f, s.radial ? 1.0f : 0.0f };
                        partMvVerts_.insert(partMvVerts_.end(), v, v + 14);
                    }
                    partMvRuns_.back().count += 6;
                }
            }
            if (wantSprites) {
                // the path tracer only takes the glowing ones
                int w = 0;
                for (int n = 0; n < sprites.length(); ++n) if (sprites[n].glow) sprites[w++] = sprites[n];
                sprites.setsize(w);
            }
            const int count = wantSprites ? min(sprites.length(), 4096) : 0;
            if (count > 0) {
                static std::vector<interop::RayTracingScene::SkinSourceVertex> sv;
                static std::vector<uint32_t> si;
                sv.assign(static_cast<size_t>(count) * 4, interop::RayTracingScene::SkinSourceVertex{});
                si.resize(static_cast<size_t>(count) * 6);
                std::unordered_map<int, uint32_t> texIndex;
                int used = 0;
                for (int n = 0; n < count; ++n) {
                    const ptglowsprite &s = sprites[n];
                    auto it = texIndex.find(s.texid);
                    if (it == texIndex.end()) it = texIndex.emplace(s.texid, skinSink.registerTexture(static_cast<uint32_t>(s.texid))).first;
                    if (it->second == 0) continue;
                    for (int k = 0; k < 4; ++k) {
                        interop::RayTracingScene::SkinSourceVertex &v = sv[static_cast<size_t>(used) * 4 + k];
                        v.px = s.c[k].x; v.py = s.c[k].y; v.pz = s.c[k].z;
                        v.nx = s.color.x; v.ny = s.color.y; v.nz = s.color.z;  // colour x fade
                        v.tx = s.radial ? 1.0f : 0.0f;  // radial falloff (explosion discs)
                        v.tw = 1.0f;
                        v.u = s.u[k]; v.v = s.v[k];
                        v.textureIndex = it->second;
                        v.materialIndex = 0xFFFFFFFEu;  // pathtrace_trace.comp PARTICLE_MATERIAL
                    }
                    const uint32_t b = static_cast<uint32_t>(used) * 4;
                    const uint32_t q[6] = { b, b + 1, b + 2, b, b + 2, b + 3 };
                    for (int k = 0; k < 6; ++k) si[static_cast<size_t>(used) * 6 + k] = q[k];
                    ++used;
                }
                if (used > 0) scene_->addParticleSpriteMesh(sv.data(), static_cast<uint32_t>(used) * 4, si.data(), static_cast<uint32_t>(used) * 6);
            }
        }
        earlyLap(1);
        sauerinterop::extractAnimatedMapModelInstances(animatedCollector_);
        earlyLap(2);
        animatedCollector_.flush(skinSink);
        earlyLap(3);
        rtPosedSink = &animatedCollector_;

        if (pathtrace) {
            return;
        }

        // camProjUnjittered is not one of Sauerbraten's globals
        matrix4 camProjUnjittered;
        camProjUnjittered.muld(projMatrixUnjittered_, cammatrix);

        prepass_.render(curWidth_, curHeight_, camprojmatrix.a.v, cammatrix.a.v,
            camProjUnjittered.a.v, prevCamProjMatrixUnjittered_.a.v);
        earlyLap(4);
        runAnimatedMotionVectorPass(camprojmatrix.a.v, cammatrix.a.v, camProjUnjittered.a.v, prevCamProjMatrixUnjittered_.a.v);
        earlyLap(5);
        earlyReport();

        pathTraceHistoryValid_ = false;

        // RTAO only runs with path tracing on (onFrameLate()'s dispatch)
        historyValid_ = false;

        mvCurrVp_ = camProjUnjittered; mvPrevVp_ = prevCamProjMatrixUnjittered_; mvFresh_ = true;
        prevCamProjMatrix_ = camprojmatrix;
        prevCamProjMatrixUnjittered_ = camProjUnjittered;
    }

    void submitGrass() {
        if (!ptgrass) return;
        static vector<ptgrassquad> quads;
        quads.setsize(0);
        ptcollectgrass(quads);
        const int count = min(quads.length(), 65536);
        if (count <= 0) { grassSubmitted_ = true; return; }  // grass on but none in range
        static std::vector<interop::RayTracingScene::SkinSourceVertex> sv;
        static std::vector<float> cutoffs;
        static std::vector<uint32_t> si;
        sv.resize(static_cast<size_t>(count) * 4);
        cutoffs.resize(static_cast<size_t>(count) * 4);
        si.resize(static_cast<size_t>(count) * 6);
        int lastTex = -1; uint32_t lastIdx = 0;
        int used = 0;
        for (int n = 0; n < count; ++n) {
            const ptgrassquad &q = quads[n];
            if (q.texid != lastTex) {
                lastTex = q.texid;
                lastIdx = 0;
                if (!scene_->isPathTraceTextureRegistered(static_cast<uint32_t>(q.texid), &lastIdx)) {
                    lastIdx = 0;
                    if (std::find(pendingGrassTex_.begin(), pendingGrassTex_.end(), q.texid) == pendingGrassTex_.end()) pendingGrassTex_.push_back(q.texid);
                }
            }
            if (lastIdx == 0) continue;
            vec t = vec(q.c[0]).sub(q.c[1]);
            if (t.iszero()) continue;
            t.normalize();
            // +2 marks grass for the shader (cutoffs are 0..1), rt_common.glsl gCandidateIsGrass
            const float cutoff = 2.0f + min(1.0f - q.fade * (1.0f - ptgrasscutoff), 1.001f);
            for (int k = 0; k < 4; ++k) {
                interop::RayTracingScene::SkinSourceVertex &v = sv[static_cast<size_t>(used) * 4 + k];
                v = interop::RayTracingScene::SkinSourceVertex{};
                v.px = q.c[k].x; v.py = q.c[k].y; v.pz = q.c[k].z;
                v.nx = q.normal.x; v.ny = q.normal.y; v.nz = q.normal.z;
                v.tx = t.x; v.ty = t.y; v.tz = t.z; v.tw = 1.0f;
                v.u = q.u[k]; v.v = q.v[k];
                v.textureIndex = lastIdx;
                v.materialIndex = 0xFFFFFFFFu;  // no VSlot, the texture is the material (same as mapmodels)
                cutoffs[static_cast<size_t>(used) * 4 + k] = cutoff;
            }
            const uint32_t b = static_cast<uint32_t>(used) * 4;
            const uint32_t tri[6] = { b, b + 1, b + 2, b, b + 2, b + 3 };
            for (int k = 0; k < 6; ++k) si[static_cast<size_t>(used) * 6 + k] = tri[k];
            ++used;
        }
        if (used > 0 && scene_->addGrassMesh(sv.data(), cutoffs.data(), static_cast<uint32_t>(used) * 4, si.data(), static_cast<uint32_t>(used) * 6))
            grassSubmitted_ = pendingGrassTex_.empty();  // any grass still waiting on its texture
    }
    // the path tracer has this frame's grass, rendergrass() skips the raster grass
    bool ptGrassActive() { return pathtrace != 0 && grassSubmitted_; }

    void onFrameLate() {
        // investigating "completely invisible to the pathtracer" report
        {
            static int lastLog = 0;
            if (lastmillis - lastLog > 1000) {
                lastLog = lastmillis;
                vlogmsg("DIAG onFrameLate top: ready_=" + std::to_string(ready_) + " hasGeometry_=" + std::to_string(hasGeometry_) +
                    " pathtrace=" + std::to_string(pathtrace));
            }
        }
        if (!ready_ || !hasGeometry_ || !pathtrace) return;

        // the TLAS build itself, deferred here from onFrameEarly()
        submitGrass();
        {
            interop::RayTracingScene::WaterPatchSettings ws;
            ws.waves = hasPtWater_ && ptwaterwaves > 0.0f;
            ws.enabled = ptwaves != 0 && ((hasPtWater_ && ptwaterwaves > 0.0f) || (hasPtLava_ && ptlavamode && ptlavadisplace));
            ws.lava0[0] = ptlavasize; ws.lava0[1] = ptlavaflow; ws.lava0[2] = ptlavarelief; ws.lava0[3] = ptlavacrack;
            ws.lava1[0] = ptlavatemp; ws.lava1[1] = ptlavacrust; ws.lava1[2] = scrollClockSeconds() * ptlavaspeed; ws.lava1[3] = ptlavacrustamt;
            ws.camX = camera1->o.x; ws.camY = camera1->o.y; ws.camZ = camera1->o.z;
            ws.time = scrollClockSeconds();
            ws.speed = ptwavespeed;
            ws.strength = ptwaterwaves;
            ws.detail = ptwavedetail;
            ws.height = ptwaveheight;
            ws.range = ptwaverange;
            ws.cellSize = ptwavegrid;
            ws.fadeWidth = ptwavefade;
            ws.windSpeed = ptwind;
            ws.windDir = ptwinddir;
            ws.fetch = ptfetch;
            ws.spread = ptspread;
            ws.chop = ptchop;
            ws.sizeMeters = ptwavesize;
            ws.depthAware = ptwavedepth != 0;
            ws.worldSize = float(worldsize);
            ws.mapStats = ptwavestats != 0;
            scene_->setWaterPatch(ws);
        }
        scene_->setForceAccelRebuild(pathtracenorefit != 0);
        bool diagBuildOk = scene_->build(/*reflectionsActive=*/false, /*rtdiActive=*/false, /*pathTraceActive=*/true);
        if (rtrigiddiag) {
            static std::string lastRigidDiagSeen;
            std::string cur = scene_->lastRigidDiag();
            if (cur != lastRigidDiagSeen) {
                lastRigidDiagSeen = cur;
                logmsg("DIAG lastRigidDiag CHANGED: " + cur);
            }
        }
        if (!diagBuildOk) {
            logmsg("RayTracingScene::build (per-frame TLAS refresh) failed: " + ctx_->lastError());
            return;
        }

        // same derivation as onFrameEarly()'s identical local
        matrix4 camProjUnjittered;
        camProjUnjittered.muld(projMatrixUnjittered_, cammatrix);

        prepass_.render(curWidth_, curHeight_, camprojmatrix.a.v, cammatrix.a.v,
            camProjUnjittered.a.v, prevCamProjMatrixUnjittered_.a.v);
        runAnimatedMotionVectorPass(camprojmatrix.a.v, cammatrix.a.v, camProjUnjittered.a.v, prevCamProjMatrixUnjittered_.a.v);

        {
            interop::PathTraceConstants c;
            matrix4 ptCamProj = camprojmatrix, ptInvCamProj = invcamprojmatrix;
            const bool sceneJittered = jitterNdcX_ != 0.0f || jitterNdcY_ != 0.0f;
            float ptJitterNdcX = 0.0f, ptJitterNdcY = 0.0f;
            if (sceneJittered) {
                ptJitterNdcX = jitterNdcX_ * pathtracejittermagnitude;
                ptJitterNdcY = jitterNdcY_ * pathtracejittermagnitude;
            } else if (rtaojitter && pathtracejittermagnitude > 0.001f && curWidth_ > 0 && curHeight_ > 0) {
                const uint32_t phases = static_cast<uint32_t>(rtaojitterphases < 2 ? 2 : rtaojitterphases);
                const uint32_t sampleIndex = (pathTraceJitterIndex_ % phases) + 1;
                ++pathTraceJitterIndex_;
                const float haltonX = haltonSequence(sampleIndex, 2) - 0.5f;
                const float haltonY = haltonSequence(sampleIndex, 3) - 0.5f;
                // path tracing renders at curWidth_/curHeight_, so one of its pixels is 2/curWidth_
                // NDC wide
                ptJitterNdcX = 2.0f * haltonX * static_cast<float>(rtaojitterscale) * pathtracejittermagnitude * pathtraceaajitter / static_cast<float>(curWidth_);
                ptJitterNdcY = 2.0f * haltonY * static_cast<float>(rtaojitterscale) * pathtracejittermagnitude * pathtraceaajitter / static_cast<float>(curHeight_);
            }
            if (!sceneJittered || pathtracejittermagnitude < 0.999f) {
                matrix4 ptProj = projMatrixUnjittered_;
                ptProj.jitter(ptJitterNdcX, ptJitterNdcY);
                ptCamProj.muld(ptProj, cammatrix);
                ptInvCamProj.invert(ptCamProj);
            }
            std::memcpy(c.invViewProj, ptInvCamProj.a.v, sizeof(float) * 16);
            ptInvCamProjFrame_ = ptInvCamProj;  // see runPtSurfaceMotion()
            // DLSS-RR specular motion vectors, see runRrSpecMotion()
            c.lavaSize = ptlavasize; c.lavaFlow = ptlavaflow; c.lavaRelief = ptlavarelief; c.lavaCrack = ptlavacrack;  // path-traced lava
            c.lavaTemp = ptlavatemp; c.lavaCrustTemp = ptlavacrust; c.lavaSpeed = ptlavaspeed; c.lavaCrustAmount = ptlavacrustamt;
            lavaTintFromMap(c.lavaTintHue, c.lavaTintSat);  // lava colour
            c.waterRays = ptwaterrays; c.waterRaysG = ptwaterraysg; c.waterRaysFocus = ptwaterfocus;  // underwater volumetrics
            // the wave clock (scrollClockSeconds() * ptwavespeed) advanced this much since last frame
            c.waterFlowDt = static_cast<float>(curtime) / 1000.0f * std::max(0.0f, static_cast<float>(ptwavespeed));
            c.rrSpecularMotion = dlssSceneRealDlssActive_ && dlssSceneConfiguredUsedRr_ && rrSpecMotionProgram_ != 0;
            // the forward (non-inverted) counterpart
            std::memcpy(c.viewProj, ptCamProj.a.v, sizeof(float) * 16);
            std::memcpy(c.invView, invcammatrix.a.v, sizeof(float) * 16);
            // primary rays always use the jittered ptCamProj (sub-pixel anti- aliasing)
            const bool resolveInJitteredGrid = sceneJittered;
            {
                matrix4 resolveInv;
                if (resolveInJitteredGrid) resolveInv = ptInvCamProj;
                else resolveInv.invert(camProjUnjittered);
                std::memcpy(c.prevViewProj, resolveInv.a.v, sizeof(float) * 16);
            }
            std::memcpy(c.prevViewProjJittered, (resolveInJitteredGrid ? prevPtCamProj_ : prevCamProjMatrixUnjittered_).a.v, sizeof(float) * 16);
            prevPtCamProj_ = ptCamProj;

            static std::vector<float> lightPacked;
            gatherPathTraceLights(lightPacked);
            const uint32_t lightCount = static_cast<uint32_t>(lightPacked.size() / 12);
            updateLightGrid(lightPacked, lastStaticLightCount_);  // rebuilt only when the static lights change

            // the froxel passes also run with fog off (at zero density) while there are lit particles
            // to light
            const bool partLightWanted = ptpartlit && (volfog || ptpartlit == 2) && ptlitparticlecount() > 0;
            partLightThisFrame_ = false;
            if (volfog || partLightWanted) {
                interop::GpuVolFogConstants vfc;
                vfc.reservedGridDim = partLightWanted ? 1u : 0u;
                // the same lights the path tracer gets
                vfc.localLightParams[0] = volfoglights;
                vfc.localLightParams[1] = pathtraceglowscale;
                scene_->setVolFogLights(lightCount ? lightPacked.data() : nullptr, lightCount);
                volFogInvViewProjUsed_ = ptInvCamProj;
                volFogCamPosUsed_ = camera1->o;
                std::memcpy(vfc.invViewProj, ptInvCamProj.a.v, sizeof(float) * 16);
                std::memcpy(vfc.prevViewProj, prevCamProjMatrixUnjittered_.a.v, sizeof(float) * 16);
                vfc.frameIndex = static_cast<uint32_t>(frameCounter_);
                vfc.shadowRefreshFreq = static_cast<uint32_t>(volfogshadowfreq);
                vfc.historyValid = volFogHistoryValid_ ? 1u : 0u;
                vfc.sunDirectionRadius[0] = -sunlightdir.x;
                vfc.sunDirectionRadius[1] = -sunlightdir.y;
                vfc.sunDirectionRadius[2] = -sunlightdir.z;
                vfc.sunDirectionRadius[3] = pathtracesunangle;
                if (sunlight) {
                    // sun irradiance in the same units the lit surfaces use
                    const float sunIrr = 2.0f * PI * volfogsun;
                    vfc.sunColorPad[0] = ptLightChannel(sunlightcolor.x / 255.0f) * sunlightscale * sunIrr;
                    vfc.sunColorPad[1] = ptLightChannel(sunlightcolor.y / 255.0f) * sunlightscale * sunIrr;
                    vfc.sunColorPad[2] = ptLightChannel(sunlightcolor.z / 255.0f) * sunlightscale * sunIrr;
                } else {
                    vfc.sunColorPad[0] = vfc.sunColorPad[1] = vfc.sunColorPad[2] = 0.0f;
                }
                vfc.skyAmbientPad[0] = ptLightChannel(skylightcolor.x / 255.0f);
                vfc.skyAmbientPad[1] = ptLightChannel(skylightcolor.y / 255.0f);
                vfc.skyAmbientPad[2] = ptLightChannel(skylightcolor.z / 255.0f);
                vfc.ambientFloor[0] = 2.0f * ptLightChannel(ambientcolor.x / 255.0f);  // particle light floor (x2 overbright)
                vfc.ambientFloor[1] = 2.0f * ptLightChannel(ambientcolor.y / 255.0f);
                vfc.ambientFloor[2] = 2.0f * ptLightChannel(ambientcolor.z / 255.0f);
                switch (volfogres) {
                    case 0: vfc.gridDimX = 80; vfc.gridDimY = 45; vfc.gridDimZ = 96; break;
                    case 2: vfc.gridDimX = 160; vfc.gridDimY = 90; vfc.gridDimZ = 320; break;
                    default: vfc.gridDimX = 160; vfc.gridDimY = 90; vfc.gridDimZ = 256; break;
                }
                {
                    static uint32_t lastGX=0,lastGY=0,lastCW=0,lastCH=0,lastSW=0,lastSH=0,lastVGX=0,lastVGY=0;
                    uint32_t curVGX = scene_ ? scene_->volFogGridX() : 0;
                    uint32_t curVGY = scene_ ? scene_->volFogGridY() : 0;
                    if (vfc.gridDimX!=lastGX || vfc.gridDimY!=lastGY || curWidth_!=lastCW || curHeight_!=lastCH ||
                        static_cast<uint32_t>(screenw)!=lastSW || static_cast<uint32_t>(screenh)!=lastSH ||
                        curVGX!=lastVGX || curVGY!=lastVGY) {
                        logmsg("volfog res check: vfc.gridDim=" + std::to_string(vfc.gridDimX) + "x" + std::to_string(vfc.gridDimY) +
                               " scene_->volFogGrid=" + std::to_string(curVGX) + "x" + std::to_string(curVGY) +
                               " curWidth_/curHeight_=" + std::to_string(curWidth_) + "x" + std::to_string(curHeight_) +
                               " screenw/screenh=" + std::to_string(screenw) + "x" + std::to_string(screenh));
                        lastGX=vfc.gridDimX; lastGY=vfc.gridDimY; lastCW=curWidth_; lastCH=curHeight_;
                        lastSW=static_cast<uint32_t>(screenw); lastSH=static_cast<uint32_t>(screenh);
                        lastVGX=curVGX; lastVGY=curVGY;
                    }
                }
                vfc.nearPlane = nearplane;
                vfc.volFogDist = volfogdist;
                // diagnostic
                vfc.reservedNearFar0 = static_cast<float>(volfogreproject);
                vfc.reservedNearFar1 = static_cast<float>(volfogdebug);
                vfc.baseDensity = volfog ? volfogdensity : 0.0f;
                vfc.fogBaseHeight = volfogheight;
                vfc.heightFalloff = volfogfalloff;
                vfc.ambientShadowFloor = volfogambientfloor;
                vfc.phaseG = volfogphase;
                vfc.noiseScale = 1.0f;
                vfc.noiseEnabled = volfognoise ? 1.0f : 0.0f;
                vfc.noiseIntensity = 0.5f;
                vfc.camWorldPos[0] = invcammatrix.d.x;
                vfc.camWorldPos[1] = invcammatrix.d.y;
                vfc.camWorldPos[2] = invcammatrix.d.z;
                vfc.camForward[0] = camdir.x;
                vfc.camForward[1] = camdir.y;
                vfc.camForward[2] = camdir.z;
                vfc.camWorldPos[3] = static_cast<float>(volfoglightexact);
                vfc.camForward[3] = static_cast<float>(volfoglightsamples);
                vfc.lava0[0] = ptlavasize; vfc.lava0[1] = ptlavaflow; vfc.lava0[2] = ptlavarelief; vfc.lava0[3] = ptlavacrack;
                vfc.lava1[0] = ptlavatemp; vfc.lava1[1] = ptlavacrust; vfc.lava1[2] = scrollClockSeconds() * ptlavaspeed; vfc.lava1[3] = ptlavacrustamt;
                vfc.lavaFog[1] = ptlavaglow * (0.2126f * sauerinterop::kLavaNominal[0] + 0.7152f * sauerinterop::kLavaNominal[1] + 0.0722f * sauerinterop::kLavaNominal[2]);
                vfc.lavaFog[2] = ptlavaglow;
                lavaTintFromMap(vfc.lavaTint[0], vfc.lavaTint[1]);  // the fog's lava light matches the surface
                vfc.emissiveFog[1] = volfogglowmis ? 1.0f : 0.0f;  // see updateVolFogDescriptorSet
                vfc.emissiveFog[2] = volfogcull ? 1.0f : 0.0f;
                vfc.emissiveFog[3] = static_cast<float>(volfogrelight);
                vfc.lavaFog[3] = (ptlava && ptlavamode && volfogglowmis) ? 1.0f : 0.0f;  // procedural lava present (else the fog keeps plain light sampling)
                volFogCamForwardUsed_ = camdir;
                int volFogVolumeMat = volFogMat_ & MATF_VOLUME;
                if (volFogVolumeMat == MAT_WATER && hasPtWater_) {
                    // path-traced water owns the underwater medium, no second fog tint
                    vfc.mediumOverride[0] = vfc.mediumOverride[1] = vfc.mediumOverride[2] = 0.0f;
                    vfc.mediumOverride[3] = 0.0f;
                } else if (volFogVolumeMat == MAT_WATER) {
                    const bvec &wcol = getwatercolor(volFogMat_);
                    vfc.mediumOverride[0] = wcol.x / 255.0f;
                    vfc.mediumOverride[1] = wcol.y / 255.0f;
                    vfc.mediumOverride[2] = wcol.z / 255.0f;
                    vfc.mediumOverride[3] = 1.0f;
                } else if (volFogVolumeMat == MAT_LAVA) {
                    const bvec &lcol = getlavacolor(volFogMat_);
                    vfc.mediumOverride[0] = lcol.x / 255.0f;
                    vfc.mediumOverride[1] = lcol.y / 255.0f;
                    vfc.mediumOverride[2] = lcol.z / 255.0f;
                    vfc.mediumOverride[3] = 1.0f;
                } else {
                    vfc.mediumOverride[0] = vfc.mediumOverride[1] = vfc.mediumOverride[2] = 0.0f;
                    vfc.mediumOverride[3] = 0.0f;
                }

                std::string vferr;
                {
                    VolFogTail& t = volFogTail_;
                    t.camPos[0] = vfc.camWorldPos[0]; t.camPos[1] = vfc.camWorldPos[1]; t.camPos[2] = vfc.camWorldPos[2];
                    const bool medium = vfc.mediumOverride[3] > 0.5f;
                    t.density[0] = vfc.baseDensity;
                    t.density[1] = vfc.fogBaseHeight;
                    t.density[2] = medium ? 0.0f : vfc.heightFalloff;  // inside water/lava
                    t.density[3] = volfogfar;
                    t.sun[0] = -vfc.sunDirectionRadius[0]; t.sun[1] = -vfc.sunDirectionRadius[1]; t.sun[2] = -vfc.sunDirectionRadius[2];
                    t.sun[3] = vfc.phaseG;
                    t.sunColor[0] = vfc.sunColorPad[0]; t.sunColor[1] = vfc.sunColorPad[1]; t.sunColor[2] = vfc.sunColorPad[2];
                    t.ambient[0] = vfc.skyAmbientPad[0] * 0.25f; t.ambient[1] = vfc.skyAmbientPad[1] * 0.25f; t.ambient[2] = vfc.skyAmbientPad[2] * 0.25f;
                    t.base[0] = medium ? vfc.mediumOverride[0] : 1.0f;
                    t.base[1] = medium ? vfc.mediumOverride[1] : 1.0f;
                    t.base[2] = medium ? vfc.mediumOverride[2] : 1.0f;
                }
                if (!frame_->executeVolumetricFog(*scene_, vfc, &vferr)) {
                    logmsg("executeVolumetricFog failed: " + vferr);
                    volFogHistoryValid_ = false;
                } else {
                    volFogHistoryValid_ = true;
                    partLightThisFrame_ = partLightWanted;
                }
            }

            c.frameIndex = static_cast<uint32_t>(frameCounter_++);
            c.bounceCount = static_cast<uint32_t>(pathtracebounces) | (static_cast<uint32_t>(pathtracespp) << 16);
            c.historyValid = pathTraceHistoryValid_;
            c.sunDirection[0] = -sunlightdir.x;
            c.sunDirection[1] = -sunlightdir.y;
            c.sunDirection[2] = -sunlightdir.z;
            c.sunAngularRadius = pathtracesunangle;
            if (sunlight) {
                c.sunColor[0] = ptLightChannel(sunlightcolor.x / 255.0f) * sunlightscale;
                c.sunColor[1] = ptLightChannel(sunlightcolor.y / 255.0f) * sunlightscale;
                c.sunColor[2] = ptLightChannel(sunlightcolor.z / 255.0f) * sunlightscale;
            } else {
                c.sunColor[0] = c.sunColor[1] = c.sunColor[2] = 0.0f;
            }
            {
                static bool haveLast = false;
                static float lastDir[3] = {0,0,0};
                static float lastColor[3] = {0,0,0};
                static int lastSunlight = -1;
                bool changed = !haveLast || lastSunlight != sunlight ||
                    lastDir[0] != c.sunDirection[0] || lastDir[1] != c.sunDirection[1] || lastDir[2] != c.sunDirection[2] ||
                    lastColor[0] != c.sunColor[0] || lastColor[1] != c.sunColor[1] || lastColor[2] != c.sunColor[2];
                if (changed) {
                    vlogmsg("pathtrace sun debug: sunlight=" + std::to_string(sunlight) +
                        " sunlightdir=(" + std::to_string(sunlightdir.x) + "," + std::to_string(sunlightdir.y) + "," + std::to_string(sunlightdir.z) + ")" +
                        " sunlightcolor=(" + std::to_string(int(sunlightcolor.x)) + "," + std::to_string(int(sunlightcolor.y)) + "," + std::to_string(int(sunlightcolor.z)) + ")" +
                        " sunlightscale=" + std::to_string(sunlightscale) +
                        " => c.sunDirection=(" + std::to_string(c.sunDirection[0]) + "," + std::to_string(c.sunDirection[1]) + "," + std::to_string(c.sunDirection[2]) + ")" +
                        " c.sunColor=(" + std::to_string(c.sunColor[0]) + "," + std::to_string(c.sunColor[1]) + "," + std::to_string(c.sunColor[2]) + ")" +
                        " c.sunAngularRadius=" + std::to_string(c.sunAngularRadius));
                    haveLast = true;
                    lastSunlight = sunlight;
                    lastDir[0]=c.sunDirection[0]; lastDir[1]=c.sunDirection[1]; lastDir[2]=c.sunDirection[2];
                    lastColor[0]=c.sunColor[0]; lastColor[1]=c.sunColor[1]; lastColor[2]=c.sunColor[2];
                }
            }
            c.skyAmbient[0] = ptLightChannel(skylightcolor.x / 255.0f);
            c.skyAmbient[1] = ptLightChannel(skylightcolor.y / 255.0f);
            c.skyAmbient[2] = ptLightChannel(skylightcolor.z / 255.0f);
            c.ambientFloor[0] = ptLightChannel(ambientcolor.x / 255.0f);
            c.ambientFloor[1] = ptLightChannel(ambientcolor.y / 255.0f);
            c.ambientFloor[2] = ptLightChannel(ambientcolor.z / 255.0f);
            c.glowScale = pathtraceglowscale;
            c.maxHistorySamples = static_cast<float>(pathtracemaxhistory);
            c.pixelSpreadAngle = atanf(2.0f * tanf(fovy * RAD * 0.5f) / static_cast<float>(curHeight_));
            c.glassThinness = pathtraceglassthinness;

            // sky hit sampling
            for (int i = 0; i < 6; ++i) c.skyFaceTexIndex[i] = skyFaceTexIndices_[i];
            c.skyValid = skyValid_;
            c.blueNoiseTexIndex = pathtracebluenoise ? blueNoiseTexIndex_ : 0xFFFFFFFFu;
            c.skyLightScale = pathtraceskylight;
            c.lightmapClamp = pathtracelightclamp != 0;
            c.plainSpecular = pathtraceplainspec;
            c.pomEnabled = pathtracepom != 0;
            c.pomMaxIterations = static_cast<uint32_t>(pathtracepomsteps);
            c.pomDepthScale = pathtracepomdepth;
            c.pomMode = static_cast<uint32_t>(pathtracepommode);
            c.glassVolumeTint = pathtraceglasstint;
            c.alphaGlassTint = pathtracealphaglasstint;
            c.modelEnvReflect = ptmodelreflect;
            c.causticsEnabled = ptcaustics != 0 && hasPtWater_;  // water caustics need path-traced water
            c.rayStats = ptstats != 0;
            c.rayStatsProfile = ptprofile != 0;
            c.lightSampleBounces = ptlightsample != 0;
            c.lightSplitExact = static_cast<uint32_t>(ptlightexact);
            c.lightSplitRandom = static_cast<uint32_t>(ptlightrandom);
            c.perfCut = static_cast<uint32_t>(ptcut);
            c.perfSkipShadow = ptcutshadow != 0;
            c.risBounceSamples = static_cast<uint32_t>(ptrisbouncesamples); c.risBounceCandidates = static_cast<uint32_t>(ptrisbouncecandidates);
            c.traceMode = static_cast<uint32_t>(ptraygen); scene_->pathTraceRgenStackMin_ = static_cast<uint32_t>(ptraygenstack); c.restirEnabled = ptrestir != 0; c.restirMinLights = ptrestir == 2 ? static_cast<uint32_t>(ptrestirmin) : 0u; c.restirSpatial = static_cast<uint32_t>(ptrestirspatial); c.restirCandidates = static_cast<uint32_t>(ptrestircandidates);
            c.risEnabled = ptris == 1 || (ptris == 2 && rrCompositeMode()); c.risImportance = ptrisgrid != 0; c.risCandidates = static_cast<uint32_t>(ptriscandidates); c.risSamples = static_cast<uint32_t>(ptrissamples);
            // camera medium from gl_drawframe()'s fogmat classification (setFogMaterialState())
            c.cameraInWater = hasPtWater_ && (volFogMat_ & MATF_VOLUME) == MAT_WATER;
            if (c.cameraInWater) {
                const bvec &wcol = getwatercolor(volFogMat_);
                c.cameraWaterColor[0] = wcol.x / 255.0f; c.cameraWaterColor[1] = wcol.y / 255.0f; c.cameraWaterColor[2] = wcol.z / 255.0f;
                const int wfog = getwaterfog(volFogMat_);
                c.cameraWaterDepth = wfog > 0 ? float(wfog) : 150.0f;
                c.cameraWaterSurfaceZ = cameraLiquidSurfaceZ_;  // gl_drawframe()'s own findsurface() - WATER_OFFSET
            }
            c.waterClarity = ptwaterclarity;
            c.waterWaves = ptwaterwaves;
            c.waterScatter = ptwaterscatter;
            c.waterHistory = static_cast<float>(ptwaterhistory);
            c.waterDebug = static_cast<float>(ptwaterdebug);
            if (ptcache && lastPtCache_ == 0) cacheClearPending_ = true;
            lastPtCache_ = ptcache;
            if (cacheClearPending_ && scene_) { scene_->clearRadianceCache(); cacheClearPending_ = false; }
            c.cacheEnabled = ptcache != 0;
            c.cacheCell = ptcachecell;
            c.cacheFrames = static_cast<float>(ptcacheframes);
            c.cacheUpdate = static_cast<float>(ptcacheupdate);
            c.cacheFloor = ptcachefloor; c.cacheFallback = ptcachefallback != 0; c.cacheCoherent = ptcachecoherent != 0;
            c.skyboxLight = ptsky != 0;
            c.physicalMaterials = ptmat != 0;
            c.particleReflect = ptpartreflect;
            c.transportAlbedo = ptalbedo != 0;
            c.transportAlbedoGamma = ptalbedogamma;
            c.timeSeconds = scrollClockSeconds();
            frame_->setGpuStatsEnabled(ptstats != 0);
            c.textureGradScale = (dlssSceneRealDlssActive_ && screenw > 0 && curWidth_ > 0)
                ? std::min(1.0f, static_cast<float>(curWidth_) / static_cast<float>(screenw)) : 1.0f;
            c.causticRadius = ptcausticradius;
            c.causticRange = ptcausticrange;
            c.causticRipple = ptcausticripple;
            c.causticPhotons = static_cast<uint32_t>(ptcausticphotons) * 1024u;
            c.skyRotationRadians = (spinsky * static_cast<float>(lastmillis) / 1000.0f + static_cast<float>(yawsky)) * -RAD;

            c.lightCount = lightCount;

            const uint32_t pathTraceSlotIndex = frame_->nextSlotIndex();
            if (!scene_->setPathTraceLights(pathTraceSlotIndex, lightCount ? lightPacked.data() : nullptr, lightCount)) {
                logmsg("RayTracingScene::setPathTraceLights failed: " + ctx_->lastError());
            }

            c.resourceEpoch = nativeResourceEpoch_;
            std::vector<interop::SharedTexture*> ptResources = {
                depthTex_.get(), normalTex_.get(), albedoTex_.get(), motionTex_.get(),
                pathTraceOutput_.get(), pathTraceHistoryPrev_.get(), pathTraceHistoryNext_.get(),
                pathTraceHistoryMetaPrev_.get(), pathTraceHistoryMetaNext_.get(),
                pathTraceRawRadiance_.get(),
                glowTex_.get(),  // appended
                pathTracePrimaryNormal_.get(),
                pathTraceLinearDepth_.get(),
            };
            std::string err;
            if (!frame_->executePathTrace(*scene_, ptResources, c, &err)) {
                logmsg("executePathTrace failed: " + err);
                return;
            }
            // accumulate, log a summary every 2 s
            if (ptstats) {
                static interop::InteropFrame::GpuStats acc;
                static int windowStart = 0;
                interop::InteropFrame::GpuStats st = frame_->takeGpuStats();
                acc.ptCausticsMs += st.ptCausticsMs; acc.ptTraceMs += st.ptTraceMs; acc.ptResolveMs += st.ptResolveMs;
                acc.fogInjectMs += st.fogInjectMs; acc.fogIntegrateMs += st.fogIntegrateMs;
                acc.ptFrames += st.ptFrames; acc.fogFrames += st.fogFrames;
                for (int i = 0; i < 64; ++i) acc.rays[i] += st.rays[i];
                if (!windowStart) windowStart = totalmillis;
                if (totalmillis - windowStart >= 2000 && acc.ptFrames > 0) {
                    const double pf = acc.ptFrames, ff = acc.fogFrames > 0 ? acc.fogFrames : 1;
                    const double pixels = acc.rays[1] > 0 ? double(acc.rays[1]) : 1.0;
                    const double lit = acc.rays[7] > 0 ? double(acc.rays[7]) : 1.0;
                    const double total = double(acc.rays[1] + acc.rays[2] + acc.rays[3] + acc.rays[4] + acc.rays[5] + acc.rays[6]);
                    char buf[1024];
                    snprintf(buf, sizeof(buf),
                        "ptstats (%u frames, %dx%d, %u lights): GPU ms/frame caustics %.2f trace %.2f resolve %.2f | fog inject %.2f integrate %.2f | "
                        "rays/pixel: primary %.2f sun %.2f lights@primary %.2f lights@bounces %.2f glow %.2f bounce %.2f = total %.2f | "
                        "lit points/pixel %.2f, lights in range per lit point %.2f, light-list entries scanned per lit point %.2f, sky pixels %.1f%%",
                        acc.ptFrames, int(curWidth_), int(curHeight_), unsigned(c.lightCount),
                        acc.ptCausticsMs / pf, acc.ptTraceMs / pf, acc.ptResolveMs / pf,
                        acc.fogInjectMs / ff, acc.fogIntegrateMs / ff,
                        acc.rays[1] / pixels, acc.rays[2] / pixels, acc.rays[3] / pixels, acc.rays[4] / pixels,
                        acc.rays[5] / pixels, acc.rays[6] / pixels, total / pixels,
                        acc.rays[7] / pixels, acc.rays[8] / lit, 8.0 * acc.rays[10] / lit, 100.0 * acc.rays[9] / pixels);
                    logmsg(buf);
                    // pathtrace_trace.comp's gRayStat[11..15]
                    snprintf(buf, sizeof(buf),
                        "ptstats transport: mean albedo at bounce hits %.3f | radiance/pixel %.4f, from vertex 4+ %.4f (%.2f%%) | camera-seen albedo %.4f/pixel",
                        acc.rays[12] > 0 ? acc.rays[11] / 256.0 / double(acc.rays[12]) : 0.0,
                        acc.rays[13] / 256.0 / pixels, acc.rays[14] / 256.0 / pixels,
                        acc.rays[13] > 0 ? 100.0 * double(acc.rays[14]) / double(acc.rays[13]) : 0.0,
                        acc.rays[15] / 256.0 / pixels);
                    logmsg(buf);
                    // path endings at the first bounce hit, pathtrace_trace.comp's gRayStat[16..19]
                    if (acc.rays[60] > 0) {
                        const double b1 = double(acc.rays[60]);
                        snprintf(buf, sizeof(buf),
                            "ptstats path endings at the first bounce hit (%.2f per pixel): radiance cache %.1f%% | cache-update paths %.1f%% | cell missing/unsettled %.1f%% | glass/water/moving %.1f%%",
                            b1 / pixels, 100.0 * acc.rays[61] / b1, 100.0 * acc.rays[62] / b1, 100.0 * acc.rays[63] / b1,
                            100.0 * (b1 - acc.rays[61] - acc.rays[62] - acc.rays[63]) / b1);
                        logmsg(buf);
                    }
                    // pathtrace_trace.comp's PT_PROFILE build (needs VK_KHR_shader_clock)
                    {
                        auto sec = [&](int i) { return double(acc.rays[16 + 2 * i]) + double(acc.rays[17 + 2 * i]) * 4294967296.0; };
                        const double tot = sec(0);
                        if (tot > 0.0) {
                            const double traceMs = acc.ptTraceMs / pf;
                            static const char* names[22] = { "total", "primary ray+fetch", "NEE camera hit (lightmap cap)", "NEE bounce hits (lightmap cap)",
                                "bounce traversal", "bounce hit fetch", "caustics gather", "RR specular ray", "pre-loop setup",
                                "loop head (bounce 0)", "loop head (bounces 1+)", "BSDF/next-dir sampling", "bounce hit shading",
                                "miss/sky shading", "final writes", "radiance-cache write-back", "floor/per-sample end",
                                "NEE sun", "NEE sky light", "NEE point lights", "NEE emissive triangle", "NEE setup" };
                            std::string line = "ptstats trace-pass breakdown (" + std::to_string(traceMs).substr(0, 6) + " ms):";
                            double known = 0.0;
                            for (int i = 1; i < 22; ++i) {
                                if (i == 9 || i == 17) { logmsg(line); line = "ptstats trace-pass breakdown (cont.):"; }  // two lines, the log truncates long ones
                                const double f = sec(i) / tot;
                                known += f;
                                char part[96];
                                snprintf(part, sizeof(part), " %s %.2f ms (%.0f%%) |", names[i], f * traceMs, 100.0 * f);
                                line += part;
                            }
                            char tail[64];
                            snprintf(tail, sizeof(tail), " unaccounted %.0f%%", 100.0 * (1.0 - known));
                            line += tail;
                            logmsg(line);
                        } else if (!ctx_->capabilities().hasShaderClock) {
                            static bool said = false;
                            if (!said) { said = true; logmsg("ptstats: no trace-pass breakdown -- this GPU/driver has no VK_KHR_shader_clock."); }
                        }
                    }
                    acc = interop::InteropFrame::GpuStats{};
                    windowStart = totalmillis;
                }
            }

            pathTraceHistoryPrev_.swap(pathTraceHistoryNext_);
            pathTraceHistoryMetaPrev_.swap(pathTraceHistoryMetaNext_);
            pathTraceHistoryValid_ = true;

        }

        if (!executeRtaoDispatch(camProjUnjittered)) return;

        mvCurrVp_ = camProjUnjittered; mvPrevVp_ = prevCamProjMatrixUnjittered_; mvFresh_ = true;
        runPtSurfaceMotion();  // water / glass volumes / lava
        prevCamProjMatrix_ = camprojmatrix;
        prevCamProjMatrixUnjittered_ = camProjUnjittered;

        {
            const uint32_t maskW = sceneTargetActive_ ? static_cast<uint32_t>(sceneTargetRenderW_) : curWidth_;
            const uint32_t maskH = sceneTargetActive_ ? static_cast<uint32_t>(sceneTargetRenderH_) : curHeight_;
            std::string maskErr;
            if (!ensureRtMask(maskW, maskH, &maskErr)) {
                logmsg("ensureRtMask (per-frame resize) failed: " + maskErr);
            }
        }

        renderRtMask();

        compositePathTrace();
    }

    bool executeRtaoDispatch(const matrix4& camProjUnjittered) {
        interop::RtaoConstants c;
        std::memcpy(c.invViewProj, invcamprojmatrix.a.v, sizeof(float) * 16);
        std::memcpy(c.invView, invcammatrix.a.v, sizeof(float) * 16);
        std::memcpy(c.currentViewProjUnjittered, camProjUnjittered.a.v, sizeof(float) * 16);
        std::memcpy(c.prevViewProjJittered, prevCamProjMatrix_.a.v, sizeof(float) * 16);
        std::memcpy(c.prevViewProjUnjittered, prevCamProjMatrixUnjittered_.a.v, sizeof(float) * 16);
        c.radius = rtaoradius;
        c.intensity = rtaointensity;
        c.bias = rtaobias;
        c.frameIndex = static_cast<uint32_t>(frameCounter_++);
        c.raysPerPixel = static_cast<uint32_t>(rtaorays);
        c.historyValid = historyValid_ && static_cast<bool>(rtaotemporal);
        c.lightDirection[0] = -sunlightdir.x;
        c.lightDirection[1] = -sunlightdir.y;
        c.lightDirection[2] = -sunlightdir.z;
        c.lightRadius = rtaoshadowradius;
        c.shadowRaysPerPixel = static_cast<uint32_t>(rtaoshadowrays);
        c.shadowEnabled = static_cast<bool>(rtaoshadow);
        c.resourceEpoch = nativeResourceEpoch_;

        std::vector<interop::SharedTexture*> resources = {
            depthTex_.get(), normalTex_.get(), motionTex_.get(),
            aoOutput_.get(), aoHistoryPrev_.get(), aoHistoryNext_.get(),
            shadowOutput_.get(), shadowHistoryPrev_.get(), shadowHistoryNext_.get(),
            momentsHistoryPrev_.get(), momentsHistoryNext_.get(),
        };
        std::string err;
        if (!frame_->executeRtao(*scene_, resources, c, &err)) {
            logmsg("executeRtao failed: " + err);
            return false;
        }

        aoHistoryPrev_.swap(aoHistoryNext_);
        shadowHistoryPrev_.swap(shadowHistoryNext_);
        momentsHistoryPrev_.swap(momentsHistoryNext_);  // SVGF-style
        historyValid_ = true;
        return true;
    }

    bool beginSceneTarget(int &w, int &h, uint32_t &fbo) {
        sceneTargetActive_ = false;
        // !hasGeometry_ too, mirroring compositeIntoScene()'s identical gate
        if (!ready_ || !hasGeometry_) return false;
        if (!sceneTargetWanted_) {
            // MSAA on the default framebuffer is meaningless once the scene is rendered offscreen
            // anyway
            if (dlssscene && fsaa > 0 && !loggedFsaaRefusal_) {
                logmsg("beginSceneTarget: refusing -- fsaa > 0 is incompatible with the offscreen scene target (dlssscene has no effect while anti-aliasing is enabled).");
                loggedFsaaRefusal_ = true;
            }
            return false;
        }
        const int rw = sceneTargetRenderW_, rh = sceneTargetRenderH_;
        std::string err;
        if (!ensureSceneTarget(static_cast<uint32_t>(rw), static_cast<uint32_t>(rh), &err)) {
            logmsg("beginSceneTarget: ensureSceneTarget failed: " + err);
            return false;
        }
        w = rw;
        h = rh;
        fbo = sceneFbo_;
        sceneTargetActive_ = true;
        return true;
    }

    void executeDlssSceneEvaluate() {
        interop::DlssFrameConstants dc;
        matrix4 camProjUnjittered; camProjUnjittered.muld(projMatrixUnjittered_, cammatrix);
        matrix4 dlssSceneInvProjUnjittered; dlssSceneInvProjUnjittered.invert(projMatrixUnjittered_);
        matrix4 dlssSceneInvCurrentVpUnjittered; dlssSceneInvCurrentVpUnjittered.invert(camProjUnjittered);
        // last frame's view-projection
        const matrix4 &dlssPrevVp = mvFresh_ ? mvPrevVp_ : prevCamProjMatrixUnjittered_;
        matrix4 dlssSceneClipToPrevClip; dlssSceneClipToPrevClip.muld(dlssPrevVp, dlssSceneInvCurrentVpUnjittered);
        matrix4 dlssScenePrevClipToClip; dlssScenePrevClipToClip.invert(dlssSceneClipToPrevClip);
        std::memcpy(dc.cameraViewToClip, projMatrixUnjittered_.a.v, sizeof(float) * 16);
        std::memcpy(dc.clipToCameraView, dlssSceneInvProjUnjittered.a.v, sizeof(float) * 16);
        std::memcpy(dc.clipToPrevClip, dlssSceneClipToPrevClip.a.v, sizeof(float) * 16);
        std::memcpy(dc.prevClipToClip, dlssScenePrevClipToClip.a.v, sizeof(float) * 16);
        // per-frame pixel-space offset
        dc.jitterX = jitterPixelX_;
        dc.jitterY = jitterPixelY_;
        dc.mvecScaleX = 1.0f;
        dc.mvecScaleY = 1.0f;
        dc.cameraPosX = invcammatrix.d.x; dc.cameraPosY = invcammatrix.d.y; dc.cameraPosZ = invcammatrix.d.z;
        dc.cameraRightX = invcammatrix.a.x; dc.cameraRightY = invcammatrix.a.y; dc.cameraRightZ = invcammatrix.a.z;
        dc.cameraUpX = invcammatrix.b.x; dc.cameraUpY = invcammatrix.b.y; dc.cameraUpZ = invcammatrix.b.z;
        dc.cameraFwdX = -invcammatrix.c.x; dc.cameraFwdY = -invcammatrix.c.y; dc.cameraFwdZ = -invcammatrix.c.z;
        dc.cameraNear = nearplane;
        dc.cameraFar = static_cast<float>(farplane);
        dc.cameraFov = fovy * RAD;  // Sauerbraten's own fovy is in degrees
        dc.cameraAspect = aspect;
        dc.reset = !dlssSceneHistoryValid_;

        std::string err;
        if (dlssSceneConfiguredUsedRr_ && dlssSplitProgram_ && normalTex_ && dlssRrDiffuseAlbedo_ && dlssRrSpecularAlbedo_) {
            runDlssSplit();
            copyDlssRrInputs(sceneColor_.get(), sceneDepthTex_.get());
            if (pathtrace) runRrSpecMotion(dlssSceneInvCurrentVpUnjittered, dlssPrevVp);
            if (pathtrace) runRrWaterGuides();  // the water's own normal/roughness/albedo guides
            const bool rrLinearized = rrCompositeMode() && runRrTone(sceneColor_->glTextureId(), dlssRrColorInput_.get(), 0);
            std::vector<interop::SharedTexture*> dlssRrResources = {
                dlssRrColorInput_.get(), dlssRrDepthInput_.get(), dlssRrMotionInput_.get(), dlssSceneColorOutput_.get(),
                dlssRrDiffuseAlbedo_.get(), dlssRrSpecularAlbedo_.get(), dlssRrNormalRoughnessInput_.get(),
                dlssRrSpecularMotionInput_.get(),
            };
            if (!dlssFrame_->executeDlssRr(dlssRrResources, dc, &err)) {
                logmsg("DLSS-scene: executeDlssRr failed: " + err);
                dlssSceneRealDlssActive_ = false;
                return;
            }
            // tonemap sandwich, step 2
            if (rrLinearized) {
                if (!rrDisplayOutput_ || rrDisplayOutput_->width() != dlssSceneColorOutput_->width() ||
                    rrDisplayOutput_->height() != dlssSceneColorOutput_->height()) {
                    rrDisplayOutput_.reset();
                    makeTexture(rrDisplayOutput_, interop::PixelFormat::Rgba16Float,
                        dlssSceneColorOutput_->width(), dlssSceneColorOutput_->height(), "rrDisplayOutput_");
                }
                rrDisplayValid_ = rrDisplayOutput_ && runRrTone(dlssSceneColorOutput_->glTextureId(), rrDisplayOutput_.get(), 1);
            }
        } else {
            // same "dlssCtx_'s separate VkDevice, dedicated copies only" reasoning as the RR branch
            // above
            copyDlssRrInputs(sceneColor_.get(), sceneDepthTex_.get());
            std::vector<interop::SharedTexture*> dlssSceneResources = {
                dlssRrColorInput_.get(), dlssRrDepthInput_.get(), dlssRrMotionInput_.get(), dlssSceneColorOutput_.get(),
            };
            if (!dlssFrame_->executeDlss(dlssSceneResources, dc, &err)) {
                logmsg("DLSS-scene: executeDlss failed: " + err);
                dlssSceneRealDlssActive_ = false;
                return;
            }
        }
        dlssSceneHistoryValid_ = true;
    }

    void presentScene() {
        if (!sceneTargetActive_) return;
        if (!sceneColor_ || !presentSceneProgram_ || !sceneDepthRestoreProgram_ || !debugVao_) return;

        // run the Streamline evaluate before the composite draws below
        rrDisplayValid_ = false;
        if (dlssSceneRealDlssActive_) executeDlssSceneEvaluate();
        interop::SharedTexture* presentSource =
            (rrDisplayValid_ && dlssSceneRealDlssActive_) ? rrDisplayOutput_.get() :
            (dlssSceneRealDlssActive_ && dlssSceneColorOutput_) ? dlssSceneColorOutput_.get() : sceneColor_.get();

        GLint prevVao = 0;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
        GLboolean prevDepthTest = glIsEnabled(GL_DEPTH_TEST);
        GLboolean prevDepthMask = GL_TRUE;
        glGetBooleanv(GL_DEPTH_WRITEMASK, &prevDepthMask);
        GLboolean prevBlend = glIsEnabled(GL_BLEND);
        GLboolean prevColorMask[4] = { GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE };
        glGetBooleanv(GL_COLOR_WRITEMASK, prevColorMask);
        GLboolean prevScissorTest = glIsEnabled(GL_SCISSOR_TEST);
        GLboolean prevCullFace = glIsEnabled(GL_CULL_FACE);

        glBindFramebuffer_(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, screenw, screenh);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_CULL_FACE);
        glDisable(GL_BLEND);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

        // increment 4
        const float outputSizeInvX = screenw > 0 ? 1.0f / static_cast<float>(screenw) : 0.0f;
        const float outputSizeInvY = screenh > 0 ? 1.0f / static_cast<float>(screenh) : 0.0f;

        bool bloomReady = false;
        if (pathtracebloom && presentSource) {
            const uint32_t bloomW = std::max(1u, static_cast<uint32_t>(presentSource->width()) / 4u);
            const uint32_t bloomH = std::max(1u, static_cast<uint32_t>(presentSource->height()) / 4u);
            std::string bloomErr;
            if (ensureBloomTarget(bloomW, bloomH, &bloomErr) && bloomThresholdProgram_ && bloomBlurProgram_) {
                const float bloomSizeInvX = 1.0f / static_cast<float>(bloomW);
                const float bloomSizeInvY = 1.0f / static_cast<float>(bloomH);
                glViewport(0, 0, static_cast<GLsizei>(bloomW), static_cast<GLsizei>(bloomH));
                glBindVertexArray_(debugVao_);

                // threshold + downsample
                glBindFramebuffer_(GL_FRAMEBUFFER, bloomFboA_);
                glUseProgram_(bloomThresholdProgram_);
                glActiveTexture_(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, presentSource->glTextureId());
                if (bloomThresholdUSceneLoc_ >= 0) glUniform1i_(bloomThresholdUSceneLoc_, 0);
                if (bloomThresholdUOutputSizeInvLoc_ >= 0) glUniform2f_(bloomThresholdUOutputSizeInvLoc_, bloomSizeInvX, bloomSizeInvY);
                if (bloomThresholdUThresholdLoc_ >= 0) glUniform1f_(bloomThresholdUThresholdLoc_, pathtracebloomthreshold);
                glDrawArrays(GL_TRIANGLES, 0, 3);

                // blur H: bloomTexA_ -> bloomTexB_
                glBindFramebuffer_(GL_FRAMEBUFFER, bloomFboB_);
                glUseProgram_(bloomBlurProgram_);
                glActiveTexture_(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, bloomTexA_);
                if (bloomBlurUSrcLoc_ >= 0) glUniform1i_(bloomBlurUSrcLoc_, 0);
                if (bloomBlurUOutputSizeInvLoc_ >= 0) glUniform2f_(bloomBlurUOutputSizeInvLoc_, bloomSizeInvX, bloomSizeInvY);
                if (bloomBlurUDirectionLoc_ >= 0) glUniform2f_(bloomBlurUDirectionLoc_, 1.0f, 0.0f);
                glDrawArrays(GL_TRIANGLES, 0, 3);

                // blur V: bloomTexB_ -> bloomTexA_. bloomTexA_ is the final bloom result Pass 1 below
                // binds
                glBindFramebuffer_(GL_FRAMEBUFFER, bloomFboA_);
                glBindTexture(GL_TEXTURE_2D, bloomTexB_);
                if (bloomBlurUDirectionLoc_ >= 0) glUniform2f_(bloomBlurUDirectionLoc_, 0.0f, 1.0f);
                glDrawArrays(GL_TRIANGLES, 0, 3);

                bloomReady = true;
                glBindFramebuffer_(GL_FRAMEBUFFER, 0);
                glViewport(0, 0, screenw, screenh);
            } else if (!bloomErr.empty()) {
                logmsg("presentScene: ensureBloomTarget failed: " + bloomErr);
            }
        }

        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glUseProgram_(presentSceneProgram_);
        glActiveTexture_(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, presentSource->glTextureId());
        if (presentSceneUSceneLoc_ >= 0) glUniform1i_(presentSceneUSceneLoc_, 0);
        glActiveTexture_(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, bloomReady ? bloomTexA_ : 0);
        if (presentSceneUBloomLoc_ >= 0) glUniform1i_(presentSceneUBloomLoc_, 1);
        if (presentSceneUBloomIntensityLoc_ >= 0) glUniform1f_(presentSceneUBloomIntensityLoc_, bloomReady ? pathtracebloomintensity : 0.0f);
        glActiveTexture_(GL_TEXTURE0);
        if (presentSceneUOutputSizeInvLoc_ >= 0) glUniform2f_(presentSceneUOutputSizeInvLoc_, outputSizeInvX, outputSizeInvY);
        glBindVertexArray_(debugVao_);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glDepthMask(GL_TRUE);
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glUseProgram_(sceneDepthRestoreProgram_);
        glActiveTexture_(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sceneDepthTex_->glTextureId());
        if (sceneDepthRestoreUDepthLoc_ >= 0) glUniform1i_(sceneDepthRestoreUDepthLoc_, 0);
        if (sceneDepthRestoreUOutputSizeInvLoc_ >= 0) glUniform2f_(sceneDepthRestoreUOutputSizeInvLoc_, outputSizeInvX, outputSizeInvY);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glBindVertexArray_(static_cast<GLuint>(prevVao));
        glUseProgram_(0);

        if (prevCullFace) glEnable(GL_CULL_FACE);
        if (prevScissorTest) glEnable(GL_SCISSOR_TEST);
        glColorMask(prevColorMask[0], prevColorMask[1], prevColorMask[2], prevColorMask[3]);
        if (prevBlend) glEnable(GL_BLEND);
        if (prevDepthTest) glEnable(GL_DEPTH_TEST);
        glDepthMask(prevDepthMask);
    }

    void drawDebugOverlay() {
        if (!ready_) return;  // onFrame() never successfully initialized
        static uint32_t tailCount = 0;
        if ((tailCount++ % 60) == 0) {
            vlogmsg("drawDebugOverlay: reached, rtaodebug=" + std::to_string(static_cast<int>(rtaodebug)));
        }
        if (rtaodebug) drawDebugBlit();
        if (mvdebug) drawMotionDebug();
    }

    void compositeIntoScene() {
        static uint32_t callCount = 0;
        if ((callCount++ % 60) == 0) {
            vlogmsg("compositeIntoScene: reached, ready_=" + std::to_string(static_cast<int>(ready_)) +
                " hasGeometry_=" + std::to_string(static_cast<int>(hasGeometry_)) +
                " rtaoblend=" + std::to_string(static_cast<int>(rtaoblend)) +
                " pathtrace=" + std::to_string(static_cast<int>(pathtrace)));
        }
        if (!ready_ || !hasGeometry_) return;
        return;
    }

    void shutdown() {
        if (!attempted_) return;
        if (debugVao_) glDeleteVertexArrays_(1, &debugVao_);
        if (debugProgram_) glDeleteProgram_(debugProgram_);
        if (mvDebugProgram_) glDeleteProgram_(mvDebugProgram_);
        mvDebugProgram_ = 0;
        if (rrWaterProgram_) glDeleteProgram_(rrWaterProgram_);
        if (rrWaterFbo_) glDeleteFramebuffers_(1, &rrWaterFbo_);
        rrWaterProgram_ = 0;
        rrWaterFbo_ = 0;
        if (ptSurfMotionProgram_) glDeleteProgram_(ptSurfMotionProgram_);
        if (ptSurfMotionFbo_) glDeleteFramebuffers_(1, &ptSurfMotionFbo_);
        ptSurfMotionProgram_ = 0;
        ptSurfMotionFbo_ = 0;
        if (compositeProgram_) glDeleteProgram_(compositeProgram_);
        if (pathTraceCompositeProgram_) glDeleteProgram_(pathTraceCompositeProgram_);  // path tracing
        if (pathTraceDepthProgram_) glDeleteProgram_(pathTraceDepthProgram_);
        if (presentSceneProgram_) glDeleteProgram_(presentSceneProgram_);  // DLSS-on-final-frame
        if (sceneDepthRestoreProgram_) glDeleteProgram_(sceneDepthRestoreProgram_);  // DLSS-on-final-frame
        if (rtMaskProgram_) glDeleteProgram_(rtMaskProgram_);
        if (dlssSplitProgram_) glDeleteProgram_(dlssSplitProgram_);  // dlss ray reconstruction
        if (dlssSplitFbo_) glDeleteFramebuffers_(1, &dlssSplitFbo_);  // dlss ray reconstruction
        dlssSplitProgram_ = 0;
        dlssSplitFbo_ = 0;
        if (rrSpecMotionProgram_) glDeleteProgram_(rrSpecMotionProgram_);
        if (rrSpecMotionFbo_) glDeleteFramebuffers_(1, &rrSpecMotionFbo_);
        rrSpecMotionProgram_ = 0;
        rrSpecMotionFbo_ = 0;
        if (rrToneProgram_) glDeleteProgram_(rrToneProgram_);
        if (rrToneFbo_) glDeleteFramebuffers_(1, &rrToneFbo_);
        rrToneProgram_ = 0;
        rrToneFbo_ = 0;
        rrDisplayOutput_.reset();
        rrDisplayValid_ = false;
        debugVao_ = 0;
        debugProgram_ = 0;
        compositeProgram_ = 0;
        pathTraceCompositeProgram_ = 0;  // path tracing
        pathTraceDepthProgram_ = 0;
        presentSceneProgram_ = 0;  // DLSS-on-final-frame
        sceneDepthRestoreProgram_ = 0;  // DLSS-on-final-frame
        rtMaskProgram_ = 0;
        destroyRtMask();  // same "before ctx_" ordering as everything else here
        destroySceneTarget();
        bloomThresholdProgram_ = 0; bloomBlurProgram_ = 0;
        // the GL objects go with the context
        aeMeterProgram_ = aeAdaptProgram_ = 0;
        aeMeterTex_ = aeMeterFbo_ = 0; aeTex_[0] = aeTex_[1] = 0; aeFbo_[0] = aeFbo_[1] = 0;
        aeHistory_ = false;
        destroyBloomTarget();
        sceneTargetActive_ = false;
        sceneTargetWanted_ = false;
        loggedFsaaRefusal_ = false;
        // DLSS-on-final-frame, SharedTexture, same "before ctx_" ordering as everything else in this
        // function
        dlssSceneColorOutput_.reset();
        dlssSceneRealDlssActive_ = false;
        dlssSceneConfigured_ = false;
        dlssSceneConfiguredOutputWidth_ = dlssSceneConfiguredOutputHeight_ = 0;
        dlssSceneConfiguredModeOrdinal_ = -1;
        dlssSceneRenderWidth_ = dlssSceneRenderHeight_ = 0;
        dlssSceneHistoryValid_ = false;
        dlssPathTraceColorOutput_.reset();
        dlssPathTraceActive_ = false;
        dlssPathTraceConfigured_ = false;
        dlssPathTraceConfiguredOutputWidth_ = dlssPathTraceConfiguredOutputHeight_ = 0;
        dlssPathTraceConfiguredModeOrdinal_ = -1;
        dlssPathTraceRenderWidth_ = dlssPathTraceRenderHeight_ = 0;
        dlssPathTraceHistoryValid_ = false;

        // destruction order matters
        frame_.reset();
        scene_.reset();
        destroyResources();
        // dlssFrame_/dlssCtx_'s VkDevice
        dlssFrame_.reset();
        dlssCtx_.reset();
        ctx_.reset();

        ready_ = false;
        hasGeometry_ = false;
        attempted_ = false;
        curWidth_ = curHeight_ = 0;
        nativeResourceEpoch_ = 0;
        historyValid_ = false;
        pathTraceHistoryValid_ = false;
    }

}  // namespace sauerinterop

bool rtQueryFreshPose(dynent *d) {
    return sauerinterop::rtFreshDynents_.count(d) != 0;
}
