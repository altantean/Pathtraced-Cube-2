#pragma once
#include <cstdint>

enum { MDL_MD2 = 0, MDL_MD3, MDL_MD5, MDL_OBJ, MDL_SMD, MDL_IQM, NUMMODELTYPES };

// per-frame GPU-skinning data sink for animated (skeletal) geometry
struct SkinSourceVertexData
{
    float px, py, pz;
    float nx, ny, nz;
    float tx, ty, tz, tw;
    float u, v;
    uint32_t blendBone0, blendBone1, blendBone2, blendBone3;
    float blendWeight0, blendWeight1, blendWeight2, blendWeight3;
    uint32_t textureIndex;
    uint32_t materialIndex;
};

struct SkinExtractSink
{
    virtual bool hasSkinSource(uint64_t modelKey, uint32_t boneCount) = 0;
    virtual void submitSkinSource(uint64_t modelKey, const SkinSourceVertexData *vertices, uint32_t vertexCount,
                                   const uint32_t *indices, uint32_t indexCount, uint32_t boneCount) = 0;
    // one-time (per distinct GL texture, cached by the implementation) bindless- texture-table
    // registration
    virtual uint32_t registerTexture(uint32_t textureGlId) = 0;
    // per-frame submission of one instance's current pose
    virtual void submitPose(uint64_t modelKey, const float *boneDualQuats, uint32_t boneCount,
                             const float worldMatrix[12]) = 0;
    virtual void submitDynamicMesh(uint64_t modelKey, const SkinSourceVertexData *vertices, uint32_t vertexCount,
                                    const uint32_t *indices, uint32_t indexCount) {}
    // perf blas-sharing instancing for rigid (non-deforming) content
    virtual bool hasRigidSource(uint64_t modelKey) { return false; }
    virtual void registerRigidSource(uint64_t modelKey, const SkinSourceVertexData *vertices, uint32_t vertexCount,
                                      const uint32_t *indices, uint32_t indexCount) {}
    virtual void submitRigidInstance(uint64_t modelKey, const float worldMatrix[12]) {}
    virtual bool lastRigidInstanceAccepted() { return false; }
    virtual bool lastPoseAccepted() { return false; }
    virtual void beginInstance(dynent *d) {}
protected:
    ~SkinExtractSink() {}
};

extern SkinExtractSink *rtPosedSink;

// rt interop hook per-frame flag SauerbratenInterop.cpp updates every frame
extern bool rtPathTraceActive;
// Set by renderhudmodel() (rendermodel.cpp) while it submits the first- person weapon/hands to the rt
// sink
extern bool rtHudModelPass;
extern bool rtHudMainPass;

// rt interop hook, per-dynent query
extern bool rtQueryFreshPose(dynent *d);

// Sauerbraten skin::alphatest lookup for a given model's texture
struct model;
extern float findMapModelAlphaTest(model *m, Texture *tex);

struct model
{
    char *name;
    float spinyaw, spinpitch, offsetyaw, offsetpitch;
    bool collide, ellipsecollide, shadow, alphadepth, depthoffset;
    float scale;
    vec translate;
    BIH *bih;
    vec bbcenter, bbradius, bbextend, collidecenter, collideradius;
    float rejectradius, eyeheight, collidexyradius, collideheight;
    int batch;

    model(const char *name) : name(name ? newstring(name) : NULL), spinyaw(0), spinpitch(0), offsetyaw(0), offsetpitch(0), collide(true), ellipsecollide(false), shadow(true), alphadepth(true), depthoffset(false), scale(1.0f), translate(0, 0, 0), bih(0), bbcenter(0, 0, 0), bbradius(-1, -1, -1), bbextend(0, 0, 0), collidecenter(0, 0, 0), collideradius(-1, -1, -1), rejectradius(-1), eyeheight(0.9f), collidexyradius(0), collideheight(0), batch(-1) {}
    virtual ~model() { DELETEA(name); DELETEP(bih); }
    virtual void calcbb(vec &center, vec &radius) = 0;
    virtual void render(int anim, int basetime, int basetime2, const vec &o, float yaw, float pitch, dynent *d, modelattach *a = NULL, const vec &color = vec(0, 0, 0), const vec &dir = vec(0, 0, 0), float transparent = 1) = 0;
    virtual bool load() = 0;
    virtual int type() const = 0;
    // per-loaded-ASSET query
    virtual bool hasRealAnimationData() const { return true; }
    virtual bool mapmodelAnimates() const { return hasRealAnimationData(); }

    // per-frame GPU-skinning data extraction for a specific pose of this model
    virtual bool extractSkinPose(int anim, int basetime, int basetime2, const vec &o, float yaw, float pitch,
                                  dynent *d, SkinExtractSink &sink) { return false; }

    virtual BIH *setBIH() { return 0; }
    virtual bool envmapped() { return false; }
    virtual bool skeletal() const { return false; }

    virtual void setshader(Shader *shader) {}
    virtual void setenvmap(float envmapmin, float envmapmax, Texture *envmap) {}
    virtual void setspec(float spec) {}
    virtual void setambient(float ambient) {}
    virtual void setglow(float glow, float glowdelta, float glowpulse) {}
    virtual void setptemissive(float strength, float lightradius, float lightintensity) {}
    virtual bool ptemissivelight(float &radius, float &intensity, Texture *&tex, Texture *&masks) { return false; }
    virtual void setglare(float specglare, float glowglare) {}
    virtual void setalphatest(float alpha) {}
    virtual void setalphablend(bool blend) {}
    virtual void setfullbright(float fullbright) {}
    virtual void setcullface(bool cullface) {}

    virtual void preloadBIH() { if(!bih) setBIH(); }
    virtual void preloadshaders(bool force = false) {}
    virtual void preloadmeshes() {}
    virtual void cleanup() {}

    virtual void startrender() {}
    virtual void endrender() {}

    void boundbox(vec &center, vec &radius)
    {
        if(bbradius.x < 0)
        {
            calcbb(bbcenter, bbradius);
            bbradius.add(bbextend);
        }
        center = bbcenter;
        radius = bbradius;
    }

    float collisionbox(vec &center, vec &radius)
    {
        if(collideradius.x < 0)
        {
            boundbox(collidecenter, collideradius);
            if(collidexyradius)
            {
                collidecenter.x = collidecenter.y = 0;
                collideradius.x = collideradius.y = collidexyradius;
            }
            if(collideheight)
            {
                collidecenter.z = collideradius.z = collideheight/2;
            }
            rejectradius = vec(collidecenter).abs().add(collideradius).magnitude();
        }
        center = collidecenter;
        radius = collideradius;
        return rejectradius;
    }

    float boundsphere(vec &center)
    {
        vec radius;
        boundbox(center, radius);
        return radius.magnitude();
    }

    float above()
    {
        vec center, radius;
        boundbox(center, radius);
        return center.z+radius.z;
    }
};

