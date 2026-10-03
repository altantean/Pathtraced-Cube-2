#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "gl_ext_loader.h"
#include <windows.h>
#include <GL/gl.h>
#include <iostream>
#include <chrono>

namespace {

// wglGetProcAddress is only contractually reliable for functions beyond OpenGL 1.1
void* resolveGlFunction(const char* name) {
    void* p = reinterpret_cast<void*>(wglGetProcAddress(name));
    if (p == nullptr || p == reinterpret_cast<void*>(0x1) || p == reinterpret_cast<void*>(0x2) ||
        p == reinterpret_cast<void*>(0x3) || p == reinterpret_cast<void*>(-1)) {
        static HMODULE opengl32 = LoadLibraryA("opengl32.dll");
        if (opengl32 != nullptr) {
            p = reinterpret_cast<void*>(GetProcAddress(opengl32, name));
        }
    }
    return p;
}

template <typename FnPtr>
bool resolveOrFail(FnPtr& outFn, const char* name, std::string* error) {
    outFn = reinterpret_cast<FnPtr>(resolveGlFunction(name));
    if (outFn == nullptr) {
        if (error != nullptr) {
            *error = std::string("Failed to resolve required GL function: ") + name +
                     " (is a GL context current on this thread, and does the driver "
                     "actually expose the extension this function belongs to?)";
        }
        return false;
    }
    return true;
}

}  // namespace

namespace interop {

bool loadGlExtFunctions(GlExtFunctions& out, std::string* error) {
    if (wglGetCurrentContext() == nullptr) {
        if (error != nullptr) {
            *error = "loadGlExtFunctions: no OpenGL context is current on this thread. "
                     "This native call must happen on the same thread that holds the "
                     "LWJGL GL context current (usually the render thread), after "
                     "GL.createCapabilities() has already run on the Java side.";
        }
        return false;
    }

    bool ok = true;
    ok &= resolveOrFail(out.glCreateTextures, "glCreateTextures", error);
    if (!ok) return false;
    ok &= resolveOrFail(out.glDeleteTextures, "glDeleteTextures", error);
    if (!ok) return false;
    ok &= resolveOrFail(out.glClearTexSubImage, "glClearTexSubImage", error);
    if (!ok) return false;
    ok &= resolveOrFail(out.glGetTextureSubImage, "glGetTextureSubImage", error);
    if (!ok) return false;
    ok &= resolveOrFail(out.glGetUnsignedBytevEXT, "glGetUnsignedBytevEXT", error);
    if (!ok) return false;
    ok &= resolveOrFail(out.glGetError, "glGetError", error);
    ok &= resolveOrFail(out.glFlush, "glFlush", error);
    ok &= resolveOrFail(out.glTextureParameteri, "glTextureParameteri", error);
    if (!ok) return false;

    ok &= resolveOrFail(out.glCreateMemoryObjectsEXT, "glCreateMemoryObjectsEXT", error);
    if (!ok) return false;
    ok &= resolveOrFail(out.glDeleteMemoryObjectsEXT, "glDeleteMemoryObjectsEXT", error);
    if (!ok) return false;
    ok &= resolveOrFail(out.glTextureStorageMem2DEXT, "glTextureStorageMem2DEXT", error);
    if (!ok) return false;
    ok &= resolveOrFail(out.glImportMemoryWin32HandleEXT, "glImportMemoryWin32HandleEXT", error);
    if (!ok) return false;
    void* volFogFn = resolveGlFunction("glTextureStorageMem3DEXT");
    out.glTextureStorageMem3DEXT = reinterpret_cast<PFNGLTEXTURESTORAGEMEM3DEXTPROC_>(volFogFn);

    ok &= resolveOrFail(out.glGenSemaphoresEXT, "glGenSemaphoresEXT", error);
    if (!ok) return false;
    ok &= resolveOrFail(out.glDeleteSemaphoresEXT, "glDeleteSemaphoresEXT", error);
    if (!ok) return false;
    ok &= resolveOrFail(out.glSignalSemaphoreEXT, "glSignalSemaphoreEXT", error);
    if (!ok) return false;
    ok &= resolveOrFail(out.glWaitSemaphoreEXT, "glWaitSemaphoreEXT", error);
    if (!ok) return false;
    ok &= resolveOrFail(out.glImportSemaphoreWin32HandleEXT, "glImportSemaphoreWin32HandleEXT", error);
    if (!ok) return false;

    return true;
}

bool queryGlDeviceLuid(const GlExtFunctions& fns, uint8_t luidOut[8], std::string* error) {
    if (fns.glGetUnsignedBytevEXT == nullptr) {
        if (error != nullptr) {
            *error = "queryGlDeviceLuid: glGetUnsignedBytevEXT was not resolved -- "
                     "GL_EXT_memory_object_win32 is likely not exposed by this driver/context.";
        }
        return false;
    }
    // GL_DEVICE_LUID_EXT_ (0x9599)
    fns.glGetUnsignedBytevEXT(GL_DEVICE_LUID_EXT_, luidOut);
    return true;
}

namespace {
const char* glErrorName(GLenum err) {
    switch (err) {
        case GL_INVALID_ENUM_: return "GL_INVALID_ENUM";
        case GL_INVALID_VALUE_: return "GL_INVALID_VALUE";
        case GL_INVALID_OPERATION_: return "GL_INVALID_OPERATION";
        case GL_STACK_OVERFLOW_: return "GL_STACK_OVERFLOW";
        case GL_STACK_UNDERFLOW_: return "GL_STACK_UNDERFLOW";
        case GL_OUT_OF_MEMORY_: return "GL_OUT_OF_MEMORY";
        case GL_INVALID_FRAMEBUFFER_OPERATION_: return "GL_INVALID_FRAMEBUFFER_OPERATION";
        default: return "unknown GL error";
    }
}
}  // namespace

int drainPendingGlErrors(const GlExtFunctions& fns, const char* beforeWhat) {
    if (fns.glGetError == nullptr) return 0;
    int count = 0;
    GLenum first = 0, err;
    while ((err = fns.glGetError()) != GL_NO_ERROR_ && count < 64) {
        if (count++ == 0) first = err;
    }
    static auto lastLog = std::chrono::steady_clock::time_point{};
    const auto now = std::chrono::steady_clock::now();
    if (count > 0 && now - lastLog > std::chrono::seconds(1)) {
        lastLog = now;
        std::cerr << "[interop][GL] " << count << " GL error(s) already pending before " << beforeWhat
                  << " (from earlier rendering, NOT the interop), first " << glErrorName(first)
                  << " (0x" << std::hex << first << std::dec << ")" << std::endl;
    }
    return count;
}

bool checkGlError(const GlExtFunctions& fns, const char* context) {
    if (fns.glGetError == nullptr) {
        std::cerr << "[interop][GL] (" << context << ") glGetError itself was not resolved -- "
                     "cannot check for errors here." << std::endl;
        return false;
    }
    bool any = false;
    GLenum err;
    while ((err = fns.glGetError()) != GL_NO_ERROR_) {
        std::cerr << "[interop][GL] (" << context << ") " << glErrorName(err)
                  << " (0x" << std::hex << err << std::dec << ")" << std::endl;
        any = true;
    }
    return any;
}

}  // namespace interop
