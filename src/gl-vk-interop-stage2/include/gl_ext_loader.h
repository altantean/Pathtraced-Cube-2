#pragma once

#include <cstdint>
#include <string>

// minimal GL type aliases (avoids depending on gl.h/glcorearb.h being present)
using GLenum = unsigned int;
using GLuint = unsigned int;
using GLint = int;
using GLsizei = int;
using GLboolean = unsigned char;
using GLuint64 = uint64_t;
using GLchar = char;
using GLfloat = float;

constexpr GLenum GL_TEXTURE_2D_ = 0x0DE1;
constexpr GLenum GL_TEXTURE_3D_ = 0x806F;
constexpr GLenum GL_RGBA8_ = 0x8058;
constexpr GLenum GL_RGBA16F_ = 0x881A;  // GL 3.0 core / ARB_texture_float, HDR scene color format
constexpr GLenum GL_RGBA32F_ = 0x8814;
constexpr GLenum GL_RG16F_ = 0x822F;
// GL 3.0 core / ARB_texture_float, the froxel sun-visibility grid's single-channel HDR format
constexpr GLenum GL_R16F_ = 0x822D;
constexpr GLenum GL_RGB16F_ = 0x881B;
constexpr GLenum GL_R8_ = 0x8229;
constexpr GLenum GL_RED_ = 0x1903;  // GL 3.0 core / ARB_texture_float, motion-vector-style format
constexpr GLenum GL_UNSIGNED_BYTE_ = 0x1401;
constexpr GLenum GL_TEXTURE_MIN_FILTER_ = 0x2801;
constexpr GLenum GL_TEXTURE_MAG_FILTER_ = 0x2800;
constexpr GLenum GL_TEXTURE_WRAP_S_ = 0x2802;
constexpr GLenum GL_TEXTURE_WRAP_T_ = 0x2803;
constexpr GLenum GL_TEXTURE_WRAP_R_ = 0x8072;
constexpr GLenum GL_LINEAR_ = 0x2601;
constexpr GLenum GL_CLAMP_TO_EDGE_ = 0x812F;
constexpr GLenum GL_FLOAT_ = 0x1406;
constexpr GLenum GL_RGBA_ = 0x1908;
constexpr GLenum GL_NO_ERROR_ = 0;
constexpr GLenum GL_INVALID_ENUM_ = 0x0500;
constexpr GLenum GL_INVALID_VALUE_ = 0x0501;
constexpr GLenum GL_INVALID_OPERATION_ = 0x0502;
constexpr GLenum GL_STACK_OVERFLOW_ = 0x0503;
constexpr GLenum GL_STACK_UNDERFLOW_ = 0x0504;
constexpr GLenum GL_OUT_OF_MEMORY_ = 0x0505;
constexpr GLenum GL_INVALID_FRAMEBUFFER_OPERATION_ = 0x0506;

constexpr GLenum GL_HANDLE_TYPE_OPAQUE_WIN32_EXT_ = 0x9587;
constexpr GLenum GL_HANDLE_TYPE_OPAQUE_WIN32_KMT_EXT_ = 0x9588;
constexpr GLenum GL_DEVICE_LUID_EXT_ = 0x9599;
constexpr GLenum GL_DEVICE_NODE_MASK_EXT_ = 0x959A;
// not a token to pass to glGetXXX
constexpr int GL_LUID_SIZE_EXT_ = 8;

constexpr GLenum GL_LAYOUT_GENERAL_EXT_ = 0x958D;
constexpr GLenum GL_DEPTH_COMPONENT32F_ = 0x8CAC;

using PFNGLCREATETEXTURESPROC_ = void (*)(GLenum target, GLsizei n, GLuint* textures);
using PFNGLDELETETEXTURESPROC_ = void (*)(GLsizei n, const GLuint* textures);
using PFNGLCLEARTEXSUBIMAGEPROC_ = void (*)(GLuint texture, GLint level,
    GLint xoffset, GLint yoffset, GLint zoffset,
    GLsizei width, GLsizei height, GLsizei depth,
    GLenum format, GLenum type, const void* data);
using PFNGLGETTEXTURESUBIMAGEPROC_ = void (*)(GLuint texture, GLint level,
    GLint xoffset, GLint yoffset, GLint zoffset,
    GLsizei width, GLsizei height, GLsizei depth,
    GLenum format, GLenum type, GLsizei bufSize, void* pixels);
using PFNGLGETUNSIGNEDBYTEVEXTPROC_ = void (*)(GLenum pname, unsigned char* data);
using PFNGLGETERRORPROC_ = GLenum (*)();
using PFNGLFLUSHPROC_ = void (*)();
// core GL 4.5 DSA texture-parameter setter
using PFNGLTEXTUREPARAMETERIPROC_ = void (*)(GLuint texture, GLenum pname, GLint param);

using PFNGLCREATEMEMORYOBJECTSEXTPROC_ = void (*)(GLsizei n, GLuint* memoryObjects);
using PFNGLDELETEMEMORYOBJECTSEXTPROC_ = void (*)(GLsizei n, const GLuint* memoryObjects);
using PFNGLTEXTURESTORAGEMEM2DEXTPROC_ = void (*)(GLuint texture, GLsizei levels,
    GLenum internalFormat, GLsizei width, GLsizei height, GLuint memory, GLuint64 offset);
using PFNGLTEXTURESTORAGEMEM3DEXTPROC_ = void (*)(GLuint texture, GLsizei levels,
    GLenum internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLuint memory, GLuint64 offset);
using PFNGLIMPORTMEMORYWIN32HANDLEEXTPROC_ = void (*)(GLuint memory, GLuint64 size,
    GLenum handleType, void* handle);

using PFNGLGENSEMAPHORESEXTPROC_ = void (*)(GLsizei n, GLuint* semaphores);
using PFNGLDELETESEMAPHORESEXTPROC_ = void (*)(GLsizei n, const GLuint* semaphores);
using PFNGLSIGNALSEMAPHOREEXTPROC_ = void (*)(GLuint semaphore,
    GLuint numBufferBarriers, const GLuint* buffers,
    GLuint numTextureBarriers, const GLuint* textures, const GLenum* dstLayouts);
using PFNGLWAITSEMAPHOREEXTPROC_ = void (*)(GLuint semaphore,
    GLuint numBufferBarriers, const GLuint* buffers,
    GLuint numTextureBarriers, const GLuint* textures, const GLenum* srcLayouts);
using PFNGLIMPORTSEMAPHOREWIN32HANDLEEXTPROC_ = void (*)(GLuint semaphore,
    GLenum handleType, void* handle);

namespace interop {

// all function pointers this PoC resolves, grouped in one place
struct GlExtFunctions {
    PFNGLCREATETEXTURESPROC_ glCreateTextures = nullptr;
    PFNGLDELETETEXTURESPROC_ glDeleteTextures = nullptr;
    PFNGLCLEARTEXSUBIMAGEPROC_ glClearTexSubImage = nullptr;
    PFNGLGETTEXTURESUBIMAGEPROC_ glGetTextureSubImage = nullptr;
    PFNGLGETUNSIGNEDBYTEVEXTPROC_ glGetUnsignedBytevEXT = nullptr;
    PFNGLGETERRORPROC_ glGetError = nullptr;
    PFNGLFLUSHPROC_ glFlush = nullptr;
    PFNGLTEXTUREPARAMETERIPROC_ glTextureParameteri = nullptr;

    PFNGLCREATEMEMORYOBJECTSEXTPROC_ glCreateMemoryObjectsEXT = nullptr;
    PFNGLDELETEMEMORYOBJECTSEXTPROC_ glDeleteMemoryObjectsEXT = nullptr;
    PFNGLTEXTURESTORAGEMEM2DEXTPROC_ glTextureStorageMem2DEXT = nullptr;
    PFNGLTEXTURESTORAGEMEM3DEXTPROC_ glTextureStorageMem3DEXT = nullptr;
    PFNGLIMPORTMEMORYWIN32HANDLEEXTPROC_ glImportMemoryWin32HandleEXT = nullptr;

    PFNGLGENSEMAPHORESEXTPROC_ glGenSemaphoresEXT = nullptr;
    PFNGLDELETESEMAPHORESEXTPROC_ glDeleteSemaphoresEXT = nullptr;
    PFNGLSIGNALSEMAPHOREEXTPROC_ glSignalSemaphoreEXT = nullptr;
    PFNGLWAITSEMAPHOREEXTPROC_ glWaitSemaphoreEXT = nullptr;
    PFNGLIMPORTSEMAPHOREWIN32HANDLEEXTPROC_ glImportSemaphoreWin32HandleEXT = nullptr;
};

bool loadGlExtFunctions(GlExtFunctions& out, std::string* error);

bool queryGlDeviceLuid(const GlExtFunctions& fns, uint8_t luidOut[8], std::string* error);

bool checkGlError(const GlExtFunctions& fns, const char* context);
int drainPendingGlErrors(const GlExtFunctions& fns, const char* beforeWhat);

}  // namespace interop
