# Building Sauerbraten RT

Only needed if you want to change the code. To just play, grab the zip from the Releases page.

## You need

- Visual Studio 2026 with the C++ desktop workload (toolset v145). Older versions work if you retarget the
  project.
- the Vulkan SDK, with `VULKAN_SDK` set (the installer does that)
- the game content: copy the `packages` folder from the release zip into the repo root

## Build

```
build_shaders.bat
powershell -ExecutionPolicy Bypass -File make_release.ps1
```

`make_release.ps1` builds the Release exe and the shaders, and puts a runnable copy plus a zip in `dist\`.

For working on the code, open `src\vcpp\sauerbraten.sln`. A build puts the exe in `bin64\` and copies the
Streamline / DLSS runtime to `bin64\streamline\`, so `sauerbraten.bat` runs it straight from the repo.

## Where things are

- `src\engine\sauerbraten_interop` - the game side of the path tracer
- `src\gl-vk-interop-stage2` - the Vulkan side, and the Streamline SDK in `third_party`
- `gl_vk_interop_v2\shaders` - the path tracing shaders (`build_shaders.bat` compiles them)

## Licenses

- Sauerbraten engine source: zlib, see `src\readme_source.txt`
- game content: see `packages\readme.txt` and the readmes inside each package
- NVIDIA Streamline and DLSS: see `src\gl-vk-interop-stage2\third_party\streamline`
- SDL2 and the other libraries in `bin64`: see their `LICENSE.*.txt` files
- the path tracing additions: MIT, see `LICENSE`
