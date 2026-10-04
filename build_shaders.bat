@echo off
rem rebuilds every .spv in gl_vk_interop_v2\shaders. needs the vulkan sdk (VULKAN_SDK set by its installer)
rem only needed if you change a shader, the compiled ones are checked in

setlocal
set GLSLANG=%VULKAN_SDK%\Bin\glslangValidator.exe
if not exist "%GLSLANG%" (
    echo cant find glslangValidator - install the vulkan sdk or set VULKAN_SDK
    exit /b 1
)
set G="%GLSLANG%" -V --target-env vulkan1.2
cd /d "%~dp0gl_vk_interop_v2\shaders"
set FAIL=0

rem the path tracer is one file built a bunch of ways
call :c -S comp pathtrace_trace.comp pathtrace_trace.spv
call :c -S comp -DRADIANCE_CACHE_PASS pathtrace_trace.comp pathtrace_cache.spv
call :c -S comp -DSKY_CDF_PASS pathtrace_trace.comp pathtrace_skycdf.spv
call :c -S comp -DCAUSTIC_PHOTON_PASS pathtrace_trace.comp pathtrace_caustics.spv
call :c -S comp -DPT_PROFILE pathtrace_trace.comp pathtrace_trace_prof.spv
call :c -S rgen -DPT_RAYGEN pathtrace_trace.comp pathtrace_trace_rgen.spv
call :c -S rgen -DPT_RAYGEN -DPT_SER -DPT_SER_NV -DPT_SER_ONCE pathtrace_trace.comp pathtrace_trace_ser.spv

rem everything else is one .comp -> one .spv
for %%f in (pathtrace_resolve rtao skin volfog_depth volfog_inject volfog_integrate water_fft water_patch water_spectrum water_wavemap reflections restirDi restirDiDebugBruteforce trivial_invert trivial_invert_rg16f trivial_invert_rgba16f trivial_invert_rgba8) do call :c -S comp %%f.comp %%f.spv

if %FAIL%==0 (echo all shaders built) else (echo some shaders failed, see above)
exit /b %FAIL%

:c
rem last two args are input / output, anything before is passed through
set ARGS=
:collect
if "%~3"=="" goto run
set ARGS=%ARGS% %1
shift
goto collect
:run
%G% %ARGS% %1 -o %2 >nul || (echo FAILED: %1 -^> %2 & %G% %ARGS% %1 -o %2 & set FAIL=1)
exit /b 0
