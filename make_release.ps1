# builds the release exe + shaders and packs a zip that runs on a clean pc
# usage: powershell -ExecutionPolicy Bypass -File make_release.ps1 [-Version 1.0] [-SkipBuild]
param(
    [string]$Version = (Get-Date -Format "yyyy.MM.dd"),
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$name = "Sauerbraten-RT-$Version"
$out  = Join-Path $root "dist\$name"
$zip  = Join-Path $root "dist\$name.zip"

if (-not $SkipBuild) {
    # msbuild from whatever visual studio is installed
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $msbuild = & $vswhere -latest -requires Microsoft.Component.MSBuild -find "MSBuild\**\Bin\MSBuild.exe" | Select-Object -First 1
    if (-not $msbuild) { throw "cant find msbuild - install visual studio with the c++ workload" }
    & $msbuild "$root\src\vcpp\sauerbraten.sln" -p:Configuration=Release -p:Platform=x64 -m -v:minimal -nologo
    if ($LASTEXITCODE -ne 0) { throw "release build failed" }
    & cmd /c "`"$root\build_shaders.bat`""
    if ($LASTEXITCODE -ne 0) { throw "shader build failed" }
}

if (Test-Path $out) { Remove-Item $out -Recurse -Force }
New-Item -ItemType Directory $out | Out-Null
function Copy-To($src, $dst) {
    $d = Join-Path $out $dst
    New-Item -ItemType Directory -Force (Split-Path $d) | Out-Null
    Copy-Item (Join-Path $root $src) $d -Recurse -Force
}

# game content, same layout as a normal sauerbraten install
Copy-To "data" "data"
Copy-To "packages" "packages"
Copy-To "docs" "docs"
Copy-To "README.html" "README.html"
Copy-To "README.md" "README.md"
Copy-To "LICENSE" "LICENSE"
Copy-To "server-init.cfg" "server-init.cfg"

# exe + runtime dlls (only what the game actually loads)
$bin = @("sauerbraten.exe", "SDL2.dll", "SDL2_image.dll", "SDL2_mixer.dll", "zlib1.dll", "smpeg2.dll",
         "libFLAC-8.dll", "libjpeg-9.dll", "libmikmod-2.dll", "libmodplug-1.dll", "libmpg123-0.dll",
         "libogg-0.dll", "libopus-0.dll", "libopusfile-0.dll", "libpng16-16.dll", "libtiff-5.dll",
         "libvorbis-0.dll", "libvorbisfile-3.dll", "libwebp-4.dll", "libwebp-7.dll", "libwinpthread-1.dll",
         "README-SDL.txt")
foreach ($f in $bin) { Copy-To "bin64\$f" "bin64\$f" }
Get-ChildItem "$root\bin64\LICENSE.*.txt" | ForEach-Object { Copy-To "bin64\$($_.Name)" "bin64\$($_.Name)" }

# dlss / streamline - straight from the checked in sdk copy (signed release dlls, not the dev ones)
$sl = "src\gl-vk-interop-stage2\third_party\streamline"
foreach ($f in @("sl.interposer.dll", "sl.common.dll", "sl.dlss.dll", "sl.dlss_d.dll", "nvngx_dlss.dll", "nvngx_dlssd.dll")) {
    Copy-To "$sl\bin\x64\$f" "bin64\streamline\$f"
}

# compiled shaders only, sources stay in the repo
New-Item -ItemType Directory -Force "$out\gl_vk_interop_v2\shaders" | Out-Null
Copy-Item "$root\gl_vk_interop_v2\shaders\*.spv" "$out\gl_vk_interop_v2\shaders\"

# licenses for the stuff that isnt ours
Copy-To "$sl\license.txt" "licenses\streamline-license.txt"
Copy-To "$sl\3rd-party-licenses.md" "licenses\streamline-3rd-party-licenses.md"
Copy-To "$sl\bin\x64\nvngx_dlss.license.txt" "licenses\nvngx_dlss.license.txt"
Copy-To "src\readme_source.txt" "licenses\sauerbraten-source-license.txt"

# launchers
@"
@echo off
rem sauerbraten with the path tracer. settings/saves go to "My Games\Sauerbraten" like normal sauer
cd /d "%~dp0"
start "" "%~dp0bin64\sauerbraten.exe" "-q`$HOME\My Games\Sauerbraten" -glog.txt %*
"@ | Set-Content "$out\sauerbraten.bat" -Encoding ASCII
Copy-To "server.bat" "server.bat"

if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $out -DestinationPath $zip -CompressionLevel Optimal
"{0}  ({1:N0} MB)" -f $zip, ((Get-Item $zip).Length / 1MB)
