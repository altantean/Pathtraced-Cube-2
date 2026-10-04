# Sauerbraten RT

Cube 2: Sauerbraten with real-time path tracing and DLSS Ray Reconstruction. Same game, same maps, just with
light that actually bounces.

**Beta:** expect rough edges, and please report anything that breaks.

## Features

- path traced lighting from the sun, sky, lights and glowing surfaces, with bounce light
- DLSS Ray Reconstruction for a clean, sharp image
- ocean water with real waves, refraction and caustics
- path traced lava, glass and reflective models
- volumetric fog
- works on all the stock maps

## Download and play

1. Get the latest zip from the [Releases](../../releases) page.
2. Extract it anywhere.
3. Run `sauerbraten.bat`.

The first launch can hang for a minute while your driver compiles the shaders. That only happens once.

## Requirements

- Windows 10 or 11, 64-bit
- NVIDIA RTX graphics card (20 series or newer)
- a recent NVIDIA driver

## Handy settings

Open the console with `T`, then type a command starting with `/`, e.g. `/dlssquality 2`.

| command | what it does |
| --- | --- |
| `dlssquality 0-4` | 0 DLAA, 1 Quality (default), 2 Balanced, 3 Performance, 4 Ultra Performance |
| `pathtrace 0/1` | path tracing off/on |
| `pathtracebounces 0-8` | light bounces, default 2 |
| `volfog 0/1` | volumetric fog |
| `ptcaustics 0/1` | water caustics |
| `ptwind 0.5-30` | wind speed for the waves |

Your settings and maps are saved in `Documents\My Games\Sauerbraten`, same as normal Sauerbraten.

## Building from source

See [BUILDING.md](BUILDING.md).

## Credits

Cube 2: Sauerbraten by Wouter van Oortmerssen, Lee Salzman, Mike Dysart, Robert Pointon, Quinton Reeves and
everyone who made content for it. DLSS and Streamline by NVIDIA. License details are in [LICENSE](LICENSE).
