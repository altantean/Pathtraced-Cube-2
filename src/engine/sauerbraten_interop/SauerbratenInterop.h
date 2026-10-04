#pragma once

#include <cstdint>

namespace sauerinterop {

	void applyCameraJitter();

	void onFrameEarly();

	void onFrameLate();

	bool beginSceneTarget(int &w, int &h, uint32_t &fbo);

	void presentScene();

	// call once per frame too, but from a new hook point in gl_drawframe()
	void compositeIntoScene();

	// call once per frame too, but from the OPPOSITE end of gl_drawframe()
	void drawDebugOverlay();

	// call once, from Sauerbraten's cleanup() (main.cpp), before SDL_Quit()
	void shutdown();


	bool volFogActive();

	// called by load_world() (worldio.cpp) once a map has fully loaded
	void onMapLoaded();

	void onMaterialsEdited();

	// true while the path tracer draws the water (pathtrace on, ptwater on, the map has water)
	bool ptWaterActive();
	// the path tracer draws the lava (the raster lava is skipped)
	bool ptLavaActive();
	// the path tracer has this frame's grass (the raster grass is skipped)
	bool ptGrassActive();

	void setFogMaterialState(int fogmat, float fogblend, int abovemat, float surfacez);

	void bindVolFogTexture();

	// writes every particle's motion into the DLSS motion vectors (call right after
	// renderparticles(true))
	void particleMotion();

}  // namespace sauerinterop
