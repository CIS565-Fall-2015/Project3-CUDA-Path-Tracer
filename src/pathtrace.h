#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
void initRay(const Camera &cam);
void camSetup(int imageCount);
void initMaterial();
void initGeom();