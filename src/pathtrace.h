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
void scan(int n, int *odata, const int *idata,int blockSize);
int compact(int n, glm::vec3 *origin, glm::vec3 *direction, int *pos,glm::vec3 *pixes,
			glm::vec3 *image,bool *inside,int *idata,int blockSize);