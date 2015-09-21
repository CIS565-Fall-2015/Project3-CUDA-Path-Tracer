#pragma once

#include <vector>
#include "scene.h"
#define MAX_DEPTH 10

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
int cullRays(int numRays);