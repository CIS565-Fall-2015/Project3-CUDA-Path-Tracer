#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene, bool strCmpt = false);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
