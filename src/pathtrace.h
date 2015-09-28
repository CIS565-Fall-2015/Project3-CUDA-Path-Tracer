#pragma once

#include <vector>
#include "scene.h"
#include "stream_compaction\efficient.h"
#include "stream_compaction\common.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
int cullRaysThrust(int numRays);
int cullRaysEfficient(int numRays);
int cullRaysEfficientSharedMemory(int numRays);