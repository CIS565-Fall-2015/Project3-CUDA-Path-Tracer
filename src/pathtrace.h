#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);


__global__ void generateRays(RayInfo* out_d_rays,
	glm::vec3 cam_pos, glm::vec3 cam_target,
	glm::vec3 V, glm::vec3 H, glm::vec2 resolution);

__global__ void iterate(RayInfo* d_rayInfo, glm::vec3* d_image, int activeRayNo, int depth, int iter,
	Geom *d_geom, int numGeom, Material *d_mat, Texture *d_texture);

__global__ void getOneZeroBit(RayInfo* d_rayInfo, int* d_sum, int noRays);

//n MUST be a power of 2 -- or else it's impossible to define block size at run time
__global__ void prescan(int* d_sum, int n, int originalSize);
__global__ void getBlockIncrements(int* d_increment, int *d_sum, int n, int sweepBlockSize);
__global__ void offsetWithIncrement(int *d_sum, int n, int* d_increment, int sweepBlockSize);
__global__ void scatter(RayInfo* out_d_rayInfo, RayInfo* d_rayInfo, int* d_sum, int rayNo);

//__global__ void scatterIntTest(char* out, char* in, int* IO, int* d_sum, int rayNo);

//n MUST be a power of 2 
void calculateSum(int *d_sum, int n, int sweepBlockSize, int noSweepBlock);