#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

inline int ilog2(int x) {
	int lg = 0;
	while (x >>= 1) {
		++lg;
	}
	return lg;
}

inline int ilog2ceil(int x) {
	return ilog2(x - 1) + 1;
}

// Kernels
__global__ void kernSharedScan(int n, int *odata, const int *idata);
template <typename T> __global__ void kernFindAlive(int n, int* odata, T* idata);
template <typename T> __global__ void kernScatter(int n, T* odata, T* idata, int* bools, int* scan);
template <typename T, typename Predicate> __global__ void kernMapToBool(int n, int* odata, T* idata, Predicate pred);
__global__ void kernInPlaceIncrementBlocks(int n, int* dev_data, int* increment);
__global__ void shared_scan(int n, int *odata, const int *idata);


// Normal functions
void blockwise_scan(int n, int* dev_odata, int* dev_idata);
template <typename T> int shared_compact(int n, T* dev_odata, T* dev_idata);