#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cmath>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
//#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define IDENTITY (0)




//const int blockSize = 192;
//const int blockSize = 128;
const int blockSize = 32;
//const int blockSize = 4;

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
//void checkCUDAErrorFn(const char *msg, const char *file = NULL, int line = -1);

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


namespace StreamCompaction {
namespace Common {
	__global__ void kernZeroArray(int n, int * data);

	__global__ void kernInclusive2Exclusive(int n, int * exclusive, const int * inclusive);

    __global__ void kernMapToBoolean(int n, int *bools, const int *idata);

    __global__ void kernScatter(int n, int *odata,
            const int *idata, const int *bools, const int *indices);
}
}
