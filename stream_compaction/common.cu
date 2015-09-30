#include "common.h"

namespace StreamCompaction {
namespace Common {

/**
 * Maps an array to an array of 0s and 1s for stream compaction. Elements
 * which map to 0 will be removed, and elements which map to 1 will be kept.
 */
__global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
    int index =  (blockIdx.x * blockDim.x) + threadIdx.x;	
	if (index < n) {
		if (idata[index] != 0) {
			bools[index] = 1;
		} else {
			bools[index] = 0;
		}
	}
}

/**
 * Maps an array to an array of 0s and 1s for stream compaction. Elements
 * which map to 0 will be removed, and elements which map to 1 will be kept.
 */
__global__ void kernMapRayToBoolean(int n, int *bools, const Ray *idata) {
    int index =  (blockIdx.x * blockDim.x) + threadIdx.x;   
    if (index < n) {
		bools[index] = (int) idata[index].isAlive;
    }
}

/**
 * Performs scatter on an array. That is, for each element in idata,
 * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
 */
__global__ void kernScatter(int n, Ray *odata,
        const Ray *idata, const int *bools, const int *indices) {
    int index =  (blockIdx.x * blockDim.x) + threadIdx.x;	
	if (index < n) {
		if (bools[index] == 1) {
			odata[indices[index]] = idata[index];
		}
	}
}

}
}
