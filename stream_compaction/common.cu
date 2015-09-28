#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"



//void checkCUDAErrorFn(const char *msg, const char *file, int line) {
//    cudaError_t err = cudaGetLastError();
//    if (cudaSuccess == err) {
//        return;
//    }
//
//    fprintf(stderr, "CUDA error");
//    if (file) {
//        fprintf(stderr, " (%s:%d)", file, line);
//    }
//    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
//    exit(EXIT_FAILURE);
//}


namespace StreamCompaction {
namespace Common {

	__global__ void kernZeroArray(int n, int * data)
	{
		int k = threadIdx.x + blockDim.x * blockIdx.x;
		if(k < n)
		{
			data[k] = 0;
		}
	}



	__global__ void kernInclusive2Exclusive(int n, int * exclusive, const int * inclusive)
	{
		int k = threadIdx.x + blockDim.x * blockIdx.x;
		if( k < n)
		{
			if(k == 0)
			{
				exclusive[k] = IDENTITY;
			}
			else
			{
				exclusive[k] = inclusive[k-1];
			}
		}
	}



	/**
	* Maps an array to an array of 0s and 1s for stream compaction. Elements
	* which map to 0 will be removed, and elements which map to 1 will be kept.
	*/
	__global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
		int k = threadIdx.x + blockDim.x * blockIdx.x;
		if( k < n )
		{
			bools[k] = idata[k] != 0 ? 1 : 0;
		}
	}

	/**
	* Performs scatter on an array. That is, for each element in idata,
	* if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
	*/
	__global__ void kernScatter(int n, int *odata,
		const int *idata, const int *bools, const int *indices) {
			int k = threadIdx.x + blockDim.x * blockIdx.x;
			if( k < n )
			{
				if(bools[k] == 1)
				{
					odata[ indices[k] ] = idata[k];
				}
			}
	}

}
}
