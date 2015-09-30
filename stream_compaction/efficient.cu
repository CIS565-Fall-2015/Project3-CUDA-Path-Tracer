#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 256
#define blockSizeHalf 128

namespace StreamCompaction {
namespace Efficient {

//code from http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
__global__ void prescan(int n, int *odata, const int *idata) {
	extern __shared__ int temp[];
	int tid = threadIdx.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = 1;

	temp[2 * tid] = idata[2 * index ];
	temp[2 * tid + 1] = idata[2 * index + 1 ];

	for (int d = n >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if (tid < d) {
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (tid == 0) {
		temp[n - 1] = 0;
	}

	for (int d = 1; d < n; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (tid < d) {
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	odata[2 * index ] = temp[2 * tid];
	odata[2 * index + 1 ] = temp[2 * tid + 1];
}

__global__ void sumEachBlock(int n, int *datasum, int *idata, int *odata) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if( index < n ) {
		datasum[index] = idata[(index + 1) * blockSize - 1] + odata[(index + 1) * blockSize - 1];
	}
}

__global__ void addIncrements(int n, int *data, int *increments) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if( index < n ) {
		data[index] = data[index] + increments[blockIdx.x];
	}
}

// scan on multiple blocks, algorithm from CIS565 lecture slides
void scan(int n, int *odata, int *idata) {

	int blocksPerGrid = (n + blockSize - 1) / blockSize;
	int n_new = blocksPerGrid * blockSize;
	int *dev_idata;
	int *dev_odata = odata;

	cudaMalloc((void**)&dev_idata, n_new * sizeof(int));
	cudaMemset(dev_idata, 0, n_new * sizeof(int));
	cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);

	//prescan<<<blocksPerGrid, blockSize>>>(blockSize, dev_odata, dev_idata);
	prescan<<<blocksPerGrid, blockSizeHalf, blockSize * sizeof(int)>>>(blockSize, dev_odata, dev_idata);

	if( blocksPerGrid > 1) {
		int *dev_sum, *dev_sum_scan;
		cudaMalloc((void**)&dev_sum, blocksPerGrid * sizeof(int));
		cudaMalloc((void**)&dev_sum_scan, blocksPerGrid * sizeof(int));

		int blocksPerGrid_new = (blocksPerGrid + blockSize - 1) / blockSize;
		sumEachBlock<<<blocksPerGrid_new, blockSize>>>(blocksPerGrid, dev_sum, dev_odata, dev_idata);
		scan(blocksPerGrid, dev_sum_scan, dev_sum);
		addIncrements<<<blocksPerGrid, blockSize>>>(n_new, dev_odata, dev_sum_scan);

		cudaFree(dev_sum);
		cudaFree(dev_sum_scan);
	}
	cudaFree(dev_idata);
}


/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, Ray *odata, Ray *idata) {

	Ray *dev_idata = idata;
	Ray *dev_odata = odata;
	int *dev_bools;
	int *dev_indices;

	int hst_bools[n];
	int hst_indices[n];

	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

    //cudaMalloc((void**)&dev_idata, n * sizeof(Ray));
    //cudaMalloc((void**)&dev_odata, n * sizeof(Ray));
    cudaMalloc((void**)&dev_bools, n * sizeof(int));
    cudaMalloc((void**)&dev_indices, n * sizeof(int));
    cudaMemset(dev_indices, 0, n * sizeof(int));

    //cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
    Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(n, dev_bools, dev_idata);
    cudaMemcpy(hst_bools, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);

	scan(n, dev_indices, dev_bools);
	//scan(n, hst_indices, hst_bools);
	//printf("n is %d \n", n);
	cudaMemcpy(hst_indices, dev_indices, n * sizeof(int), cudaMemcpyDeviceToHost);

	Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);

	//cudaFree(dev_idata);
	//cudaFree(dev_odata);
	cudaFree(dev_bools);
	cudaFree(dev_indices);

	if(hst_bools[n-1] == 0) {
		return hst_indices[n-1];
	} else {
		return hst_indices[n-1] + 1;
	}
	//return n;
}

}
}
