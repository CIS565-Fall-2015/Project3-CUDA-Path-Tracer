#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

int* g_odata;
int* g_idata;
int* dev_bools;
int* dev_indices;

__global__ void generate_zeros(int *data) {
	int i = threadIdx.x;
	data[i] = 0;
}

__global__ void set_zero(int size, int n, int *data) {
	int i = threadIdx.x;
	if (i >= n - 1) {
		data[i] = 0;
	}
}
__global__ void kern_up_sweep(int n, int *odata, const int *idata, int layer) {
	int thrId = threadIdx.x + (blockIdx.x * blockDim.x);
	if ((thrId < n) && (thrId%layer == 0)) {
		odata[thrId + layer - 1] += idata[thrId + (layer / 2) - 1];
	}
}

__global__ void kern_down_sweep(int n, int *odata, const int *idata, int layer) {
	int thrId = threadIdx.x + (blockIdx.x * blockDim.x);
	if ((thrId < n) && (thrId%layer == 0)) {
		int temp = idata[thrId + (layer / 2) - 1];
		odata[thrId + (layer / 2) - 1] = idata[thrId + layer - 1];
		odata[thrId + layer - 1] += temp;
	}
	
	
}
/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int blockSize = 128;
	int numBlocks = ceil((float)n / (float)blockSize);
	int powTwo = pow(2, ilog2ceil(n));
	dim3 fullBlocksPerGrid((powTwo + blockSize - 1) / blockSize);
	cudaMalloc((void**)&g_odata, powTwo * sizeof(int));
	cudaMalloc((void**)&g_idata, powTwo  * sizeof(int));

	generate_zeros<<<1, powTwo>>>(g_odata);
	generate_zeros<<<1, powTwo>>>(g_idata);

	int* scanArray = new int[n];
	//scanArray[0] = 0;
	for (int i = 0; i < n; i++) {
		scanArray[i] = idata[i];
	}

	cudaMemcpy(g_odata, odata, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(g_idata, scanArray, n*sizeof(int), cudaMemcpyHostToDevice);
	
	cudaEventRecord(start);
    for (int d = 0; d <= ilog2ceil(n) - 1; d++) {
		int layer = pow(2, d + 1);
		g_odata = g_idata;
		
		kern_up_sweep<<<fullBlocksPerGrid, blockSize>>>(powTwo, g_odata, g_idata, layer);
		g_idata = g_odata;
	}


	set_zero<<<1, powTwo>>>(powTwo, n, g_idata);

	for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
		int layer = pow(2, d + 1);
		g_odata = g_idata;
		kern_down_sweep<<<fullBlocksPerGrid, blockSize>>>(powTwo, g_odata, g_idata, layer);
		g_idata = g_odata;
	}
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("%f - ", milliseconds);
	cudaMemcpy(odata, g_odata, n*sizeof(int), cudaMemcpyDeviceToHost);
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
int compact(int n, int *odata, const int *idata) {	
	int blockSize = 128;
	int numBlocks = ceil((float)n / (float)blockSize);
	int powTwo = pow(2, ilog2ceil(n));
	dim3 fullBlocksPerGrid((powTwo + blockSize - 1) / blockSize);

    cudaMalloc((void**)&g_odata, powTwo * sizeof(int));
	cudaMalloc((void**)&g_idata, powTwo  * sizeof(int));
	cudaMalloc((void**)&dev_bools, powTwo * sizeof(int));
	cudaMalloc((void**)&dev_indices, powTwo * sizeof(int));

	int* indices = new int[n];
	int* bools = new int[n];

	cudaMemcpy(g_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);

	Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(powTwo, dev_bools, g_idata);
	cudaMemcpy(bools, dev_bools, n*sizeof(int), cudaMemcpyDeviceToHost);

	scan(n, indices, bools);

	cudaMemcpy(dev_indices, indices, n*sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(g_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);
	Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(powTwo, g_odata, g_idata, dev_bools, dev_indices);

	cudaMemcpy(odata, g_odata, n*sizeof(int), cudaMemcpyDeviceToHost);
	
    return indices[n-1] + bools[n-1];
}

}
}
