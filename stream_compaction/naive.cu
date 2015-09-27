#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

#define blockSize 128
int *scan_result;
int *temp_scan;
int *shifted_result;

// TODO: __global__

__global__ void prefixSum(int n, int d, int *o_data, int *i_data) {
	int index =  (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (index < n) {
		if (index >= (int)pow(2.0, d-1)) {
			o_data[index] = i_data[index - (int)pow(2.0, d-1)] + i_data[index];
		} 
	}
}



/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {

	int d = ilog2ceil(n);

	cudaMalloc((void**)&scan_result, n * sizeof(int));
	cudaMalloc((void**)&temp_scan, n * sizeof(int));
	cudaMalloc((void**)&shifted_result, n * sizeof(int));
	
	cudaMemcpy(temp_scan, idata, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(scan_result, idata, n * sizeof(int), cudaMemcpyHostToDevice);

	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	for (int i = 1; i <= d; i++) {
		prefixSum<<<fullBlocksPerGrid, blockSize>>>(n, i, scan_result, temp_scan);
		temp_scan = scan_result;
	}
	cudaEventRecord(stop);

	cudaMemcpy(odata, scan_result, n * sizeof(int), cudaMemcpyDeviceToHost);

	//shift right
	for(int i = n-1; i >= 0; i--) {
		odata[i] = odata[i-1];
	}
	odata[0] = 0;

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f milliseconds for naive \n", milliseconds);
	
	cleanUp();
   
}

void cleanUp() {
	cudaFree(scan_result);
	cudaFree(temp_scan);
}

}
}
