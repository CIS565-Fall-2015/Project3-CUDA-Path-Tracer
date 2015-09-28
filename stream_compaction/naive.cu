#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

cudaEvent_t start, stop;

static void setup_timer_events() {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
}

static float teardown_timer_events() {
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return milliseconds;
}

// TODO: __global__

__global__ void naive_scan_step(int offset, int *x_1, int *x_2) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	if (i >= offset) {
		x_2[i] = x_1[i - offset] + x_1[i];
	}
	else {
		x_2[i] = x_1[i];
	}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // copy everything in idata over to the GPU
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int tpb = deviceProp.maxThreadsPerBlock;
	int blockWidth = fmin(n, tpb);
	int blocks = 1;
	if (blockWidth != n) {
		blocks = n / tpb;
		if (n % tpb) {
			blocks++;
		}
	}

	dim3 dimBlock(blockWidth);
	dim3 dimGrid(blocks);

	int *dev_x;
	int *dev_x_next;
	cudaMalloc((void**)&dev_x, sizeof(int) * n);
	cudaMalloc((void**)&dev_x_next, sizeof(int) * n);

	cudaMemcpy(dev_x, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x_next, dev_x, sizeof(int) * n, cudaMemcpyDeviceToDevice);

	if (BENCHMARK) {
		setup_timer_events();
	}

	// run steps.
	// no need to pad with 0s to get a power of 2 array here,
	// this can be an "unbalanced" binary tree of ops.
	int logn = ilog2ceil(n);
	for (int d = 1; d <= logn; d++) {
		int offset = powf(2, d - 1);
		naive_scan_step <<<dimGrid, dimBlock >>>(offset, dev_x, dev_x_next);
		int *temp = dev_x_next;
		dev_x_next = dev_x;
		dev_x = temp;
	}
	if (BENCHMARK) {
		printf("%f microseconds.\n",
			teardown_timer_events() * 1000.0f);
	}

	cudaMemcpy(odata + 1, dev_x, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
	odata[0] = 0;
	
	cudaFree(dev_x);
	cudaFree(dev_x_next);
}

}
}
