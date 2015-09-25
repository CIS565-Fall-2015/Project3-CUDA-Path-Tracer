#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

// TODO: __global__
__global__ void kernScan(int n, int powd, int* odata, int* idata){
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	//int powd = (int)pow(2,d);

	if (i < n){
		if (i >= powd){
			odata[i] = idata[i - powd] + idata[i];
		} else {
			odata[i] = idata[i];
		}
	}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	// Initialize
	int td = ilog2ceil(n);
	int n2 = (int)pow(2,td);
	int numBlocks = (n2-1) / MAXTHREADS + 1;

	int n_size = n * sizeof(int);
	int n2_size = n2 * sizeof(int);

	int* dev_idata;
	int* dev_odata;

	cudaMalloc((void**)&dev_idata, n2_size);
	cudaMalloc((void**)&dev_odata, n2_size);
	cudaMemcpy(dev_idata, idata, n_size, cudaMemcpyHostToDevice);
	cudaMemset(dev_idata+n, 0, n2_size-n_size);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// Scan
	cudaEventRecord(start);
	for(int d=1; d<=td; d++){
		int powd = 1 << (d-1);
		kernScan<<<numBlocks,MAXTHREADS>>>(n2, powd, dev_odata, dev_idata);
		cudaThreadSynchronize();
		cudaMemcpy(dev_idata, dev_odata, n2_size, cudaMemcpyDeviceToDevice);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("naive time(s): %f\n", ms/1000.0);

	// Remove leftover (from the log-rounded portion)
	// Do a shift right to make it an exclusive sum
	odata[0] = 0;
	cudaMemcpy(odata+1, dev_odata, n_size-sizeof(int), cudaMemcpyDeviceToHost);
}

}
}
