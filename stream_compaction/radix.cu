#include "radix.h"
#include "common.h"
#include "efficient.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace StreamCompaction {
namespace Radix {

/**
 * Maps an array to an array of 0s and 1s for stream compaction. Elements
 * which map to 0 will be removed, and elements which map to 1 will be kept.
 */
__global__ void kernMapToBits(int n, int bit, int *bools, const int *idata) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);

	if (k < n){
		int b = 1 << bit;
		bools[k] = (b & idata[k]) >> bit;
	}
}

__global__ void kernNegate(int n, int* odata, const int* idata){
	int k = threadIdx.x + (blockIdx.x * blockDim.x);

	if (k < n){
		odata[k] = !idata[k];
	}
}

__global__ void kernSumFalses(int n, int totalFalses, int* odata, const int* idata){
	int k = threadIdx.x + (blockIdx.x * blockDim.x);

	if (k < n){
		odata[k] = k - idata[k] + totalFalses;
	}
}

__global__ void kernMux(int n, int* odata, const int* bdata, const int* tdata, const int* fdata){
	int k = threadIdx.x + (blockIdx.x * blockDim.x);

	if (k < n){
		odata[k] = bdata[k] ? tdata[k] : fdata[k];
	}
}

__global__ void kernScatter(int n, int *odata,
	const int *idata, const int *indices) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);

	if (k < n){
		odata[indices[k]] = idata[k];
	}
}

void split(int n, int bit, int* odata, int* idata){
	int numBlocks = (n - 1) / MAXTHREADS + 1;
	int n_size = n * sizeof(int);

	int* hst_e;
	int* hst_f;
	int* dev_idata;
	int* dev_odata;
	int* dev_b;
	int* dev_e;
	int* dev_f;
	int* dev_t;
	int* dev_d;
	int totalFalses;

	hst_e = (int*)malloc(n_size);
	hst_f = (int*)malloc(n_size);
	cudaMalloc((void**)&dev_idata, n_size);
	cudaMalloc((void**)&dev_odata, n_size);
	cudaMalloc((void**)&dev_b, n_size);
	cudaMalloc((void**)&dev_e, n_size);
	cudaMalloc((void**)&dev_f, n_size);
	cudaMalloc((void**)&dev_t, n_size);
	cudaMalloc((void**)&dev_d, n_size);

	cudaMemcpy(dev_idata, idata, n_size, cudaMemcpyHostToDevice);

	// b - get bits
	kernMapToBits<<<numBlocks, MAXTHREADS>>>(n, bit, dev_b, dev_idata);
	cudaDeviceSynchronize();

	int* hst_b = (int*)malloc(n_size);

	// e - negate bits
	kernNegate<<<numBlocks, MAXTHREADS>>>(n, dev_e, dev_b);
	cudaDeviceSynchronize();

	// f - exclusive scan on e
	cudaMemcpy(hst_e, dev_e, n_size, cudaMemcpyDeviceToHost);
	StreamCompaction::Efficient::scan(n, hst_f, hst_e);
	cudaMemcpy(dev_f, hst_f, n_size, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	// t - t[i] = i - f[i] + totalFalses
	int en1;
	cudaMemcpy(&en1, dev_e + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
	totalFalses = en1 + hst_f[n - 1];
	kernSumFalses<<<numBlocks, MAXTHREADS>>>(n, totalFalses, dev_t, dev_f);
	cudaDeviceSynchronize();

	// d - d[i] = b[i] ? t[i] : f[i]
	kernMux<<<numBlocks, MAXTHREADS>>>(n, dev_d, dev_b, dev_t, dev_f);
	cudaDeviceSynchronize();

	// scatter idata according to d
	kernScatter<<<numBlocks, MAXTHREADS>>>(n, dev_odata, dev_idata, dev_d);
	cudaDeviceSynchronize();

	cudaMemcpy(odata, dev_odata, n_size, cudaMemcpyDeviceToHost);
}

void sort(int n, int* odata, const int* idata){
	int numBits = 8 * sizeof(int); //TODO: probably only need log size of largest int in the array
	int n_size = n*sizeof(int);
	
	int* hst_idata = (int*)malloc(n_size);
	int* hst_odata = (int*)malloc(n_size);
	int* tmp;
	memcpy(hst_idata, idata, n_size);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	for (int i = 0; i < numBits; i++){
		split(n, i, hst_odata, hst_idata);
		cudaDeviceSynchronize();
		memcpy(hst_idata, hst_odata, n_size);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("radix sort time (s): %f\n", ms / 1000.0);

	memcpy(odata, hst_odata, n_size);
}

}
}
