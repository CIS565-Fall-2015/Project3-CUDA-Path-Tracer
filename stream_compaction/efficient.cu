#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

// TODO: __global__

__global__ void upsweep(int n, int powd, int powd1, int* idata){
	int k = threadIdx.x + (blockIdx.x * blockDim.x);

	if (k < n){
		if (k % (powd1) == 0){
			idata[k + powd1 - 1] += idata[k + powd - 1];
		}
	}
}

__global__ void downsweep(int n, int powd, int powd1, int* idata){
	int k = threadIdx.x + (blockIdx.x * blockDim.x);

	if (k < n){
		if (k % (powd1) == 0){
			int t = idata[k + powd - 1];
			idata[k + powd - 1] = idata[k + powd1 - 1];
			idata[k + powd1 - 1] += t;
		}
	}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // Compute log-rounded n
	int td = ilog2ceil(n);
	int n2 = (int)pow(2,td);

	int numBlocks = (n2 - 1) / MAXTHREADS + 1;

	int n_size = n * sizeof(int);
	int n2_size = n2 * sizeof(int);

	int* hst_idata2 = new int[n2]();
	memcpy(hst_idata2, idata, n_size);

	int* dev_idata;
	cudaMalloc((void**)&dev_idata, n2_size);
	cudaMemcpy(dev_idata, hst_idata2, n2_size, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	// Scan
	int powd, powd1;
	for(int d=0; d<td; d++){
		powd = (int)pow(2,d);
		powd1 = (int)pow(2,d+1);
		upsweep<<<numBlocks,MAXTHREADS>>>(n2, powd, powd1, dev_idata);
	}

	cudaMemset((void*)&dev_idata[n2-1],0,sizeof(int));
	for(int d=td-1; d>=0; d--){
		powd = (int)pow(2,d);
		powd1 = (int)pow(2,d+1);
		downsweep<<<numBlocks,MAXTHREADS>>>(n2, powd, powd1, dev_idata);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	//printf("efficient scan time (s): %f\n",ms/1000.0);

	// Remove leftover (from the log-rounded portion)
	// No need to shift in this one I guess?
	cudaMemcpy(odata, dev_idata, n_size, cudaMemcpyDeviceToHost);
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
	int n_size = n*sizeof(int);
	int numBlocks = (n - 1) / MAXTHREADS + 1;
	int on = -1;

	// Initialize memory
	int* hst_nz = (int*)malloc(n_size);
	
	int* dev_nz;
	int* dev_idata;
	int* dev_scan;
	int* dev_odata;
	
	cudaMalloc((void**)&dev_nz, n_size);
	cudaMalloc((void**)&dev_idata, n_size);
	cudaMalloc((void**)&dev_scan, n_size);
	cudaMalloc((void**)&dev_odata, n_size);

	// Nonzero
	cudaMemcpy(dev_idata, idata, n_size, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	StreamCompaction::Common::kernMapToBoolean<<<numBlocks,MAXTHREADS>>>(n, dev_nz, dev_idata);
	cudaDeviceSynchronize();

	// TODO: technically only need the last element here
	cudaMemcpy(hst_nz, dev_nz, n_size, cudaMemcpyDeviceToHost);

	// Scan
	int* hst_scan = (int*)malloc(n_size);
	scan(n, hst_scan, hst_nz);
	on = hst_scan[n-1] + hst_nz[n-1];

	// Scatter
	cudaMemcpy(dev_scan, hst_scan, n_size, cudaMemcpyHostToDevice);
	StreamCompaction::Common::kernScatter<<<numBlocks,MAXTHREADS>>>(n, dev_odata, dev_idata, dev_nz, dev_scan);
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("efficient compact time (s): %f\n",ms/1000.0);

	cudaMemcpy(odata, dev_odata, n_size, cudaMemcpyDeviceToHost);

	return on;
}

}
}
