#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define MAX_THREADS 512

namespace StreamCompaction {
namespace Efficient {

// TODO: __global__

__global__ void upsweep(int n, int d, int* idata){
	int k = threadIdx.x + (blockIdx.x * blockDim.x);

	if (k < n){
		int d1 = d + 1;
		int powd = 1 << d;
		int powd1 = 1 << d1;
		if (k % (powd1) == 0){
			idata[k + powd1 - 1] += idata[k + powd - 1];
		}
	}
}

__global__ void downsweep(int n, int d, int* idata){
	int k = threadIdx.x + (blockIdx.x * blockDim.x);

	if (k < n){
		int d1 = d + 1;
		int powd = 1 << d;
		int powd1 = 1 << d1;
		if (k % (powd1) == 0){
			int t = idata[k + powd - 1];
			idata[k + powd - 1] = idata[k + powd1 - 1];
			idata[k + powd1 - 1] += t;
		}
	}
}

/*
* Exclusive scan on idata, stores into odata, using shared memory
*/
__global__ void shared_scan(int n, int *odata, const int *idata){
	__shared__ int* temp;

	int index = threadIdx.x;
	int offset = 1;

	temp[2 * index] = idata[2 * index];
	temp[2 * index + 1] = idata[2 * index + 1];

	for (int d = n >> 1; d > 0; d >>= 1){
		__syncthreads();
		if (index < d){
			int ai = offset*(2 * index + 1) - 1;
			int bi = offset*(2 * index + 2) - 1;
			temp[bi] += temp[ai];
		}
	}
	offset *= 2;
	if (index == 0){
		temp[n - 1] = 0;
	}

	for (int d = 1; d < n; d *= 2){
		offset >>= 1;
		__syncthreads();
		if (index < d){
			int ai = offset*(2 * index + 1) - 1;
			int bi = offset*(2 * index + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	odata[2 * index] = temp[2 * index];
	odata[2 * index + 1] = temp[2 * index + 1];
}

template <typename T, typename Predicate> __global__ void kernMapToBoolean(int n, int* odata, T* idata, Predicate pred){
	int index = (blockIdx.x*blockDim.x)+threadIdx.x;

	if (index < n){
		odata[index] = pred(idata[index]);
	}
}

template <typename T> __global__ void kernScatter(int n, T* odata, T* idata, int* bools, int* scan){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (index < n){
		if (bools[index] == 1){
			odata[scan[index]] = idata[index];
		}
	}
}

template <typename T, typename Predicate> int shared_compact(int n, T* dev_odata, T* dev_idata, Predicate pred){
	// Returns the number of elements remaining, elements after the return value in odata are undefined
	// Assumes device memory
	int td = ilog2ceil(n);
	int n2 = (int)pow(2, td);

	int numBlocks = (n - 1) / MAX_THREADS + 1;
	int numBlocks2 = (n2 - 1) / MAX_THREADS + 1;
	int n_size = n * sizeof(int);
	int n2_size = n2 * sizeof(int);
	int out_size = 0;

	int* dev_temp;
	int* dev_temp_n2;
	int* dev_scan;

	cudaMalloc((void**)&dev_temp, n_size);
	cudaMalloc((void**)&dev_temp_n2, n2_size);
	cudaMalloc((void**)&dev_scan, n2_size);

	// Compute temp (binary)
	kernMapToBoolean<<<numBlocks, MAX_THREADS>>>(n, dev_temp, dev_idata, pred);

	// Scan on temp
	cudaMemcpy(dev_temp_n2, dev_temp, n_size, cudaMemcpyDeviceToDevice); // Grow temp
	cudaMemset(dev_temp_n2 + n, 0, n2_size - n_size); // Pad with 0's
	shared_scan<<<numBlocks2, MAX_THREADS>>>(n2, dev_scan, dev_temp_n2);
	
	// Scatter on scan
	kernScatter<<<numBlocks, MAX_THREADS>>>(n, dev_odata, dev_idata, dev_temp, dev_scan);

	// Compute outsize
	int lastnum;
	int lastbool;
	cudaMemcpy(&lastnum, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&lastbool, dev_temp + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	out_size = lastnum + lastbool;
	return out_size;
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
		upsweep<<<numBlocks,MAXTHREADS>>>(n2, d, dev_idata);
	}

	cudaMemset((void*)&dev_idata[n2-1],0,sizeof(int));
	for(int d=td-1; d>=0; d--){
		downsweep<<<numBlocks,MAXTHREADS>>>(n2, d, dev_idata);
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
