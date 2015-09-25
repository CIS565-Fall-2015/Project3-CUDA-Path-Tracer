#include <cuda.h>
#include <cuda_runtime.h>
#include "efficient.h"
#include <cmath>

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

#define BLOCKSIZE 128
#define CEILING 7

#define FILENAME1 (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError1(msg) checkCUDAErrorFn1(msg, FILENAME1, __LINE__)
#define ERRORCHECK1 0
void checkCUDAErrorFn1(const char *msg, const char *file, int line) {
#if ERRORCHECK1
	+ cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
	exit(EXIT_FAILURE);
#endif ERRORCHECK1
}

namespace StreamCompaction {
namespace Efficient {

	__global__ void filter(int *odata, const PathRay *idata, const int n){
		int k = blockIdx.x*blockDim.x + threadIdx.x;
		if (k < n){
			odata[k] = (int)!idata[k].terminate;
		}
	}

	__global__ void scatter(PathRay *odata, const PathRay *idata, const int *filter, const int *idx, const int n){
		int k = blockIdx.x*blockDim.x + threadIdx.x;

		if (k < n){
			int f = filter[k];
			int i = idx[k];
			PathRay p = idata[k];
			if (f == 1){
				odata[i] = p;
			}
		}
	}

	__global__ void countF(int *c, const int *f, const int *idx, const int n){
		c[0] = f[n - 1] + idx[n - 1];
	}

	/**
	* Scan for one single block
	*/
	__global__ void smallScan(int *small_scan, const int *idata, const int n){
		__shared__ int scanBlock[BLOCKSIZE];
		__shared__ int dArray[CEILING+1];
		int k = blockIdx.x*blockDim.x + threadIdx.x;

		//int ceiling = ilog2ceil(BLOCKSIZE);

		if (threadIdx.x <= CEILING){
			dArray[threadIdx.x] = lround(pow((double)2, (double)(threadIdx.x)));
		}

		__syncthreads();

		if (k < n){
			scanBlock[threadIdx.x] = idata[k];
		}
		else {
			scanBlock[threadIdx.x] = 0;
		}

		// Up sweep
		/*
		int p1, p2;
		for (int d = 0; d < ceiling; d++){
			__syncthreads();
			p1 = lround(pow((double)2, (double)(d + 1)));
			p2 = lround(pow((double)2, (double)d));
			if (threadIdx.x % p1 == 0){
				scanBlock[threadIdx.x - 1 + p1] += scanBlock[threadIdx.x - 1 + p2];
			}
		}
		*/
		int p1, p2, tminus = threadIdx.x - 1, d;
		for (d = 0; d < CEILING; d++){
			__syncthreads();
			if (threadIdx.x %  dArray[d + 1] == 0){
				p2 = tminus + dArray[d];
				p1 = tminus + dArray[d+1];
				scanBlock[p1] += scanBlock[p2];
			}
		}

		// Reset root
		if (threadIdx.x == 0){
			scanBlock[BLOCKSIZE-1] = 0;
		}

		/*
		for (int d = ceiling - 1; d >= 0; d--){
			__syncthreads();
			p1 = lround(pow((double)2, (double)(d + 1)));
			p2 = lround(pow((double)2, (double)d));
			if (threadIdx.x % p1 == 0){
				int tmp = scanBlock[threadIdx.x - 1 + p2];
				scanBlock[threadIdx.x - 1 + p2] = scanBlock[threadIdx.x - 1 + p1];
				scanBlock[threadIdx.x - 1 + p1] += tmp;
			}
		}
		*/
		for (d = CEILING - 1; d >= 0; d--){
			__syncthreads();
			if (threadIdx.x % dArray[d + 1] == 0){
				p2 = tminus + dArray[d];
				p1 = tminus + dArray[d+1];
				int tmp = scanBlock[p2];
				scanBlock[p2] = scanBlock[p1];
				scanBlock[p1] += tmp;
			}
		}
		__syncthreads();

		small_scan[k] = scanBlock[threadIdx.x];
	}

	__global__ void getBlockTotal(int *block_total, const int *scans, const int * data_in, const int B, const int n){
		int totalIndex = (blockIdx.x + 1)*B - 1;
		if (totalIndex < n){
			block_total[blockIdx.x] = scans[totalIndex] + data_in[totalIndex];
		}
		else {
			block_total[blockIdx.x] = scans[totalIndex];
		}
	}

	__global__ void blockIncrement(int *odata, const int *block_incr, const int *small_scan, const int n){
		int k = blockIdx.x*blockDim.x + threadIdx.x;
		if (k < n){
			odata[k] = block_incr[blockIdx.x] + small_scan[k];
		}
	}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	int *scan_out;
	if (n <= BLOCKSIZE){
		// Base case
		cudaMalloc((void**)&scan_out, BLOCKSIZE * sizeof(int));
		// Scan on each block & get total sum
		smallScan << <1, BLOCKSIZE >> >(scan_out, idata, n);
		cudaMemcpy(odata, scan_out, n * sizeof(int), cudaMemcpyDeviceToHost);

		checkCUDAError1("SC small scan base case");
	}
	else {
		// Divide into blocks and padding
		//int gridSize = lround(ceil((double)n / (double)BLOCKSIZE));
		int gridSize = (n + BLOCKSIZE - 1) / BLOCKSIZE;

		int *block_total;

		// Padding zero
		cudaMalloc((void**)&scan_out, gridSize * BLOCKSIZE * sizeof(int));
		cudaMalloc((void**)&block_total, gridSize * sizeof(int));

		smallScan << <gridSize, BLOCKSIZE >> >(scan_out, idata, n);
		checkCUDAError1("SC small scan");

		getBlockTotal << <gridSize, 1 >> >(block_total, scan_out, idata, BLOCKSIZE, n);

		// Scan and get block increment; recursively
		scan(gridSize, block_total, block_total);

		// Increment block
		blockIncrement << <gridSize, BLOCKSIZE >> >(odata, block_total, scan_out, n);
		checkCUDAError1("SC block increment");

		cudaFree(block_total);
	}
	cudaFree(scan_out);
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
void compact(int n, int* f, int* idx, PathRay *dv_out, PathRay *idata, int* c) {
	// Padding
	//int gridSize = (n + BLOCKSIZE - 1) / BLOCKSIZE;
	//int m = (int)pow((double)2, (double)ilog2ceil(gridSize));
	//int tsize = m > gridSize ? m : gridSize;
	//int tsize = gridSize;
	int tsize = (n + BLOCKSIZE - 1) / BLOCKSIZE;

	// Filter
	filter << <tsize, BLOCKSIZE >> >(f, idata, n);
	checkCUDAError1("SC filter");

	// Scan
	scan(n, idx, f);

	// Scatter
	scatter << <tsize /2, BLOCKSIZE * 2>> >(dv_out, idata, f, idx, n);
	checkCUDAError1("SC scatter");

	// Get new array size
	countF<<<1, 1>>>(c, f, idx, n);
}

}
}
