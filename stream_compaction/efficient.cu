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

#define BLOCKSIZE 64

#define FILENAME1 (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError1(msg) checkCUDAErrorFn1(msg, FILENAME1, __LINE__)
#define ERRORCHECK1 1
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

	__global__ void filter(int *odata, PathRay *idata, const int n){
		int k = blockIdx.x*blockDim.x + threadIdx.x;
		if (k < n){
			odata[k] = (int)!idata[k].terminate;
		}
	}

	__global__ void scatter(PathRay *odata, PathRay *idata, int *filter, int *idx, const int n){
		int k = blockIdx.x*blockDim.x + threadIdx.x;
		if (k < n){
			if (filter[k] == 1){
				odata[idx[k]] = idata[k];
			}
		}
	}



	__global__ void countF(int *c, int *f, int *idx, const int n){
		c[0] = f[n - 1] + idx[n - 1];
	}

	__global__ void countS(PathRay *dv, int n, int *f, int *idx){
		int count = 0;
		for (int i = 0; i < n; i++){
			if (f[i] != 0){
				count++;
			}
		}
		printf("%d : %d, %d\n", count, idx[0], idx[n - 1]);
	}

	/**
	* Scan for one single block
	*/
	__global__ void smallScan(int *small_scan, const int *idata, const int n){
		__shared__ int scanBlock[BLOCKSIZE];
		//__shared__ int oBlock[BLOCKSIZE];
		int k = blockIdx.x*blockDim.x + threadIdx.x;
		int t = threadIdx.x;

		int ceiling = ilog2ceil(BLOCKSIZE);

		if (k < n){
			scanBlock[t] = idata[k];
			//oBlock[t] = idata[k];
		}
		else {
			scanBlock[t] = 0;
			//oBlock[t] = 0;
		}

		// Up sweep
		int p1, p2;
		for (int d = 0; d < ceiling; d++){
			__syncthreads();
			p1 = lround(pow((double)2, (double)(d + 1)));
			p2 = lround(pow((double)2, (double)d));
			if (t % p1 == 0){
				scanBlock[t - 1 + p1] += scanBlock[t - 1 + p2];
			}
		}

		__syncthreads();
		// Reset root
		if (t == 0){
			scanBlock[BLOCKSIZE-1] = 0;
		}

		__syncthreads();

		for (int d = ceiling - 1; d >= 0; d--){
			__syncthreads();
			p1 = lround(pow((double)2, (double)(d + 1)));
			p2 = lround(pow((double)2, (double)d));
			if (t % p1 == 0){
				int tmp = scanBlock[t - 1 + p2];
				scanBlock[t - 1 + p2] = scanBlock[t - 1 + p1];
				scanBlock[t - 1 + p1] += tmp;
			}
		}
		__syncthreads();

		if (k < n){
			small_scan[k] = scanBlock[t];
		}
	}

	__global__ void getBlockTotal(int *block_total, int *scans, const int B){
		int totalIndex = (blockIdx.x + 1)*B - 1;
		block_total[blockIdx.x] = scans[totalIndex];
	}

	__global__ void blockIncrement(int *odata, int *block_incr, int *small_scan, const int n){
		int k = blockIdx.x*blockDim.x + threadIdx.x;
		if (k < n){
			odata[k] = block_incr[blockIdx.x] + small_scan[k];
		}
	}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	if (n <= BLOCKSIZE){
		// Base case
		// Scan on each block & get total sum
		smallScan << <1, BLOCKSIZE >> >(odata, idata, n);

		checkCUDAError1("SC small scan base case");
	}
	else {
		// Divide into blocks and padding
		int gridSize = ceil(n / BLOCKSIZE);

		int *scan_out;
		int *block_total, *block_scan;

		// Padding zero
		cudaMalloc((void**)&scan_out, gridSize * BLOCKSIZE * sizeof(int));
		cudaMalloc((void**)&block_total, gridSize * sizeof(int));
		cudaMemset(block_total, 0, gridSize * sizeof(int));
		cudaMalloc((void**)&block_scan, gridSize * sizeof(int));
		cudaMemset(block_scan, 0, gridSize * sizeof(int));

		smallScan << <gridSize, BLOCKSIZE >> >(scan_out, idata, n);
		checkCUDAError1("SC small scan");

		getBlockTotal << <gridSize, 1 >> >(block_total, scan_out, BLOCKSIZE);

		// Scan and get block increment; recursively
		scan(gridSize, block_scan, block_total);

		// Increment block
		blockIncrement << <gridSize, 1 >> >(odata, block_scan, scan_out, n);
		checkCUDAError1("SC block increment");
		cudaFree(scan_out);
		cudaFree(block_total);
		cudaFree(block_scan);
	}
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, PathRay *idata) {
	int *f;
	// Padding
	int gridSize = (n + BLOCKSIZE - 1) / BLOCKSIZE;
	int m = (int)pow((double)2, (double)ilog2ceil(gridSize));
	int tsize = m > gridSize ? m : gridSize;

	// Filter
	cudaMalloc((void**)&f, n * sizeof(int));
	filter << <tsize, BLOCKSIZE >> >(f, idata, n);
	checkCUDAError1("SC filter");

	// Scan
	int *idx;
	cudaMalloc((void**)&idx, n * sizeof(int));
	scan(n, idx, f);

	// Scatter
	PathRay *dv_out;
	cudaMalloc((void**)&dv_out, n * sizeof(PathRay));
	cudaMemset(dv_out, 0, n*sizeof(PathRay));

	scatter << <tsize, BLOCKSIZE >> >(dv_out, idata, f, idx, n);
	checkCUDAError1("SC scatter");

	countS << <1, 1 >> >(dv_out, n, f, idx);

	// Get new array size
	int count = 0;
	int *c;
	cudaMalloc((void**)&c, sizeof(int));
	countF<<<1, 1>>>(c, f, idx, n);
	cudaMemcpy(&count, &c[0], sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(idata, dv_out, count*sizeof(PathRay), cudaMemcpyDeviceToDevice);

	cudaFree(f);
	cudaFree(idx);
	cudaFree(dv_out);
	cudaFree(c);

	return count;
}

}
}
