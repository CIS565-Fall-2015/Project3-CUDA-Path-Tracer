#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

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

__global__ void upsweep_step(int d_offset_plus, int d_offset, int *x) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);
	if (k % d_offset_plus) {
		return;
	}
	x[k + d_offset_plus - 1] += x[k + d_offset - 1];
}

__global__ void downsweep_step(int d_offset_plus, int d_offset, int *x) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);
	if (k % d_offset_plus) {
		return;
	}
	int t = x[k + d_offset - 1];
	x[k + d_offset - 1] = x[k + d_offset_plus - 1];
	x[k + d_offset_plus - 1] += t;
}

__global__ void fill_by_value(int val, int *x) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);
	x[k] = val;
}

static void setup_dimms(dim3 &dimBlock, dim3 &dimGrid, int n) {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int tpb = deviceProp.maxThreadsPerBlock;
	int blockWidth = fmin(n, tpb);
	int blocks = 1;
	if (blockWidth != n) {
		blocks = n / tpb;
		if (n % tpb) {
			blocks ++;
		}
	}

	dimBlock = dim3(blockWidth);
	dimGrid = dim3(blocks);
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {

	// we'll need to pad the device memory with 0s to get a power of 2 array size.
	int logn = ilog2ceil(n);
	int pow2 = (int)pow(2, logn);

	dim3 dimBlock;
	dim3 dimGrid;
	setup_dimms(dimBlock, dimGrid, pow2);

	int *dev_x;
	cudaMalloc((void**)&dev_x, sizeof(int) * pow2);
	fill_by_value <<<dimGrid, dimBlock >>>(0, dev_x);
	// copy everything in idata over to the GPU.
	cudaMemcpy(dev_x, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

	// up sweep and down sweep
	up_sweep_down_sweep(pow2, dev_x, -1, -1);

	cudaMemcpy(odata, dev_x, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaFree(dev_x);
}

// exposed up sweep and down sweep. expects powers of two!
void up_sweep_down_sweep(int n, int *dev_data1, int blocksPerGrid, int blockSize) {
	int logn = ilog2ceil(n);

	dim3 dimBlock(blockSize, 1);
	dim3 dimGrid(blocksPerGrid, 1);
	if (blockSize < 0 && blocksPerGrid < 0)
		setup_dimms(dimBlock, dimGrid, n);

	// Up Sweep
	for (int d = 0; d < logn; d++) {
		int d_offset_plus = (int)pow(2, d + 1);
		int d_offset = (int)pow(2, d);
		upsweep_step << <dimGrid, dimBlock >> >(d_offset_plus, d_offset, dev_data1);
	}

	// Down-Sweep
	cudaMemset(&dev_data1[n - 1], 0, sizeof(int) * 1);
	for (int d = logn - 1; d >= 0; d--) {
		int d_offset_plus = (int)pow(2, d + 1);
		int d_offset = (int)pow(2, d);
		downsweep_step << <dimGrid, dimBlock >> >(d_offset_plus, d_offset, dev_data1);
	}
}

__global__ void temporary_array(int *x, int *temp) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);
	temp[k] = (x[k] != 0);
}

__global__ void scatter(int *x, int *trueFalse, int* scan, int *out) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);
	if (trueFalse[k]) {
		out[scan[k]] = x[k];
	}
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
	int logn = ilog2ceil(n);
	int pow2 = (int)pow(2, logn);

	dim3 dimBlock;
	dim3 dimGrid;
	setup_dimms(dimBlock, dimGrid, pow2);

	int *dev_x;
	int *dev_tmp;
	int *dev_scatter;
	int *dev_scan;

	cudaMalloc((void**)&dev_x, sizeof(int) * pow2);
	cudaMalloc((void**)&dev_tmp, sizeof(int) * pow2);
	cudaMalloc((void**)&dev_scan, sizeof(int) * pow2);
	cudaMalloc((void**)&dev_scatter, sizeof(int) * pow2);

	// 0 pad up to a power of 2 array length.
	// copy everything in idata over to the GPU.
	fill_by_value << <dimGrid, dimBlock >> >(0, dev_x);
	cudaMemcpy(dev_x, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

    // Step 1: compute temporary true/false array
	temporary_array <<<dimGrid, dimBlock >>>(dev_x, dev_tmp);

	// Step 2: run efficient scan on the tmp array
	cudaMemcpy(dev_scan, dev_tmp, sizeof(int) * pow2, cudaMemcpyDeviceToDevice);
	up_sweep_down_sweep(pow2, dev_scan, -1, -1);

	// Step 3: scatter
	scatter <<<dimGrid, dimBlock >>>(dev_x, dev_tmp, dev_scan, dev_scatter);

	cudaMemcpy(odata, dev_scatter, sizeof(int) * n, cudaMemcpyDeviceToHost);

	int last_index;
	cudaMemcpy(&last_index, dev_scan + (n - 1), sizeof(int),
		cudaMemcpyDeviceToHost);

	int last_true_false;
	cudaMemcpy(&last_true_false, dev_tmp + (n - 1), sizeof(int),
		cudaMemcpyDeviceToHost);

	cudaFree(dev_x);
	cudaFree(dev_tmp);
	cudaFree(dev_scan);
	cudaFree(dev_scatter);

	return last_index + last_true_false;
}

__global__ void block_upsweep(int n, int *dev_data) {
	// parallel reduction with some modifications
	// in place of:
	// 0  1  2  3  4  5  6  7   stride = 1  
	// 1     5     9     13     stride = 2 
	// 6           22           stride = 4 
	// 28                              
	//
	// we want:
	// 0  1  2  3  4  5  6  7   stride = 1
	//    1     5     9     13  stride = 2
	//          6           22  stride = 4
	//                      28
	//
	// want to do stuff at indices:
	// 1  3  5  6  7 -> stride to get here is 1
	// 3 7 -> stride to get here is 2
	// 7 -> stride to get here is 4
	//
	// use if((t + 2) % (2 * stride) == 1)
	// needs to produce something more like an "upsweep" than a traditional parallel reduction

	unsigned int t = threadIdx.x; // we're indexing shared memory, so no need for +(blockIdx.x * blockDim.x);

	// load into shared memory from provided pointer
	// we know dev_data is spread over the entire grid
	// so start is blockId.x * blockDim.x, size i blockDim.x
	__shared__ int block_data[blockDim.x];
	block_data[t] = dev_data[t + blockIdx.x * blockDim.x];
	// for each stage:
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		// syncthreads to make sure all threads have transferred relevant data
		__syncthreads();
		// compute partial
		if ((t + 2) % (2 * stride) == 1) {
			partialSum[t] += partialSum[t - stride];
		}
	}
	__syncthreads(); // make sure all threads are done computing
	// write the data out
	dev_data[t + blockIdx.x * blockDim.x] = block_data[t];
}

__global__ void block_downsweep() {

}

void efficient_scan(int n, int *dev_data, int blocksPerGrid, int blockSize) {
	// break up into blocks. pad with zeros so we have pow 2
	// run scan on each block (upsweep downsweep)
	// accumulate block sums into an array of sums.
	// scan block sums to compute block increments. if it's too big for one block, recurse (omg)
	// add block increments to each element in the corresponding block. stop at n, don't pile on zeros
	// return!
}

}
}
