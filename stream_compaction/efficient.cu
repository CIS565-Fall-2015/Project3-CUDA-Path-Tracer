#include <cuda.h>
#include <cuda_runtime.h>
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
namespace Efficient {

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
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
}

/**
* Maps an array to an array of 0s and 1s for stream compaction. Elements
* which map to 0 will be removed, and elements which map to 1 will be kept.
*/
__global__ void KernMapToBoolean(int n, int *bools, const Ray *idata) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	bools[index] = !!idata[index].alive;
}

/**
* Performs scatter on an array. That is, for each element in idata,
* if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
*/
__global__ void KernScatter(int n, Ray *odata, const Ray *idata, const int *bools, const int *indices) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (bools[index] == 1) {
		odata[indices[index]] = idata[index];
	}
}

/**
* Accumulates the new count of threads for a block with the original.
*/
__global__ void KernGetBlockCount(int n, int *odata, const int *idata1, const int *idata2) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	odata[index] = idata1[(index + 1) * blockSize - 1] + idata2[(index + 1) * blockSize - 1];
}

/**
* Increments the block count.
*/
__global__ void KernIncrementBlock(int n, int *data, const int *increments) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	data[index] = data[index] + increments[blockIdx.x];
}

/*
* Performs work efficient scan on data in a single GPU block using shared memory.
* Based on the GPU Gems 3 Code found here:
* http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
*/
__global__ void KernScan(int n, int *odata, const int *idata) {
	int index = threadIdx.x;
	int offset = 1;
	extern __shared__ int temp[];

	// Copy input data to shared memory
	temp[2 * index] = idata[2 * index + (blockIdx.x * blockDim.x * 2)];
	temp[2 * index + 1] = idata[2 * index + 1 + (blockIdx.x * blockDim.x * 2)];

	// Up sweep
	for (int d = n >> 1; d > 0; d >>= 1) {
		__syncthreads();

		if (index < d) {
			int ai = offset * (2 * index + 1) - 1;
			int bi = offset * (2 * index + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	// Clear the root
	if (index == 0) {
		temp[n - 1] = 0;
	}

	// Down sweep
	for (int d = 1; d < n; d *= 2) {
		offset >>= 1;
		__syncthreads();

		if (index < d) {
			int ai = offset * (2 * index + 1) - 1;
			int bi = offset * (2 * index + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	// Write to output array
	odata[2 * index + (blockIdx.x * blockDim.x * 2)] = temp[2 * index];
	odata[2 * index + 1 + (blockIdx.x * blockDim.x * 2)] = temp[2 * index + 1];
}

void Scan(int n, int *odata, int *idata) {
	int blocksPerGrid = (n - 1) / blockSize + 1;
	int *dev_idata, *dev_odata; // Padded device memory to handle non power of 2 cases

	cudaMalloc((void**)&dev_idata, blocksPerGrid * blockSize * sizeof(int));
	cudaMemset(dev_idata, 0, blocksPerGrid * blockSize * sizeof(int));
	cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);

	cudaMalloc((void**)&dev_odata, blocksPerGrid * blockSize * sizeof(int));

	if (blocksPerGrid == 1) {
		KernScan<<<1, blockSize / 2, blockSize * sizeof(int)>>>(blockSize, dev_odata, dev_idata);
		checkCUDAError("KernScan");

		cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);
	}
	else {
		int *dev_increments, *dev_scannedIncrements;

		cudaMalloc((void**)&dev_increments, blocksPerGrid * sizeof(int));
		cudaMalloc((void**)&dev_scannedIncrements, blocksPerGrid * sizeof(int));

		KernScan<<<blocksPerGrid, blockSize / 2, blockSize * sizeof(int)>>>(blockSize, dev_odata, dev_idata);
		checkCUDAError("KernScan");

		int tempBlocksPerGrid = (blocksPerGrid - 1) / blockSize + 1;
		KernGetBlockCount<<<tempBlocksPerGrid, blockSize>>>(blocksPerGrid, dev_increments, dev_odata, dev_idata);
		checkCUDAError("KernGetBlockCount");

		// Recursive scan call until we can fit on a single block
		Scan(blocksPerGrid, dev_scannedIncrements, dev_increments);

		KernIncrementBlock<<<blocksPerGrid, blockSize>>>(blocksPerGrid * blockSize, dev_odata, dev_scannedIncrements);
		checkCUDAError("KernIncrementBlock");

		cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);

		cudaFree(dev_increments);
		cudaFree(dev_scannedIncrements);
	}

	cudaFree(dev_idata);
	cudaFree(dev_odata);
}

int Compact(int n, Ray *odata, Ray *idata) {
	int blocksPerGrid = (n - 1) / blockSize + 1;
	int rayCount = 0;
	int *dev_bools, *dev_scanData;

	cudaMalloc((void**)&dev_bools, n * sizeof(int));
	cudaMalloc((void**)&dev_scanData, blocksPerGrid * blockSize * sizeof(int));

	// Map input to boolean values
	KernMapToBoolean<<<blocksPerGrid, blockSize>>>(n, dev_bools, idata);
	checkCUDAError("KernMapToBoolean");

	// Scan
	Scan(n, dev_scanData, dev_bools);

	// Scatter
	KernScatter<<<blocksPerGrid, blockSize>>>(n, odata, idata, dev_bools, dev_scanData);
	checkCUDAError("KernScatter");

	// Get number of rays remaining
	int lastScanDataElem, lastBoolElem;
	cudaMemcpy(&lastScanDataElem, dev_scanData + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&lastBoolElem, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	rayCount = lastScanDataElem + lastBoolElem;

	cudaFree(dev_bools);
	cudaFree(dev_scanData);

	return rayCount;
}
}
}
