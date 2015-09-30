#include <cuda.h>
#include <cuda_runtime.h>
#include <src/sceneStructs.h>
#include "common.h"
#include "shared.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn2(msg, FILENAME, __LINE__)
void checkCUDAErrorFn2(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
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
#endif //ERRORCHECK
}

namespace StreamCompaction {
namespace Shared {

/**
 * Maps an array to an array of 0s and 1s for stream compaction. Elements
 * which map to 0 will be removed, and elements which map to 1 will be kept.
 */
__global__ void kernMapToBoolean(int n, int valid, int *bools, Pixel *idata) {
    int k = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (k < valid) {
        bools[k] = (idata[k].terminated == false) ? 1 : 0;
    } else if (k < n) {
        bools[k] = 0;
    }
}

__global__ void kernScatter(int n, int valid, Pixel *odata, int *indices, Pixel *idata) {
    int k = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (k >= n) { return; }
    if (k == n-1) {
        // always take the last element
        // `compact` will adjust size appropriately
        int index = indices[k];
        if (index < valid) {
            odata[index] = idata[k];
        }
    } else if (indices[k] != indices[k+1]) {
        int index = indices[k];
        if (index < valid) {
            odata[index] = idata[k];
        }
    }
}

int BLOCK_SIZE = 1 << 8;

__global__ void kUpSweep(int *data, int n) {
    int t = threadIdx.x;
    int start = blockDim.x * blockIdx.x;

    // Load into shared memory
    extern __shared__ int shared[];
    shared[t] = data[start+t];
    __syncthreads();

    n = blockDim.x;
    int iters = ilog2ceil(n);

    for (int d = 0; d < iters; d++) {
        int exp_d1 = (int)exp2f(d+1);
        int k = t * exp_d1;

        if (k + exp_d1 - 1 < n) {
            int exp_d  = (int)exp2f(d);
            shared[k + exp_d1 - 1] += shared[k + exp_d - 1];
        }
        __syncthreads();
    }

    // Load back into global memory
    data[start+t] = shared[t];
    __syncthreads();
}

__global__ void kStoreZero(int *data, int count, int blockSize) {
    int t = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (t < count) {
        int index = (t+1)*blockSize - 1;
        data[index] = 0;
    }
}

__global__ void kDownSweep(int *data, int n) {
    int t = threadIdx.x;
    int start = blockDim.x * blockIdx.x;

    // Load into shared memory
    extern __shared__ int shared[];
    shared[t] = data[start+t];
    __syncthreads();

    n = blockDim.x;
    int iters = ilog2ceil(n)-1;
    for (int d = iters; d >= 0; d--) {

        int k = t * (int)exp2f(d+1);
        int left  = k + (int)exp2f(d)   - 1;
        int right = k + (int)exp2f(d+1) - 1;

        if (k < n && right < n) {
            int left_data  = shared[left];

            shared[left]   = shared[right];
            shared[right] += left_data;
        }
        __syncthreads();
    }

    // Load back into global memory
    if (t < n) {
        data[start+t] = shared[t];
    }
    __syncthreads();
}

__global__ void kStoreLastElt(int *odata, int *idata, int numBlocks, int blockSize) {
    int t = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (t < numBlocks) {
        int index = (t+1)*blockSize - 1;
        odata[t] = idata[index];
    }
}

__global__ void kAddLastElt(int *odata, int *idata, int numBlocks, int blockSize) {
    int t = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (t < numBlocks) {
        int index = (t+1)*blockSize - 1;
        odata[t] += idata[index];
    }
}

__global__ void kAddRunningBlockTotal(int *data, int *totals) {
    int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    data[idx] += totals[blockIdx.x];
}

/*
 * In-place scan on `dev_data`, which must be a device memory pointer.
 */
void dv_scan(int n, int *dev_data) {
    // Number of blocks of data
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Number of blocks, when operating on the set of blocks
    int numBlocksForBlocks = (numBlocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

            checkCUDAError("1");
    int *increments;
    cudaMalloc((void**) &increments, numBlocks*sizeof(int));
            checkCUDAError("1");

    // Store last value, to add into block increments later
    kStoreLastElt<<<numBlocksForBlocks, BLOCK_SIZE>>>(increments, dev_data, numBlocks, BLOCK_SIZE);
            checkCUDAError("1");

    // Run EXCLUSIVE scan on each block
    kUpSweep<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(int)>>>(dev_data, n);
            checkCUDAError("1");
    kStoreZero<<<numBlocksForBlocks, BLOCK_SIZE>>>(dev_data, numBlocks, BLOCK_SIZE);
            checkCUDAError("1");
    kDownSweep<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(int)>>>(dev_data, n);
            checkCUDAError("1");

    // Build array of sums from each block
    kAddLastElt<<<numBlocksForBlocks, BLOCK_SIZE>>>(increments, dev_data, numBlocks, BLOCK_SIZE);
            checkCUDAError("1");

    // Find block increments (EXclusive scan)
    if (numBlocks > BLOCK_SIZE) {
        dv_scan(numBlocks, increments);
            checkCUDAError("1");
    } else {
        kUpSweep<<<1, numBlocks, numBlocks*sizeof(int)>>>(increments, numBlocks);
            checkCUDAError("1");
        kStoreZero<<<1, 1>>>(increments, 1, numBlocks);
            checkCUDAError("1");
        kDownSweep<<<1, numBlocks, numBlocks*sizeof(int)>>>(increments, numBlocks);
            checkCUDAError("1");
    }

    // Add block increments back into each blocks
        checkCUDAError("1");
    kAddRunningBlockTotal<<<numBlocks, BLOCK_SIZE>>>(dev_data, increments);
        checkCUDAError("1");

    //cudaFree(increments);
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
int compact(int size, Pixel *input) {
    checkCUDAError("1");
    int n;

    if (size & (size-1) != 0) { // if size is not a power of 2
        n = (int)exp2f(ilog2ceil(size));
    } else {
        n = size;
    }

    int array_size = n * sizeof(Pixel);

    int *dev_indices;
    cudaMalloc((void**) &dev_indices, n * sizeof(int));
    checkCUDAError("1");

    Pixel *dev_odata;
    cudaMalloc((void**) &dev_odata, array_size);
    checkCUDAError("1");

    Pixel *dev_idata;
    cudaMalloc((void**) &dev_idata, array_size);
    cudaMemcpy(dev_idata, input, size * sizeof(Pixel), cudaMemcpyHostToDevice);
    checkCUDAError("1");

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    checkCUDAError("1");
    kernMapToBoolean<<<numBlocks, BLOCK_SIZE>>>(n, size, dev_indices, dev_idata);
    checkCUDAError("1");

    int last;
    cudaMemcpy(&last, dev_indices + size-1, sizeof(int), cudaMemcpyDeviceToHost);

    checkCUDAError("1");
    dv_scan(n, dev_indices);
    checkCUDAError("1");

    int streamSize;
    cudaMemcpy(&streamSize, dev_indices + size-1, sizeof(int), cudaMemcpyDeviceToHost);

    kernScatter<<<numBlocks, BLOCK_SIZE>>>(n, size, dev_odata, dev_indices, dev_idata);
    checkCUDAError("1");

    cudaMemcpy(input, dev_odata, size * sizeof(Pixel), cudaMemcpyDeviceToHost);
    checkCUDAError("1");

    // The kernel always copies the last elt.
    // Adjust the size to include it if desired.
    if (last == 1) {
        streamSize++;
    }

    cudaFree(dev_indices);
    cudaFree(dev_odata);
    cudaFree(dev_idata);

    return streamSize;
}

}
}
