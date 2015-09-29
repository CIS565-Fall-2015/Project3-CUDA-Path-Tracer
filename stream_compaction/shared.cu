#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "shared.h"

namespace StreamCompaction {
namespace Shared {

/**
 * Maps an array to an array of 0s and 1s for stream compaction. Elements
 * which map to 0 will be removed, and elements which map to 1 will be kept.
 */
__global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
    int k = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (k < n) {
        bools[k] = (idata[k] != 0) ? 1 : 0;
    }
}

__global__ void kernScatter(int n, int *odata, int *indices, int *idata) {
    int k = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (k >= n) { return; }
    if (k == n-1) {
        // always take the last element
        // `compact` will adjust size appropriately
        odata[indices[k]] = idata[k];
    } else if (indices[k] != indices[k+1]) {
        odata[indices[k]] = idata[k];
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

    int *increments;
    cudaMalloc((void**) &increments, numBlocks*sizeof(int));

    // Store last value, to add into block increments later
    kStoreLastElt<<<numBlocksForBlocks, BLOCK_SIZE>>>(increments, dev_data, numBlocks, BLOCK_SIZE);

    // Run EXCLUSIVE scan on each block
    kUpSweep<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(int)>>>(dev_data, n);
    kStoreZero<<<numBlocksForBlocks, BLOCK_SIZE>>>(dev_data, numBlocks, BLOCK_SIZE);
    kDownSweep<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(int)>>>(dev_data, n);

    // Build array of sums from each block
    kAddLastElt<<<numBlocksForBlocks, BLOCK_SIZE>>>(increments, dev_data, numBlocks, BLOCK_SIZE);

    // Find block increments (EXclusive scan)
    if (numBlocks > BLOCK_SIZE) {
        dv_scan(numBlocks, increments);
    } else {
        kUpSweep<<<1, numBlocks, numBlocks*sizeof(int)>>>(increments, numBlocks);
        kStoreZero<<<1, 1>>>(increments, 1, numBlocks);
        kDownSweep<<<1, numBlocks, numBlocks*sizeof(int)>>>(increments, numBlocks);
    }

    // Add block increments back into each blocks
    kAddRunningBlockTotal<<<numBlocks, BLOCK_SIZE>>>(dev_data, increments);

    cudaFree(increments);
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int size, int *odata, int *input) {
    int *idata;
    int n;

    if (size & (size-1) != 0) { // if size is not a power of 2
        n = (int)exp2f(ilog2ceil(size));
        idata = (int*)malloc(n * sizeof(int));
        memcpy(idata, input, n * sizeof(int));
        for (int j = size; j < n; j++) {
            idata[j] = 0;
        }
    } else {
        n = size;
        idata = input;
    }

    int array_size = n * sizeof(int);
    int *dv_idata;

    cudaMalloc((void**) &dv_idata, array_size);
    cudaMemcpy(dv_idata, idata, array_size, cudaMemcpyHostToDevice);

    dv_scan(n, dv_idata);

    cudaMemcpy(odata, dv_idata, array_size, cudaMemcpyDeviceToHost);
    cudaFree(dv_idata);
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
int compact(int size, int *odata, const int *input) {
    int *idata;
    int n;

    if (size & (size-1) != 0) { // if size is not a power of 2
        n = (int)exp2f(ilog2ceil(size));
        idata = (int*)malloc(n * sizeof(int));
        memcpy(idata, input, n * sizeof(int));
        for (int j = size; j < n; j++) {
            idata[j] = 0;
        }
    } else {
        n = size;
        idata = (int*)malloc(n * sizeof(int));
        memcpy(idata, input, n * sizeof(int));
    }

    int *dev_indices;
    int *dev_odata;
    int *dev_idata;
    int array_size = n * sizeof(int);
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc((void**) &dev_indices, array_size);
    cudaMalloc((void**) &dev_odata, array_size);

    cudaMalloc((void**) &dev_idata, array_size);
    cudaMemcpy(dev_idata, idata, array_size, cudaMemcpyHostToDevice);

    kernMapToBoolean<<<numBlocks, BLOCK_SIZE>>>(n, dev_indices, dev_idata);

    int last;
    cudaMemcpy(&last, dev_indices + n-1, sizeof(int), cudaMemcpyDeviceToHost);

    dv_scan(n, dev_indices);
    int streamSize;
    cudaMemcpy(&streamSize, dev_indices + n-1, sizeof(int), cudaMemcpyDeviceToHost);

    kernScatter<<<numBlocks, BLOCK_SIZE>>>(n, dev_odata, dev_indices, dev_idata);

    cudaMemcpy(odata, dev_odata, array_size, cudaMemcpyDeviceToHost);

    // The kernel always copies the last elt.
    // Adjust the size to include it if desired.
    if (last == 1) {
        streamSize++;
    }

    return streamSize;
}

}
}
