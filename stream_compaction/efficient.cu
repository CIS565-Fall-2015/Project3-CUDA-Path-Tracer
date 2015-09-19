#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

__global__ void up_sweep(int n, int d, int *data) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);

	if (k < n) {
		int p2d = pow(2.0, (double)d);
		int p2da1 = pow(2.0, (double)(d + 1));

		if (k % p2da1 == 0) {
			data[k + p2da1 - 1] += data[k + p2d - 1];
		}
	}	
}

__global__ void down_sweep(int n, int d, int *data) {
	int k = threadIdx.x + (blockIdx.x * blockDim.x);

	if (k < n) {
		int p2d = pow(2.0, (double)d);
		int p2da1 = pow(2.0, (double)(d + 1));

		if (k % p2da1 == 0) {
			int temp = data[k + p2d - 1];
			data[k + p2d - 1] = data[k + p2da1 - 1];
			data[k + p2da1 - 1] += temp;
		}
	}
}

void padArrayRange(int start, int end, int *a) {
	for (int i = start; i < end; i++) {
		a[i] = 0;
	}
}
/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
float scan(int n, int *odata, const int *idata) {
	int m = pow(2, ilog2ceil(n));
	int *new_idata = (int*)malloc(m * sizeof(int));
	dim3 fullBlocksPerGrid((m + blockSize - 1) / blockSize);
	dim3 threadsPerBlock(blockSize);

	cudaEvent_t start, stop;
	float ms_time = 0.0f;
	float ms_total_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Expand array to next power of 2 size
	for (int i = 0; i < n; i++) {
		new_idata[i] = idata[i];
	}
	padArrayRange(n, m, new_idata);

	// Can use one array for input and output in this implementation
	int *dev_data;
	cudaMalloc((void**)&dev_data, m * sizeof(int));
	cudaMemcpy(dev_data, new_idata, m * sizeof(int), cudaMemcpyHostToDevice);

	// Execute scan on device
	cudaEventRecord(start);
	for (int d = 0; d < ilog2ceil(n); d++) {
		up_sweep<<<fullBlocksPerGrid, threadsPerBlock>>>(n, d, dev_data);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms_time, start, stop);
	ms_total_time += ms_time;
	ms_time = 0.0f;

	cudaMemset((void*)&dev_data[m - 1], 0, sizeof(int));
	cudaEventRecord(start);
	for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
		down_sweep<<<fullBlocksPerGrid, threadsPerBlock>>>(n, d, dev_data);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms_time, start, stop);
	ms_total_time += ms_time;

	cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_data);
	free(new_idata);

	return ms_total_time;
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
	int *bools = (int*)malloc(n * sizeof(int));
	int *scan_data = (int*)malloc(n * sizeof(int));
	int num_remaining = -1;
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
	dim3 threadsPerBlock(blockSize);

	cudaEvent_t start, stop;
	float ms_time = 0.0f;
	float ms_total_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int *dev_bools;
	int *dev_idata;
	int *dev_odata;
	int *dev_scan_data;

	cudaMalloc((void**)&dev_bools, n * sizeof(int));
	cudaMalloc((void**)&dev_idata, n * sizeof(int));
	cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_odata, n * sizeof(int));
	cudaMalloc((void**)&dev_scan_data, n * sizeof(int));

	// Map to boolean
	cudaEventRecord(start);
	StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGrid, threadsPerBlock>>>(n, dev_bools, dev_idata);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms_time, start, stop);
	ms_total_time += ms_time;
	ms_time = 0.0f;

	cudaMemcpy(bools, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);

	// Execute the scan
	ms_total_time += scan(n, scan_data, bools);
	num_remaining = scan_data[n - 1] + bools[n - 1];

	// Execute the scatter
	cudaMemcpy(dev_scan_data, scan_data, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	StreamCompaction::Common::kernScatter<<<fullBlocksPerGrid, threadsPerBlock>>>(n, dev_odata, dev_idata, dev_bools, dev_scan_data);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms_time, start, stop);
	ms_total_time += ms_time;
	printf("CUDA execution time for stream compaction: %.5fms\n", ms_total_time);

	cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_bools);
	cudaFree(dev_idata);
	cudaFree(dev_odata);
	cudaFree(dev_scan_data);
	free(bools);
	free(scan_data);

	return num_remaining;
}

}
}
