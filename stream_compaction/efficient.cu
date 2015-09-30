#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

#define blockSize 1024
int *temp_scan;
int *scan_result;
Ray *rays;

__global__ void upSweep(int n, int d, int *o_data, int *i_data) {
	int index =  (blockIdx.x * blockDim.x) + threadIdx.x;	
	if (index <= n) {
		if (index % (int)pow(2.0, d+1) == 0) {
			o_data[index-1] = (int)i_data[index - 1 - (int)pow(2.0, d)] + (int)i_data[index - 1];
		} 
	}
}

__global__ void downSweep(int n, int d, int *o_data, int *i_data) {
	int index =  (blockIdx.x * blockDim.x) + threadIdx.x;
	int temp = 0;
	if (index <= n) {
		if (index % (int)pow(2.0, d+1) == 0) {
			temp = i_data[index - 1 - (int)pow(2.0, d)];
			o_data[index - 1 - (int)pow(2.0, d)] = i_data[index-1];
			o_data[index-1] = temp + i_data[index - 1];
		} 
	}

}

__global__ void rayToInt(int n, int *o_data, Ray *i_data) {
	int index =  (blockIdx.x * blockDim.x) + threadIdx.x;
	int temp = 0;
	if (index <= n) {
		o_data[index] = (int) i_data[index].isAlive;
	}
}

void scan(int n, int *odata, const int *idata) {
    int d = ilog2ceil(n);
	int total = (int) pow(2.0, d);

	cudaMalloc((void**)&scan_result, total * sizeof(int));
	cudaMalloc((void**)&temp_scan, total * sizeof(int));
	cudaMemcpy(scan_result, idata, total * sizeof(int), cudaMemcpyHostToDevice);

	dim3 fullBlocksPerGrid((total + blockSize - 1) / blockSize);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	for (int i = 0; i < d; i++) {
		upSweep<<<fullBlocksPerGrid, blockSize>>>(total, i, scan_result, temp_scan);
		temp_scan = scan_result;
	}

	
	scan_result[total-1] = 0;
	cudaMemcpy(odata, scan_result, total * sizeof(int), cudaMemcpyDeviceToHost);
	

	cudaMemcpy(scan_result, odata, total * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(temp_scan, odata, total * sizeof(int), cudaMemcpyHostToDevice);

	for (int i = d-1; i >= 0; i--) {
		downSweep<<<fullBlocksPerGrid, blockSize>>>(total, i, scan_result, temp_scan);
		temp_scan = scan_result;
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f milliseconds for efficient \n", milliseconds);

	cudaMemcpy(odata, scan_result, total * sizeof(int), cudaMemcpyDeviceToHost);
	printf("odata[n-1] %d \n", odata[total-1]);
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
int compact(int n, Ray *odata, const Ray *idata) {
    
	int d = ilog2ceil(n);
	int total = (int) pow(2.0, d);

	int *predicate_array;
	int *hst_predicate_array;
	Ray *dev_idata;
	Ray *compacted_rays;
	int *compact_array;

	int *hst_indices;
	int *dev_indices;

	cudaMalloc((void**)&predicate_array, total * sizeof(int));
	cudaMalloc((void**)&hst_predicate_array, total * sizeof(int));
	cudaMalloc((void**)&dev_idata, total * sizeof(Ray));
	cudaMalloc((void**)&compact_array, total * sizeof(int));
	cudaMalloc((void**)&hst_indices, total * sizeof(int));
	cudaMalloc((void**)&dev_indices, total * sizeof(int));

	cudaMemcpy(dev_idata, idata, total * sizeof(Ray), cudaMemcpyHostToDevice);

	dim3 fullBlocksPerGrid((total + blockSize - 1) / blockSize);

	Common::kernMapRayToBoolean<<<fullBlocksPerGrid, blockSize>>>(total, predicate_array, 
		dev_idata);

	cudaMemcpy(hst_predicate_array, predicate_array, 
		total * sizeof(int), cudaMemcpyDeviceToHost);
	
	scan(total, hst_indices, hst_predicate_array);
	cudaMemcpy(dev_indices, hst_indices, total * sizeof(int), cudaMemcpyHostToDevice);

	int totalAfterCompaction = hst_indices[total-1];
	
	cudaMalloc((void**)&compacted_rays, totalAfterCompaction * sizeof(Ray));
	Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(total, odata,
        dev_idata, predicate_array, dev_indices);

    return totalAfterCompaction;
}


}
}
