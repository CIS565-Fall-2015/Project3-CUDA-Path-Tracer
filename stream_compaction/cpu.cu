#include <cstdio>
#include <cstdlib>
#include "cpu.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
float scan(int n, int *odata, const int *idata) {
	cudaEvent_t start, stop;
	float ms_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	odata[0] = 0;
	for (int i = 1; i < n; i++) {
		odata[i] = odata[i - 1] + idata[i - 1];
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms_time, start, stop);
	return ms_time;
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	cudaEvent_t start, stop;
	float ms_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	int j = 0;
	for (int i = 0; i < n; i++) {
		if (idata[i] != 0) {
			odata[j] = idata[i];
			j++;
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms_time, start, stop);
	printf("CPU execution time for compact without scan: %.5fms\n", ms_time);

    return j;
}

void zeroArray(int n, int *a) {
	for (int i = 0; i < n; i++) {
		a[i] = 0;
	}
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
	int *temp = (int*)malloc(n * sizeof(int));
	zeroArray(n, temp);
	int *scan_output = (int*)malloc(n * sizeof(int));
	zeroArray(n, scan_output);

	cudaEvent_t start, stop;
	float ms_time = 0.0f;
	float ms_total_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Compute temporary array
	for (int i = 0; i < n; i++) {
		if (idata[i] != 0) {
			temp[i] = 1;
		}
	}

	// Run exclusive scan on the temporary array
	ms_time = scan(n, scan_output, temp);
	ms_total_time += ms_time;
	ms_time = 0.0f;

	// Scatter
	cudaEventCreate(&start);
	for (int i = 0; i < n; i++) {
		if (temp[i] == 1) {
			odata[scan_output[i]] = idata[i];
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms_time, start, stop);
	ms_total_time += ms_time;
	printf("CPU execution time for compact with scan: %.5fms\n", ms_total_time);

	return scan_output[n - 1] + temp[n - 1];
}

}
}
