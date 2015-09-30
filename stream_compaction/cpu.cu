#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
    
	
	//cuda event init
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	cudaEventRecord(start);

	odata[0] = 0;
	for (int i = 1; i<n; i++)
	{
		odata[i] = odata[i - 1] + idata[i - 1];
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "cpu method: " << milliseconds << "ms" << std::endl;
	
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
    // TODO
    
	//iterate over the indata
	int cur_index = 0;
	for (int i = 0; i < n; i++)
	{
		if (idata[i]!=0)
		{
			odata[cur_index++] = idata[i]; 
		}
	}

	return cur_index;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
    // TODO

	int* idata_map = new int[n];
	int* scan_sum = new int[n];
	for (int i = 0; i<n; i++)
	{
		idata_map[i] = (idata[i] == 0) ? 0 : 1;
	}
    
	scan(n, scan_sum, idata_map);
	int num_remain = scatter(n, odata, scan_sum, idata_map,idata);
	
	return num_remain;
}

int scatter(int n, int *odata, const int *scan_sum, const int *idata_map, const int *idata)
{
	int cur_num = 0;
	for (int i = 0; i<n; i++)
	{
		if (idata_map[i] == 1)
		{
			odata[scan_sum[i]] = idata[i];
			cur_num++;
		}	
	}

	return cur_num;

}

}
}
