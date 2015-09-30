#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "common.h"
#include "naive.h"




int* dev_array1;
int* dev_array2;


namespace StreamCompaction {
namespace Naive {

	__global__ void kern_naive_scan(int d, int n, int m_power,int* array_1, int* array_2)
	{
		int index = threadIdx.x + (blockIdx.x * blockDim.x);

		if (index >= m_power && index < n)
		{
			array_2[index] = array_1[index] + array_1[index - m_power];
		}

	}
/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	//compute the size of the intermediate array
	int m_power = ilog2ceil(n);
	int new_n = pow(2,m_power);

	dim3 fullBlocksPerGrid((new_n + blockSize - 1) / blockSize);
	dim3 threadsPerBlock(blockSize);

	//init the array
	cudaMalloc((void**)&dev_array1, new_n * sizeof(int));
	checkCUDAErrorFn("cudaMalloc dev_array1 failed!");

	cudaMalloc((void**)&dev_array2, new_n * sizeof(int));
	checkCUDAErrorFn("cudaMalloc dev_array1 failed!");

	cudaMemset(dev_array1, 0, new_n * sizeof(int));
	checkCUDAErrorFn("cudaMemset dev_array1 failed!");
	cudaMemset(dev_array2, 0, new_n * sizeof(int));
	checkCUDAErrorFn("cudaMemset dev_array2 failed!");

	int* tmp_data = new int[n];
	tmp_data[0] = 0;
	
	for (int i = 1; i<n; i++)
	{
		tmp_data[i] = idata[i-1];
	}
	
	cudaMemcpy(dev_array1, tmp_data, n*sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAErrorFn("cudaMemcpy dev_array1 failed!");
	
	//cuda event init
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	
	cudaEventRecord(start);
	//invoke the kernel function m_power times
	for (int d = 1; d<=m_power; d++)
	{
		int m_power = pow(2, d-1);
		kern_naive_scan << <fullBlocksPerGrid, threadsPerBlock >> > (d, new_n, m_power, dev_array1, dev_array2);
		//copy array2 to array1
		cudaMemcpy(dev_array1,dev_array2,new_n*sizeof(int),cudaMemcpyDeviceToDevice);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "naive method: " << milliseconds << "ms"<<std::endl;

	cudaMemcpy(odata, dev_array2, n*sizeof(int), cudaMemcpyDeviceToHost);
	
}

}
}
