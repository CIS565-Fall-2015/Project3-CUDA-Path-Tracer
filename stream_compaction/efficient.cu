#include <cuda.h>
#include <cuda_runtime.h>
#include "efficient.h"
#include <iostream>

namespace StreamCompaction {
namespace Efficient {

const int threadCount = 256;

void printArray(int n, int * a)
{
	printf("\n");
	for(int i=0; i<n; ++i)
		printf("%d ", a[i]);
	printf("\n");
}

__global__ void setK(int * k, int * data, int index)
{
	(*k) = data[index];
}

__global__ void blockWiseScan(int n, int *odata, int *idata)
{
	//Reference-> http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html

	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if(index < n)
	{
		//Do block exclusive scans
		__shared__ int data[threadCount];

		unsigned int t = threadIdx.x;
		n = blockDim.x;

		data[t] = idata[index];
		int offset = 1;

		for (int d = n>>1; d > 0; d >>= 1)  // build sum in place up the tree
		{
			__syncthreads();
			if (t < d)
			{
				int ai = offset * ((t<<1)+1) - 1;
				int bi = offset * ((t<<1)+2) - 1;

				data[bi] += data[ai];
			}
			offset <<= 1;
		}

		if (t == 0) { data[n - 1] = 0; } // clear the last element

		for (int d = 1; d < n; d <<= 1) // traverse down tree & build scan
		{
		     offset >>= 1;
		     __syncthreads();

		     if (t < d)
		     {
		    	 int ai = offset * ((t<<1)+1) - 1;
		    	 int bi = offset * ((t<<1)+2) - 1;

		    	 float t = data[ai];
		    	 data[ai] = data[bi];
		    	 data[bi] += t;
		     }
		}

		odata[index] = data[t];
	}
}

__global__ void createTemp(int * odata, int *idata, int * temp, int numThreads)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	temp[index] = odata[(index+1) * numThreads - 1] + idata[(index+1) * numThreads - 1];
}

__global__ void updateidata(int n, int *odata, int *temp_data)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	odata[index] += temp_data[blockIdx.x];
}

void exclusiveScan(int n, int *odata, int *idata, int numBlocks, int numThreads)
{
	blockWiseScan<<<numBlocks, numThreads>>>(n, odata, idata);

	//Then we have to recurse and solve the odata array, So create a new array and solve.
	int *dev_temp,
		*dev_odata;
	int p = ilog2ceil(numBlocks);
	int	fullN = pow(2, p);

	cudaMalloc((void**)&dev_temp, fullN * sizeof(int));
	cudaMalloc((void**)&dev_odata, fullN * sizeof(int));

	cudaMemset(dev_temp, 0, fullN * sizeof(int));
	createTemp<<<1, numBlocks>>>(odata, idata, dev_temp, numThreads);

	int newN = numBlocks;
	int newNumBlocks = (numBlocks + numThreads -1) / numThreads;

	if(numBlocks > numThreads)
	{
		exclusiveScan(newN, dev_odata, dev_temp, newNumBlocks, numThreads);
	}

	else
	{
		blockWiseScan<<<newNumBlocks, numThreads>>>(newN, dev_odata, dev_temp);
	}


	updateidata<<<numBlocks, numThreads>>>(n, odata, dev_odata);
	cudaFree(dev_temp);
	cudaFree(dev_odata);
}

int compact(int n, RayState *odata, RayState *idata) {

	std::cout<<n<<std::endl;
	int oriN = n;

	int p = ilog2ceil(n);
	n = pow(2, p);
	std::cout<<n<<std::endl;

	int numThreads = threadCount,
		numBlocks = (n + numThreads - 1) / numThreads;

	int	*dev_k,
		*dev_scanData,
		*dev_temp;
	int *printData = new int[n];

	cudaMalloc((void**)&dev_k, sizeof(int));
	cudaMalloc((void**)&dev_scanData, n * sizeof(int));
	cudaMalloc((void**)&dev_temp, n * sizeof(int));


	StreamCompaction::Common::kernMapToBoolean<<<numBlocks, numThreads>>>(n, dev_scanData, idata);

	cudaMemcpy(printData, dev_scanData, n * sizeof(int), cudaMemcpyDeviceToHost);
	printArray(n, printData);

	exclusiveScan(n, dev_temp, dev_scanData, numBlocks, numThreads);

	cudaMemcpy(printData, dev_temp, n * sizeof(int), cudaMemcpyDeviceToHost);
	printArray(n, printData);

	setK<<<1,1>>>(dev_k, dev_temp, n-1);
	int *k = new int;
	cudaMemcpy(k, dev_k, sizeof(int), cudaMemcpyDeviceToHost);

	StreamCompaction::Common::kernScatter<<<numBlocks, numThreads>>>(n, odata, idata, dev_scanData, dev_temp);

	cudaMemcpy(idata, odata, oriN * sizeof(RayState), cudaMemcpyDeviceToDevice);

	std::cout<<*k<<std::endl;

	cudaFree(dev_scanData);
	cudaFree(dev_k);
	cudaFree(dev_temp);
	return (*k);
}


}
}


namespace StreamCompaction {
namespace Common {

__global__ void kernMapToBoolean(int n, int *bools, const RayState *idata) {

	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if(index < n)
	{
//		if(idata[index].isAlive)
//		{
//			printf("Here\n");
//		}
	bools[index] = (idata[index].isAlive) ? 1 : 0;
	}
}

/**
 * Performs scatter on an array. That is, for each element in idata,
 * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
 */
__global__ void kernScatter(int n, RayState *odata,
        const RayState *idata, const int *bools, const int *indices) {
    // TODO

	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if(index < n)
	{
		if(bools[index] == 1)
		{
			int i = indices[index];
			odata[i].isAlive = idata[index].isAlive;
			odata[i].pixelIndex = idata[index].pixelIndex;
			odata[i].rayColor = idata[index].rayColor;
			odata[i].ray.direction = idata[index].ray.direction;
			odata[i].ray.origin = idata[index].ray.origin;
		}
	}
}

}
}
