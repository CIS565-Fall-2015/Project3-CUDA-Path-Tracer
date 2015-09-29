#include <cuda.h>
#include <cuda_runtime.h>
#include "efficient.h"
#include <iostream>

#define DEBUG 0

namespace StreamCompaction {
namespace Efficient {

const int threadCount = 32;

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
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


void printArray(int n, int * a)
{
	printf("\n");
	for(int i=0; i<n; ++i)
		printf("%d ", a[i]);
	printf("\n");
}

__global__ void setK(int * k, int * data, int *bool_data, int index)
{
	(*k) = data[index] + bool_data[index];
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

__global__ void updateidata(int n, int *odata, int *temp_data, int numThreads)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	odata[index] += temp_data[(index / numThreads)];
}

void exclusiveScan(int n, int *odata, int *idata, int numBlocks, int numThreads)
{
	blockWiseScan<<<numBlocks, numThreads>>>(n, odata, idata);
	checkCUDAError("BlockWiseScan1");

	int *printData = new int[n];
//	if(DEBUG)
//	{
//		std::cout<<"\nblockWiseScan";
//		cudaMemcpy(printData, odata, n * sizeof(int), cudaMemcpyDeviceToHost);
//		printArray(n, printData);
//	}

	//Then we have to recurse and solve the odata array, So create a new array and solve.
	int *dev_temp,
		*dev_odata;
	int p = ilog2ceil(numBlocks);
	int	fullN = pow(2, p);

	cudaMalloc((void**)&dev_temp, fullN * sizeof(int));
	cudaMemset(dev_temp, 0, fullN * sizeof(int));

	int newN = numBlocks;
	int newNumBlocks = (numBlocks + numThreads - 1) / numThreads;

	createTemp<<<newNumBlocks, numThreads>>>(odata, idata, dev_temp, numThreads);
	checkCUDAError("createTemp");

//	if(DEBUG)
//	{
//		std::cout<<"\ncreateTemp";
//		cudaMemcpy(printData, dev_temp, newN * sizeof(int), cudaMemcpyDeviceToHost);
//		printArray(newN, printData);
//	}

	cudaMalloc((void**)&dev_odata, fullN * sizeof(int));

	if(numBlocks > numThreads)
	{
		exclusiveScan(newN, dev_odata, dev_temp, newNumBlocks, numThreads);
	}

	else
	{
		blockWiseScan<<<newNumBlocks, numThreads>>>(newN, dev_odata, dev_temp);
		checkCUDAError("BlockWiseScan2");
	}

	updateidata<<<numBlocks, numThreads>>>(n, odata, dev_odata, numThreads);
	checkCUDAError("updateidata");

//	if(DEBUG)
//	{
//		std::cout<<"\nupdate idata";
//		cudaMemcpy(printData, odata, n * sizeof(int), cudaMemcpyDeviceToHost);
//		printArray(n, printData);
//	}

	cudaFree(dev_temp);
	cudaFree(dev_odata);
	delete(printData);
}

int compact(int n, RayState *idata)
{

	RayState * hst_idata = new RayState[n];

	cudaMemcpy(hst_idata, idata, n * sizeof(RayState), cudaMemcpyDeviceToHost);

	int i, count = 0;
	for(i=0; i<n; ++i)
	{
		if(hst_idata[i].isAlive)
			count++;
	}

//	if(DEBUG)
//	{
//		std::cout<<"Count Alive: "<<count<<std::endl;
//	}

	int oriN = n;

	int p = ilog2ceil(n);
	n = pow(2, p);

	int numThreads = threadCount,
		numBlocks = (n + numThreads - 1) / numThreads;

	RayState *dev_odata;

	int	*dev_k = NULL,
		*dev_bool = NULL,
		*dev_temp = NULL;
	int *printData = new int[n];

	cudaMalloc((void**)&dev_k, sizeof(int));
	cudaMalloc((void**)&dev_bool, n * sizeof(int));
	cudaMalloc((void**)&dev_temp, n * sizeof(int));

	StreamCompaction::Common::kernMapToBoolean<<<numBlocks, numThreads>>>(oriN, dev_bool, idata);
	checkCUDAError("kernMapToBool");

//	if(DEBUG)
//	{
//		std::cout<<"\nBools : ";
//		cudaMemcpy(printData, dev_bool, n * sizeof(int), cudaMemcpyDeviceToHost);
//		printArray(n, printData);
//	}

	exclusiveScan(n, dev_temp, dev_bool, numBlocks, numThreads);
	checkCUDAError("Exclusive Scan");

	setK<<<1,1>>>(dev_k, dev_temp, dev_bool, n-1);
	int *k = new int;
	cudaMemcpy(k, dev_k, sizeof(int), cudaMemcpyDeviceToHost);

	cudaMalloc((void**)&dev_odata, (*k) * sizeof(RayState));
	StreamCompaction::Common::kernScatter<<<numBlocks, numThreads>>>(n, dev_odata, idata, dev_bool, dev_temp);
	checkCUDAError("kernScatter");

	cudaMemcpy(idata, dev_odata, (*k) * sizeof(RayState), cudaMemcpyDeviceToDevice);

//	if(DEBUG)
//	{
//		std::cout<<"K :"<<*k<<std::endl;
//	}

	cudaFree(dev_bool);
	cudaFree(dev_k);
	cudaFree(dev_temp);
	cudaFree(dev_odata);
	delete(printData);
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
		bools[index] = (idata[index].isAlive) ? 1 : 0;
	}
	else
	{
		bools[index] = 0;
	}
}

/**
 * Performs scatter on an array. That is, for each element in idata,
 * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
 */
__global__ void kernScatter(int n, RayState *odata,
        const RayState *idata, const int *bools, const int *indices) {

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
