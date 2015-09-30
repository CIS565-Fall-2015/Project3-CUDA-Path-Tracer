#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "radixsort.h"
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
	namespace RadixSort {


		__global__ void kern_get_k_bit_array(int n, int k, int* odata, const int* idata)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (index < n)
			{
				odata[index] = (idata[index] & (1 << k)) >> k; //get the kth bit of the cur int
			}
		}

		__global__ void kern_inv_array(int n, int* odata, const int* idata) //1-->0  0-->1
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (index < n)
			{
				odata[index] = std::abs(idata[index]-1);
			}
		}

		__global__ void kern_get_totalFalses(int n, int* e, int *f, int* totalFalse)
		{
			*totalFalse = e[n - 1] + f[n - 1];
		}

		__global__ void kern_compute_t_array(int n,int * f,int *t,int* totalFalse)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (index < n)
			{
				t[index] = index - f[index]+ *totalFalse;
			}
		}

		__global__ void kern_compute_the_d_array(int n, int *b, int *t, int *f, int *d)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (index < n)
			{
				d[index] = b[index] ? t[index] : f[index];
			}
		}

		__global__ void kern_get_output(int n,int * d,int * odata,int* idata)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (index < n)
			{
				odata[d[index]] = idata[index];
			}
		}




		void radixsort(int n, int *odata, const int *idata) //assume that all the bits are 32 bits
		{
			
			
			
			dim3 threadsPerBlock(blockSize);
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			
			int * dev_b_array;
			cudaMalloc((void**)&dev_b_array, n*sizeof(int));
			int * dev_e_array;
			cudaMalloc((void**)&dev_e_array, n*sizeof(int));
			int * dev_f_array;
			cudaMalloc((void**)&dev_f_array, n*sizeof(int));
			int * dev_t_array;
			cudaMalloc((void**)&dev_t_array, n*sizeof(int));
			int * dev_d_array;
			cudaMalloc((void**)&dev_d_array, n*sizeof(int));

			int * dev_idata;
			cudaMalloc((void**)&dev_idata, n*sizeof(int));

			int * dev_odata;
			cudaMalloc((void**)&dev_odata, n*sizeof(int));

			
			cudaMemcpy(dev_idata,idata,n*sizeof(int),cudaMemcpyHostToDevice);


			int* dev_totalFalse;
			cudaMalloc((void**)&dev_totalFalse, 1*sizeof(int));


			int* host_f_array = new int[n];
			
			int* host_e_array = new int[n];
			
			
			//cuda event init
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			float milliseconds = 0;

			cudaEventRecord(start);
			
			for (int k = 0; k<32; k++)
			{
				//get b array
				kern_get_k_bit_array << <fullBlocksPerGrid, threadsPerBlock >> > (n, k, dev_b_array,dev_idata);

				//get e array
				kern_inv_array << <fullBlocksPerGrid, threadsPerBlock >> >(n, dev_e_array, dev_b_array);

				//get f data
				
				cudaMemcpy(host_e_array, dev_e_array, n*sizeof(int), cudaMemcpyDeviceToHost);
				StreamCompaction::Thrust::scan(n, host_f_array, host_e_array); 
				cudaMemcpy(dev_f_array, host_f_array, n*sizeof(int), cudaMemcpyHostToDevice);

				//get t array
				//comptue the totalFalse
				kern_get_totalFalses << <1, 1 >> >(n, dev_e_array, dev_f_array, dev_totalFalse);

				kern_compute_t_array << < fullBlocksPerGrid, threadsPerBlock >> > (n, dev_f_array, dev_t_array, dev_totalFalse);

				//get the d array 
				kern_compute_the_d_array << < fullBlocksPerGrid, threadsPerBlock >> > (n, dev_b_array, dev_t_array, dev_f_array, dev_d_array);

				//get the current output
				kern_get_output << < fullBlocksPerGrid, threadsPerBlock >> > (n, dev_d_array, dev_odata,dev_idata);

				//update the idata
				cudaMemcpy(dev_idata, dev_odata, n*sizeof(int), cudaMemcpyDeviceToDevice);

			}

			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << "radix sort method: " << milliseconds << "ms" << std::endl;

			cudaMemcpy(odata, dev_odata, n*sizeof(int), cudaMemcpyDeviceToHost);

		}

	
	}
}