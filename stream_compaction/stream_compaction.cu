#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "stream_compaction.h"


namespace StreamCompaction {
	namespace Efficient {
		//const int blockSize = 128;



		__global__ void kernUpSweep(int size, int step, int * data)
		{
			//step = 2^(d+1)
			int k = threadIdx.x + blockDim.x * blockIdx.x;

			if (k < size)
			{
				if (k % step == 0)
				{
					data[k + step - 1] += data[k + (step >> 1) - 1];
				}
			}

		}

		__global__ void kernDownSweep(int size, int step, int * data)
		{
			//step = 2^(d+1)
			int k = threadIdx.x + blockDim.x * blockIdx.x;

			if (k < size)
			{
				if (k % step == 0)
				{
					int left_child = data[k + (step >> 1) - 1];
					data[k + (step >> 1) - 1] = data[k + step - 1];
					data[k + step - 1] += left_child;
				}
			}
		}


		__global__ void kernSetRootZero(int rootId, int * data)
		{
			int k = threadIdx.x + blockDim.x * blockIdx.x;
			if (k == rootId)
			{
				data[k] = 0;
			}
		}






		//combine upsweep, downsweep ... -> one kernel function
		//one block
		//g_odata can = g_idata
		__global__ void kernScan(int N , int * g_odata, const int *g_idata)
		{
			extern __shared__ int s_idata[];

			int n = blockDim.x * 2;		//data size
			int blockOffset = n * blockIdx.x;
			int thid = threadIdx.x;	// +blockDim.x * blockIdx.x;

			//if (thid + blockOffset < N)
			//{
				
			//int n_data = n * 2;
			int offset = 1;

			int ai = thid;
			int bi = thid + (n / 2);
			int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
			int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

			//copy data from global to shared
			s_idata[ai + bankOffsetA] = g_idata[ai + blockOffset];
			s_idata[bi + bankOffsetB] = g_idata[bi + blockOffset];

			//if (ai + blockOffset >= N)
			//{
			//	s_idata[ai + bankOffsetA] = 0;
			//}
			//else
			//{
			//	s_idata[ai + bankOffsetA] = g_idata[ai + blockOffset];
			//}

			//if (bi + blockOffset >= N)
			//{
			//	s_idata[bi + bankOffsetB] = 0;
			//}
			//else
			//{
			//	s_idata[bi + bankOffsetB] = g_idata[bi + blockOffset];
			//}




			//UpSweep
			for (int d = n >> 1; d > 0; d >>= 1)
			{
				__syncthreads();
				if (thid < d)
				{
					int ai = offset * (2 * thid + 1) - 1;
					int bi = offset * (2 * thid + 2) - 1;
					ai += CONFLICT_FREE_OFFSET(ai);
					bi += CONFLICT_FREE_OFFSET(bi);

					s_idata[bi] += s_idata[ai];
				}
				offset <<= 1; // * 2
			}

			
			//assign 0 to the root
			if (thid == 0)
			{
				s_idata[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
			}



			//DownSweep
			for (int d = 1; d < n; d <<= 1)
			{
				offset >>= 1;
				__syncthreads();
				if (thid < d)
				{
					int ai = offset * (2 * thid + 1) - 1;
					int bi = offset * (2 * thid + 2) - 1;
					ai += CONFLICT_FREE_OFFSET(ai);
					bi += CONFLICT_FREE_OFFSET(bi);

					int t = s_idata[ai];
					s_idata[ai] = s_idata[bi];
					s_idata[bi] += t;
				}
			}

			g_odata[ai + blockOffset] = s_idata[ai + bankOffsetA];
			g_odata[bi + blockOffset] = s_idata[bi + bankOffsetB];
			//}

			//test
			//printf("%d : %d\n", ai + blockOffset, s_idata[ai + bankOffsetA]);
			//printf("%d : %d\n", bi + blockOffset, s_idata[bi + bankOffsetB]);
		}




		__global__ void kernCopyToBlockSumArray(int N,int blockArraySize, int * g_sum ,const int * g_bools,const int * g_scans)
		{
			//turn from exclusive result to inclusive
			int k = threadIdx.x + blockDim.x * blockIdx.x;

			if (k < N)
			{
				int this_block_last_id = (k)*blockArraySize + blockArraySize - 1;
				g_sum[k] = g_scans[this_block_last_id] + g_bools[this_block_last_id];
			}
			else
			{
				g_sum[k] = 0;
			}
		}


		__global__ void kernAddSumArrayBack(int N,int blockArraySize, int * g_odata, const int * g_sum)
		{
			int k = threadIdx.x + blockDim.x * blockIdx.x;
			if (k < N)
			{

				//all element after the first block array
				
				g_odata[k] += g_sum[(k / blockArraySize)];
				
			}
			
		}



		__global__ void kernAssignBool(int N, int * istrues, Path* paths)
		{
			int k = threadIdx.x + blockDim.x * blockIdx.x;
			if (k < N)
			{
				istrues[k] = (int)(!paths[k].terminated);
			}
			else
			{
				istrues[k] = 0;
			}
		}


		__global__ void kernScatter(int n, Path *odata,
			const Path *idata, const int *bools, const int *indices)
		{
			int k = threadIdx.x + blockDim.x * blockIdx.x;
			if (k < n)
			{
				if (bools[k] == 1)
				{
					odata[indices[k]] = idata[k];
				}
			}
		}




		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		* input g_idata is on global memory on device
		*/

		



		/**
		* size - size of the bools, has already been extended to 2^c
		* scans, bools - device memory, memory has been allocated
		*/
		void scan(int size, int * scans, const int * bools)
		{
			dim3 fullBlocksPerGrid_2((size / 2 + blockSize - 1) / blockSize);
			dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);


			if (fullBlocksPerGrid_2.x == 1)
			{
				// one block scan
				kernScan << <fullBlocksPerGrid_2, blockSize, 2 * blockSize*sizeof(int) >> >(size, scans, bools);
				cudaDeviceSynchronize();
				return;
			}


			//multiblock scan
			int * block_sum_array;
			int * block_sum_array_bools;

			//per block scan
			kernScan << <fullBlocksPerGrid_2, blockSize, 2 * blockSize*sizeof(int) >> >(size, scans, bools);
			
			//cudaDeviceSynchronize();


			//int ceil_log2n_sum = ilog2ceil(fullBlocksPerGrid.x);
			//int size_sum = 1 << ceil_log2n_sum;
			int size_sum = fullBlocksPerGrid_2.x;

			dim3 numBlocks_2((size_sum / 2 + blockSize - 1) / blockSize);
			dim3 numBlocks((size_sum + blockSize - 1) / blockSize);

			cudaMalloc(&block_sum_array, size_sum * sizeof(int));

			cudaDeviceSynchronize();
			kernCopyToBlockSumArray << <numBlocks, blockSize >> >(size_sum, 2 * blockSize, block_sum_array, bools, scans);
			cudaDeviceSynchronize();


			cudaMalloc(&block_sum_array_bools, size_sum * sizeof(int));
			cudaMemcpy(block_sum_array_bools, block_sum_array, size_sum * sizeof(int), cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			//if (numBlocks_2.x == 1)
			//{
			scan(size_sum, block_sum_array, block_sum_array_bools);
			//}
			cudaDeviceSynchronize();
			

			kernAddSumArrayBack << <fullBlocksPerGrid, blockSize >> >(size, 2 * blockSize, scans, block_sum_array);
			cudaDeviceSynchronize();


			cudaFree(block_sum_array);
			cudaFree(block_sum_array_bools);

			cudaDeviceSynchronize();
		}




		//multiple blocks, shared memory, resolve bank conflict
		int compact(int n, Path* path)
		{
			//int * block_sum_array;	//global
			int * exist;
			int * scans;
			Path * tmp_path;

			cudaMalloc(&tmp_path, n * sizeof(Path));
			cudaMemcpy(tmp_path, path, n*sizeof(Path), cudaMemcpyDeviceToDevice);

			int ceil_log2n = ilog2ceil(n);
			int size = 1 << ceil_log2n;

			dim3 fullBlocksPerGrid_2((size / 2 + blockSize - 1) / blockSize);
			dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);

			cudaMalloc(&exist, size*sizeof(int));
			kernAssignBool << <fullBlocksPerGrid, blockSize >> >(n, exist, path);
			cudaMalloc(&scans, size*sizeof(int));

			

			

			scan(size, scans, exist);


			//compact

			

			kernScatter << <fullBlocksPerGrid, blockSize >> >(size, path,
				tmp_path, exist, scans);
			cudaDeviceSynchronize();

			int hos_sum;
			cudaMemcpy(&hos_sum, scans + size - 1, sizeof(int), cudaMemcpyDeviceToHost);

			int hos_last;
			cudaMemcpy(&hos_last, exist + size - 1, sizeof(int), cudaMemcpyDeviceToHost);


			cudaFree(tmp_path);
			//cudaFree(block_sum_array);
			cudaFree(exist);
			cudaFree(scans);

			return hos_sum + hos_last;
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

		//int compact(int n, int *odata, const int *idata) {
		//	int hos_scans;
		//	int hos_bools;
		//	int * dev_bools;
		//	int * dev_scans;
		//	int * dev_idata;
		//	int * dev_odata;
		//	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

		//	cudaMalloc((void**)&dev_bools, n * sizeof(int));
		//	//checkCUDAError("cudaMalloc dev_bools failed");
		//	cudaMalloc((void**)&dev_scans, n * sizeof(int));
		//	//checkCUDAError("cudaMalloc dev_scans failed");
		//	cudaMalloc((void**)&dev_idata, n * sizeof(int));
		//	//checkCUDAError("cudaMalloc dev_idata failed");
		//	cudaMalloc((void**)&dev_odata, n * sizeof(int));
		//	//checkCUDAError("cudaMalloc dev_odata failed");

		//	cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
		//	//checkCUDAError("cudaMemcpy from data to dev_data failed");
		//	cudaDeviceSynchronize();

		//	Common::kernMapToBoolean << < fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);
		//	cudaDeviceSynchronize();

		//	//cudaMemcpy(hos_bools,dev_bools, n * sizeof(int),cudaMemcpyDeviceToHost);
		//	//checkCUDAError("cudaMemcpy from data to dev_data failed");
		//	//cudaDeviceSynchronize();

		//	scan(n, dev_scans, dev_bools, true);

		//	//cudaMemcpy(dev_scans,hos_scans, n * sizeof(int),cudaMemcpyHostToDevice);
		//	//checkCUDAError("cudaMemcpy from hos_scans to dev_scans failed");
		//	//cudaDeviceSynchronize();

		//	Common::kernScatter << < fullBlocksPerGrid, blockSize >> >(n, dev_odata,
		//		dev_idata, dev_bools, dev_scans);
		//	cudaDeviceSynchronize();

		//	cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
		//	//checkCUDAError("cudaMemcpy from dev_odata to odata failed");
		//	//cudaDeviceSynchronize();

		//	cudaMemcpy(&hos_scans, dev_scans + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
		//	//checkCUDAError("cudaMemcpy scans[n-1] failed");

		//	cudaMemcpy(&hos_bools, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
		//	//checkCUDAError("cudaMemcpy bools[n-1] failed");

		//	cudaDeviceSynchronize();



		//	cudaFree(dev_idata);
		//	cudaFree(dev_odata);
		//	cudaFree(dev_bools);
		//	cudaFree(dev_scans);

		//	//int num = hos_scans[n-1] + hos_bools[n-1];
		//	int num = hos_scans + hos_bools;
		//	//delete[] hos_scans;
		//	//delete[] hos_bools;

		//	return num;
		//}



		

		/**
		* CPU stream compaction without using the scan function.
		*
		* @returns the number of elements remaining after compaction.
		* odata can = idata, no race condition
		*/
		int compactWithoutScan(int n, Path *odata, const Path *idata) {
			int r = 0;
			for (int i = 0; i < n; i++)
			{
				if (!idata[i].terminated)
				{
					odata[r] = idata[i];
					r++;
				}
			}
			return r;
		}


		int cmpArrays(int n, Path *a, Path *b) {
			int r = 0;
			for (int i = 0; i < n; i++) {
				if (a[i].image_index != b[i].image_index) {
					printf("    a[%d] = %d, b[%d] = %d\n", i, (int)a[i].image_index, i, (int)b[i].image_index);
					//return 1;
					r = 1;
				}
			}
			return r;
		}
		


		void printArray(int n, Path *a, bool abridged) {
			printf("    [ ");
			for (int i = 0; i < n; i++) {
				if (abridged && i + 2 == 15 && n > 16) {
					i = n - 2;
					printf("... ");
				}
				//printf("%d ", (int)a[i].terminated);
				printf("%d ", a[i].image_index);
			}
			printf("]\n");
		}






	}
}
