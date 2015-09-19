#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {

struct is_terminated {
	__host__ __device__
	bool operator()(const Ray ray) {
		return !ray.alive;
	}
};

/*
int compact(int n, Ray *data) {
	thrust::host_vector<Ray> hst_in(data, data + n);
	thrust::device_vector<Ray> dev_data = hst_in;

	thrust::device_vector<Ray>::iterator dev_data_end = thrust::remove_if(dev_data.begin(), dev_data.end(), is_terminated());
	dev_data.resize(dev_data_end - dev_data.begin()); // the output is still the old size though

	thrust::host_vector<Ray> hst_out = dev_data;

	//Free old odata, allocate correct new size
	int num_remaining = hst_out.size();
	free(data);
	data = (Ray*)malloc(num_remaining * sizeof(Ray));
	for (int i = 0; i < num_remaining; i++) {
		data[i] = hst_out[i];
	}

	return num_remaining;
}
*/

/**
* Performs stream compaction on idata, removing terminated rays, returns number remaining.
*/
int compact(int n, Ray *data) {
	thrust::device_vector<Ray> dev_data(data, data + (n * sizeof(Ray))); //WARNING: TRIPLE CHECK THIS IS RIGHT SIZE

	thrust::device_vector<Ray>::iterator dev_data_end = thrust::remove_if(dev_data.begin(), dev_data.end(), is_terminated());
	dev_data.resize(dev_data_end - dev_data.begin()); // the output is still the old size though

	thrust::host_vector<Ray> hst_out = dev_data;

	//Free old odata, allocate correct new size
	int num_remaining = hst_out.size();
	free(data);
	data = (Ray*)malloc(num_remaining * sizeof(Ray));
	for (int i = 0; i < num_remaining; i++) {
		data[i] = hst_out[i];
	}

	return num_remaining;
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	thrust::host_vector<int> hst_in(idata, idata + n);
	thrust::device_vector<int> dev_in = hst_in;
	thrust::device_vector<int> dev_out(n);

	thrust::exclusive_scan(dev_in.begin(), dev_in.end(), dev_out.begin());
	thrust::host_vector<int> hst_out = dev_out;

	for (int i = 0; i < n; i++) {
		odata[i] = hst_out[i];
	}
}

}
}
