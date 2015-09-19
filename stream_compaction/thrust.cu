#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
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

/**
* Performs stream compaction on array of Rays, removing those that are terminated.
*/
Ray *compact(Ray *data, Ray *data_end) {
	return thrust::remove_if(thrust::device, data, data_end, is_terminated());
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
