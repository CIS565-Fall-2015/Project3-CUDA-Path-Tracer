#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {

cudaEvent_t start, stop;

static void setup_timer_events() {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
}

static float teardown_timer_events() {
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return milliseconds;
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // use `thrust::exclusive_scan`
    // example: for device_vectors dv_in and dv_out:
    // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

	// Create a thrust::device_vector from a thrust::host_vector
	thrust::host_vector<int> v_in(idata, idata + n);
	thrust::device_vector<int> device_v_in(v_in);
	thrust::device_vector<int> device_v_out(n);

	if (BENCHMARK) {
		setup_timer_events();
	}

	thrust::exclusive_scan(device_v_in.begin(), device_v_in.end(),
		device_v_out.begin());

	if (BENCHMARK) {
		printf("%f microseconds.\n",
			teardown_timer_events() * 1000.0f);
	}

	// copy back over
	for (int i = 0; i < n; i++) {
		odata[i] = device_v_out[i];
	}
}

}
}
