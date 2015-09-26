#pragma once
#include <src/sceneStructs.h>

__host__ __device__ inline int ilog2(int x) {
	int lg = 0;
	while (x >>= 1) {
		++lg;
	}
	return lg;
}

__host__ __device__ inline int ilog2ceil(int x) {
	return ilog2(x - 1) + 1;
}

namespace StreamCompaction {
namespace Efficient {
    void scan(int n, int *odata, const int *idata);

    void compact(int n, int *f, int *idx, PathRay *dv_out_tmp, PathRay *idata, int *c);
}
}
