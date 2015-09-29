#pragma once

#include <cstdio>
#include <cstring>
#include <cmath>

__device__ __host__ inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

__device__ __host__ inline int ilog2ceil(int x) {
    return ilog2(x - 1) + 1;
}
