#pragma once

namespace StreamCompaction {
namespace Shared {
    __global__ void kernMapToBoolean(int n, int *bools, const int *idata);

    __global__ void kernScatter(int n, int *odata, int *indices, int *idata);

    void dv_scan(int n, int *odata);
    void scan(int n, int *odata, int *idata);

    int compact(int n, int *odata, const int *idata);
}
}
