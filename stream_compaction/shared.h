#pragma once

#include <src/sceneStructs.h>

namespace StreamCompaction {
namespace Shared {
    __global__ void kernMapToBoolean(int n, int valid, int *bools, Pixel *idata);

    __global__ void kernScatter(int n, Pixel *odata, int *indices, Pixel *idata);

    void dv_scan(int n, int *odata);

    int compact(int n, Pixel *input);
}
}
