#ifndef __EFFICIENT_H__
#define __EFFICIENT_H__

#include <src/sceneStructs.h>

namespace StreamCompaction {
namespace Efficient {

    int compact(int n, RayState *idata);//, RayState *odata);
}
}

inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(int x) {
    return ilog2(x - 1) + 1;
}


namespace StreamCompaction {
namespace Common {
    __global__ void kernMapToBoolean(int n, int *bools, const RayState *idata);

    __global__ void kernScatter(int n, RayState *odata,
            const RayState *idata, const int *bools, const int *indices);
}
}

#endif
