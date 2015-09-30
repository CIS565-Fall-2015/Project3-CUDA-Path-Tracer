#pragma once
#include "../src/sceneStructs.h"

namespace StreamCompaction {
namespace Efficient {
    void scan(int n, int *odata, const int *idata);
    int compact(int n, Ray *odata, Ray *idata);
}
}
