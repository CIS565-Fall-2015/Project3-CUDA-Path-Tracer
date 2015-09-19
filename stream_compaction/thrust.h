#pragma once
#include "../src/sceneStructs.h"

namespace StreamCompaction {
namespace Thrust {
	Ray *compact(Ray *data, Ray* data_end);
    void scan(int n, int *odata, const int *idata);
}
}
