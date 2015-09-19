#pragma once
#include "../src/sceneStructs.h"

namespace StreamCompaction {
namespace Thrust {
	int compact(Ray n, Ray *odata, const Ray *idata);
    void scan(int n, int *odata, const int *idata);
}
}
