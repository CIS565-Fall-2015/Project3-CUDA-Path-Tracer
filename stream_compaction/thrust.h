#pragma once
#include "../src/sceneStructs.h"

namespace StreamCompaction {
namespace Thrust {
	int compact(int n, Ray *data);
    void scan(int n, int *odata, const int *idata);
}
}
