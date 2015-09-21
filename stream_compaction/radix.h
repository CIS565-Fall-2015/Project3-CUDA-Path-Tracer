#pragma once

#include <cstdio>
#include <cstring>
#include <cmath>

#define MAXTHREADS 1024

namespace StreamCompaction {
namespace Radix {
	void split(int n, int bit, int* odata, int* idata);
	void sort(int n, int* odata, const int* idata);
}
}
