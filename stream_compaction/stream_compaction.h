#pragma once

#include <src/sceneStructs.h>

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4 
//#define CONFLICT_FREE_OFFSET(n) \
//	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

#define CONFLICT_FREE_OFFSET(n) \
	(0)

namespace StreamCompaction {
namespace Efficient {
	void scan(int size, int * scans, const int * bools);

	int compact(int n, Path* path);


	//cpu, used for test
	int compactWithoutScan(int n, Path *odata, const Path *idata);
	int cmpArrays(int n, Path *a, Path *b);

	//int printArray(int n, Path *a);

	void printArray(int n, Path *a, bool abridged = true);

    //int compact(int n, int *odata, const int *idata);

	//void radixSortLauncher(int n, int *odata, const int *idata, int msb,int lsb);
}
}
