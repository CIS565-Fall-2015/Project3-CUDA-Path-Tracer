#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
	odata[0] = 0;
	for (int i = 1; i < n; i++) {
		odata[i] = idata[i-1] + odata[i-1];
	}
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	int cnt = 0;
	for(int i = 0; i < n; i++) {
		if (idata[i] != 0) {
			cnt++;
			odata[i] = 1;
		} else {
			odata[i] = 0;
		}
	}
	return cnt;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
    for(int i = 0; i < n; i++) {
		if (idata[i] != 0) {
			odata[i] = 1;
		} else {
			odata[i] = 0;
		}
	}
	int* result = new int[n];
	scan(n, result, odata);
	return result[n-1];
}

}
}
