#include <cstdio>
#include "cpu.h"
#include "common.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
	std::chrono::high_resolution_clock::time_point t1;
	if (BENCHMARK) {
		t1 = std::chrono::high_resolution_clock::now();
	}


    // Implement exclusive serial scan on CPU
	odata[0] = 0;
	for (int i = 1; i < n; i++) {
		odata[i] = odata[i - 1] + idata[i - 1];
	}

	if (BENCHMARK) {
		std::chrono::high_resolution_clock::time_point t2 =
			std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::cout << duration << " microseconds.\n";
	}
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	std::chrono::high_resolution_clock::time_point t1;
	if (BENCHMARK) {
		t1 = std::chrono::high_resolution_clock::now();
	}

    // remove all 0s from the array of ints
	int odataIndex = 0;
	for (int i = 0; i < n; i++) {
		if (idata[i] == 0) {
			continue;
		}
		odata[odataIndex] = idata[i];
		odataIndex++;
	}

	if (BENCHMARK) {
		std::chrono::high_resolution_clock::time_point t2 =
			std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::cout << duration << " microseconds.\n";
	}

	return odataIndex;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
	int *trueArray = new int[n];
	int *trueScan = new int[n];

	std::chrono::high_resolution_clock::time_point t1;
	if (BENCHMARK) {
		t1 = std::chrono::high_resolution_clock::now();
	}


    // Step 1: Compute temporary values in odata
	for (int i = 0; i < n; i++) {
		if (idata[i] == 0) {
			trueArray[i] = 0;
		}
		else {
			trueArray[i] = 1;
		}
	}
	// Step 2: Run exclusive scan on temporary array
	scan(n, trueScan, trueArray);

	// Step 3: Scatter
	for (int i = 0; i < n; i++) {
		if (trueArray[i]) {
			odata[trueScan[i]] = idata[i];
		}
	}
	int numRemaining = trueScan[n - 1] + trueArray[n - 1];

	if (BENCHMARK) {
		std::chrono::high_resolution_clock::time_point t2 =
			std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::cout << duration << " microseconds.\n";
	}

	delete trueArray;
	delete trueScan;
	return numRemaining;
}

}
}
