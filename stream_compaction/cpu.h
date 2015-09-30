#pragma once

namespace StreamCompaction {
namespace CPU {
    void scan(int n, int *odata, const int *idata);

    int compactWithoutScan(int n, int *odata, const int *idata);

    int compactWithScan(int n, int *odata, const int *idata);

	int scatter(int n, int *odata, const int *scan_sum, const int *idata_map, const int *idata);
}
}
