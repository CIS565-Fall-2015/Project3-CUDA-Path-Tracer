#pragma once

namespace StreamCompaction {
namespace CPU {
    void scan(int n, int *odata, const int *idata);

    int compactWithoutScan(int n, int *odata, const int *idata);

    int compactWithScan(int n, int *odata, const int *idata);
}
}
