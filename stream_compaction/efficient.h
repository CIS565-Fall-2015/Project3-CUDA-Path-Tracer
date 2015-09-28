#pragma once

namespace StreamCompaction {
namespace Efficient {
    void scan(int n, int *odata, const int *idata);

	void up_sweep_down_sweep(int n, int *dev_data, int blocksPerGrid, int blockSize);

    int compact(int n, int *odata, const int *idata);
}
}
