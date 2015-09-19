#pragma once

namespace StreamCompaction {
namespace Efficient {
    float scan(int n, int *odata, const int *idata);

    int compact(int n, int *odata, const int *idata);
}
}
