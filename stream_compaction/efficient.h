#pragma once

namespace StreamCompaction {
namespace Efficient {
    void scan(int n, int *odata, const int *idata);
    int compact(int n, int *odata, const int *idata);
	template <typename T, typename Predicate> int shared_compact(int n, T* odata, T* idata, Predicate pred);
}
}
