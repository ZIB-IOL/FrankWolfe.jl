#include <cstdint>
#include <omp.h>
#include <vector>
#include <limits>
#include <iostream>

#include <oracle/linear_lop_oracle.impl.hpp>

typedef double real_t;

extern "C" void call_lop_oracle(
    const uint64_t num_items,
    const real_t * objective,
    uint64_t * v_order,
    uint64_t * v_ix,
    real_t * v_obj)
{
    LinearLOPOracle<real_t> lop(num_items);
    lop.call_linear_parallel(objective);

    *v_obj = lop.call_value();
    *v_ix = lop.call_index();
    std::copy(lop.call_order().begin(), lop.call_order().end(), v_order);
}