#ifndef _LINEAR_LOP_ORACLE_HPP
#define _LINEAR_LOP_ORACLE_HPP

#include <cstdint>
#include <omp.h>
#include <vector>

template <typename real_t> class LinearLOPOracle
{
public:
    LinearLOPOracle(const uint64_t num_items);
    ~LinearLOPOracle();

    void   call_linear_parallel(const real_t * objective);

    real_t call_value();
    std::vector<uint64_t> & call_order();
    uint64_t                call_index();

protected:
    uint64_t factorial(const uint64_t n);

    const uint64_t        m_num_items;
    const uint64_t        m_num_orders;
    std::vector<uint64_t> m_divisors;
    std::vector<uint64_t> m_oracle_order;
    real_t                m_oracle_val;
    uint64_t              m_oracle_index;
};

#endif // _LINEAR_LOP_ORACLE_HPP
