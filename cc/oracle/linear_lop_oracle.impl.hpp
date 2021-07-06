#ifndef _LINEAR_LOP_ORACLE_IMPL_HPP
#define _LINEAR_LOP_ORACLE_IMPL_HPP

#include <oracle/linear_lop_oracle.hpp>

#include <omp.h>

#include <iostream>
#include <limits>

template <typename real_t>
LinearLOPOracle<real_t>::LinearLOPOracle(const size_t num_items)
    : m_num_items(num_items), m_num_orders(factorial(num_items)),
      m_oracle_index(0)
{
    m_divisors.resize(num_items);
    m_oracle_order.resize(num_items);

    for(uint64_t i = 0; i < num_items; ++i)
    {
        m_divisors[i] = factorial(num_items - i - 1);
    }
}

template <typename real_t> LinearLOPOracle<real_t>::~LinearLOPOracle() {}

template <typename real_t>
void
LinearLOPOracle<real_t>::call_linear_parallel(const real_t * objective)
{
    // CLEAR_TIMER("parallel");
    // START_TIMER("parallel");
    auto obj_access = [&](const uint64_t i, const uint64_t j) {
        return objective[i * m_num_items + j];
    };
    
    // create a job pool
    uint64_t pool_size = omp_get_max_threads();

    // create array for the individual threads
    std::vector<uint8_t> choice(m_num_items * pool_size);
    std::vector<uint64_t> order(m_num_items * pool_size);
    std::vector<uint64_t> min_order(m_num_items * pool_size);
    std::vector<real_t> min_obj(pool_size);
    std::vector<uint64_t> min_ix(pool_size);
    std::fill(min_obj.begin(), min_obj.end(), std::numeric_limits<real_t>::max());
    std::fill(min_ix.begin(), min_ix.end(), std::numeric_limits<uint64_t>::max());

    // compute thread-wise minima
    #pragma omp parallel for num_threads(pool_size)
    for(uint64_t order_ix = 0; order_ix < m_num_orders; ++order_ix)
    {
        uint64_t tid = omp_get_thread_num();
        uint8_t * t_choice = choice.data() + tid * m_num_items;
        uint64_t * t_order = order.data() + tid * m_num_items;
        uint64_t * t_min_order = min_order.data() + tid * m_num_items;

        std::fill(t_choice, t_choice + m_num_items, true);

        // order_ix to actual order
        uint64_t rem = order_ix;
        uint64_t div, ix;
        for(uint64_t i = 0; i < m_num_items; ++i)
        {
            div = rem / m_divisors[i];
            rem = rem % m_divisors[i];

            ix = 0;
            for(uint64_t j = 0; j < m_num_items; ++j)
            {
                if(t_choice[j])
                {
                    if(div == 0)
                    {
                        ix = j;
                        break;
                    }
                    --div;
                }
            }

            t_order[i]   = ix;
            t_choice[ix] = false;
        }

        // compute objective function
        real_t objval = 0.0;
        for(uint64_t i = 0; i < m_num_items; ++i)
        {
            for(uint64_t j = i + 1; j < m_num_items; ++j)
            {
                objval += obj_access(t_order[i], t_order[j]);
            }
        }

        if(objval < min_obj[tid])
        {
            min_obj[tid] = objval;
            min_ix[tid] = order_ix;
            std::copy(t_order, t_order + m_num_items, t_min_order);
        }
    }

    // final, global reduction
    m_oracle_val = std::numeric_limits<real_t>::max();
    for(int i = 0; i < pool_size; ++i)
    {
        if(min_obj[i] < m_oracle_val)
        {
            m_oracle_val = min_obj[i];
            m_oracle_index = min_ix[i];
            for(int j = 0; j < m_num_items; ++j)
            {
                m_oracle_order[j] = min_order[i * m_num_items + j];
            }
        }
    }
    // STOP_TIMER("parallel");
    // PRINT_TIMER("parallel");
}

template <typename real_t>
real_t
LinearLOPOracle<real_t>::call_value()
{
    return m_oracle_val;
}

template <typename real_t>
std::vector<uint64_t> &
LinearLOPOracle<real_t>::call_order()
{
    return m_oracle_order;
}

template <typename real_t>
uint64_t
LinearLOPOracle<real_t>::call_index()
{
    return m_oracle_index;
}

template <typename real_t>
uint64_t
LinearLOPOracle<real_t>::factorial(const uint64_t n)
{
    uint64_t fn = 1;
    for(uint64_t i = 2; i <= n; ++i)
        fn *= i;

    return fn;
}

#endif // _LINEAR_LOP_ORACLE_IMPL_HPP