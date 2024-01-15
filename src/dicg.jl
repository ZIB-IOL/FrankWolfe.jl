
## interface functions for LMOs that are supported by the decomposition-invariant algorithm

"""
    is_decomposition_invariant_oracle(lmo)

Function to indicate whether the given LMO supports the decomposition-invariant interface
"""
is_decomposition_invariant_oracle(::LinearMinimizationOracle) = false

"""
    dicg_fix_variable!(lmo, variable_idx::Int, value::Real)

Fixes a variable to a value (its lower or upper bound).
"""
function dicg_fix_variable!(lmo, variable_idx, value) end

"""
    dicg_unfix_variable!(lmo, variable_idx::Int, lb::Real, ub::Real)

Unfixes a variable that was fixed before, resetting its lower and upper bounds to the provided ones.
"""
function dicg_unfix_variable!(lmo, variable_idx, lb, ub) end

"""
    dicg_maximum_step(lmo, x, direction)

Given `x` the current iterate and `direction` the negative of the direction towards which the iterate will move,
determine a maximum step size `gamma_max`, such that `x - gamma_max * direction` is in the polytope.
"""
function dicg_maximum_step(lmo, x, direction) end

struct ZeroOneHypercube
    fixed_to_one::Set{Int}
    fixed_to_zero::Set{Int}
end

ZeroOneHypercube() = ZeroOneHypercube(Set{Int}(), Set{Int}())

function FrankWolfe.compute_extreme_point(lmo::ZeroOneHypercube, direction; kwargs...)
    d = BitVector(signbit(di) for di in direction)
    for idx in lmo.fixed_to_one
        d[idx] = true
    end
    for idx in lmo.fixed_to_zero
        d[idx] = false
    end
    return d
end

is_decomposition_invariant_oracle(::ZeroOneHypercube) = true

function dicg_fix_variable!(lmo::ZeroOneHypercube, variable_idx::Int, value)
    if value ≈ 0
        delete!(lmo.fixed_to_one, variable_idx)
        push!(lmo.fixed_to_zero, variable_idx)
    else
        @assert value ≈ 1
        delete!(lmo.fixed_to_zero, variable_idx)
        push!(lmo.fixed_to_zero, variable_idx)
    end
    return nothing
end

function dicg_unfix_variable!(lmo::ZeroOneHypercube, variable_idx::Int, lb=0, ub=1)
    delete!(lmo.fixed_to_one, variable_idx)
    delete!(lmo.fixed_to_zero, variable_idx)
    return nothing
end

function dicg_maximum_step(lmo::ZeroOneHypercube, x, direction)
    
end
