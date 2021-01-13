
"""
Supertype for linear minimization oracles.

All LMOs must implement `compute_extreme_point(lmo::LMO, direction)`
and return a vector `v` of the appropriate type.
"""
abstract type LinearMinimizationOracle
end

"""
    compute_extreme_point(lmo::LinearMinimizationOracle, direction; kwargs...)

Computes the point `argmin_{v ∈ C} v ⋅ direction`
with `C` the set represented by the LMO.
"""
function compute_extreme_point end
