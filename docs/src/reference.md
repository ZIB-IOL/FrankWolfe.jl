# Algorithms

This section contains all algorithms of the [`FrankWolfe.jl`](https://github.com/ZIB-IOL/FrankWolfe.jl) package.

## Functions

```@docs
blended_conditional_gradient
compute_extreme_point
frank_wolfe
lazified_conditional_gradient
away_frank_wolfe

```


# LMOs

The Linear Minimization Oracle (LMO) is an integral part of the iterative step in the FW algorithm. Given $d\in \mathcal{X}$, it returns:
```math
v\in \argmin_{x\in \mathcal{C}} \langle d,x \rangle.
```
[`FrankWolfe.jl`](https://github.com/ZIB-IOL/FrankWolfe.jl) features the following common LMOs out of the box:

- probability simplex: [`FrankWolfe.ProbabilitySimplexOracle`](@ref)
- unit simplex: [`FrankWolfe.UnitSimplexOracle`](@ref)
- ``K``-sparse polytope: [`FrankWolfe.KSparseLMO`](@ref)
- ``K``-norm ball: [`FrankWolfe.KNormBallLMO`](@ref)
- ``L^p``-norm ball: [`FrankWolfe.LpNormLMO`](@ref)
- Birkhoff polytope: [`FrankWolfe.BirkhoffPolytopeLMO`](@ref)

All of them are subtypes of [`FrankWolfe.LinearMinimizationOracle`](@ref) and implement the [`compute_extreme_point`](@ref) method.

## Functions and Structures

```@docs
FrankWolfe.LinearMinimizationOracle
FrankWolfe.CachedLinearMinimizationOracle
FrankWolfe.SingleLastCachedLMO
FrankWolfe.MultiCacheLMO
FrankWolfe.VectorCacheLMO
FrankWolfe.ProductLMO
compute_extreme_point(lmo::FrankWolfe.ProductLMO, direction::Tuple; kwargs...)
compute_extreme_point(lmo::FrankWolfe.ProductLMO{N},direction::AbstractArray;storage=similar(direction),direction_indices,kwargs...,) where {N}
FrankWolfe.UnitSimplexOracle
compute_extreme_point(lmo::FrankWolfe.UnitSimplexOracle{T}, direction) where {T}
FrankWolfe.compute_dual_solution(::FrankWolfe.UnitSimplexOracle{T}, direction, primalSolution) where {T}
FrankWolfe.ProbabilitySimplexOracle
compute_extreme_point(lmo::FrankWolfe.ProbabilitySimplexOracle{T}, direction; kwargs...) where {T}
FrankWolfe.compute_dual_solution(::FrankWolfe.ProbabilitySimplexOracle{T},direction,primal_solution;kwargs...,) where {T}
FrankWolfe.KSparseLMO
FrankWolfe.BirkhoffPolytopeLMO
FrankWolfe.LpNormLMO
FrankWolfe.KNormBallLMO
FrankWolfe.NuclearNormLMO
compute_extreme_point(lmo::FrankWolfe.NuclearNormLMO, direction::AbstractMatrix; tol=1e-8, kwargs...)
FrankWolfe.MathOptLMO
FrankWolfe.convert_mathopt
```


# Components

This section gathers all additional relevant components of the [`FrankWolfe.jl`](https://github.com/ZIB-IOL/FrankWolfe.jl) package.

## Functions and Structures

```@docs
FrankWolfe.ActiveSet
FrankWolfe.active_set_update!
FrankWolfe.compute_active_set_iterate
FrankWolfe.active_set_argmin
FrankWolfe.active_set_argminmax
FrankWolfe.find_minmax_directions
FrankWolfe.minimize_over_convex_hull!
FrankWolfe.build_reduced_problem(atoms::AbstractVector{<:FrankWolfe.ScaledHotVector},hessian,weights,gradient,tolerance)
FrankWolfe.build_reduced_problem(atoms::AbstractVector{<:SparseArrays.AbstractSparseArray},hessian,weights,gradient,tolerance)
FrankWolfe.build_reduced_problem(atoms::AbstractVector{<:Array},hessian,weights,gradient,tolerance)
FrankWolfe.Strong_Frank_Wolfe_gap
FrankWolfe.accelerated_simplex_gradient_descent_over_probability_simplex
FrankWolfe.simplex_gradient_descent_over_probability_simplex
FrankWolfe.projection_simplex_sort
FrankWolfe.Strong_Frank_Wolfe_gap_probability_simplex
FrankWolfe.simplex_gradient_descent_over_convex_hull
FrankWolfe.lp_separation_oracle
FrankWolfe.LineSearchMethod
FrankWolfe.Emphasis
FrankWolfe.ObjectiveFunction
FrankWolfe.compute_value_gradient
FrankWolfe.StochasticObjective
FrankWolfe.ScaledHotVector
FrankWolfe.RankOneMatrix
FrankWolfe.line_search_wrapper
FrankWolfe.adaptive_step_size
FrankWolfe.plot_results
FrankWolfe._unsafe_equal
FrankWolfe.check_gradients
FrankWolfe.trajectory_callback
```
