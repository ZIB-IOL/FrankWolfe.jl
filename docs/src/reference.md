
# Algorithms

This section contains all main algorithms of the [`FrankWolfe.jl`](https://github.com/ZIB-IOL/FrankWolfe.jl) package. These are the ones typical users will call.

```@docs
frank_wolfe
lazified_conditional_gradient
away_frank_wolfe
blended_conditional_gradient
FrankWolfe.stochastic_frank_wolfe
```

# Linear Minimization Oracle

The Linear Minimization Oracle (LMO) is a key component called at each iteration of the FW algorithm. Given ``d\in \mathcal{X}``, it returns a vertex of the feasible set:
```math
v\in \argmin_{x\in \mathcal{C}} \langle d,x \rangle.
```

```@docs
FrankWolfe.LinearMinimizationOracle
```

All of them are subtypes of [`FrankWolfe.LinearMinimizationOracle`](@ref) and implement the following method:
```@docs
compute_extreme_point
```

The package features the following common LMOs out of the box:

```@docs
FrankWolfe.BirkhoffPolytopeLMO
FrankWolfe.KNormBallLMO
FrankWolfe.KSparseLMO
FrankWolfe.LpNormLMO
FrankWolfe.NuclearNormLMO
FrankWolfe.ProbabilitySimplexOracle
FrankWolfe.ScaledBoundL1NormBall
FrankWolfe.ScaledBoundLInfNormBall
FrankWolfe.SpectraplexLMO
FrankWolfe.UnitSimplexOracle
FrankWolfe.UnitSpectrahedronLMO
FrankWolfe.MathOptLMO
```

It also contains some meta-LMOs wrapping another one with extended behavior:
```@docs
FrankWolfe.CachedLinearMinimizationOracle
FrankWolfe.ProductLMO
FrankWolfe.SingleLastCachedLMO
FrankWolfe.MultiCacheLMO
FrankWolfe.VectorCacheLMO
```

See [Combettes, Pokutta 2021](https://arxiv.org/abs/2101.10040) for references on most LMOs
implemented in the package and their comparison with projection operators.

## Functions and Structures

```@docs
compute_extreme_point(lmo::FrankWolfe.ProductLMO, direction::Tuple; kwargs...)
compute_extreme_point(lmo::FrankWolfe.ProductLMO{N},direction::AbstractArray;storage=similar(direction),direction_indices,kwargs...) where {N}
compute_extreme_point(lmo::FrankWolfe.UnitSimplexOracle{T}, direction) where {T}
FrankWolfe.compute_dual_solution(::FrankWolfe.UnitSimplexOracle{T}, direction, primalSolution) where {T}
compute_extreme_point(lmo::FrankWolfe.ProbabilitySimplexOracle{T}, direction; kwargs...) where {T}
FrankWolfe.compute_dual_solution(::FrankWolfe.ProbabilitySimplexOracle{T},direction,primal_solution;kwargs...) where {T}
FrankWolfe.convert_mathopt
```

# Components

This section gathers all additional relevant components of the package.

## Active set management

The active set represents an iterate as a convex combination of atoms.
It maintains a vector of atoms, the corresponding weights, and the current iterate.

```@autodocs
Modules = [FrankWolfe]
Pages = ["active_set.jl"]
```

## Step size computation

For all Frank-Wolfe algorithms, a step size must be determined to move from the
current iterate to the next one. This step size can be determined by exact line search
or any other rule represented by a subtype of `LineSearchMethod` which
must implement `perform_line_search`.

## Functions and Structures

```@docs
FrankWolfe.minimize_over_convex_hull!
FrankWolfe.build_reduced_problem(atoms::AbstractVector{<:FrankWolfe.ScaledHotVector},hessian,weights,gradient,tolerance)
FrankWolfe.strong_frankwolfe_gap
FrankWolfe.accelerated_simplex_gradient_descent_over_probability_simplex
FrankWolfe.simplex_gradient_descent_over_probability_simplex
FrankWolfe.projection_simplex_sort
FrankWolfe.strong_frankwolfe_gap_probability_simplex
FrankWolfe.simplex_gradient_descent_over_convex_hull
FrankWolfe.lp_separation_oracle
FrankWolfe.MemoryEmphasis
FrankWolfe.ObjectiveFunction
FrankWolfe.compute_value_gradient
FrankWolfe.StochasticObjective
FrankWolfe.trajectory_callback
```

A note on iterates precision in algorithms depending on an active set:  
The weights in the active set are currently defined as `Float64` in the algorithm.
This means that even with vertices using a lower precision, the iterate `sum_i(lambda_i * v_i)`
will be upcast to `Float64`. One reason for keeping this as-is for now is the
higher precision required by the computation of iterates from their barycentric decomposition.

## Custom extreme point types

For some feasible sets, the extreme points of the feasible set returned by
the LMO possess a specific structure that can be represented in an efficient
manner both for storage and for common operations like scaling and addition with an iterate. They are presented below:

```@docs
FrankWolfe.ScaledHotVector
FrankWolfe.RankOneMatrix
```

## Batch and momentum iterators

```@docs
FrankWolfe.momentum_iterate
FrankWolfe.ExpMomentumIterator
FrankWolfe.ConstantMomentumIterator
FrankWolfe.batchsize_iterate
FrankWolfe.ConstantBatchIterator
FrankWolfe.IncrementBatchIterator
```

## Oracle counting trackers

The following structures are wrapping given oracles to behave similarly but additionally track the number of calls.

```@docs
FrankWolfe.TrackingObjective
FrankWolfe.TrackingGradient
FrankWolfe.TrackingLMO
FrankWolfe.tracking_trajectory_callback
```

Also see the example "Tracking number of calls to different oracles".
