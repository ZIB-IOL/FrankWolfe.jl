# API Reference

## Algorithms

This section contains all main algorithms of the [`FrankWolfe.jl`](https://github.com/ZIB-IOL/FrankWolfe.jl) package. These are the ones typical users will call.

```@docs
frank_wolfe
lazified_conditional_gradient
away_frank_wolfe
blended_conditional_gradient
FrankWolfe.blended_pairwise_conditional_gradient
FrankWolfe.stochastic_frank_wolfe
```

## Linear Minimization Oracles

```@docs
FrankWolfe.LinearMinimizationOracle
compute_extreme_point
```

### Implemented LMOs

```@docs
FrankWolfe.BirkhoffPolytopeLMO
FrankWolfe.KNormBallLMO
FrankWolfe.KSparseLMO
FrankWolfe.L1ballDense
FrankWolfe.LpNormLMO
FrankWolfe.NuclearNormLMO
FrankWolfe.ProbabilitySimplexOracle
FrankWolfe.ScaledBoundL1NormBall
FrankWolfe.ScaledBoundLInfNormBall
FrankWolfe.SpectraplexLMO
FrankWolfe.UnitSimplexOracle
FrankWolfe.UnitSpectrahedronLMO
```

### MathOptInterface compatibility

```@docs
FrankWolfe.MathOptLMO
```

### Wrappers

```@docs
FrankWolfe.CachedLinearMinimizationOracle
FrankWolfe.ProductLMO
FrankWolfe.SingleLastCachedLMO
FrankWolfe.MultiCacheLMO
FrankWolfe.VectorCacheLMO
```

### Associated methods

```@docs
compute_extreme_point(lmo::FrankWolfe.ProductLMO, direction::Tuple; kwargs...)
compute_extreme_point(lmo::FrankWolfe.ProductLMO{N},direction::AbstractArray;storage=similar(direction),direction_indices,kwargs...,) where {N}
compute_extreme_point(lmo::FrankWolfe.UnitSimplexOracle{T}, direction) where {T}
FrankWolfe.compute_dual_solution(::FrankWolfe.UnitSimplexOracle{T}, direction, primalSolution) where {T}
compute_extreme_point(lmo::FrankWolfe.ProbabilitySimplexOracle{T}, direction; kwargs...) where {T}
FrankWolfe.compute_dual_solution(::FrankWolfe.ProbabilitySimplexOracle{T},direction,primal_solution;kwargs...,) where {T}
compute_extreme_point(lmo::FrankWolfe.NuclearNormLMO, direction::AbstractMatrix; tol=1e-8, kwargs...)
FrankWolfe.convert_mathopt
```

## Backend components

### Active set management

```@docs
FrankWolfe.ActiveSet
FrankWolfe.active_set_update!
FrankWolfe.compute_active_set_iterate
FrankWolfe.active_set_argmin
FrankWolfe.active_set_argminmax
FrankWolfe.find_minmax_directions
FrankWolfe.active_set_initialize!
```

### Step size computation

```@docs
FrankWolfe.line_search_wrapper
FrankWolfe.LineSearchMethod
FrankWolfe.adaptive_step_size
FrankWolfe.MonotonousStepSize
FrankWolfe.MonotonousNonConvexStepSize
```

### Custom extreme point types

```@docs
FrankWolfe.ScaledHotVector
FrankWolfe.RankOneMatrix
```

### Batch and momentum iterators

```@docs
FrankWolfe.momentum_iterate
FrankWolfe.ExpMomentumIterator
FrankWolfe.ConstantMomentumIterator
FrankWolfe.batchsize_iterate
FrankWolfe.ConstantBatchIterator
FrankWolfe.IncrementBatchIterator
```

### Miscellaneous

```@docs
FrankWolfe.compute_value
FrankWolfe.compute_gradient
FrankWolfe.compute_value_gradient
FrankWolfe.check_gradients
FrankWolfe.minimize_over_convex_hull!
FrankWolfe.build_reduced_problem(atoms::AbstractVector{<:FrankWolfe.ScaledHotVector},hessian,weights,gradient,tolerance)
FrankWolfe.strong_frankwolfe_gap
FrankWolfe.accelerated_simplex_gradient_descent_over_probability_simplex
FrankWolfe.simplex_gradient_descent_over_probability_simplex
FrankWolfe.projection_simplex_sort
FrankWolfe.strong_frankwolfe_gap_probability_simplex
FrankWolfe.simplex_gradient_descent_over_convex_hull
FrankWolfe.lp_separation_oracle
FrankWolfe.Emphasis
FrankWolfe.ObjectiveFunction
FrankWolfe.SimpleFunctionObjective
FrankWolfe.StochasticObjective
FrankWolfe.plot_results
FrankWolfe.trajectory_callback
```
