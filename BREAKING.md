# Noteworthy changes in v0.6

- The function `FrankWolfe.update_iterate` was renamed to `FrankWolfe.update_block_iterate` to be more specific about it belonging to the block coordinate interface and avoid confusion.
- A `status` field was added to the named tuple return of all algorithms, corresponding to a `FrankWolfe.ExecutionStatus` enum value indicating why the algorithm stopped.
- The internal function `fast_dot(a, b)` was removed, now that the fast implementation of sparse dot products was added to `SparseArrays`. The quadratic form `fast_dot(a, Q, b)` still exists.
- The deprecated `MonotonousStepSize` was removed.
- Some features for which we do not want to commit to SemVer non-breaking rules were moved to an `Experimental` submodule.
- The following oracle types have been renamed for consistency:
    - CachedLinearMinimizationOracle -> CachedLMO
    - ConvexHullOracle -> ConvexHullLMO
    - HyperSimplexOracle -> HyperSimplexLMO
    - LpNormLMO -> LpNormBallLMO
    - NuclearNormLMO -> NuclearNormBallLMO
    - OrderWeightNormLMO -> OrderWeightNormBallLMO
    - ProbabilitySimplexOracle -> ProbabilitySimplexLMO
    - ScaledBoundL1NormBall -> DiamondLMO
    - ScaledBoundLInfNormBall -> BoxLMO
    - UnitHyperSimplexOracle -> UnitHyperSimplexLMO
    - UnitSimplexOracle -> UnitSimplexLMO
    - ZeroOneHypercube -> ZeroOneHypercubeLMO

# Noteworthy changes from v0.4 to v0.5

- change keyword argument `lazy_tolerance` to `sparsity_control` in all algorithms except `away_frank_wolfe` [PR556](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/556)

# Noteworthy changes from v0.3 to v0.4

- Changed `tt` to `step_type` everywhere, including as a type in the `CallbackState` object.
- `FrankWolfe.st` is now `FrankWolfe.steptype_string`
- all step types were renamed to clearly appear as constants, with a naming convention `ST_NAME`.

# Noteworthy changes from v0.1 to v0.2

- clean up `active_set.jl` by renaming `compute_active_set_iterate` to `get_active_set_iterate` and merging `find_minmax_direction` and `active_set_argminmax` [PR258](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/258)
- change keyword argument `K` to `lazy_tolerance` in all algorithms [PR255](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/255)
- remove L1ballDense [PR276](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/276)
- add `perform_line_search` function to all step size strategies and a workspace to line searches (to store intermediate arrays) [PR259](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/259)
- add type `MemoryEmphasis` and subtypes `InplaceEmphasis` and `OutplaceEmphasis` [PR277](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/277)- use `GenericSchur` to have type stable eigenvalues and compute min and max eigenvalues
- remove heavy dependencies, notably Plots.jl [PR245](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/245). The plotting functions are now in `examples/plot_utils.jl` and must be included separately.
- add step counter feature [PR301](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/301)
- call `@memory_mode` from within a function so new methods can easily be implemented [PR302](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/302)
- add struct `CallbackStructure` for state of callbacks. Callbacks can now return `false` to terminate the FW algorithm.
- add unified callback for verbose printcallback, user given stop criteria and trajectory tracking with counters [PR313](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/313)
- start with `t=1` and pass `t` instead of `t-1` in callbacks [PR333](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/333)
