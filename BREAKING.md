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
