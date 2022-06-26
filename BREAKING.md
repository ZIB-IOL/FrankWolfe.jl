# Noteworthy changes from the last version

- clean up `active_set.jl` by renaming `compute_active_set_iterate` to `get_active_set_iterate` and merging `find_minmax_direction` and `active_set_argminmax` [PR258](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/258)
- change keyword argument `K` to `lazy_tolerance` in all algorithms [PR255](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/255)
- remove L1 LMO [PR276](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/276)
- add `perform_line_search` function to all step size structures and a workspace to line searches (to store variables) [PR259](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/259)
- add type `MemoryEmphasis` and subtypes `InplaceEmphasis` and `OutplaceEmphasis` [PR277](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/277)
- fix header [PR278](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/278)
- use `GenericSchur` to have type stable eigenvalues and compute min and max eigenvalues
- add test of accelerated blended CG for generic types
- use `SnoopCompile` to analyze which functions are called during compilation to add them to the precompilation file [PR281](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/281)
- check what costs a lot in inference and compilation time and remove heavy dependencies [PR245](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/245)
- add norms for `ScaledHotVector` and `RankOneMatrix`
- add step counter feature [PR301](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/301)
- call `@memory_mode` from within a function so the function can be easily modified [PR302](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/302)
- re-add `gradient` to `frank_wolfe` [PR314](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/314)
- add struct `CallbackStructure` for state of callbacks
- add unified callback for verbose printcallback, user given stop criteria and trajectory tracking with counters [PR313](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/313)
- start with `t=1` and pass `t` instead of `t-1` in callbacks [PR333](https://github.com/ZIB-IOL/FrankWolfe.jl/pull/333)
- add optional arguments to callback calls
