# Utilities and data structures

## Active set

```@autodocs
Modules = [FrankWolfe]
Pages = ["active_set.jl"]
```

## Functions and gradients

```@autodocs
Modules = [FrankWolfe]
Pages = ["function_gradient.jl"]
```

## Callbacks

```@docs
FrankWolfe.CallbackState
```

## Custom vertex storage

## Custom extreme point types

For some feasible sets, the extreme points of the feasible set returned by
the LMO possess a specific structure that can be represented in an efficient
manner both for storage and for common operations like scaling and addition with an iterate. They are presented below:

```@docs
FrankWolfe.ScaledHotVector
FrankWolfe.RankOneMatrix
```

```@autodocs
Modules = [FrankWolfe]
Pages = ["types.jl"]
```

## Utils

```@autodocs
Modules = [FrankWolfe]
Pages = ["utils.jl"]
```

## Oracle counting trackers

The following structures are wrapping given oracles to behave similarly but additionally track the number of calls.

```@docs
FrankWolfe.TrackingObjective
FrankWolfe.TrackingGradient
FrankWolfe.TrackingLMO
```

Also see the example [Tracking, counters and custom callbacks for Frank Wolfe](@ref).

## Update order for block-coordinate methods

Block-coordinate methods can be run with different update orders. All update orders are subtypes of [`FrankWolfe.BlockCoordinateUpdateOrder`](@ref). They have to implement the method [`FrankWolfe.select_update_indices`](@ref) which selects which blocks to update in what order.

```@docs
FrankWolfe.BlockCoordinateUpdateOrder
FrankWolfe.select_update_indices
FrankWolfe.FullUpdate
FrankWolfe.CyclicUpdate
FrankWolfe.StochasticUpdate
```

## Update step for block-coordinate Frank-Wolfe

Block-coordinate Frank-Wolfe (BCFW) can run different FW algorithms on different blocks. All update steps are subtypes of [`FrankWolfe.UpdateStep`](@ref) and implement [`FrankWolfe.update_iterate`](@ref) which defines one iteration of the corresponding method.

```@docs
FrankWolfe.UpdateStep
FrankWolfe.update_iterate
FrankWolfe.FrankWolfeStep
FrankWolfe.BPCGStep
```

## Block vector
```@docs
FrankWolfe.BlockVector
```

## Index

```@index
Pages = ["3_backend.md"]
```
