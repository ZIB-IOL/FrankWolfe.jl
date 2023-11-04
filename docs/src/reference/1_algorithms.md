# Algorithms

This section contains all main algorithms of the package. These are the ones typical users will call.

The typical signature for these algorithms is:
```julia
my_algorithm(f, grad!, lmo, x0)
```

## Standard algorithms

```@autodocs
Modules = [FrankWolfe]
Pages = ["fw_algorithms.jl"]
```

```@docs
FrankWolfe.block_coordinate_frank_wolfe
```

## Active-set based methods

The following algorithms maintain the representation of the iterates
as a convex combination of vertices.

### Away-step

```@autodocs
Modules = [FrankWolfe]
Pages = ["afw.jl"]
```

### Blended Conditional Gradient

```@autodocs
Modules = [FrankWolfe]
Pages = ["blended_cg.jl"]
```

### Blended Pairwise Conditional Gradient

```@autodocs
Modules = [FrankWolfe]
Pages = ["pairwise.jl"]
```

## Alternating Methods

Problems over intersections of convex sets, i.e.
```math
\min_{x \in \bigcap_{i=1}^n P_i} f(x),
```
pose a challenge as one has to combine the information of two or more LMOs.

[`FrankWolfe.alternating_linear_minimization`](@ref) converts the problem into a series of subproblems over single sets. To find a point within the intersection, one minimizes both the distance to the iterates of the other subproblems and the original objective function.

[`FrankWolfe.alternating_projections`](@ref) solves feasibility problems over intersections of feasible regions.

```@autodocs
Modules = [FrankWolfe]
Pages = ["alternating_methods.jl"]
```

## Index

```@index
Pages = ["2_algorithms.md"]
```
