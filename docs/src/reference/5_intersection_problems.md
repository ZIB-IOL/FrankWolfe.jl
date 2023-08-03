# Intersection problems

Problems over intersections of convex sets, i.e. 
```math
\min_{x \in \bigcap_{i=1}^n P_i} f(x),
```
pose a challenge as one has to combine the information of two or more LMOs. The key idea of `AlternatingLinearMinimization` is to convert the problem into a series of subproblems over single sets. To find a point within the intersection, one minimizes both the distance to the iterates of the other subproblems and the original objective function.

[`FrankWolfe.AlternatingLinearMinimization`](@ref) calls a block-coordinate method to solve the subproblems parallelly. All block-coordinate methods are subtypes of [`FrankWolfe.BlockCoordinateMethod`](@ref) and implement [`FrankWolfe.perform_bc_updates`](@ref) which solves the subproblems.


```@docs
FrankWolfe.AlternatingLinearMinimization
FrankWolfe.BlockCoordinateMethod
FrankWolfe.perform_bc_updates
```

```@autodocs
Modules = [FrankWolfe]
Pages = ["alm.jl", "block_coordinate_algorithms.jl"]
```

See [Beck, Pauwels, Sabach 2015](https://arxiv.org/abs/1502.03716) for more details about the underlying theory.

## Index

```@index
Pages = ["5_intersection_problems.md"]
```