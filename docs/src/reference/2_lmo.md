# Linear Minimization Oracles

The Linear Minimization Oracle (LMO) is a key component called at each iteration of the FW algorithm. Given ``d\in \mathcal{X}``, it returns a vertex of the feasible set:
```math
v\in \argmin_{x\in \mathcal{C}} \langle d,x \rangle.
```

See [Combettes, Pokutta 2021](https://arxiv.org/abs/2101.10040) for references on essential LMOs
implemented in the package and their comparison with projection operators.

## Interface and wrappers

```@docs
FrankWolfe.LinearMinimizationOracle
```

All of them are subtypes of [`FrankWolfe.LinearMinimizationOracle`](@ref) and implement the following method:
```@docs
compute_extreme_point
```

Optionally, an LMO can implement a weak separation procedure based either on a heuristic or on an approximation algorithm:
```@docs
compute_weak_separation_point
```

Weak separation procedures will be used in the methods using an active set and lazified variants only.

We also provide some meta-LMOs wrapping another one with extended behavior:
```@docs
FrankWolfe.CachedLinearMinimizationOracle
FrankWolfe.ProductLMO
FrankWolfe.SingleLastCachedLMO
FrankWolfe.MultiCacheLMO
FrankWolfe.VectorCacheLMO
```

## Norm balls

```@autodocs
Modules = [FrankWolfe]
Pages = ["norm_oracles.jl"]
```

## Simplex

```@autodocs
Modules = [FrankWolfe]
Pages = ["simplex_oracles.jl"]
```

## Polytope

```@autodocs
Modules = [FrankWolfe]
Pages = ["polytope_oracles.jl"]
```

## MathOptInterface

```@autodocs
Modules = [FrankWolfe]
Pages = ["moi_oracle.jl"]
```

## Index

```@index
Pages = ["1_lmo.md"]
```
