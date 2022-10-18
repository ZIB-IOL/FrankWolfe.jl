# Algorithms

This section contains all main algorithms of the package. These are the ones typical users will call.

The typical signature for these algorithms is:
```julia
my_algorithm(f, grad_iip!, lmo, x0)
```

## Standard algorithms

```@autodocs
Modules = [FrankWolfe]
Pages = ["fw_algorithms.jl"]
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

## Index

```@index
Pages = ["2_algorithms.md"]
```
