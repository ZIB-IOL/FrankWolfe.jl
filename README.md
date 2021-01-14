# Frank Wolfe and Conditional Gradients : a Julia package

This package defines a generic interface and several implementations for the
Frank-Wolfe algorithms.
The main entry point is the `fw` function running the algorithm.

```julia
FrankWolfe.fw(f,grad,lmo,x0,maxIt=1000,stepSize=FrankWolfe.agnostic,verbose=true)
```

## LMO

Several oracles are implemented, all are subtypes of `LinearMinimizationOracle`
and implement the method:

```julia
compute_extreme_point(lmo::LMO, direction; kwargs...) -> v
```

which takes a minimization direction and returns the point minimizing in the direction
over the set LMO represents.
