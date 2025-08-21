# Adaptive Proximal Gradient Descent Methods

This package implements several variants of adaptive proximal gradient descent methods.
Their primary use is internal to FrankWolfe.jl, specifically for the Blended Conditional Gradients algorithm, but they can also be used as standalone algorithms.

For now however, they remain in the `Experimental` submodule to highlight that they may be subject to breaking changes without changing the major version.

```@autodocs
Modules = [FrankWolfe.Experimental]
Pages = ["gradient_descent.jl"]
```
