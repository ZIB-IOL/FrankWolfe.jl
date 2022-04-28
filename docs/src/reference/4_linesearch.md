# Line search and step size settings

The following are step size selection rules for Frank Wolfe /
Conditional Gradient algorithms. The step size dictates how
far one traverses along a local descent direction. Some
methodologies (e.g. `FixedStep` and `Agnostic`) are deterministic,
while others (e.g. `GoldenSearch` and `Adaptive`) change according
to local information about the function; the adaptive methods
often require extra function and/or gradient computations. The
"vanilla" option for convex optimization is the `Agnostic` method.

## Line search and step size methods

```@autodocs
Modules = [FrankWolfe]
Pages = ["defs.jl"]
```

## Index

```@index
Pages = ["4_linesearch.md"]
```
