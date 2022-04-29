# Line search and step size settings


The step size dictates how far one traverses along a local descent direction.
More specifically, the step size $gamma_t$ is used at each iteration to determine
how much the next iterate moves towards the new vertex:  
$$x_{t+1} = x_t - \gamma_t (x_t - v_t).$$
  
$\gamma_t = 1$ implies that the next iterate is exactly the vertex,
a zero $\gamma_t$ implies that the iterate is not moving.  

The following are step size selection rules for Frank Wolfe algorithms.
Some methodologies (e.g. `FixedStep` and `Agnostic`) depend only on the iteration number and induce series $\gamma_t$
that are independent of the problem data,
while others (e.g. `GoldenSearch` and `Adaptive`) change according
to local information about the function; the adaptive methods
often require extra function and/or gradient computations. The
typical options for convex optimization are `Agnostic` or `Adaptive`.  

All step size computation strategies are subtypes of [LineSearchMethod](@ref).

## Line search and step size methods

```@autodocs
Modules = [FrankWolfe]
Pages = ["defs.jl"]
```

## Index

```@index
Pages = ["4_linesearch.md"]
```
