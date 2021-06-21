# Introduction

```@contents
```

This package defines a generic interface and several implementations for [Frank-Wolfe algorithms](https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm).
A paper presenting the package and explaining the algorithms and numerous examples in detail can be found here:
[FrankWolfe.jl: A high-performance and flexible toolbox for Frank-Wolfe algorithms and Conditional Gradients](https://arxiv.org/pdf/2104.06675.pdf).
The package features four algorithms: a *standard Frank-Wolfe* implementation ([`frank_wolfe`](@ref), FW), *Away-step Frank-Wolfe* ([`away_frank_wolfe`](@ref), AFW), *Blended Conditional
Gradient* ([`blended_conditional_gradient`](@ref), BCG), and *Stochastic Frank-Wolfe* ([`FrankWolfe.stochastic_frank_wolfe`](@ref), SFW).
While the standard Frank-Wolfe algorithm can only move *towards* extreme points of the compact, convex set ``\mathcal{C}``, Away-step Frank-Wolfe can move *away* 
from them. The following figure from [FrankWolfe.jl: A high-performance and flexible toolbox
for Frank-Wolfe algorithms and Conditional Gradients](https://arxiv.org/pdf/2104.06675.pdf) schematizes this behaviour:
![FW vs AFW](./fw_vs_afw.PNG) \
The algorithms minimize a quadratic function (contour lines depicted) over a simple polytope. As the minimizer lies on a face, the standard Frank-Wolfe algorithm
zig-zags towards the solution. \
The following table compares the characteristics of the algorithms presented in the package:

| Algorithm | Progress/Iteration | Time/Iteration | Sparsity | Numerical Stability | Active Set | Lazifiable |
|:---------:|:------------------:|:--------------:|:--------:|:-------------------:|:----------:|:----------:|
| **FW**    | Low                | Low            | Low      | High                | No         | Yes        |
| **AFW**   | Medium             | Medium-High    | Medium   | Medium-High         | Yes        | Yes        |
| **BCG**   | High               | Medium-High    | High     | Medium              | Yes        | By design  |
| **SFW**   | Low                | Low            | Low      | High                | No         | No         |
