# FrankWolfe.jl

[![Build Status](https://github.com/ZIB-IOL/FrankWolfe.jl/workflows/CI/badge.svg)](https://github.com/ZIB-IOL/FrankWolfe.jl/actions)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://zib-iol.github.io/FrankWolfe.jl/dev/)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://zib-iol.github.io/FrankWolfe.jl/stable/)
[![Coverage](https://codecov.io/gh/ZIB-IOL/FrankWolfe.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ZIB-IOL/FrankWolfe.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12720673.svg)](https://doi.org/10.5281/zenodo.12720673)

This package is a toolbox for Frank-Wolfe and conditional gradients algorithms.

## Overview

Frank-Wolfe algorithms were designed to solve optimization problems of the form 
```math
\min_{x ∈ C} f(x),
```
where $f$ is a differentiable convex function and $C$ is a convex and compact set.
They are especially useful when we know how to optimize a linear function over $C$ in an efficient way.

A paper presenting the package with mathematical explanations and numerous examples can be found here:

> [FrankWolfe.jl: A high-performance and flexible toolbox for Frank-Wolfe algorithms and Conditional Gradients](https://arxiv.org/abs/2104.06675).

## Installation

The most recent release is available via the julia package manager, e.g., with

```julia
using Pkg
Pkg.add("FrankWolfe")
```

or the master branch:

```julia
Pkg.add(url="https://github.com/ZIB-IOL/FrankWolfe.jl", rev="master")
```

## Getting started

Let's say we want to solve the following minimization problem 
```math
\min_{p \in  Δ(n)} p_1^2 + \dots + p_n^2,
```
where $`Δ(n)= \{p \in R^n_{\geq 0} | p_1 + \dots + p_n =1\}`$ is the _probability simplex_.

Using `FrankWolfe.jl`, let's write a minimal code solving this problem in dimension $n=3$.
The main function is **`FrankWolfe.frank_wolfe`** and it requires: 

* a **function `f`** that computes the values of the objective function $f$;
* a **function `grad!`** that computes in-place the gradient of the objective function $f$;
* a **subtype of `FrankWolfe.LinearMinimizationOracle`** for which a method of        `FrankWolfe.compute_extreme_point` has been implemented (see [here](https://zib-iol.github.io/FrankWolfe.jl/dev/basics/#Linear-Minimization-Oracles));
* a **starting vector `p0`**.

```julia
julia> using FrankWolfe

# #objective function f(p) = p_1^2 + ... + p_n^2
julia> f(p) = sum(abs2, p)

# #in-place gradient computation for f thanks to '.='
julia> grad!(storage, p) = storage .= 2p  

# #pre-defined type for which a method computes a solution of min ⟨p,d⟩ st. p ∈ Δ 
julia> lmo = FrankWolfe.ProbabilitySimplexOracle(1.)

# #starting vector (of dimension n=3)
julia> p0 = [1., 0., 0.]

# #an optimal solution is returned in p_opt
julia> p_opt, _ = frank_wolfe(f, grad!, lmo, p0; verbose=true);

Vanilla Frank-Wolfe Algorithm.
MEMORY_MODE: FrankWolfe.InplaceEmphasis() STEPSIZE: Adaptive EPSILON: 1.0e-7 MAXITERATION: 10000 TYPE: Float64
MOMENTUM: nothing GRADIENTTYPE: Nothing
[ Info: In memory_mode memory iterates are written back into x0!

-------------------------------------------------------------------------------------------------
  Type     Iteration         Primal           Dual       Dual Gap           Time         It/sec
-------------------------------------------------------------------------------------------------
     I             1   1.000000e+00  -1.000000e+00   2.000000e+00   0.000000e+00            Inf
  Last            24   3.333333e-01   3.333332e-01   9.488992e-08   1.533181e+00   1.565373e+01
-------------------------------------------------------------------------------------------------

julia> p_opt
3-element Vector{Float64}:
 0.33333334349923327
 0.33333332783841896
 0.3333333286623478
```

**Note** that active-set based methods like Away Frank-Wolfe and Blended Pairwise Conditional Gradient also include a post processing step. 
In post-processing all values are recomputed and in particular the dual gap is computed at the current FW vertex, which might be slightly larger than the best dual gap observed as the gap is not monotonic. This is expected behavior.


## Documentation and examples

To explore the content of the package, go to the [documentation](https://zib-iol.github.io/FrankWolfe.jl/dev/).

Beyond those presented in the documentation, many more use cases are implemented in the `examples` folder.
To run them, you will need to activate the test environment, which can be done simply with [TestEnv.jl](https://github.com/JuliaTesting/TestEnv.jl) (we recommend you install it in your base Julia).

```julia
julia> using TestEnv

julia> TestEnv.activate()
"/tmp/jl_Ux8wKE/Project.toml"

# necessary for plotting
julia> include("examples/plot_utils.jl")
julia> include("examples/linear_regression.jl")
...
```

If you need the plotting utilities in your own code, make sure Plots.jl is included in your current project and run:

```julia
using Plots
using FrankWolfe

include(joinpath(dirname(pathof(FrankWolfe)), "../examples/plot_utils.jl"))
```
