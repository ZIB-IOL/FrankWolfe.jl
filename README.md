# FrankWolfe.jl

[![Build Status](https://github.com/ZIB-IOL/FrankWolfe.jl/workflows/CI/badge.svg)](https://github.com/ZIB-IOL/FrankWolfe.jl/actions)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://zib-iol.github.io/FrankWolfe.jl/dev/)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://zib-iol.github.io/FrankWolfe.jl/stable/)
[![Coverage](https://codecov.io/gh/ZIB-IOL/FrankWolfe.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ZIB-IOL/FrankWolfe.jl)
[![Genie Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/FrankWolfe)](https://pkgs.genieframework.com?packages=FrankWolfe)

This package is a toolbox for Frank-Wolfe and conditional gradients algorithms.

## Overview

Frank-Wolfe algorithms were designed to solve optimization problems of the form `min_{x ∈ C} f(x)`, where `f` is a differentiable convex function and `C` is a convex and compact set.
They are especially useful when we know how to optimize a linear function over `C` in an efficient way.

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

Let's say we want to minimize the Euclidian norm over the probability simplex `Δ`. Using `FrankWolfe.jl`, this is what the code looks like (in dimension 3):

```julia
julia> using FrankWolfe

julia> f(p) = sum(abs2, p)  # objective function

julia> grad!(storage, p) = storage .= 2p  # in-place gradient computation

# # function d ⟼ argmin ⟨p,d⟩ st. p ∈ Δ
julia> lmo = FrankWolfe.ProbabilitySimplexOracle(1.)

julia> p0 = [1., 0., 0.]

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
