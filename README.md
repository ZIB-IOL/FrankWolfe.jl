# Frank-Wolfe and Conditional Gradient algorithms

[![Build Status](https://github.com/ZIB-IOL/FrankWolfe.jl/workflows/CI/badge.svg)](https://github.com/ZIB-IOL/FrankWolfe.jl/actions)
[![Coverage](https://codecov.io/gh/ZIB-IOL/FrankWolfe.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ZIB-IOL/FrankWolfe.jl)

This package defines a generic interface and several implementations for
Frank-Wolfe algorithms.
The main entry point is the `frank_wolfe` function running the algorithm.

```julia
FrankWolfe.frank_wolfe(f, grad!, lmo, x0, max_iteration=1000, line_search=FrankWolfe.agnostic, verbose=true)
```

where `f(x)` is the objective function, `grad!(storage, x)` is the inplace gradient.
`lmo` is a structure implementing the Linear Minimization Oracle interface presented below.

## LMO

Several oracles are implemented, all are subtypes of `LinearMinimizationOracle`
and implement the method:

```julia
compute_extreme_point(lmo::LMO, direction; kwargs...) -> v
```

which takes a minimization direction and returns the point `v` minimizing in the direction
over the set represented by the LMO.

## Implemented Methods

### Conditional Gradient algorithms

- Basic Frank-Wolfe Algorithm (see <http://proceedings.mlr.press/v28/jaggi13.html> for an overview)
  - works both for convex and non-convex function (use step size rule `FrankWolfe.nonconvex`)
- Stochastic Frank-Wolfe 
- Away-Step Frank-Wolfe (see <https://arxiv.org/abs/1511.05932> for an overview)
- Blended Conditional Gradients (see <https://arxiv.org/abs/1805.07311>)
  - build-in stability feature that temporarily increases accuracy
- Pairwise Conditional Gradients to come (see <https://arxiv.org/abs/1511.05932> for an overview)

- Most algorithms also have a lazified version (see <https://arxiv.org/abs/1610.05120>)

### LMOs

Several common LMOs are available out-of-the-box
- Probability simplex
- Unit simplex
- K-sparse polytope
- K-norm ball
- L_p-norm ball
- Birkhoff polytope

See <https://arxiv.org/pdf/2010.07243.pdf> and <https://arxiv.org/abs/2101.10040> for details

Moreover: 
- you can simply define your own LMOs directly 
- you can use an LP oracle defined via an LP solver (e.g., `glop`, `scip`, `soplex`) with `MathOptInferface`

### General Features

#### Multi-precision 

All algorithms can run in various precisions modes: `Float16, Float32, Float64, BigFloat` and also for rationals based on various integer types `Int32, Int64, BigInt` (see e.g., the approximate Carathéodory example)

#### Line Search Strategies

Most common strategies and some more particular ones:

- Agnostic: 2/(2+t) rule for FW 
- Nonconvex: 1/sqrt rule for nonconvex function and vanilla FW
- Fixed: fixed stepsize of a given value. Useful for nonconvex and stochastic or more generally when we know the total number of iterations
- Short-step rule: basically minimizing the smoothness inequality -> requires knowledge of (an estimate of) L
- Golden ratio linesearch
- Backtracking line search
- Rational Short-step rule: some as short-step rule but all computations are kept rational if inputs are rational. useful for the rational variants
- Adaptive FW: starts with an estimate for L and then refine it dynamically (see <https://arxiv.org/pdf/1806.05123.pdf> and also the survey *** to be added *** )

#### Callbacks

All top-level algorithms can take an optional `callback` argument, which must be a function
taking a named tuple as argument of the form:
```julia
state = (
    t = t,
    primal = primal,
    dual = primal - dual_gap,
    dual_gap = phi_value,
    time = (time_ns() - time_start) / 1e9,
    x = x,
    v = vertex,
    active_set_length = active_set_length,
)
```

Some fields of the named tuple can vary, but the 6 first are always present in this order.
The callback can be used to log additional information or store some values of interest in an external array.
If a callback is passed, the `trajectory` keyword is ignored since it is a special case of callback pushing the 5 first elements of the
state to an array returned from the algorithm.

#### Other 

- Emphasis: All solvers support emphasis (parameter `Emphasis`) to either exploit vectorized linear algebra or be memory efficient, e.g., for extreme large-scale instances
- Various caching strategies for the lazy implementations. Unbounded cache sizes (can get slow), bounded cache sizes as well as early returns once any sufficient vertex is found in the cache.
- (to come:) when the LMO can compute dual prices then the Frank-Wolfe algorithms return dual prices for the (approximately) optimal solutions (see <https://arxiv.org/abs/2101.02087>)
- optionally all algorithms can be endowed with gradient momentum. This might help convergence especially in the stochastic context.

## Cool Examples

### Approximate Carathéodory with rational arithmetic

Example: `examples/approximateCaratheodory.jl`

We can solve the approximate Carathéodory problem with rational arithmetic to obtain rational approximations; see <https://arxiv.org/abs/1911.04415> for some background about approximate Carathéodory and Conditioanl Gradients. We consider the simple instance of approximating the `0` over the probability simplex here:

<p class="aligncenter">
<img src="https://render.githubusercontent.com/render/math?math=\min_{x \in \Delta(n)} \|x\|^2">
</p>

with n = 100.

````
Vanilla Frank-Wolfe Algorithm.
EMPHASIS: blas STEPSIZE: rationalshortstep EPSILON: 1.0e-7 max_iteration: 100 TYPE: Rational{BigInt}

───────────────────────────────────────────────────────────────────────────────────
  Type     Iteration         Primal           Dual       Dual Gap           Time
───────────────────────────────────────────────────────────────────────────────────
     I             0   1.000000e+00  -1.000000e+00   2.000000e+00   1.540385e-01
    FW            10   9.090909e-02  -9.090909e-02   1.818182e-01   2.821186e-01
    FW            20   4.761905e-02  -4.761905e-02   9.523810e-02   3.027964e-01
    FW            30   3.225806e-02  -3.225806e-02   6.451613e-02   3.100331e-01
    FW            40   2.439024e-02  -2.439024e-02   4.878049e-02   3.171654e-01
    FW            50   1.960784e-02  -1.960784e-02   3.921569e-02   3.244207e-01
    FW            60   1.639344e-02  -1.639344e-02   3.278689e-02   3.326185e-01
    FW            70   1.408451e-02  -1.408451e-02   2.816901e-02   3.418239e-01
    FW            80   1.234568e-02  -1.234568e-02   2.469136e-02   3.518750e-01
    FW            90   1.098901e-02  -1.098901e-02   2.197802e-02   3.620287e-01
  Last                 1.000000e-02   1.000000e-02   0.000000e+00   4.392171e-01
───────────────────────────────────────────────────────────────────────────────────

  0.600608 seconds (3.83 M allocations: 111.274 MiB, 12.97% gc time)
  
Output type of solution: Rational{BigInt}
````
The solution returned is rational as we can see and in fact the exactly optimal solution:

```
x = Rational{BigInt}[1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100]
```

### Large-Scale problems

Example: `examples/largeScale.jl`

The package is build to scale well, for those conditional gradients variants that can scale well. For exampple, Away-Step Frank-Wolfe and Pairwise Conditional Gradients do in most cases *not scale well* because they need to maintain active sets and maintaining them can be very expensive. Similarly, line search methods might become prohibitive at large sizes. However if we consider scale-friendly variants, e.g., the vanilla Frank-Wolfe algorithm with the agnostic step size rule or short step rule, then these algorithms can scale well to extreme sizes esentially only limited by the amount of memory that you have available. However even for these methods that tend to scale well, allocation of memory itself can be very slow when you need to allocate gigabytes of memory for a single gradient computation. 

The package is build to support extreme sizes with a special memory efficient emphasis `Emphasis=FrankWolfe.memory`, which minimizes very expensive allocation memory and performs as many operations as possible in-place. 

Here is an example of a run with 1e9 variables (that is *one billion* variables). Each gradient is around 7.5 GB in size. Here is the output of the run broken down into pieces:

````
Size of single vector (Float64): 7629.39453125 MB                                                                                                                                    
Testing f... 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:23
Testing grad... 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:23
Testing lmo... 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:29
Testing dual gap... 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:46
Testing update... (Emphasis: blas) 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:01:35
Testing update... (Emphasis: memory) 100%|█████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:58
 ──────────────────────────────────────────────────────────────────────────
                                   Time                   Allocations      
                           ──────────────────────   ───────────────────────
     Tot / % measured:           278s / 31.4%            969GiB / 30.8%    

 Section           ncalls     time   %tot     avg     alloc   %tot      avg
 ──────────────────────────────────────────────────────────────────────────
 update (blas)         10    36.1s  41.3%   3.61s    149GiB  50.0%  14.9GiB
 lmo                   10    18.4s  21.1%   1.84s     0.00B  0.00%    0.00B
 grad                  10    12.8s  14.6%   1.28s   74.5GiB  25.0%  7.45GiB
 f                     10    12.7s  14.5%   1.27s   74.5GiB  25.0%  7.45GiB
 update (memory)       10    5.00s  5.72%   500ms     0.00B  0.00%    0.00B
 dual gap              10    2.40s  2.75%   240ms     0.00B  0.00%    0.00B
 ──────────────────────────────────────────────────────────────────────────
````

The above is the optional benchmarking of the oracles that we provide to understand how fast crucial parts of the algorithms are, mostly notably oracle evaluations, the update of the iterate and the computation of the dual gap. As you can see if you compare `update (blas)` vs. `update (memory)`, the normal update when we use BLAS requires an additional 14.9GB of memory on top of the gradient etc whereas the `update (memory)` (the memory emphasis mode) does not consume any extra memory. This is also reflected in the computational times: the BLAS version requires 3.61 seconds on average to update the iterate, while the memory emphasis version requires only 500ms. In fact none of the crucial components in the algorithm consume any memory when run in memory efficient mode. Now let us look at the actual footprint of the whole algorithm:

````
Vanilla Frank-Wolfe Algorithm.
EMPHASIS: memory STEPSIZE: agnostic EPSILON: 1.0e-7 MAXITERATION: 1000 TYPE: Float64
MOMENTUM: nothing GRADIENTTYPE: Nothing
WARNING: In memory emphasis mode iterates are written back into x0!

─────────────────────────────────────────────────────────────────────────────────────────────────
  Type     Iteration         Primal           Dual       Dual Gap           Time         It/sec
─────────────────────────────────────────────────────────────────────────────────────────────────
     I             0   1.000000e+00  -1.000000e+00   2.000000e+00   8.783523e+00   0.000000e+00
    FW           100   1.326732e-02  -1.326733e-02   2.653465e-02   4.635923e+02   2.157068e-01
    FW           200   6.650080e-03  -6.650086e-03   1.330017e-02   9.181294e+02   2.178342e-01
    FW           300   4.437059e-03  -4.437064e-03   8.874123e-03   1.372615e+03   2.185609e-01
    FW           400   3.329174e-03  -3.329180e-03   6.658354e-03   1.827260e+03   2.189070e-01
    FW           500   2.664003e-03  -2.664008e-03   5.328011e-03   2.281865e+03   2.191190e-01
    FW           600   2.220371e-03  -2.220376e-03   4.440747e-03   2.736387e+03   2.192672e-01
    FW           700   1.903401e-03  -1.903406e-03   3.806807e-03   3.190951e+03   2.193703e-01
    FW           800   1.665624e-03  -1.665629e-03   3.331253e-03   3.645425e+03   2.194532e-01
    FW           900   1.480657e-03  -1.480662e-03   2.961319e-03   4.099931e+03   2.195159e-01
    FW          1000   1.332665e-03  -1.332670e-03   2.665335e-03   4.554703e+03   2.195533e-01
  Last          1000   1.331334e-03  -1.331339e-03   2.662673e-03   4.559822e+03   2.195261e-01
─────────────────────────────────────────────────────────────────────────────────────────────────

4560.661203 seconds (7.41 M allocations: 112.121 GiB, 0.01% gc time)
````

As you can see the algorithm ran for about 4600 secs (single-thread run) allocating 112.121 GiB of memory throughout. So how does this average out to the per-iteration cost in terms of memory: `112.121 / 7.45 / 1000 = 0.0151` so about 15.1MiB per iteration which is much less than the size of the gradient and in fact only stems from the reporting here. 

**NB.** This example highlights also one of the great features of first-order methods and conditional gradients in particular: We have dimension-independent convergence rates. In fact, we contract the primal gap as `2LD^2 / (t+2)` (for the simple agnostic rule) and, e.g., if the feasible region is the probability simplex with `D = sqrt(2)` and the function has bounded Lipschitzness, e.g., the function `|| x - xp ||^2` has `L = 2`, then the convergence rate is completely independent of the input size. The only thing that limits scaling is how much memory you have available and whether you can stomach the (linear) per-iteration cost.


