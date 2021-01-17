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

## Implemented Methods

*** mention survey once done as general reference ***

### Conditional Gradient algorithms

- Basic Frank-Wolfe Algorithm (see <http://proceedings.mlr.press/v28/jaggi13.html> for an overview)
  - works both for convex and non-convex function (use step size rule `FrankWolfe.nonconvex`)
- (to come:) Stochastic Frank-Wolfe 
- (to come:) Away-Step Frank-Wolfe and Pairwise Conditional Gradients (see <https://arxiv.org/abs/1511.05932> for an overview)
- (to come:) Blended Conditional Gradients (see <https://arxiv.org/abs/1805.07311>)

- all algorithms also have a lazified version (see <https://arxiv.org/abs/1610.05120>)

### LMOs

Several common LMOs are available out-of-the-box
- Probability simplex
- Unit simplex
- K-sparse polytope
- K-norm ball
- L_p-norm ball
- (to come:) Permutahedron
- (to come:) Birkhoff polytope
- (to come:) Flow Polytope

See <https://arxiv.org/pdf/2010.07243.pdf> and *** add LP complexity paper *** for details

Moreover: 
- you can simply define your own LMOs directly 
- (to come:) via an LP solver (e.g., `glop`, `scip`, `soplex`)

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
- (to come:) adaptive FW: starts with an estimate for L and then refine it dynamically (see <https://arxiv.org/pdf/1806.05123.pdf> and also the survey *** to be added *** )

#### Other 

- Emphasis: All solvers support emphasis (parameter `emph`) to either exploit vectorized linear algebra or be memory efficient, e.g., for extreme large-scale instances
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
EMPHASIS: blas STEPSIZE: rationalshortstep EPSILON: 1.0e-7 MAXIT: 100 TYPE: Rational{BigInt}

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

The package is build to support extreme sizes with a special memory efficient emphasis `emph=FrankWolfe.memory`, which minimizes very expensive allocation memory and performs as many operations as possible in-place. 

Here is an example of a run with 1e9 variables (that is *one billion* variables). Each gradient is around 7.5 GB in size. Here is the output of the run broken down into pieces:

````
Size of single vector (Float64): 7629.39453125 MB                                                                                                                                    
Testing f... 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:23
Testing grad... 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:23
Testing lmo... 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:29
Testing dual gap... 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:46
Testing update... (emph: blas) 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:01:35
Testing update... (emph: memory) 100%|█████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:58
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

The above is the optional benchmarking of the oracles that we provide to understand how fast crucial parts of the algorithms are, mostly notably oracle evaluations, the update of the iterate and the computation of the dual gap. As you can see if you compare `update (blas)` vs. `update (memory)`, the normal update when we use BLAS requires an additional 14.9GB of memory on top of the gradient etc whereas the `update (memory)` (the memory emphasis mode) does not consume any extra memory. This is also reflected in the times: the BLAS version requires 3.61 seconds on average to update the iterate, while the memory emphasis version requires only 500ms. In fact none of the crucial components in the algorithm consume any memory when run in memory efficient mode. Now let us look at the actual footprint of the whole algorithm:

````
Vanilla Frank-Wolfe Algorithm.
EMPHASIS: memory STEPSIZE: agnostic EPSILON: 1.0e-7 MAXIT: 1000 TYPE: Float64

───────────────────────────────────────────────────────────────────────────────────
  Type     Iteration         Primal           Dual       Dual Gap           Time
───────────────────────────────────────────────────────────────────────────────────
     I             0   1.000000e+00  -1.000000e+00   2.000000e+00   1.065668e+01
    FW           100   1.326732e-02  -1.326733e-02   2.653465e-02   4.943181e+02
    FW           200   6.650080e-03  -6.650086e-03   1.330017e-02   9.776392e+02
    FW           300   4.437059e-03  -4.437064e-03   8.874123e-03   1.460490e+03
    FW           400   3.329174e-03  -3.329180e-03   6.658354e-03   1.943530e+03
    FW           500   2.664003e-03  -2.664008e-03   5.328011e-03   2.426762e+03
    FW           600   2.220371e-03  -2.220376e-03   4.440747e-03   2.910154e+03
    FW           700   1.903401e-03  -1.903406e-03   3.806807e-03   3.394155e+03
    FW           800   1.665624e-03  -1.665629e-03   3.331253e-03   3.879587e+03
    FW           900   1.480657e-03  -1.480662e-03   2.961319e-03   4.362325e+03
    FW          1000   1.332665e-03  -1.332670e-03   2.665335e-03   4.845915e+03
  Last                 1.331334e-03  -1.331339e-03   2.662673e-03   4.852188e+03
───────────────────────────────────────────────────────────────────────────────────

4853.033417 seconds (4.75 M allocations: 14.683 TiB, 0.50% gc time)
````

As you can see the algorithm ran for about 4900 secs (single-thread run) allocating 14.683TiB of memory throughout. So how does this average out to the per-iteration cost in terms of memory: `14.683 * 1024 / 7.45 / 1000 = 2.0181734228` so about 2 times the size of our gradient per iteration; the little extra is due to the logging (that requires compuation of f(x) which costs another 7.45 GB every 100 iterations) and overhead from other minor allocations. This factor 2 is essentially as good as it gets: you need one 7.45 GB to hold the gradient and another 7.45GB to hold the iterate.

**NB.** This example highlights also one of the great features of first-order methods and conditional gradients in particular: We have dimension-independent convergence rates. In fact, we contract the primal gap as `2LD^2 / (t+2)` (for the simple agnostic rule) and, e.g., if the feasible region is the probability simplex with `D = sqrt(2)` and the function has bounded Lipschitzness, e.g., the function `|| x - xp ||^2` has `L = 2`, then the convergence rate is completely independent of the input size. The only thing that limits scaling is how much memory you have available and whether you can stomach the (linear) per-iteration cost.


