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
- (to come:) optionally all algorithms can be endowed with gradient momentum. This might help convergence especially in the stochastic context.

## Cool Examples

### Approximate Carathéodory with rational arithmetic

Example: `examples/approximateCaratheodory.jl`

We can solve the approximate Carathéodory problem with rational arithmetic to obtain rational approximations; see <https://arxiv.org/abs/1911.04415> for some background about approximate Carathéodory and Conditioanl Gradients. 

<p class="aligncenter">
<img src="https://render.githubusercontent.com/render/math?math=\min_{x \in \Delta(n)} \|x\|^2">
</p>

with n = 100 here.

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

xxx





