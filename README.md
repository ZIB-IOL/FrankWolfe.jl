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

### Conditional Gradient algorithms

### LMOs

### General Features

#### Multi-precision 

All algorithms can run in various precisions modes: `Float16, Float32, Float64, BigFloat` and also for rationals based on various integer types `Int32, Int64, BigInt` (see e.g., the approximate Carathéodory example)

#### Line Search Strategies


## Cool Examples

### Approximate Carathéodory with rational arithmetic

We can solve the approximate Carathéodory problem with rational arithmetic to obtain rational approximations; see <https://arxiv.org/abs/1911.04415> for some background about approximate Carathéodory and Conditioanl Gradients. 

See full example: `examples/approximateCaratheodory.jl`

<img src="https://render.githubusercontent.com/render/math?math=\min_{n in \Delta(n)} \|x\|^2">

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
The solution returned it rational as we can see up there and:

```
x = Rational{BigInt}[1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100, 1//100]
```







