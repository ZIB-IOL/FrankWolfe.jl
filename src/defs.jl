
"""
Line search method to apply once the direction is computed.
"""
abstract type LineSearchMethod end

"""
Computes step size: `2/(2 + t)` at iteration `t`.
"""
struct Agnostic <: LineSearchMethod end

"""
Computes step size via the [Backtracking Line
Search](https://arxiv.org/pdf/1806.05123.pdf) method.
"""
struct Backtracking <: LineSearchMethod end

"""
Computes a step size via [Golden Section
Search](https://en.wikipedia.org/wiki/Golden-section_search).
"""
struct Goldenratio <: LineSearchMethod end

"""
Computes a step size for nonconvex functions: `1/sqrt(t + 1)`.
"""
struct Nonconvex <: LineSearchMethod end

"""
Computes the 'Short step' step size:
`dual_gap / (L * norm(x - v)^2)`,
where `L` is the Lipschitz constant of the gradient, `x` is the
current iterate, and `v` is the current Frank-Wolfe vertex.
"""
struct Shortstep <: LineSearchMethod end

"""
Constant step size given by `gamma0`
"""
struct FixedStep <: LineSearchMethod end

"""
Computes a 'Rational Short step' step size:
`sum((x - v) .* gradient ) // (L * sum((x - v) .^ 2))`,
where `L` is the Lipschitz constant of the gradient, `x` is the
current iterate, and `v` is the current Frank-Wolfe vertex.
"""
struct RationalShortstep <: LineSearchMethod end

"""
Slight modification of
Adaptive Step Size strategy from this
[paper](https://arxiv.org/pdf/1806.05123.pdf).
"""
struct Adaptive <: LineSearchMethod end

"""
Emphasis given to the algorithm for memory-saving or not.
The memory-saving mode may not be faster than the default
blas mode for small dimensions.
"""
@enum Emphasis blas = 1 memory = 2

@enum StepType begin
    initial = 1
    regular = 2
    lazy = 3
    lazylazy = 4
    dualstep = 5
    away = 6
    pairwise = 7
    drop = 8
    simplex_descent = 101
    gap_step = 102
    last = 1000
    pp = 1001
end

const st = (
    initial="I",
    regular="FW",
    lazy="L",
    lazylazy="LL",
    dualstep="LD",
    away="A",
    pairwise="P",
    drop="D",
    simplex_descent="SD",
    gap_step="GS",
    last="Last",
    pp="PP",
)
