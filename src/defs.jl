
"""
Line search method to apply once the direction is computed.
"""
abstract type LineSearchMethod end

struct Agnostic <: LineSearchMethod end
struct Backtracking <: LineSearchMethod end
struct Goldenratio <: LineSearchMethod end
struct Nonconvex <: LineSearchMethod end
struct Shortstep <: LineSearchMethod end
struct FixedStep <: LineSearchMethod end
struct RationalShortstep <: LineSearchMethod end
struct Adaptive <: LineSearchMethod end

"""
Emphasis given to the algorithm for memory-saving or not.
The memory-saving mode may not be faster than the default
OutplaceEmphasis mode for small dimensions.
"""
struct InplaceEmphasis end
struct OutplaceEmphasis end

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
