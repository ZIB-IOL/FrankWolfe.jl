
"""
Line search method to apply once the direction is computed.
"""
@enum LineSearchMethod begin
    agnostic = 1
    backtracking = 2
    goldenratio = 3
    nonconvex = 4
    shortstep = 5
    fixed = 6
    rationalshortstep = 7
    adaptive = 8
end

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
    local_fw = 7
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
    local_fw="locFW",
    simplex_descent="SD",
    gap_step="GS",
    last="Last",
    pp="PP",
)
