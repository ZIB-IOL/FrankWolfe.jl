# Common parts of algorithm docstrings, to be interpolated in each

const RETURN = """
# Return

Returns a tuple `(x, v, primal, dual_gap, traj_data)` with:
- `x`: the final iterate
- `v`: the last vertex from the linear minimization oracle
- `primal`: the final primal value `f(x)`
- `dual_gap`: the final Frank-Wolfe gap
- `traj_data`: a vector of trajectory information, each element being the output of [`callback_state`](@ref).
"""

const RETURN_ACTIVESET = replace(RETURN, "traj_data)" => "traj_data, active_set)") * """
- `active_set`: the computed active set of vertices, of which the solution is a convex combination
"""

const COMMON_ARGS = """
# Common arguments

These positional arguments are common to most Frank-Wolfe variants:

- `f`: a function `f(x)` computing the value of the objective to minimize at point `x`
- `grad!`: a function `grad!(g, x)` overwriting `g` with the gradient of `f` at point `x`
- `lmo`: a linear minimization oracle, subtyping [`LinearMinimizationOracle`](@ref)
- `x0`: a starting point for the optimization (will be modified in-place)
"""

const COMMON_KWARGS = """
# Common keyword arguments

These keyword arguments are common to most Frank-Wolfe variants.

!!! warning
    The current variant may have additional keyword arguments, documented elsewhere, or it may only use a subset of the ones listed below.
    The default values of these arguments may also vary between variants, and thus are not part of the public API.

- `line_search::LineSearchMethod`: an object specifying the line search and its parameters (see [`LineSearchMethod`](@ref))
- `momentum::Union{Real,Nothing}=nothing`: constant momentum to apply to the gradient
- `epsilon::Real`: absolute dual gap threshold at which the algorithm is interrupted
- `max_iteration::Integer`: maximum number of iterations after which the algorithm is interrupted
- `print_iter::Integer`: interval between two consecutive log prints, expressed in number of iterations
- `trajectory::Bool=false`: whether to record the trajectory of algorithm states (through callbacks)
- `verbose::Bool`: whether to print periodic logs (through callbacks)
- `memory_mode::MemoryEmphasis`: an object dictating whether the algorithm operates in-place or out-of-place (see [`MemoryEmphasis`](@ref))
- `gradient=nothing`: pre-allocated container for the gradient
- `callback=nothing`: function called on a [`CallbackState`](@ref) at each iteration
- `traj_data=[]`: pre-allocated storage for the trajectory of algorithm states
- `timeout::Real=Inf`: maximum time after which the algorithm is interrupted (in nanoseconds)
- `linesearch_workspace=nothing`: pre-allocated workspace for the line search 
- `dual_gap_compute_frequency::Integer=1`: frequency of dual gap computation, 
"""
