

"""
Emphasis given to the algorithm for memory-saving or not.
The default memory-saving mode may be slower than
OutplaceEmphasis mode for small dimensions.
"""
abstract type MemoryEmphasis end

struct InplaceEmphasis <: MemoryEmphasis end
struct OutplaceEmphasis <: MemoryEmphasis end

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


struct CallbackState{XT,VT,FT,GT,LMO<:LinearMinimizationOracle}
    t::Int64
    primal::Float64
    dual::Float64
    dual_gap::Float64
    time::Float64
    x::XT
    v::VT
    gamma::Float64
    f::FT
    grad!::GT
    lmo::LMO
    gradient::XT
end

struct CallbackActiveSetState{AT <: ActiveSet}
    base_state::CallbackState
    active_set::AT
end

function Base.getproperty(state::CallbackActiveSetState, f::Symbol)
    if f === :active_set
        return getfield(state, f)
    end
    base_state = getfield(state, :base_state)
    return getproperty(base_state, f)
end
