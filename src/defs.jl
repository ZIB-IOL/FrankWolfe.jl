

"""
    MemoryEmphasis

Emphasis given to the algorithm for memory-saving or not.

Concrete subtypes:

- [`InplaceMemoryEmphasis`](@ref) (the default, meant to save memory)
- [`OutplaceMemoryEmphasis`](@ref) (may be faster for small dimensions)
"""
abstract type MemoryEmphasis end

"""
    InplaceEmphasis

In-place version of [`MemoryEmphasis`](@ref).
"""
struct InplaceEmphasis <: MemoryEmphasis end

"""
    OutplaceEmphasis

Out-of-place version of [`MemoryEmphasis`](@ref).
"""
struct OutplaceEmphasis <: MemoryEmphasis end

@enum StepType begin
    ST_INITIAL = 1
    ST_REGULAR = 2
    ST_LAZY = 3
    ST_LAZYSTORAGE = 4
    ST_DUALSTEP = 5
    ST_AWAY = 6
    ST_PAIRWISE = 7
    ST_DROP = 8
    ST_SIMPLEXDESCENT = 101
    ST_DCA_OUTER = 201
    ST_LAST = 1000
    ST_POSTPROCESS = 1001
end

const steptype_string = (
    ST_INITIAL="I",
    ST_REGULAR="FW",
    ST_LAZY="L",
    ST_LAZYSTORAGE="LL",
    ST_DUALSTEP="LD",
    ST_AWAY="A",
    ST_PAIRWISE="P",
    ST_DROP="D",
    ST_SIMPLEXDESCENT="SD",
    ST_DCA_OUTER="DCA",
    ST_LAST="Last",
    ST_POSTPROCESS="PP",
)

"""
    CallbackState

Main structure created before and passed to the callback in first position.

# Fields

- `t`
- `primal`
- `dual`
- `dual_gap`
- `time`
- `x`
- `v`
- `d`
- `gamma`
- `f`
- `grad!`
- `lmo`
- `gradient`
- `step_type`
"""
struct CallbackState{TP,TDV,TDG,XT,VT,DT,TG,FT,GFT,LMO,GT}
    t::Int
    primal::TP
    dual::TDV
    dual_gap::TDG
    time::Float64
    x::XT
    v::VT
    d::DT
    gamma::TG
    f::FT
    grad!::GFT
    lmo::LMO
    gradient::GT
    step_type::StepType
end

"""
    callback_state(state::CallbackState)

Select a subset of fields from [`CallbackState`](@ref) to include in the trajectory: `(t, primal, dual, dual_gap, time)`.
"""
function callback_state(state::CallbackState)
    return (state.t, state.primal, state.dual, state.dual_gap, state.time)
end
