struct CallbackState
    state
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
