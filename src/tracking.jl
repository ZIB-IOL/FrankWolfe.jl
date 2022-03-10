"""
A function acting like the normal `grad!`
but tracking the number of calls.
"""
mutable struct TrackingGradient{G} <: Function
    grad!::G
    counter::Int
end

TrackingGradient(grad!) = TrackingGradient(grad!, 0)

function (tg::TrackingGradient)(storage, x)
    tg.counter += 1
    return tg.grad!(storage, x)
end


"""
A function acting like the normal objective `f`
but tracking the number of calls.
"""
mutable struct TrackingObjective{F} <: Function
    f::F
    counter::Int
end

TrackingObjective(f) = TrackingObjective(f, 0)

function (tf::TrackingObjective)(x)
    tf.counter += 1
    return tf.f(x)
end

function wrap_objective(to::TrackingObjective)
    function f(x)
        to.counter += 1
        return to.f(x)
    end
    function grad!(storage,x)
        to.counter += 1
        return to.g(storage ,x)
    end
    return (f,grad!)
end

"""
    TrackingLMO{LMO}(lmo)

An LMO wrapping another one and tracking the number of calls.
"""
mutable struct TrackingLMO{LMO} <: LinearMinimizationOracle
    lmo::LMO
    counter::Int
end

function compute_extreme_point(lmo::TrackingLMO, x; kwargs...)
    lmo.counter += 1
    return compute_extreme_point(lmo.lmo, x)
end

is_tracking_lmo(lmo) = false
is_tracking_lmo(lmo::TrackingLMO) = true

TrackingLMO(lmo) = TrackingLMO(lmo, 0)


"""
A function acting like the passed callback,
    but adding the state to the storage variable.
    The state data is only the 5 first fields, gamma and 3 call counters, usually
`(t, primal, dual, dual_gap, time, gamma, function_calls, gradient_calls, lmo_calls)`
"""
mutable struct TrackingCallback{C} <: Function
    callback::C
    storage::Vector
end

TrackingCallback(callback) = TrackingCallback(callback, [])
TrackingCallback() = TrackingCallback(state->false, [])

function (tc::TrackingCallback)(state)
    base_tuple = Tuple(state)[1:5]
    complete_tuple = tuple(base_tuple..., state.gamma, state.f.counter, state.grad.counter, state.lmo.counter)
    push!(tc.storage, complete_tuple)
    return tc.callback(state)
end

"""
A function acting like the passed callback for cached LMOs,
    but adding the state to the storage variable.
    The state data is only the 5 first fields, gamma and 3 call counters, usually
`(t, primal, dual, dual_gap, time, gamma, function_calls, gradient_calls, lmo_calls)`
"""
mutable struct TrackingCachedCallback{C} <: Function
    callback::C
    storage::Vector
end

TrackingCachedCallback(callback) = TrackingCachedCallback(callback, [])
TrackingCachedCallback() = TrackingCachedCallback(state->false, [])

function (tc::TrackingCachedCallback)(state)
    base_tuple = Tuple(state)[1:5]
    complete_tuple = tuple(base_tuple..., state.gamma, state.f.counter, state.grad.counter, state.lmo.inner.counter)
    push!(tc.storage, complete_tuple)
    return tc.callback(state)
end


"""

Runs a function at a commit on a repo and afterwards goes back
to the original commit / branch.
"""
function withcommit(f, repo, commit)
    original_commit = shastring(repo, "HEAD")
    LibGit2.transact(repo) do r
        branch = try LibGit2.branch(r) catch err; nothing end
        try
            LibGit2.checkout!(r, shastring(r, commit))
            f()
        catch err
            rethrow(err)
        finally
            if branch !== nothing
                LibGit2.branch!(r, branch)
            else
                LibGit2.checkout!(r, original_commit)
            end
        end
    end
end

shastring(r::LibGit2.GitRepo, targetname) = string(LibGit2.revparseid(r, targetname))
shastring(dir::AbstractString, targetname) = LibGit2.with(r -> shastring(r, targetname), LibGit2.GitRepo(dir))