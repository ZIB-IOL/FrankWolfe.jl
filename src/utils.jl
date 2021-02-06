
"""
Slight modification of
Adaptive Step Size strategy from https://arxiv.org/pdf/1806.05123.pdf

Note: direction is opposite to the improving direction
norm(gradient, direction) > 0
TODO: 
- make emphasis aware and optimize
"""
function adaptive_step_size(f, gradient, x, direction, L_est; eta=0.9, tau=2, gamma_max=1)
    M = eta * L_est
    dot_dir = dot(gradient, direction)
    ndir2 = norm(direction)^2
    gamma = min(
        dot_dir / (M * ndir2),
        gamma_max,
    )
    while f(x - gamma * direction) - f(x) >
          -gamma * dot_dir +
          gamma^2 * ndir2 * M / 2
        M *= tau
        gamma = min(
            dot_dir / (M * ndir2),
            gamma_max,
        )
    end
    return M, gamma
end

# simple backtracking line search (not optimized)
# TODO:
# - code needs optimization

function backtrackingLS(
    f,
    grad_direction,
    x,
    y;
    line_search=true,
    linesearch_tol=1e-10,
    step_lim=20,
    lsTau=0.5,
)
    gamma = one(lsTau)
    d = y - x
    i = 0

    dot_gdir = dot(grad_direction, d)
    @assert dot_gdir ≤ 0
    if dot_gdir ≥ 0
        @warn "Non-improving"
        return i, 0 * gamma
    end

    oldVal = f(x)
    newVal = f(x + gamma * d)
    while newVal - oldVal > linesearch_tol * gamma * dot_gdir
        if i > step_lim
            if oldVal - newVal >= 0
                return i, gamma
            else
                return i, 0 * gamma
            end
        end
        gamma *= lsTau
        newVal = f(x + gamma * d)
        i += 1
    end
    return i, gamma
end

# simple golden-ratio based line search (not optimized)
# based on boostedFW paper code and adapted for julia
# TODO:
# - code needs optimization 

function segmentSearch(f, grad, x, y; line_search=true, linesearch_tol=1e-10)
    # restrict segment of search to [x, y]
    d = (y - x)
    left, right = copy(x), copy(y)

    # if the minimum is at an endpoint
    if dot(d, grad(x)) * dot(d, grad(y)) >= 0
        if f(y) <= f(x)
            return y, 1
        else
            return x, 0
        end
    end

    # apply golden-section method to segment
    gold = (1.0 + sqrt(5)) / 2.0
    improv = Inf
    while improv > linesearch_tol
        old_left, old_right = left, right
        new = left + (right - left) / (1.0 + gold)
        probe = new + (right - new) / 2.0
        if f(probe) <= f(new)
            left, right = new, right
        else
            left, right = left, probe
        end
        improv = norm(f(right) - f(old_right)) + norm(f(left) - f(old_left))
    end

    x_min = (left + right) / 2.0

    # compute step size gamma
    gamma = 0
    if line_search === true
        for i in 1:length(d)
            if d[i] != 0
                gamma = (x_min[i] - x[i]) / d[i]
                break
            end
        end
    end

    return x_min, gamma
end

"""
    MaybeHotVector{T}

Represents a vector of at most one value different from 0.
"""
struct MaybeHotVector{T} <: AbstractVector{T}
    active_val::T
    val_idx::Int
    len::Int
end

Base.size(v::MaybeHotVector) = (v.len,)

@inline function Base.getindex(v::MaybeHotVector{T}, idx::Integer) where {T}
    @boundscheck if !(1 ≤ idx ≤ length(v))
        throw(BoundsError(v, idx))
    end
    if v.val_idx != idx
        return zero(T)
    end
    return v.active_val
end

Base.sum(v::MaybeHotVector) = v.active_val

function LinearAlgebra.dot(v1::MaybeHotVector, v2::AbstractVector)
    return v1.active_val * v2[v1.val_idx]
end

LinearAlgebra.dot(v1::AbstractVector, v2::MaybeHotVector) = dot(v2, v1)

# warning, no bound check
function LinearAlgebra.dot(v1::MaybeHotVector, v2::MaybeHotVector)
    if length(v1) != length(v2)
        throw(DimensionMismatch("v1 and v2 do not have matching sizes"))
    end
    return v1.active_val * v2.active_val * (v1.val_idx == v2.val_idx)
end

function Base.:*(v::MaybeHotVector, x::Number)
    return MaybeHotVector(v.active_val * x, v.val_idx, v.len)
end

Base.:*(x::Number, v::MaybeHotVector) = v * x

##############################
### emphasis macro
##############################


macro emphasis(Emphasis, ex)
    return esc(quote
        if $Emphasis === memory
            @. $ex
        else
            $ex
        end
    end)
end

##############################
### Visualization etc
##############################

function plot_trajectories(data, label; filename=nothing)
    theme(:dark)
    # theme(:vibrant)
    gr()

    x = []
    y = []
    pit = nothing
    pti = nothing
    dit = nothing
    dti = nothing
    offset = 2
    for i in 1:length(data)
        trajectory = data[i]
        x = [trajectory[j][1] for j in offset:length(trajectory)]
        y = [trajectory[j][2] for j in offset:length(trajectory)]
        if i == 1
            pit = plot(
                x,
                y,
                label=label[i],
                xaxis=:log,
                yaxis=:log,
                ylabel="Primal",
                legend=:topright,
                yguidefontsize=8,
                xguidefontsize=8,
                legendfontsize=8,
            )
        else
            plot!(x, y, label=label[i])
        end
    end
    for i in 1:length(data)
        trajectory = data[i]
        x = [trajectory[j][5] for j in offset:length(trajectory)]
        y = [trajectory[j][2] for j in offset:length(trajectory)]
        if i == 1
            pti = plot(
                x,
                y,
                label=label[i],
                legend=false,
                xaxis=:log,
                yaxis=:log,
                yguidefontsize=8,
                xguidefontsize=8,
            )
        else
            plot!(x, y, label=label[i])
        end
    end
    for i in 1:length(data)
        trajectory = data[i]
        x = [trajectory[j][1] for j in offset:length(trajectory)]
        y = [trajectory[j][4] for j in offset:length(trajectory)]
        if i == 1
            dit = plot(
                x,
                y,
                label=label[i],
                legend=false,
                xaxis=:log,
                yaxis=:log,
                ylabel="Dual Gap",
                xlabel="Iterations",
                yguidefontsize=8,
                xguidefontsize=8,
            )
        else
            plot!(x, y, label=label[i])
        end
    end
    for i in 1:length(data)
        trajectory = data[i]
        x = [trajectory[j][5] for j in offset:length(trajectory)]
        y = [trajectory[j][4] for j in offset:length(trajectory)]
        if i == 1
            dti = plot(
                x,
                y,
                label=label[i],
                legend=false,
                xaxis=:log,
                yaxis=:log,
                xlabel="Time",
                yguidefontsize=8,
                xguidefontsize=8,
            )
        else
            plot!(x, y, label=label[i], legend=:topright)
        end
    end
    fp = plot(pit, pti, dit, dti, layout=(2, 2)) # layout = @layout([A{0.01h}; [B C; D E]]))
    plot!(size=(600, 400))
    if filename !== nothing
        savefig(fp, filename)
    end
    return fp
end
