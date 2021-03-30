using FiniteDifferences

"""
line search wrapper
NOTE: The stepsize is defined as x - gamma * d

Returns the step size gamma and the Lipschitz estimate L
"""
function line_search_wrapper(
    line_search,
    t,
    f,
    grad!,
    x,
    d,
    gradient,
    dual_gap,
    L,
    gamma0,
    linesearch_tol,
    step_lim,
    gamma_max,
)
    if line_search == agnostic
        gamma = 2 // (2 + t)
    elseif line_search == goldenratio # FIX for general d
        gamma, _ = segment_search(
            f,
            grad!,
            x,
            d,
            gamma_max,
            linesearch_tol=linesearch_tol,
            inplace_gradient=true,
        )
    elseif line_search == backtracking # FIX for general d
        gamma, _ = backtrackingLS(
            f,
            gradient,
            x,
            d,
            gamma_max,
            linesearch_tol=linesearch_tol,
            step_lim=step_lim,
        )
    elseif line_search == nonconvex
        gamma = 1 / sqrt(t + 1)
    elseif line_search == shortstep
        gamma = min(max(fast_dot(gradient, d) * inv(L * norm(d)^2), 0), gamma_max)
    elseif line_search == rationalshortstep
        gamma = min(max(fast_dot(gradient, d) * inv(L * fast_dot(d, d)), 0), gamma_max)
    elseif line_search == fixed
        gamma = min(gamma0, gamma_max)
    elseif line_search == adaptive
        gamma, L = adaptive_step_size(f, grad!, gradient, x, d, L, gamma_max=gamma_max)
    end
    return gamma, L
end



"""
Slight modification of
Adaptive Step Size strategy from https://arxiv.org/pdf/1806.05123.pdf

Note: direction is opposite to the improving direction
norm(gradient, direction) > 0
TODO: 
- make emphasis aware and optimize
"""
function adaptive_step_size(
    f,
    grad!,
    gradient,
    x,
    direction,
    L_est;
    eta=0.9,
    tau=2,
    gamma_max=1,
    upgrade_accuracy=false,
)
    #If there is no initial smoothness estimate
    #try to build one from the definition.
    if isnothing(L_est) || !isfinite(L_est)
        epsilon_step = min(1.0e-3, gamma_max)
        gradient_stepsize_estimation = similar(gradient)
        grad!(gradient_stepsize_estimation, x - epsilon_step * direction)
        L_est = norm(gradient - gradient_stepsize_estimation) / (epsilon_step * norm(direction))
    end
    M = eta * L_est
    if !upgrade_accuracy
        dot_dir = fast_dot(gradient, direction)
        ndir2 = norm(direction)^2
    else
        direction = big.(direction)
        x = big.(x)
        dot_dir = fast_dot(big.(gradient), direction)
        ndir2 = norm(direction)^2
    end

    gamma = min(max(dot_dir / (M * ndir2), 0.0), gamma_max)
    while f(x - gamma * direction) - f(x) > -gamma * dot_dir + gamma^2 * ndir2 * M / 2
        M *= tau
        gamma = min(max(dot_dir / (M * ndir2), 0.0), gamma_max)
    end
    return gamma, M
end

# simple backtracking line search (not optimized)
# TODO:
# - code needs optimization

function backtrackingLS(
    f,
    grad_direction,
    x,
    d,
    gamma_max;
    line_search=true,
    linesearch_tol=1e-10,
    step_lim=20,
    lsTau=0.5,
)
    gamma = gamma_max * one(lsTau)
    i = 0

    dot_gdir = fast_dot(grad_direction, d)
    if dot_gdir ≤ 0
        @warn "Non-improving"
        return zero(gamma), i
    end

    oldVal = f(x)
    newVal = f(x - gamma * d)
    while newVal - oldVal > -linesearch_tol * gamma * dot_gdir
        if i > step_lim
            if oldVal - newVal >= 0
                return gamma, i
            else
                return zero(gamma), i
            end
        end
        gamma *= lsTau
        newVal = f(x - gamma * d)
        i += 1
    end
    return gamma, i
end

# simple golden-ratio based line search (not optimized)
# based on boostedFW paper code and adapted for julia
# TODO:
# - code needs optimization.
# In particular, passing a gradient container instead of allocating

function segment_search(
    f,
    grad,
    x,
    d,
    gamma_max;
    line_search=true,
    linesearch_tol=1e-10,
    inplace_gradient=true,
)
    # restrict segment of search to [x, y]
    y = x - gamma_max * d
    left, right = copy(x), copy(y)

    if inplace_gradient
        gradient = similar(d)
        grad(gradient, x)
        dgx = fast_dot(d, gradient)
        grad(gradient, y)
        dgy = fast_dot(d, gradient)
    else
        gradient = grad(x)
        dgx = fast_dot(d, gradient)
        gradient = grad(y)
        dgy = fast_dot(d, gradient)
    end

    # if the minimum is at an endpoint
    if dgx * dgy >= 0
        if f(y) <= f(x)
            return one(eltype(d)), y
        else
            return zero(eltype(d)), x
        end
    end

    # apply golden-section method to segment
    gold = (1 + sqrt(5)) / 2
    improv = Inf
    while improv > linesearch_tol
        old_left, old_right = left, right
        new = left + (right - left) / (1 + gold)
        probe = new + (right - new) / 2
        if f(probe) <= f(new)
            left, right = new, right
        else
            left, right = left, probe
        end
        improv = norm(f(right) - f(old_right)) + norm(f(left) - f(old_left))
    end

    x_min = (left + right) / 2


    # compute step size gamma
    gamma = zero(eltype(d))
    if line_search
        for i in eachindex(d)
            if d[i] != 0
                gamma = (x[i] - x_min[i]) / d[i]
                break
            end
        end
    end

    return gamma, x_min
end

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

function plot_trajectories(
    data,
    label;
    filename=nothing,
    xscalelog=false,
    legend_position=:topright,
    lstyle=fill(:solid, length(data)),
)
    # theme(:dark)
    # theme(:vibrant)
    gr()

    x = []
    y = []
    pit = nothing
    pti = nothing
    dit = nothing
    dti = nothing
    offset = 2
    xscale = xscalelog ? :log : :identity
    for i in 1:length(data)
        trajectory = data[i]
        x = [trajectory[j][1] for j in offset:length(trajectory)]
        y = [trajectory[j][2] for j in offset:length(trajectory)]
        if i == 1
            pit = plot(
                x,
                y,
                label=label[i],
                xaxis=xscale,
                yaxis=:log,
                ylabel="Primal",
                legend=legend_position,
                yguidefontsize=8,
                xguidefontsize=8,
                legendfontsize=8,
                width=1.3,
                linestyle=lstyle[i],
            )
        else
            plot!(x, y, label=label[i], width=1.3, linestyle=lstyle[i])
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
                xaxis=xscale,
                yaxis=:log,
                yguidefontsize=8,
                xguidefontsize=8,
                width=1.3,
                linestyle=lstyle[i],
            )
        else
            plot!(x, y, label=label[i], width=1.3, linestyle=lstyle[i])
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
                xaxis=xscale,
                yaxis=:log,
                ylabel="Dual Gap",
                xlabel="Iterations",
                yguidefontsize=8,
                xguidefontsize=8,
                width=1.3,
                linestyle=lstyle[i],
            )
        else
            plot!(x, y, label=label[i], width=1.3, linestyle=lstyle[i])
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
                xaxis=xscale,
                yaxis=:log,
                xlabel="Time",
                yguidefontsize=8,
                xguidefontsize=8,
                width=1.3,
                linestyle=lstyle[i],
            )
        else
            plot!(x, y, label=label[i], width=1.3, linestyle=lstyle[i])
        end
    end
    fp = plot(pit, pti, dit, dti, layout=(2, 2)) # layout = @layout([A{0.01h}; [B C; D E]]))
    plot!(size=(600, 400))
    if filename !== nothing
        savefig(fp, filename)
    end
    return fp
end

function plot_sparsity(data, label; filename=nothing, xscalelog=false, legend_position=:topright)
    # theme(:dark)
    # theme(:vibrant)
    gr()

    x = []
    y = []
    ps = nothing
    ds = nothing
    offset = 2
    xscale = xscalelog ? :log : :identity
    for i in 1:length(data)
        trajectory = data[i]
        x = [trajectory[j][6] for j in offset:length(trajectory)]
        y = [trajectory[j][2] for j in offset:length(trajectory)]
        if i == 1
            ps = plot(
                x,
                y,
                label=label[i],
                xaxis=xscale,
                yaxis=:log,
                ylabel="Primal",
                legend=legend_position,
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
        x = [trajectory[j][6] for j in offset:length(trajectory)]
        y = [trajectory[j][4] for j in offset:length(trajectory)]
        if i == 1
            ds = plot(
                x,
                y,
                label=label[i],
                legend=false,
                xaxis=xscale,
                yaxis=:log,
                ylabel="Dual",
                yguidefontsize=8,
                xguidefontsize=8,
            )
        else
            plot!(x, y, label=label[i])
        end
    end

    fp = plot(ps, ds, layout=(1, 2)) # layout = @layout([A{0.01h}; [B C; D E]]))
    plot!(size=(600, 200))
    if filename !== nothing
        savefig(fp, filename)
    end
    return fp
end

##############################################################
# simple benchmark of elementary costs of oracles and 
# critical components
##############################################################

# TODO: add actual use of T for the rand(n)

function benchmark_oracles(f, grad!, x_gen, lmo; k=100, nocache=true)
    x = x_gen()
    sv = sizeof(x) / 1024^2
    println("\nSize of single atom ($(eltype(x))): $sv MB\n")
    to = TimerOutput()
    @showprogress 1 "Testing f... " for i in 1:k
        x = x_gen()
        @timeit to "f" temp = f(x)
    end
    @showprogress 1 "Testing grad... " for i in 1:k
        x = x_gen()
        temp = similar(x)
        @timeit to "grad" grad!(temp, x)
    end
    @showprogress 1 "Testing lmo... " for i in 1:k
        x = x_gen()
        @timeit to "lmo" temp = compute_extreme_point(lmo, x)
    end
    @showprogress 1 "Testing dual gap... " for i in 1:k
        x = x_gen()
        gradient = similar(x)
        grad!(gradient, x)
        v = compute_extreme_point(lmo, gradient)
        @timeit to "dual gap" begin
            dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
        end
    end
    @showprogress 1 "Testing update... (Emphasis: blas) " for i in 1:k
        x = x_gen()
        gradient = similar(x)
        grad!(gradient, x)
        v = compute_extreme_point(lmo, gradient)
        gamma = 1 / 2
        @timeit to "update (blas)" @emphasis(blas, x = (1 - gamma) * x + gamma * v)
    end
    @showprogress 1 "Testing update... (Emphasis: memory) " for i in 1:k
        x = x_gen()
        gradient = similar(x)
        grad!(gradient, x)
        v = compute_extreme_point(lmo, gradient)
        gamma = 1 / 2
        # TODO: to be updated to broadcast version once data structure ScaledHotVector allows for it
        @timeit to "update (memory)" @emphasis(memory, x = (1 - gamma) * x + gamma * v)
    end
    if !nocache
        @showprogress 1 "Testing caching 100 points... " for i in 1:k
            @timeit to "caching 100 points" begin
                cache = [gen_x() for _ in 1:100]
                x = gen_x()
                gradient = similar(x)
                grad!(gradient, x)
                v = compute_extreme_point(lmo, gradient)
                gamma = 1 / 2
                test = (x -> fast_dot(x, gradient)).(cache)
                v = cache[argmin(test)]
                val = v in cache
            end
        end
    end
    print_timer(to)
    return nothing
end

"""
`isequal` without the checks. Assumes a and b have the same axes.
"""
function _unsafe_equal(a::AbstractArray, b::AbstractArray)
    if a === b
        return true
    end
    @inbounds for idx in eachindex(a)
        if a[idx] != b[idx]
            return false
        end
    end
    return true
end

_unsafe_equal(a, b) = isequal(a, b)

fast_dot(A, B) = dot(A, B)

fast_dot(B::SparseArrays.SparseMatrixCSC, A::Matrix) = fast_dot(A, B)

function fast_dot(A::Matrix{T1}, B::SparseArrays.SparseMatrixCSC{T2}) where {T1,T2}
    T = promote_type(T1, T2)
    (m, n) = size(A)
    if (m, n) != size(B)
        throw(DimensionMismatch("Size mismatch"))
    end
    s = zero(T)
    if m * n == 0
        return s
    end
    rows = SparseArrays.rowvals(B)
    vals = SparseArrays.nonzeros(B)
    for j in 1:n
        for ridx in SparseArrays.nzrange(B, j)
            i = rows[ridx]
            v = vals[ridx]
            s += v * A[i, j]
        end
    end
    return s
end

"""
Check if the gradient using finite differences matches the grad! provided.
"""
function check_gradients(grad!, f, gradient, num_tests=10, tolerance=1.0e-5)
    for i in 1:num_tests
        random_point = rand(length(gradient))
        grad!(gradient, random_point)
        if norm(grad(central_fdm(5, 1), f, random_point)[1] - gradient) > tolerance
            @warn "There is a noticeable difference between the gradient provided and
            the gradient computed using finite differences."
        end
    end
end

"""
    trajectory_callback(storage)

Callback pushing the state at each iteration to the passed storage.
The state data is only the 5 first fields, usually:
`(t,primal,dual,dual_gap,time)`
"""
function trajectory_callback(storage)
    function push_trajectory!(data)
        push!(storage, Tuple(data)[1:5])
    end
end
