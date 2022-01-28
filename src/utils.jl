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
    if line_search isa Agnostic
        gamma = 2 // (2 + t)
    elseif line_search isa Goldenratio # FIX for general d
        gamma, _ = segment_search(
            f,
            grad!,
            x,
            d,
            gamma_max,
            linesearch_tol=linesearch_tol,
            inplace_gradient=true,
        )
    elseif line_search isa Backtracking # FIX for general d
        gamma, _ = backtrackingLS(
            f,
            gradient,
            x,
            d,
            gamma_max,
            linesearch_tol=linesearch_tol,
            step_lim=step_lim,
        )
    elseif line_search isa Nonconvex
        gamma = 1 / sqrt(t + 1)
    elseif line_search isa Shortstep
        gamma = min(max(fast_dot(gradient, d) * inv(L * norm(d)^2), 0), gamma_max)
    elseif line_search isa RationalShortstep
        gamma = min(max(fast_dot(gradient, d) * inv(L * fast_dot(d, d)), 0), gamma_max)
    elseif line_search isa FixedStep
        gamma = min(gamma0, gamma_max)
    elseif line_search isa Adaptive
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
    if norm(direction) > 0
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
    else
        return 0, L_est
    end
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

"""
plot_results

Given a series of list, generate subplots.
list_data_y -> contains a list of a list of lists (where each list refers to a subplot, and a list of lists refers to the y-values of the series inside a subplot).
list_data_x -> contains a list of a list of lists (where each list refers to a subplot, and a list of lists refers to the x-values of the series inside a subplot).
So if we have one plot with two series, these might look like:
    list_data_y = [[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]]
    list_data_x = [[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]]

And if we have two plots, each with two series, these might look like:
    list_data_y = [[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]], [[7, 8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12]]]
    list_data_x = [[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]], [[7, 8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12]]]

list_label -> contains the labels for the series that will be plotted,
which has to have a length equal to the number of series that are being plotted:
    list_label = ["Series 1", "Series 2"]

list_axis_x -> contains the labels for the x-axis that will be plotted, 
which has to have a length equal to the number of subplots: 
    list_axis_x = ["x-axis plot 1", "x-axis plot 1"]

list_axis_y -> Same as list_axis_x but for the y-axis

xscalelog -> A list of values indicating the type of axes to use in each subplot,
must be equal to the number of subplots:
    xscalelog = [:log, :identity]

yscalelog -> Same as xscalelog but for the y-axis

"""
function plot_results(
    list_data_y,
    list_data_x,
    list_label,
    list_axis_x,
    list_axis_y;
    filename=nothing,
    xscalelog=nothing,
    yscalelog=nothing,
    legend_position=nothing,
    list_style=fill(:solid, length(list_label)),
    list_color=get_color_palette(:auto, plot_color(:white)),
    list_markers=[
        :circle,
        :rect,
        :utriangle,
        :diamond,
        :hexagon,
        :+,
        :x,
        :star5,
        :cross,
        :xcross,
        :dtriangle,
        :rtriangle,
        :ltriangle,
        :pentagon,
        :heptagon,
        :octagon,
        :star4,
        :star6,
        :star7,
        :star8,
        :vline,
        :hline,
    ],
    number_markers_per_line=10,
)
    line_width = 3.0
    marker_size = 5.0
    transparency_markers = 0.45
    font_size_axis = 12
    font_size_legend = 9
    gr()
    plt = nothing
    list_plots = Plots.Plot{Plots.GRBackend}[]
    #Plot an appropiate number of plots
    for i in 1:length(list_data_x)
        for j in 1:length(list_data_x[i])
            if isnothing(xscalelog)
                xscale = :identity
            else
                xscale = xscalelog[i]
            end
            if isnothing(yscalelog)
                yscale = :log
            else
                yscale = yscalelog[i]
            end
            if isnothing(legend_position)
                position_legend = :best
                legend_display = true
            else
                position_legend = legend_position[i]
                if isnothing(position_legend)
                    legend_display = false
                else
                    legend_display = true
                end
            end
            if j == 1
                if legend_display
                    plt = plot(
                        list_data_x[i][j],
                        list_data_y[i][j],
                        label="",
                        xaxis=xscale,
                        yaxis=yscale,
                        ylabel=list_axis_y[i],
                        xlabel=list_axis_x[i],
                        legend=position_legend,
                        yguidefontsize=font_size_axis,
                        xguidefontsize=font_size_axis,
                        legendfontsize=font_size_legend,
                        width=line_width,
                        linestyle=list_style[j],
                        color=list_color[j],
                        grid=true,
                    )
                else
                    plt = plot(
                        list_data_x[i][j],
                        list_data_y[i][j],
                        label="",
                        xaxis=xscale,
                        yaxis=yscale,
                        ylabel=list_axis_y[i],
                        xlabel=list_axis_x[i],
                        yguidefontsize=font_size_axis,
                        xguidefontsize=font_size_axis,
                        width=line_width,
                        linestyle=list_style[j],
                        color=list_color[j],
                        grid=true,
                    )
                end
            else
                if legend_display
                    plot!(
                        list_data_x[i][j],
                        list_data_y[i][j],
                        label="",
                        width=line_width,
                        linestyle=list_style[j],
                        color=list_color[j],
                        legend=position_legend,
                    )
                else
                    plot!(
                        list_data_x[i][j],
                        list_data_y[i][j],
                        label="",
                        width=line_width,
                        linestyle=list_style[j],
                        color=list_color[j],
                    )
                end
            end
            if xscale == :log
                indices =
                    round.(
                        Int,
                        10 .^ (range(
                            log10(1),
                            log10(length(list_data_x[i][j])),
                            length=number_markers_per_line,
                        )),
                    )
                scatter!(
                    list_data_x[i][j][indices],
                    list_data_y[i][j][indices],
                    markershape=list_markers[j],
                    markercolor=list_color[j],
                    markersize=marker_size,
                    markeralpha=transparency_markers,
                    label=list_label[j],
                    legend=position_legend,
                )
            else
                scatter!(
                    view(
                        list_data_x[i][j],
                        1:length(list_data_x[i][j])÷number_markers_per_line:length(
                            list_data_x[i][j],
                        ),
                    ),
                    view(
                        list_data_y[i][j],
                        1:length(list_data_y[i][j])÷number_markers_per_line:length(
                            list_data_y[i][j],
                        ),
                    ),
                    markershape=list_markers[j],
                    markercolor=list_color[j],
                    markersize=marker_size,
                    markeralpha=transparency_markers,
                    label=list_label[j],
                    legend=position_legend,
                )
            end
        end
        push!(list_plots, plt)
    end
    fp = plot(list_plots..., layout=length(list_plots))
    plot!(size=(600, 400))
    if filename !== nothing
        savefig(fp, filename)
    end
    return fp
end


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

fast_dot(B::SparseArrays.SparseMatrixCSC, A::Matrix) = conj(fast_dot(A, B))

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
    @inbounds for j in 1:n
        for ridx in SparseArrays.nzrange(B, j)
            i = rows[ridx]
            v = vals[ridx]
            s += v * conj(A[i, j])
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
    return function push_trajectory!(data)
        return push!(storage, Tuple(data)[1:5])
    end
end

# default printing for LineSearchMethod is just showing the type
Base.print(io::IO, ls::LineSearchMethod) = print(io, split(string(typeof(ls)), ".")[end])

"""
    MonotonousStepSize{F}

Represents a monotonous open-loop step size.
Contains a halving factor `N` increased at each iteration until there is primal progress
`gamma = 2 / (t + 2) * 2^(-N)`
"""
mutable struct MonotonousStepSize{F} <: LineSearchMethod
    domain_oracle::F
    factor::Int
end

MonotonousStepSize(f::F) where {F<:Function} = MonotonousStepSize{F}(f, 0)
MonotonousStepSize() = MonotonousStepSize(x -> true)

Base.print(io::IO, ::MonotonousStepSize) = print(io, "MonotonousStepSize")

function line_search_wrapper(
    line_search::MonotonousStepSize,
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
    gamma = 2.0^(1 - line_search.factor) / (2 + t)
    xnew = x - gamma * d
    f0 = f(x)
    while !line_search.domain_oracle(xnew) || f(xnew) > f0
        line_search.factor += 1
        gamma = 2.0^(1 - line_search.factor) / (2 + t)
        @. xnew = x - gamma * d
    end
    return gamma, L
end

"""
    MonotonousNonConvexStepSize{F}

Represents a monotonous open-loop non-convex step size.
Contains a halving factor `N` increased at each iteration until there is primal progress
`gamma = 1 / sqrt(t + 1) * 2^(-N)`
"""
mutable struct MonotonousNonConvexStepSize{F} <: LineSearchMethod
    domain_oracle::F
    factor::Int
end

MonotonousNonConvexStepSize(f::F) where {F<:Function} = MonotonousNonConvexStepSize{F}(f, 0)
MonotonousNonConvexStepSize() = MonotonousNonConvexStepSize(x -> true)

Base.print(io::IO, ::MonotonousNonConvexStepSize) = print(io, "MonotonousNonConvexStepSize")

function line_search_wrapper(
    line_search::MonotonousNonConvexStepSize,
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
    gamma = 2.0^(-line_search.factor) / sqrt(1 + t)
    xnew = x - gamma * d
    f0 = f(x)
    while !line_search.domain_oracle(xnew) || f(xnew) > f0
        line_search.factor += 1
        gamma = 2.0^(-line_search.factor) / sqrt(1 + t)
        @. xnew = x - gamma * d
    end
    return gamma, L
end


function print_callback(data, format_string; print_header=false, print_footer=false)
    print_formatted(fmt, args...) = @eval @printf($fmt, $(args...))
    if print_header || print_footer
        temp = strip(format_string, ['\n'])
        temp = replace(temp, "%" => "")
        temp = replace(temp, "e" => "")
        temp = replace(temp, "i" => "")
        temp = replace(temp, "s" => "")
        temp = split(temp, " ")
        len = 0
        for i in temp
            len = len + parse(Int, i)
        end
        lenHeaderFooter = len + 2 + length(temp) - 1
        if print_footer
            line = "-"^lenHeaderFooter
            @printf("%s\n\n", line)
        end
        if print_header
            line = "-"^lenHeaderFooter
            @printf("\n%s\n", line)
            s_format_string = replace(format_string, "e" => "s")
            s_format_string = replace(s_format_string, "i" => "s")
            print_formatted(s_format_string, data...)
            @printf("%s\n", line)
        end
    else
        print_formatted(format_string, data...)
    end
end

"""
    momentum_iterate(iter::MomentumIterator) -> ρ

Method to implement for a type `MomentumIterator`.
Returns the next momentum value `ρ` and updates the iterator internal state.
"""
function momentum_iterate end

"""
    ExpMomentumIterator{T}

Iterator for the momentum used in the variant of Stochastic Frank-Wolfe.
Momentum coefficients are the values of the iterator:
`ρ_t = 1 - num / (offset + t)^exp`

The state corresponds to the iteration count.

Source:
Stochastic Conditional Gradient Methods: From Convex Minimization to Submodular Maximization
Aryan Mokhtari, Hamed Hassani, Amin Karbasi, JMLR 2020.
"""
mutable struct ExpMomentumIterator{T}
    exp::T
    num::T
    offset::T
    iter::Int
end

ExpMomentumIterator() = ExpMomentumIterator(2/3, 4.0, 8.0, 0)

function momentum_iterate(em::ExpMomentumIterator)
    em.iter += 1
    return 1 - em.num / (em.offset + em.iter)^(em.exp)
end

"""
    ConstantMomentumIterator{T}

Iterator for momentum with a fixed damping value, always return the value and a dummy state.
"""
struct ConstantMomentumIterator{T}
    v::T
end

momentum_iterate(em::ConstantMomentumIterator) = em.v

# batch sizes

"""
    batchsize_iterate(iter::BatchSizeIterator) -> b

Method to implement for a batch size iterator of type `BatchSizeIterator`.
Calling `batchsize_iterate` returns the next batch size and typically update the internal state of `iter`.
"""
function batchsize_iterate end

"""
    ConstantBatchIterator(batch_size)

Batch iterator always returning a constant batch size.
"""
struct ConstantBatchIterator
    batch_size::Int
end
 
batchsize_iterate(cbi::ConstantBatchIterator) = cbi.batch_size

"""
    IncrementBatchIterator(starting_batch_size, max_batch_size, [increment = 1])

Batch size starting at starting_batch_size and incrementing by `increment` at every iteration.
"""
mutable struct IncrementBatchIterator
    starting_batch_size::Int
    max_batch_size::Int
    increment::Int
    iter::Int
    maxreached::Bool
end

function IncrementBatchIterator(starting_batch_size::Int, max_batch_size::Int, increment::Int)
    return IncrementBatchIterator(starting_batch_size, max_batch_size, increment, 0, false)
end

function IncrementBatchIterator(starting_batch_size::Int, max_batch_size::Int)
    return IncrementBatchIterator(starting_batch_size, max_batch_size, 1, 0, false)
end

function batchsize_iterate(ibi::IncrementBatchIterator)
    if ibi.maxreached
        return ibi.max_batch_size
    end
    new_size = ibi.starting_batch_size + ibi.iter * ibi.increment
    ibi.iter += 1
    if new_size > ibi.max_batch_size
        ibi.maxreached = true
        return ibi.max_batch_size
    end
    return new_size
end
