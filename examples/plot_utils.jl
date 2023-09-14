using Plots
using FiniteDifferences

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
    line_width=3.0,
    marker_size=5.0,
    transparency_markers=0.45,
    font_size_axis=12,
    font_size_legend=9,
)
    gr()
    plt = nothing
    list_plots = Plots.Plot{Plots.GRBackend}[]
    #Plot an appropiate number of plots
    for i in eachindex(list_data_x)
        for j in eachindex(list_data_x[i])
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

# Recipe for plotting markers in plot_trajectories
@recipe function f(::Type{Val{:samplemarkers}}, x, y, z; n_markers=10, log=false)
    n = length(y)

    # Choose datapoints for markers
    if log
        xmin = log10(x[1])
        xmax = log10(x[end])
        thresholds = collect(xmin:(xmax-xmin)/(n_markers-1):xmax)
        indices = [argmin(i -> abs(t - log10(x[i])), eachindex(x)) for t in thresholds]
    else
        indices = 1:Int(ceil(length(x) / n_markers)):n
    end
    sx, sy = x[indices], y[indices]

    # add an empty series with the correct type for legend markers
    @series begin
        seriestype := :path
        markershape --> :auto
        x := []
        y := []
    end
    # add a series for the line
    @series begin
        primary := false # no legend entry
        markershape := :none # ensure no markers
        seriestype := :path
        seriescolor := get(plotattributes, :seriescolor, :auto)
        x := x
        y := y
    end
    # return  a series for the sampled markers
    primary := false
    seriestype := :scatter
    markershape --> :auto
    x := sx
    y := sy
end

function plot_trajectories(
    data,
    label;
    filename=nothing,
    xscalelog=false,
    legend_position=:topright,
    lstyle=fill(:solid, length(data)),
    marker_shapes=nothing,
    n_markers=10,
    reduce_size=false,
    primal_offset=1e-8,
    line_width=1.3,
)
    # theme(:dark)
    # theme(:vibrant)
    Plots.gr()

    x = []
    y = []
    offset = 2

    function sub_plot(idx_x, idx_y; legend=false, xlabel="", ylabel="", y_offset=0)

        fig = nothing

        for (i, trajectory) in enumerate(data)

            l = length(trajectory)
            if reduce_size && l > 1000
                indices = Int.(round.(collect(1:l/1000:l)))
                trajectory = trajectory[indices]
            end

            x = [trajectory[j][idx_x] for j in offset:length(trajectory)]
            y = [trajectory[j][idx_y] + y_offset for j in offset:length(trajectory)]

            if marker_shapes !== nothing && n_markers >= 2
                marker_args = Dict(
                    :st => :samplemarkers,
                    :n_markers => n_markers,
                    :shape => marker_shapes[i],
                    :log => xscalelog,
                )
            else
                marker_args = Dict()
            end

            if i == 1
                fig = plot(
                    x,
                    y,
                    label=label[i],
                    xaxis=xscalelog ? :log : :identity,
                    yaxis=:log,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    legend=legend,
                    yguidefontsize=8,
                    xguidefontsize=8,
                    legendfontsize=8,
                    width=line_width,
                    linestyle=lstyle[i];
                    marker_args...,
                )
            else
                plot!(x, y, label=label[i], width=line_width, linestyle=lstyle[i]; marker_args...)
            end
        end
        return fig
    end

    pit = sub_plot(1, 2; legend=legend_position, ylabel="Primal", y_offset=primal_offset)
    pti = sub_plot(5, 2; y_offset=primal_offset)
    dit = sub_plot(1, 4; xlabel="Iterations", ylabel="Dual Gap")
    dti = sub_plot(5, 4; xlabel="Time")

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

"""
Check if the gradient using finite differences matches the grad! provided.
"""
function check_gradients(grad!, f, gradient, num_tests=10, tolerance=1.0e-5)
    for i in 1:num_tests
        random_point = similar(gradient)
        random_point .= rand(length(gradient))
        grad!(gradient, random_point)
        if norm(grad(central_fdm(5, 1), f, random_point)[1] - gradient) > tolerance
            @warn "There is a noticeable difference between the gradient provided and
            the gradient computed using finite differences.:\n$(norm(grad(central_fdm(5, 1), f, random_point)[1] - gradient))"
        end
    end
end
