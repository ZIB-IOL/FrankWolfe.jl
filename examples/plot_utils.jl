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
    Plots.gr()

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