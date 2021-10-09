using Pkg
Pkg.activate(@__DIR__)

using Random

# for bug with display
ENV["GKSwstype"] = "100"

example_files = filter(readdir(@__DIR__, join=true)) do f
    endswith(f, ".jl") && !occursin("large", f) && !occursin("result", f) && !occursin("activate.jl", f) && !occursin("plot_utils.jl", f)
end

example_shuffle = randperm(length(example_files))

for file in example_files[example_shuffle[1:2]]
    @info "running example $file"
    run(`julia $file`)
end
