using Pkg
Pkg.activate(dirname(@__DIR__))
using TestEnv
TestEnv.activate()

using Random

# for bug with display
ENV["GKSwstype"] = "100"

example_files = filter(readdir(@__DIR__; join=true)) do f
    endswith(f, ".jl") && !occursin("large", f) && !occursin("result", f) && f != "activate.jl"
end

example_shuffle = randperm(length(example_files))

for file in example_files[example_shuffle[1:2]]
    @info "Including example $file"
    include(file)
end
