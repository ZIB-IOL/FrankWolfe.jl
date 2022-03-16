using Random

# for bug with display
ENV["GKSwstype"] = "100"

example_files = filter(readdir(@__DIR__; join=true)) do f
    endswith(f, ".jl") && !occursin("large", f) && !occursin("result", f) && f != "activate.jl"
end

example_shuffle = randperm(length(example_files))

activate_file = joinpath(@__DIR__, "activate.jl")

for file in example_files[example_shuffle[1:2]]
    @info "Including example $file"
    instruction = """include("$activate_file"); include("$file")"""
    run(`julia -e $instruction`)
end
