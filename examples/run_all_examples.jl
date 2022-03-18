using Random

# for bug with display
ENV["GKSwstype"] = "100"

example_files = filter(readdir(@__DIR__, join=true)) do f
    endswith(f, ".jl") && !occursin("large", f) && !occursin("result", f) && !occursin("activate.jl", f) && !occursin("plot_utils.jl", f)
end

example_shuffle = randperm(length(example_files))

if !haskey(ENV, "ALL_EXAMPLES")
    example_shuffle = example_shuffle[1:2]
else
    @info "Running all examples"
end


for file in example_files[example_shuffle]
    @info "Including example $file"
    instruction = """include("$activate_file"); include("$file")"""
    run(`julia -e $instruction`)
end
