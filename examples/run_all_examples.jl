using Random

example_files = filter(readdir(@__DIR__)) do f
    occursin(".jl", f) && !occursin("large", f) && !occursin("result", f) && f != "activate.jl"
end

example_shuffle = randperm(length(example_files))

for file in example_files[example_shuffle[1:2]]
    @info "running example $file"
    run(`julia $file`)
end
