
example_files = filter(readdir(@__DIR__)) do f
    occursin(".jl", f) && !occursin("large", f) && !occursin("result", f) && f != "activate.jl"
end

for file in example_files
    @info "running $file"
    run(`julia $file`)
end
