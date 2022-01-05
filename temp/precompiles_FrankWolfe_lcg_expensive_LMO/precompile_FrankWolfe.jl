function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :epsilon, :memory_mode, :trajectory, :verbose), Tuple{Int64, Adaptive{Float64, Int64}, Float64, Float64, InplaceEmphasis, Bool, Bool}},typeof(frank_wolfe),Function,Function,BirkhoffPolytopeLMO,SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.77472156
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, SparseArrays.SparseMatrixCSC{Float64, Int64}, SparseArrays.SparseMatrixCSC{Float64, Int64}, Float64}}})   # time: 0.08597099
    Base.precompile(Tuple{Type{Adaptive}})   # time: 0.007466047
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.001471335
end
