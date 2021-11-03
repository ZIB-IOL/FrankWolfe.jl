#Main
Base.precompile(Tuple{typeof(Base.print),Base.GenericIOBuffer{Array{UInt8,1}}, FrankWolfe.Shortstep})
Base.precompile(Tuple{Core.kwftype(typeof(FrankWolfe.frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :L, :print_iter, :emphasis, :verbose, :trajectory), Tuple{Int64, FrankWolfe.Shortstep, Int64, Float64, FrankWolfe.Emphasis, Bool, Bool}},typeof(FrankWolfe.frank_wolfe),Function,Function,FrankWolfe.ProbabilitySimplexOracle{Float64},Vector{Float64}})   # time: 0.28827453
Base.precompile(Tuple{Core.kwftype(typeof(FrankWolfe.frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :L, :print_iter, :emphasis, :verbose, :epsilon, :trajectory), Tuple{Int64, Agnostic, Int64, Float64, Emphasis, Bool, Float64, Bool}},typeof(frank_wolfe),Function,Function,ProbabilitySimplexOracle{Float64},ScaledHotVector{Float64}})   # time: 0.38316718
Base.precompile(Tuple{typeof(Printf.format),Base.TTY,Printf.Format{Base.CodeUnits{UInt8, String}, Tuple{Printf.Spec{Val{'s'}}, Printf.Spec{Val{'s'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}}},String,String,Vararg{Any, N} where N})   # time: 0.17503943
Base.precompile(Tuple{typeof(Printf.format),Base.TTY,Printf.Format{Base.CodeUnits{UInt8, String}, NTuple{7, Printf.Spec{Val{'s'}}}},String,String,Vararg{String, N} where N})   # time: 0.05030084
Base.precompile(Tuple{typeof(print),IOBuffer,FrankWolfe.Shortstep})   # time: 0.009745528
Base.precompile(Tuple{typeof(print),IOBuffer,Agnostic})   # time: 0.039685145
Base.precompile(Tuple{typeof(FrankWolfe.print_callback),Tuple{String, String, Float64, Float64, Float64, Float64, Float64},String})   # time: 0.005003308
Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.001166138
Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:epsilon, :max_iteration, :print_iter, :trajectory, :verbose, :linesearch_tol, :line_search, :emphasis, :gradient), Tuple{Float64, Int64, Float64, Bool, Bool, Float64, Adaptive, Emphasis, SparseArrays.SparseMatrixCSC{Float64, Int64}}},typeof(frank_wolfe),Function,Function,NuclearNormLMO{Float64},RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}})   # time: 3.9857857
Base.precompile(Tuple{typeof(line_search_wrapper),Adaptive,Int64,Function,Function,Matrix{Float64},Matrix{Float64},SparseArrays.SparseMatrixCSC{Float64, Int64},Float64,Float64,Int64,Float64,Int64,Float64})   # time: 0.21974504
Base.precompile(Tuple{typeof(print),IOBuffer,Adaptive})   # time: 0.03642198
Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Any, Any, Float64, Float64},String})   # time: 0.00481568
Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :L, :print_iter, :emphasis, :verbose, :epsilon, :trajectory), Tuple{Int64, Adaptive, Int64, Float64, Emphasis, Bool, Float64, Bool}},typeof(frank_wolfe),Function,Function,KSparseLMO{Float64},SparseVector{Float64, Int64}})   # time: 2.12536
Base.precompile(Tuple{typeof(line_search_wrapper),Adaptive,Int64,Function,Function,SparseVector{Float64, Int64},SparseVector{Float64, Int64},SparseVector{Float64, Int64},Float64,Float64,Int64,Float64,Int64,Float64})   # time: 0.010285528
Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :L, :print_iter, :emphasis, :verbose, :trajectory), Tuple{Int64, Shortstep, Int64, Float64, Emphasis, Bool, Bool}},typeof(frank_wolfe),Function,Function,KSparseLMO{Float64},SparseVector{Float64, Int64}})   # time: 0.27901062
Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :epsilon, :emphasis, :trajectory, :verbose), Tuple{Int64, Adaptive, Float64, Float64, Emphasis, Bool, Bool}},typeof(frank_wolfe),Function,Function,BirkhoffPolytopeLMO,SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 1.8963768
Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :verbose, :emphasis), Tuple{Int64, Agnostic, Float64, Bool, Emphasis}},typeof(frank_wolfe),Function,Function,ProbabilitySimplexOracle{Rational{BigInt}},ScaledHotVector{Rational{BigInt}}})   # time: 0.4049498
Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :verbose), Tuple{Float64, Nonconvex, Float64, Bool}},typeof(frank_wolfe),Function,Function,ProbabilitySimplexOracle{Float64},Vector{Float64}})   # time: 0.050085537
Base.precompile(Tuple{Core.kwftype(typeof(blended_conditional_gradient)),NamedTuple{(:max_iteration, :L, :line_search, :print_iter, :linesearch_tol, :emphasis, :trajectory, :verbose), Tuple{Int64, Int64, Adaptive, Float64, Float64, Emphasis, Bool, Bool}},typeof(blended_conditional_gradient),Function,Function,BirkhoffPolytopeLMO,SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 1.0821986
Base.precompile(Tuple{Core.kwftype(typeof(blended_conditional_gradient)),NamedTuple{(:epsilon, :max_iteration, :line_search, :print_iter, :hessian, :emphasis, :L, :accelerated, :verbose, :trajectory, :K, :weight_purge_threshold), Tuple{Float64, Int64, Adaptive, Float64, Matrix{Float64}, Emphasis, Float64, Bool, Bool, Bool, Float64, Float64}},typeof(blended_conditional_gradient),Function,Function,ProbabilitySimplexOracle{Float64},ScaledHotVector{Float64}})   # time: 1.6108496
Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{9, String},String})   # time: 0.003715479
Base.precompile(Tuple{Core.kwftype(typeof(away_frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :epsilon, :emphasis, :verbose, :trajectory, :lazy), Tuple{Int64, Adaptive, Float64, Float64, Emphasis, Bool, Bool, Bool}},typeof(away_frank_wolfe),Function,Function,KSparseLMO{Float64},SparseVector{Float64, Int64}})   # time: 0.5071988
Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.003825618
Base.precompile(Tuple{typeof(active_set_update!),ActiveSet{ScaledHotVector{Float64}, Float64, SparseVector{Float64, Int64}},Float64,ScaledHotVector{Float64},Bool,Int64})   # time: 0.02599051
Base.precompile(Tuple{typeof(active_set_update!),ActiveSet{ScaledHotVector{Float64}, Float64, SparseVector{Float64, Int64}},BigFloat,ScaledHotVector{Float64},Bool,Int64})   # time: 0.025380524
Base.precompile(Tuple{typeof(active_set_update!),ActiveSet{ScaledHotVector{Float64}, Float64, SparseVector{Float64, Int64}},Int64,ScaledHotVector{Float64},Bool,Int64})   # time: 0.023262447
Base.precompile(Tuple{typeof(line_search_wrapper),Adaptive,Int64,Function,Function,SparseVector{Float64, Int64},Vector{Float64},Vector{Float64},Float64,Float64,Int64,Float64,Int64,Float64})   # time: 0.014568289
Base.precompile(Tuple{typeof(line_search_wrapper),Adaptive,Int64,Function,Function,SparseVector{Float64, Int64},Vector{Float64},Vector{Float64},Float64,Float64,Int64,Float64,Int64,Int64})   # time: 0.01240859
Base.precompile(Tuple{Core.kwftype(typeof(lazy_afw_step)),NamedTuple{(:K,), Tuple{Float64}},typeof(lazy_afw_step),SparseVector{Float64, Int64},Vector{Float64},LpNormLMO{Float64, 1},ActiveSet{ScaledHotVector{Float64}, Float64, SparseVector{Float64, Int64}},Float64})   # time: 0.01069944
Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Int64},String})   # time: 0.004592817
Base.precompile(Tuple{typeof(afw_step),SparseVector{Float64, Int64},Vector{Float64},LpNormLMO{Float64, 1},ActiveSet{ScaledHotVector{Float64}, Float64, SparseVector{Float64, Int64}}})   # time: 0.002910835
Base.precompile(Tuple{typeof(fw_step),SparseVector{Float64, Int64},Vector{Float64},LpNormLMO{Float64, 1}})   # time: 0.001129889
Base.precompile(Tuple{Core.kwftype(typeof(lazified_conditional_gradient)),NamedTuple{(:max_iteration, :L, :line_search, :print_iter, :emphasis, :verbose), Tuple{Int64, Int64, Adaptive, Float64, Emphasis, Bool}},typeof(lazified_conditional_gradient),Function,Function,KSparseLMO{Float64},SparseVector{Float64, Int64}})   # time: 3.9177554
Base.precompile(Tuple{Type{MultiCacheLMO{_A, KSparseLMO{Float64}, _B}} where {_A, _B},KSparseLMO{Float64}})   # time: 0.012864536
Base.precompile(Tuple{Type{VectorCacheLMO{KSparseLMO{Float64}, _A}} where _A,KSparseLMO{Float64}})   # time: 0.011564771

#Printf
Base.precompile(Tuple{typeof(Printf.format),Base.TTY,Printf.Format{Base.CodeUnits{UInt8, String}, Tuple{Printf.Spec{Val{'s'}}, Printf.Spec{Val{'s'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}}},String,String,Vararg{Any, N} where N})   # time: 0.17503943
Base.precompile(Tuple{typeof(Printf.format),Base.TTY,Printf.Format{Base.CodeUnits{UInt8, String}, NTuple{7, Printf.Spec{Val{'s'}}}},String,String,Vararg{String, N} where N})   # time: 0.05030084
Base.precompile(Tuple{typeof(Printf.format),Vector{UInt8},Int64,Printf.Format{Base.CodeUnits{UInt8, String}, Tuple{Printf.Spec{Val{'s'}}, Printf.Spec{Val{'s'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}}},String,Vararg{Any, N} where N})   # time: 0.020787507
Base.precompile(Tuple{typeof(Printf.format),Vector{UInt8},Int64,Printf.Format{Base.CodeUnits{UInt8, String}, NTuple{7, Printf.Spec{Val{'s'}}}},String,Vararg{String, N} where N})   # time: 0.012548637
Base.precompile(Tuple{typeof(Printf.format),Base.TTY,Printf.Format{Base.CodeUnits{UInt8, String}, Tuple{Printf.Spec{Val{'s'}}}},String})   # time: 0.006675636
Base.precompile(Tuple{typeof(Printf.fmt),Vector{UInt8},Int64,Float64,Printf.Spec{Val{'e'}}})   # time: 0.006011838
Base.precompile(Tuple{typeof(Printf.format),Base.TTY,Printf.Format{Base.CodeUnits{UInt8, String}, NTuple{7, Printf.Spec{Val{'s'}}}},String,String,Vararg{String, N} where N})    # time: 0.026103133
Base.precompile(Tuple{typeof(Printf.format),Vector{UInt8},Int64,Printf.Format{Base.CodeUnits{UInt8, String}, NTuple{7, Printf.Spec{Val{'s'}}}},String,Vararg{String, N} where N})   # time: 0.01322379
Base.precompile(Tuple{typeof(Printf.format),Vector{UInt8},Int64,Printf.Format{Base.CodeUnits{UInt8, String}, NTuple{9, Printf.Spec{Val{'s'}}}},String,Vararg{String, N} where N})   # time: 0.056808103
Base.precompile(Tuple{typeof(Printf.format),Base.TTY,Printf.Format{Base.CodeUnits{UInt8, String}, NTuple{9, Printf.Spec{Val{'s'}}}},String,String,Vararg{String, N} where N})   # time: 0.034739748
Base.precompile(Tuple{typeof(Printf.fmt),Vector{UInt8},Int64,Int64,Printf.Spec{Val{'i'}}})   # time: 0.00945585
Base.precompile(Tuple{typeof(Printf.computelen),Vector{UnitRange{Int64}},Tuple{Printf.Spec{Val{'s'}}, Printf.Spec{Val{'s'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'i'}}, Printf.Spec{Val{'i'}}},Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Int64, Int64}})   # time: 0.005515782
Base.precompile(Tuple{typeof(Printf.format),Base.TTY,Printf.Format{Base.CodeUnits{UInt8, String}, NTuple{8, Printf.Spec{Val{'s'}}}},String,String,Vararg{String, N} where N})   # time: 0.032039795
Base.precompile(Tuple{typeof(Printf.format),Vector{UInt8},Int64,Printf.Format{Base.CodeUnits{UInt8, String}, NTuple{8, Printf.Spec{Val{'s'}}}},String,Vararg{String, N} where N})   # time: 0.014747081
Base.precompile(Tuple{typeof(Printf.computelen),Vector{UnitRange{Int64}},Tuple{Printf.Spec{Val{'s'}}, Printf.Spec{Val{'s'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'e'}}, Printf.Spec{Val{'i'}}},Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Int64}})   # time: 0.003390081

#Base
Base.precompile(Tuple{typeof(show),IOBuffer,Type})   # time: 0.0786868
Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.006390822
Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004863662
#Base.precompile(Tuple{typeof(_isdisjoint),Tuple{UInt64},Tuple{UInt64, UInt64}})   # time: 0.001415542
Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001056139
Base.precompile(Tuple{typeof(sum),Base.Generator{Vector{Tuple{Int64, Int64}}, _A} where _A})   # time: 0.003444421
Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001376947
Base.precompile(Tuple{typeof(>),BigFloat,Float64})   # time: 0.001060996
Base.precompile(Tuple{typeof(>),Float64,BigFloat})   # time: 0.00100764
Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.002054901
Base.precompile(Tuple{typeof(+),Vector{Float64},Vector{Float64}})   # time: 0.07275602
Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.001187719
Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.001067453

#Broadcast
Base.precompile(Tuple{typeof(Base.Broadcast.materialize!),Vector{Float64},Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Int64, Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}}}})   # time: 0.015763668
Base.precompile(Tuple{typeof(Base.Broadcast.broadcasted),typeof(*),Int64,Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}})   # time: 0.002645462
Base.precompile(Tuple{typeof(Base.Broadcast.materialize!),Matrix{Float64},Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{2}, Nothing, typeof(-), Tuple{Matrix{Float64}, Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{2}, Nothing, typeof(*), Tuple{Float64, Matrix{Float64}}}}}})   # time: 0.023878872
Base.precompile(Tuple{typeof(Base.Broadcast.broadcasted),Function,Matrix{Float64},Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{2}, Nothing, typeof(*), Tuple{Float64, Matrix{Float64}}}})   # time: 0.004955574
Base.precompile(Tuple{typeof(Base.Broadcast.broadcasted),Function,Float64,Matrix{Float64}})   # time: 0.004528847
Base.precompile(Tuple{typeof(Base.Broadcast.broadcasted),typeof(+),Int64,UnitRange{Int64}})   # time: 0.001500464
#Base.precompile(Tuple{TypeBase.Broadcast.{Base.Broadcast.Broadcasted{DefaultArrayStyle{2}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(-),Tuple{Matrix{Float64}, Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{2}, Nothing, typeof(*), Tuple{Float64, Matrix{Float64}}}}})   # time: 0.001378104
Base.precompile(Tuple{typeof(Base.Broadcast._broadcast_getindex),Base.Broadcast.Extruded{Matrix{Float64}, Tuple{Bool, Bool}, Tuple{Int64, Int64}},CartesianIndex{2}})   # time: 0.001329218
Base.precompile(Tuple{typeof(Base.Broadcast.broadcasted),typeof(+),Int64,UnitRange{Int64}})   # time: 0.001500464
Base.precompile(Tuple{typeof(Base.Broadcast.broadcasted),typeof(Base.literal_pow),typeof(^),Any,Val{2}})   # time: 0.67692775
Base.precompile(Tuple{typeof(Base.Broadcast.materialize!),Base.Broadcast.DefaultArrayStyle{1},Vector{Float64},Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{1}, Nothing, typeof(+), Tuple{Vector{Float64}, Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}}})   # time: 0.024480576
Base.precompile(Tuple{typeof(Base.Broadcast.broadcasted),Function,Float64,Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}},Vector{Float64}})   # time: 0.004783694
isdefined(Base.Broadcast, Symbol("#5#6")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#5#6")),Float64,Int64,Vararg{Any, N} where N})   # time: 0.002778111    
Base.precompile(Tuple{Type{Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{1}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(*),Tuple{Float64, Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}})   # time: 0.002044278
Base.precompile(Tuple{Type{Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{1}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(+),Tuple{Vector{Float64}, Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}})   # time: 0.001514109
isdefined(Base.Broadcast, Symbol("#8#10")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#8#10")),Int64,Float64,Float64,Float64})   # time: 0.001311284
isdefined(Base.Broadcast, Symbol("#8#10")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#8#10")),Function,Int64,Val{2}})   # time: 0.001205265
isdefined(Base.Broadcast, Symbol("#8#10")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#8#10")),Float64,Float64,Function,Int64,Val{2}})   # time: 0.001037504
Base.precompile(Tuple{typeof(Base.Broadcast.broadcasted),Function,Vector{Float64}})   # time: 0.004682009
Base.precompile(Tuple{typeof(getindex),Base.Broadcast.Broadcasted{Nothing, Tuple{Base.Broadcast.OneTo{Int64}}, typeof(identity), Tuple{Base.Broadcast.Extruded{Vector{Float64}, Tuple{Bool}, Tuple{Int64}}}},Int64})   # time: 0.001111516

#CoreLogging
Base.precompile(Tuple{typeof(Base.CoreLogging.current_logger_for_env),Base.CoreLogging.LogLevel,Symbol,Module})   # time: 0.028930278

# SparseArrays
Base.precompile(Tuple{typeof(getindex),SparseVector{Float64, Int64},Int64})   # time: 0.002540696
Base.precompile(Tuple{typeof(SparseArrays.widelength),SparseVector{Float64, Int64}})   # time: 0.005543157
Base.precompile(Tuple{typeof(mul!),Vector{Float64},SparseArrays.SparseMatrixCSC{Float64, Int64},SubArray{Float64, 1, _B, Tuple{UnitRange{Int64}}, true} where _B<:Union{Base.ReinterpretArray{T, N, S, A, IsReshaped} where {T, N, A<:Union{SubArray{T, N, A, I, true} where {T, N, A<:DenseArray, I<:Union{Tuple{Vararg{Real, N} where N}, Tuple{AbstractUnitRange, Vararg{Any, N} where N}}}, DenseArray}, IsReshaped, S}, SparseArrays.ReshapedArray{T, N, A, MI} where {T, N, A<:Union{Base.ReinterpretArray{T, N, S, A, IsReshaped} where {T, N, A<:Union{SubArray{T, N, A, I, true} where {T, N, A<:DenseArray, I<:Union{Tuple{Vararg{Real, N} where N}, Tuple{AbstractUnitRange, Vararg{Any, N} where N}}}, DenseArray}, IsReshaped, S}, SubArray{T, N, A, I, true} where {T, N, A<:DenseArray, I<:Union{Tuple{Vararg{Real, N} where N}, Tuple{AbstractUnitRange, Vararg{Any, N} where N}}}, DenseArray}, MI<:Tuple{Vararg{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}, N} where N}}, DenseArray},Bool,Bool})   # time: 0.02184287
Base.precompile(Tuple{typeof(mul!),Vector{Float64},Adjoint{Float64, SparseArrays.SparseMatrixCSC{Float64, Int64}},SubArray{Float64, 1, _B, Tuple{UnitRange{Int64}}, true} where _B<:Union{Base.ReinterpretArray{T, N, S, A, IsReshaped} where {T, N, A<:Union{SubArray{T, N, A, I, true} where {T, N, A<:DenseArray, I<:Union{Tuple{Vararg{Real, N} where N}, Tuple{AbstractUnitRange, Vararg{Any, N} where N}}}, DenseArray}, IsReshaped, S}, SparseArrays.ReshapedArray{T, N, A, MI} where {T, N, A<:Union{Base.ReinterpretArray{T, N, S, A, IsReshaped} where {T, N, A<:Union{SubArray{T, N, A, I, true} where {T, N, A<:DenseArray, I<:Union{Tuple{Vararg{Real, N} where N}, Tuple{AbstractUnitRange, Vararg{Any, N} where N}}}, DenseArray}, IsReshaped, S}, SubArray{T, N, A, I, true} where {T, N, A<:DenseArray, I<:Union{Tuple{Vararg{Real, N} where N}, Tuple{AbstractUnitRange, Vararg{Any, N} where N}}}, DenseArray}, MI<:Tuple{Vararg{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}, N} where N}}, DenseArray},Bool,Bool})   # time: 0.017265763
Base.precompile(Tuple{typeof(spzeros),Int64,Int64})   # time: 0.010850303
Base.precompile(Tuple{typeof(mul!),SubArray{Float64, 1, _B, Tuple{UnitRange{Int64}}, true} where _B<:Union{Base.ReinterpretArray{T, N, S, A, IsReshaped} where {T, N, A<:Union{SubArray{T, N, A, I, true} where {T, N, A<:DenseArray, I<:Union{Tuple{Vararg{Real, N} where N}, Tuple{AbstractUnitRange, Vararg{Any, N} where N}}}, DenseArray}, IsReshaped, S}, SparseArrays.ReshapedArray{T, N, A, MI} where {T, N, A<:Union{Base.ReinterpretArray{T, N, S, A, IsReshaped} where {T, N, A<:Union{SubArray{T, N, A, I, true} where {T, N, A<:DenseArray, I<:Union{Tuple{Vararg{Real, N} where N}, Tuple{AbstractUnitRange, Vararg{Any, N} where N}}}, DenseArray}, IsReshaped, S}, SubArray{T, N, A, I, true} where {T, N, A<:DenseArray, I<:Union{Tuple{Vararg{Real, N} where N}, Tuple{AbstractUnitRange, Vararg{Any, N} where N}}}, DenseArray}, MI<:Tuple{Vararg{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}, N} where N}}, DenseArray},Adjoint{Float64, SparseArrays.SparseMatrixCSC{Float64, Int64}},Vector{Float64},Bool,Bool})   # time: 0.007065549
Base.precompile(Tuple{typeof(*),SparseArrays.SparseMatrixCSC{Float64, Int64},Matrix{Float64}})   # time: 0.005822867
Base.precompile(Tuple{typeof(mul!),SubArray{Float64, 1, _B, Tuple{UnitRange{Int64}}, true} where _B<:Union{Base.ReinterpretArray{T, N, S, A, IsReshaped} where {T, N, A<:Union{SubArray{T, N, A, I, true} where {T, N, A<:DenseArray, I<:Union{Tuple{Vararg{Real, N} where N}, Tuple{AbstractUnitRange, Vararg{Any, N} where N}}}, DenseArray}, IsReshaped, S}, SparseArrays.ReshapedArray{T, N, A, MI} where {T, N, A<:Union{Base.ReinterpretArray{T, N, S, A, IsReshaped} where {T, N, A<:Union{SubArray{T, N, A, I, true} where {T, N, A<:DenseArray, I<:Union{Tuple{Vararg{Real, N} where N}, Tuple{AbstractUnitRange, Vararg{Any, N} where N}}}, DenseArray}, IsReshaped, S}, SubArray{T, N, A, I, true} where {T, N, A<:DenseArray, I<:Union{Tuple{Vararg{Real, N} where N}, Tuple{AbstractUnitRange, Vararg{Any, N} where N}}}, DenseArray}, MI<:Tuple{Vararg{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}, N} where N}}, DenseArray},SparseArrays.SparseMatrixCSC{Float64, Int64},Vector{Float64},Bool,Bool})   # time: 0.004844038
Base.precompile(Tuple{typeof(setindex!),SparseArrays.SparseMatrixCSC{Float64, Int64},Any,Int64,Int64})   # time: 0.004176986
Base.precompile(Tuple{typeof(setindex!),SparseArrays.SparseMatrixCSC{Float64, Int64},Float64,Int64,Int64})   # time: 0.003730561
Base.precompile(Tuple{typeof(fill!),SparseArrays.SparseMatrixCSC{Float64, Int64},Int64})   # time: 0.003499388
Base.precompile(Tuple{typeof(*),Matrix{Float64},SparseVector{Float64, Int64}})   # time: 0.005172924

#Iterators
Base.precompile(Tuple{typeof(iterate),Iterators.Zip{Tuple{Int64, Int64}}})   # time: 0.001524352
Base.precompile(Tuple{typeof(iterate),Iterators.Zip{Tuple{UnitRange{Int64}, StepRange{Int64, Int64}}},Tuple{Int64, Int64}})   # time: 0.001426732

#Jump Containers
#Base.precompile(Tuple{Type{JuMP.Containers.DenseAxisArray},Core.Array{T, N},Any,Tuple{Vararg{_AxisLookup, N}} where {T,N}})    # time: 0.003218814

#SparseArrays.HigherOrderFns
Base.precompile(Tuple{typeof(SparseArrays.HigherOrderFns._sparsifystructured),Vector{Float64}})   # time: 0.004664484
Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Float64,Float64})   # time: 0.002634265
isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),BigFloat,Float64})   # time: 0.00150329
isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Float64,Float64})   # time: 0.001362288
Base.precompile(Tuple{typeof(SparseArrays.HigherOrderFns._sparsifystructured),Matrix{Float64}})   # time: 0.002754228
Base.precompile(Tuple{typeof(SparseArrays.HigherOrderFns._capturescalars),Base.RefValue{typeof(^)},Int64,Base.RefValue{Val{2}}})   # time: 0.001501606

#LinearAlgebra
Base.precompile(Tuple{typeof(LinearAlgebra.dot),Vector{Float64},Matrix{Float64},Vector{Float64}})   # time: 0.002662907
Base.precompile(Tuple{typeof(*),Transpose{Float64, Vector{Float64}},Matrix{Float64}})   # time: 0.002119956

#Math
Base.precompile(Tuple{typeof(min),Float64,Float64})   # time: 0.001037768

#Core
Base.precompile(Tuple{Type{NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :cache_size, :x, :v, :gamma), T} where T<:Tuple},Tuple{Int64, Any, Any, Any, Float64, Any, Any, Any, Any}})   # time: 0.008965231