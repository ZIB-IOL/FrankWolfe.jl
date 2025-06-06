Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (:max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :trajectory),
            Tuple{Int64,Shortstep{Float64},Float64,InplaceEmphasis,Bool,Bool},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        KSparseLMO{Float64},
        SparseVector{Float64,Int64},
    },
)   # time: 0.31842098
Base.precompile(Tuple{Type{Shortstep},Float64})   # time: 0.010073511
Base.precompile(
    Tuple{
        Core.kwftype(typeof(print_callback)),
        NamedTuple{(:print_header,),Tuple{Bool}},
        typeof(print_callback),
        Vector{String},
        String,
    },
)   # time: 0.001591955
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (:max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :trajectory),
            Tuple{Int64,Adaptive{Float64,Int64},Float64,InplaceEmphasis,Bool,Bool},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        KSparseLMO{Float64},
        SparseVector{Float64,Int64},
    },
)   # time: 0.061905436
Base.precompile(Tuple{Type{Adaptive}})   # time: 0.003947135
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (:max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :trajectory),
            Tuple{Int64,Agnostic{Float64},Float64,InplaceEmphasis,Bool,Bool},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        KSparseLMO{Float64},
        SparseVector{Float64,Int64},
    },
)   # time: 0.06750039
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (:max_iteration, :line_search, :print_iter, :verbose, :memory_mode),
            Tuple{Int64,Agnostic{Float64},Float64,Bool,OutplaceEmphasis},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        ProbabilitySimplexOracle{Rational{BigInt}},
        ScaledHotVector{Rational{BigInt}},
    },
)   # time: 0.2661169
Base.precompile(
    Tuple{typeof(print_callback),Tuple{String,String,Float64,Any,Float64,Float64,Float64},String},
)   # time: 0.004444123
Base.precompile(
    Tuple{typeof(dot),ScaledHotVector{Rational{BigInt}},ScaledHotVector{Rational{BigInt}}},
)   # time: 0.001534546
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (:max_iteration, :line_search, :print_iter, :memory_mode, :verbose),
            Tuple{Float64,Agnostic{Float64},Float64,InplaceEmphasis,Bool},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        ProbabilitySimplexOracle{Rational{BigInt}},
        ScaledHotVector{Rational{BigInt}},
    },
)   # time: 0.43984142
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (
                :max_iteration,
                :line_search,
                :print_iter,
                :memory_mode,
                :verbose,
                :epsilon,
                :trajectory,
            ),
            Tuple{Int64,Adaptive{Float64,Int64},Float64,InplaceEmphasis,Bool,Float64,Bool},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        KSparseLMO{Float64},
        SparseVector{Float64,Int64},
    },
)   # time: 0.26637408
Base.precompile(
    Tuple{Core.kwftype(typeof(Type)),NamedTuple{(:L_est,),Tuple{Float64}},Type{Adaptive}},
)   # time: 0.00490745
Base.precompile(
    Tuple{
        Core.kwftype(typeof(print_callback)),
        NamedTuple{(:print_header,),Tuple{Bool}},
        typeof(print_callback),
        NTuple{9,String},
        String,
    },
)   # time: 0.005392282
Base.precompile(
    Tuple{
        Core.kwftype(typeof(lp_separation_oracle)),
        NamedTuple{(:inplace_loop, :force_fw_step),Tuple{Bool,Bool}},
        typeof(lp_separation_oracle),
        BirkhoffPolytopeLMO,
        ActiveSet{
            SparseArrays.SparseMatrixCSC{Float64,Int64},
            Float64,
            SparseArrays.SparseMatrixCSC{Float64,Int64},
        },
        SparseArrays.SparseMatrixCSC{Float64,Int64},
        Float64,
        Float64,
    },
)   # time: 0.001709011
Base.precompile(
    Tuple{
        Core.kwftype(typeof(blended_conditional_gradient)),
        NamedTuple{
            (
                :epsilon,
                :max_iteration,
                :line_search,
                :print_iter,
                :memory_mode,
                :verbose,
                :trajectory,
                :sparsity_control,
                :weight_purge_threshold,
            ),
            Tuple{
                Float64,
                Int64,
                Adaptive{Float64,Int64},
                Float64,
                InplaceEmphasis,
                Bool,
                Bool,
                Float64,
                Float64,
            },
        },
        typeof(blended_conditional_gradient),
        Function,
        Function,
        ProbabilitySimplexOracle{Float64},
        ScaledHotVector{Float64},
    },
)   # time: 0.5107306
Base.precompile(
    Tuple{
        Core.kwftype(typeof(blended_conditional_gradient)),
        NamedTuple{
            (
                :epsilon,
                :max_iteration,
                :line_search,
                :print_iter,
                :hessian,
                :memory_mode,
                :accelerated,
                :verbose,
                :trajectory,
                :sparsity_control,
                :weight_purge_threshold,
            ),
            Tuple{
                Float64,
                Int64,
                Adaptive{Float64,Int64},
                Float64,
                Matrix{Float64},
                InplaceEmphasis,
                Bool,
                Bool,
                Bool,
                Float64,
                Float64,
            },
        },
        typeof(blended_conditional_gradient),
        Function,
        Function,
        KSparseLMO{Float64},
        SparseVector{Float64,Int64},
    },
)   # time: 0.44398972
Base.precompile(
    Tuple{
        Core.kwftype(typeof(blended_conditional_gradient)),
        NamedTuple{
            (
                :epsilon,
                :max_iteration,
                :line_search,
                :print_iter,
                :memory_mode,
                :verbose,
                :trajectory,
                :sparsity_control,
                :weight_purge_threshold,
            ),
            Tuple{
                Float64,
                Int64,
                Adaptive{Float64,Int64},
                Float64,
                InplaceEmphasis,
                Bool,
                Bool,
                Float64,
                Float64,
            },
        },
        typeof(blended_conditional_gradient),
        Function,
        Function,
        KSparseLMO{Float64},
        SparseVector{Float64,Int64},
    },
)   # time: 0.38173005
Base.precompile(
    Tuple{
        Core.kwftype(typeof(blended_pairwise_conditional_gradient)),
        NamedTuple{
            (
                :epsilon,
                :max_iteration,
                :line_search,
                :print_iter,
                :memory_mode,
                :verbose,
                :trajectory,
            ),
            Tuple{Float64,Int64,Adaptive{Float64,Int64},Float64,InplaceEmphasis,Bool,Bool},
        },
        typeof(blended_pairwise_conditional_gradient),
        Function,
        Function,
        KSparseLMO{Float64},
        SparseVector{Float64,Int64},
    },
)   # time: 0.18068784
Base.precompile(
    Tuple{
        Core.kwftype(typeof(print_callback)),
        NamedTuple{(:print_header,),Tuple{Bool}},
        typeof(print_callback),
        NTuple{8,String},
        String,
    },
)   # time: 0.005311987
Base.precompile(
    Tuple{
        typeof(active_set_update!),
        ActiveSet{SparseVector{Float64,Int64},Float64,SparseVector{Float64,Int64}},
        Float64,
        SparseVector{Float64,Int64},
        Bool,
        Nothing,
    },
)   # time: 0.3297556
Base.precompile(
    Tuple{
        typeof(perform_line_search),
        Adaptive{Float64,Int64},
        Int64,
        Function,
        Function,
        SparseVector{Float64,Int64},
        SparseVector{Float64,Int64},
        Vector{Float64},
        Float64,
        SparseVector{Float64,Int64},
    },
)   # time: 0.0803171
Base.precompile(Tuple{Type{ActiveSet},Vector{Tuple{Float64,SparseVector{Float64,Int64}}}})   # time: 0.06371654
Base.precompile(
    Tuple{
        Core.kwftype(typeof(lazy_afw_step)),
        NamedTuple{(:sparsity_control,),Tuple{Float64}},
        typeof(lazy_afw_step),
        SparseVector{Float64,Int64},
        SparseVector{Float64,Int64},
        KSparseLMO{Float64},
        ActiveSet{SparseVector{Float64,Int64},Float64,SparseVector{Float64,Int64}},
        Float64,
        SparseVector{Float64,Int64},
    },
)   # time: 0.032522447
Base.precompile(
    Tuple{
        typeof(perform_line_search),
        Adaptive{Float64,Int64},
        Int64,
        Function,
        Function,
        SparseVector{Float64,Int64},
        SparseVector{Float64,Int64},
        SparseVector{Float64,Int64},
        Float64,
        SparseVector{Float64,Int64},
    },
)   # time: 0.007260863
Base.precompile(
    Tuple{
        typeof(active_set_update!),
        ActiveSet{SparseVector{Float64,Int64},Float64,SparseVector{Float64,Int64}},
        Float64,
        SparseVector{Float64,Int64},
        Bool,
        Int64,
    },
)   # time: 0.007143604
Base.precompile(
    Tuple{typeof(print_callback),Tuple{String,String,Any,Any,Float64,Float64,Float64,Int64},String},
)   # time: 0.003918484
Base.precompile(
    Tuple{
        typeof(afw_step),
        SparseVector{Float64,Int64},
        SparseVector{Float64,Int64},
        KSparseLMO{Float64},
        ActiveSet{SparseVector{Float64,Int64},Float64,SparseVector{Float64,Int64}},
        SparseVector{Float64,Int64},
    },
)   # time: 0.002095744
Base.precompile(
    Tuple{
        typeof(active_set_update_iterate_pairwise!),
        SparseVector{Float64,Int64},
        Float64,
        SparseVector{Float64,Int64},
        SparseVector{Float64,Int64},
    },
)   # time: 0.22706518
Base.precompile(
    Tuple{
        typeof(active_set_initialize!),
        ActiveSet{SparseVector{Float64,Int64},Float64,SparseVector{Float64,Int64}},
        SparseVector{Float64,Int64},
    },
)   # time: 0.07041422
Base.precompile(
    Tuple{
        typeof(deleteat!),
        ActiveSet{SparseVector{Float64,Int64},Float64,SparseVector{Float64,Int64}},
        Int64,
    },
)   # time: 0.001469705
Base.precompile(
    Tuple{
        Core.kwftype(typeof(away_frank_wolfe)),
        NamedTuple{
            (
                :max_iteration,
                :line_search,
                :print_iter,
                :epsilon,
                :memory_mode,
                :verbose,
                :trajectory,
                :lazy,
            ),
            Tuple{Int64,Adaptive{Float64,Int64},Float64,Float64,InplaceEmphasis,Bool,Bool,Bool},
        },
        typeof(away_frank_wolfe),
        Function,
        Function,
        KSparseLMO{Float64},
        SparseVector{Float64,Int64},
    },
)   # time: 1.2606552
Base.precompile(
    Tuple{
        Core.kwftype(typeof(away_frank_wolfe)),
        NamedTuple{
            (
                :max_iteration,
                :line_search,
                :print_iter,
                :epsilon,
                :memory_mode,
                :verbose,
                :away_steps,
                :trajectory,
            ),
            Tuple{Int64,Adaptive{Float64,Int64},Float64,Float64,InplaceEmphasis,Bool,Bool,Bool},
        },
        typeof(away_frank_wolfe),
        Function,
        Function,
        KSparseLMO{Float64},
        SparseVector{Float64,Int64},
    },
)   # time: 0.025728334
Base.precompile(
    Tuple{
        Core.kwftype(typeof(away_frank_wolfe)),
        NamedTuple{
            (
                :max_iteration,
                :line_search,
                :print_iter,
                :memory_mode,
                :verbose,
                :epsilon,
                :trajectory,
                :away_steps,
            ),
            Tuple{Int64,Adaptive{Float64,Int64},Float64,InplaceEmphasis,Bool,Float64,Bool,Bool},
        },
        typeof(away_frank_wolfe),
        Function,
        Function,
        KSparseLMO{Float64},
        SparseVector{Float64,Int64},
    },
)   # time: 0.042767525
Base.precompile(
    Tuple{
        Core.kwftype(typeof(lazified_conditional_gradient)),
        NamedTuple{
            (:max_iteration, :line_search, :print_iter, :memory_mode, :verbose),
            Tuple{Int64,Adaptive{Float64,Int64},Float64,InplaceEmphasis,Bool},
        },
        typeof(lazified_conditional_gradient),
        Function,
        Function,
        KSparseLMO{Float64},
        SparseVector{Float64,Int64},
    },
)   # time: 0.71887785
Base.precompile(
    Tuple{Type{MultiCacheLMO{_A,KSparseLMO{Float64},_B}} where {_A,_B},KSparseLMO{Float64}},
)   # time: 0.007678914
Base.precompile(
    Tuple{typeof(print_callback),Tuple{String,String,Float64,Any,Any,Float64,Float64,Any},String},
)   # time: 0.007181576
Base.precompile(Tuple{Type{VectorCacheLMO{KSparseLMO{Float64},_A}} where _A,KSparseLMO{Float64}})   # time: 0.005332965
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (
                :max_iteration,
                :line_search,
                :print_iter,
                :epsilon,
                :memory_mode,
                :trajectory,
                :verbose,
            ),
            Tuple{Int64,Adaptive{Float64,Int64},Float64,Float64,InplaceEmphasis,Bool,Bool},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        BirkhoffPolytopeLMO,
        SparseArrays.SparseMatrixCSC{Float64,Int64},
    },
)   # time: 0.77472156
Base.precompile(
    Tuple{
        Core.kwftype(typeof(lazified_conditional_gradient)),
        NamedTuple{
            (
                :max_iteration,
                :epsilon,
                :line_search,
                :print_iter,
                :memory_mode,
                :trajectory,
                :verbose,
            ),
            Tuple{Int64,Float64,Adaptive{Float64,Int64},Float64,InplaceEmphasis,Bool,Bool},
        },
        typeof(lazified_conditional_gradient),
        Function,
        Function,
        BirkhoffPolytopeLMO,
        SparseArrays.SparseMatrixCSC{Float64,Int64},
    },
)   # time: 0.326898
Base.precompile(
    Tuple{typeof(print_callback),Tuple{String,String,Any,Any,Any,Float64,Float64,Any},String},
)   # time: 0.010050932
Base.precompile(
    Tuple{Type{MultiCacheLMO{_A,BirkhoffPolytopeLMO,_B}} where {_A,_B},BirkhoffPolytopeLMO},
)   # time: 0.007305136
Base.precompile(Tuple{Type{VectorCacheLMO{BirkhoffPolytopeLMO,_A}} where _A,BirkhoffPolytopeLMO})   # time: 0.005527968
Base.precompile(
    Tuple{
        Core.kwftype(typeof(lazified_conditional_gradient)),
        NamedTuple{
            (
                :max_iteration,
                :line_search,
                :print_iter,
                :epsilon,
                :memory_mode,
                :trajectory,
                :cache_size,
                :verbose,
            ),
            Tuple{Int64,Adaptive{Float64,Int64},Float64,Float64,InplaceEmphasis,Bool,Int64,Bool},
        },
        typeof(lazified_conditional_gradient),
        Function,
        Function,
        BirkhoffPolytopeLMO,
        SparseArrays.SparseMatrixCSC{Float64,Int64},
    },
)   # time: 0.15854251
Base.precompile(
    Tuple{
        Core.kwftype(typeof(compute_extreme_point)),
        NamedTuple{(:threshold, :greedy),Tuple{Float64,Bool}},
        typeof(compute_extreme_point),
        MultiCacheLMO{_A,BirkhoffPolytopeLMO} where _A,
        SparseArrays.SparseMatrixCSC{Float64,Int64},
    },
)   # time: 0.031452376
Base.precompile(Tuple{typeof(length),MultiCacheLMO{_A,BirkhoffPolytopeLMO} where _A})   # time: 0.01073467
Base.precompile(
    Tuple{
        Core.kwftype(typeof(compute_extreme_point)),
        NamedTuple{(:threshold, :greedy),_A} where _A<:Tuple{Any,Bool},
        typeof(compute_extreme_point),
        MultiCacheLMO{_A,BirkhoffPolytopeLMO} where _A,
        SparseArrays.SparseMatrixCSC{Float64,Int64},
    },
)   # time: 0.010322329
Base.precompile(
    Tuple{
        Core.kwftype(typeof(compute_extreme_point)),
        NamedTuple{(:threshold, :greedy),Tuple{Float64,Bool}},
        typeof(compute_extreme_point),
        MultiCacheLMO{500,BirkhoffPolytopeLMO,SparseArrays.SparseMatrixCSC{Float64,Int64}},
        SparseArrays.SparseMatrixCSC{Float64,Int64},
    },
)   # time: 0.003286707
Base.precompile(
    Tuple{
        Core.kwftype(typeof(away_frank_wolfe)),
        NamedTuple{
            (
                :max_iteration,
                :line_search,
                :print_iter,
                :epsilon,
                :memory_mode,
                :lazy,
                :trajectory,
                :verbose,
            ),
            Tuple{Int64,Adaptive{Float64,Int64},Float64,Float64,InplaceEmphasis,Bool,Bool,Bool},
        },
        typeof(away_frank_wolfe),
        Function,
        Function,
        BirkhoffPolytopeLMO,
        SparseArrays.SparseMatrixCSC{Float64,Int64},
    },
)   # time: 1.1986277
Base.precompile(
    Tuple{
        Core.kwftype(typeof(blended_conditional_gradient)),
        NamedTuple{
            (
                :max_iteration,
                :line_search,
                :print_iter,
                :epsilon,
                :memory_mode,
                :trajectory,
                :verbose,
            ),
            Tuple{Int64,Adaptive{Float64,Int64},Float64,Float64,InplaceEmphasis,Bool,Bool},
        },
        typeof(blended_conditional_gradient),
        Function,
        Function,
        BirkhoffPolytopeLMO,
        SparseArrays.SparseMatrixCSC{Float64,Int64},
    },
)   # time: 1.7452017
Base.precompile(
    Tuple{
        Core.kwftype(typeof(print_callback)),
        NamedTuple{(:print_footer,),Tuple{Bool}},
        typeof(print_callback),
        Nothing,
        String,
    },
)   # time: 0.007996668
Base.precompile(
    Tuple{
        typeof(perform_line_search),
        Nonconvex{Float64},
        Int64,
        Nothing,
        Nothing,
        Vector{Float64},
        Vector{Float64},
        Vector{Float64},
        Float64,
        Nothing,
    },
)   # time: 0.002293631
Base.precompile(Tuple{typeof(fast_dot),Vector{Float64},Int64})   # time: 0.004132393
Base.precompile(Tuple{typeof(compute_extreme_point),LpNormLMO{Float64,2},Int64})   # time: 0.001986683
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (:verbose, :line_search, :max_iteration, :print_iter, :trajectory),
            Tuple{Bool,Adaptive{Float64,Int64},Int64,Float64,Bool},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        LpNormLMO{Float64,2},
        Vector{Float64},
    },
)   # time: 0.23070359
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (
                :max_iteration,
                :line_search,
                :print_iter,
                :memory_mode,
                :verbose,
                :epsilon,
                :trajectory,
            ),
            Tuple{Int64,Agnostic{Float64},Float64,InplaceEmphasis,Bool,Float64,Bool},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        ProbabilitySimplexOracle{Float64},
        ScaledHotVector{Float64},
    },
)   # time: 0.7329921
Base.precompile(
    Tuple{
        Core.kwftype(typeof(away_frank_wolfe)),
        NamedTuple{
            (
                :max_iteration,
                :line_search,
                :print_iter,
                :memory_mode,
                :verbose,
                :epsilon,
                :trajectory,
            ),
            Tuple{Int64,Adaptive{Float64,Int64},Float64,InplaceEmphasis,Bool,Float64,Bool},
        },
        typeof(away_frank_wolfe),
        Function,
        Function,
        ProbabilitySimplexOracle{Float64},
        ScaledHotVector{Float64},
    },
)   # time: 0.746063
Base.precompile(
    Tuple{
        Core.kwftype(typeof(blended_conditional_gradient)),
        NamedTuple{
            (
                :max_iteration,
                :line_search,
                :print_iter,
                :memory_mode,
                :verbose,
                :epsilon,
                :trajectory,
            ),
            Tuple{Int64,Adaptive{Float64,Int64},Float64,InplaceEmphasis,Bool,Float64,Bool},
        },
        typeof(blended_conditional_gradient),
        Function,
        Function,
        ProbabilitySimplexOracle{Float64},
        ScaledHotVector{Float64},
    },
)   # time: 1.6212598
Base.precompile(
    Tuple{
        Core.kwftype(typeof(lp_separation_oracle)),
        NamedTuple{(:inplace_loop, :force_fw_step),Tuple{Bool,Bool}},
        typeof(lp_separation_oracle),
        ProbabilitySimplexOracle{Float64},
        ActiveSet{ScaledHotVector{Float64},Float64,Vector{Float64}},
        SparseVector{Float64,Int64},
        Float64,
        Float64,
    },
)   # time: 0.001872497
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (:max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :trajectory),
            Tuple{Int64,Shortstep{Float64},Float64,InplaceEmphasis,Bool,Bool},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        ProbabilitySimplexOracle{Float64},
        Vector{Float64},
    },
)   # time: 0.19531982
Base.precompile(
    Tuple{
        typeof(perform_line_search),
        Shortstep{Float64},
        Int64,
        Function,
        Function,
        Vector{Float64},
        Vector{Float64},
        Vector{Float64},
        Float64,
        Nothing,
    },
)   # time: 0.001291007
Base.precompile(
    Tuple{typeof(print_callback),Tuple{String,String,Float64,Any,Any,Float64,Float64},String},
)   # time: 0.00497477
Base.precompile(
    Tuple{typeof(print_callback),Tuple{String,String,Any,Any,Any,Float64,Float64},String},
)   # time: 0.003270714
Base.precompile(Tuple{typeof(fast_dot),AbstractVector,Vector{Float64}})   # time: 0.002228764
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (:max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :trajectory),
            Tuple{Int64,Shortstep{Float64},Float64,OutplaceEmphasis,Bool,Bool},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        ProbabilitySimplexOracle{Float64},
        ScaledHotVector{Float64},
    },
)   # time: 0.14588983
Base.precompile(Tuple{typeof(-),ScaledHotVector{Float64},ScaledHotVector})   # time: 0.04572138
Base.precompile(
    Tuple{
        typeof(perform_line_search),
        Shortstep{Float64},
        Int64,
        Function,
        Function,
        SparseVector{Float64,Int64},
        ScaledHotVector{Float64},
        Any,
        Float64,
        Nothing,
    },
)   # time: 0.036659826
Base.precompile(
    Tuple{
        typeof(perform_line_search),
        Shortstep{Float64},
        Int64,
        Function,
        Function,
        SparseVector{Float64,Int64},
        Any,
        Any,
        Float64,
        Nothing,
    },
)   # time: 0.005265721
Base.precompile(Tuple{typeof(fast_dot),Any,SparseVector{Float64,Int64}})   # time: 0.001681489
Base.precompile(
    Tuple{
        typeof(perform_line_search),
        Shortstep{Float64},
        Int64,
        Function,
        Function,
        SparseVector{Float64,Int64},
        ScaledHotVector{Float64},
        Vector{Float64},
        Float64,
        Nothing,
    },
)   # time: 0.001473502
Base.precompile(
    Tuple{
        typeof(perform_line_search),
        Shortstep{Float64},
        Int64,
        Function,
        Function,
        SparseVector{Float64,Int64},
        Vector{Float64},
        Vector{Float64},
        Float64,
        Nothing,
    },
)   # time: 0.001434466
Base.precompile(Tuple{typeof(fast_dot),AbstractVector,SparseVector{Float64,Int64}})   # time: 0.001320551
Base.precompile(
    Tuple{
        typeof(perform_line_search),
        Adaptive{Float64,Int64},
        Int64,
        Function,
        Function,
        SparseArrays.SparseMatrixCSC{Float64,Int64},
        Matrix{Float64},
        Matrix{Float64},
        Float64,
        Matrix{Float64},
    },
)   # time: 0.024287857
Base.precompile(
    Tuple{typeof(print_callback),Tuple{String,String,Any,Any,Float64,Float64,Float64},String},
)   # time: 0.001288076
Base.precompile(
    Tuple{
        Core.kwftype(typeof(compute_extreme_point)),
        NamedTuple{(:threshold, :greedy),Tuple{Float64,Bool}},
        typeof(compute_extreme_point),
        MultiCacheLMO{
            _A,
            NuclearNormLMO{Float64},
            RankOneMatrix{Float64,Vector{Float64},Vector{Float64}},
        } where _A,
        SparseArrays.SparseMatrixCSC{Float64,Int64},
    },
)   # time: 0.029216602
Base.precompile(
    Tuple{
        Core.kwftype(typeof(compute_extreme_point)),
        NamedTuple{(:threshold, :greedy),_A} where _A<:Tuple{Any,Bool},
        typeof(compute_extreme_point),
        VectorCacheLMO{
            NuclearNormLMO{Float64},
            RankOneMatrix{Float64,Vector{Float64},Vector{Float64}},
        },
        SparseArrays.SparseMatrixCSC{Float64,Int64},
    },
)   # time: 0.010391894
Base.precompile(
    Tuple{
        Core.kwftype(typeof(compute_extreme_point)),
        NamedTuple{(:threshold, :greedy),_A} where _A<:Tuple{Any,Bool},
        typeof(compute_extreme_point),
        MultiCacheLMO{
            _A,
            NuclearNormLMO{Float64},
            RankOneMatrix{Float64,Vector{Float64},Vector{Float64}},
        } where _A,
        SparseArrays.SparseMatrixCSC{Float64,Int64},
    },
)   # time: 0.005250637
Base.precompile(
    Tuple{
        typeof(length),
        MultiCacheLMO{
            _A,
            NuclearNormLMO{Float64},
            RankOneMatrix{Float64,Vector{Float64},Vector{Float64}},
        } where _A,
    },
)   # time: 0.004705316
Base.precompile(
    Tuple{typeof(print_callback),Tuple{String,String,Any,Any,Any,Float64,Float64,Int64},String},
)   # time: 0.00435821
Base.precompile(
    Tuple{Type{MultiCacheLMO{_A,NuclearNormLMO{Float64},_B}} where {_A,_B},NuclearNormLMO{Float64}},
)   # time: 0.00359918
Base.precompile(
    Tuple{
        Type{
            MultiCacheLMO{
                _A,
                NuclearNormLMO{Float64},
                RankOneMatrix{Float64,Vector{Float64},Vector{Float64}},
            },
        } where _A,
        NuclearNormLMO{Float64},
    },
)   # time: 0.003546448
Base.precompile(
    Tuple{Type{VectorCacheLMO{NuclearNormLMO{Float64},_A}} where _A,NuclearNormLMO{Float64}},
)   # time: 0.003143188
Base.precompile(
    Tuple{
        Core.kwftype(typeof(compute_extreme_point)),
        NamedTuple{(:threshold, :greedy),Tuple{Float64,Bool}},
        typeof(compute_extreme_point),
        VectorCacheLMO{
            NuclearNormLMO{Float64},
            RankOneMatrix{Float64,Vector{Float64},Vector{Float64}},
        },
        SparseArrays.SparseMatrixCSC{Float64,Int64},
    },
)   # time: 0.002158991
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (:max_iteration, :line_search, :print_iter, :verbose),
            Tuple{Float64,Nonconvex{Float64},Float64,Bool},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        ProbabilitySimplexOracle{Float64},
        Vector{Float64},
    },
)   # time: 0.2234393
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (
                :epsilon,
                :max_iteration,
                :print_iter,
                :trajectory,
                :verbose,
                :line_search,
                :memory_mode,
                :gradient,
            ),
            Tuple{
                Float64,
                Int64,
                Float64,
                Bool,
                Bool,
                Adaptive{Float64,Int64},
                InplaceEmphasis,
                SparseArrays.SparseMatrixCSC{Float64,Int64},
            },
        },
        typeof(frank_wolfe),
        Function,
        Function,
        NuclearNormLMO{Float64},
        RankOneMatrix{Float64,Vector{Float64},Vector{Float64}},
    },
)   # time: 0.6235743
Base.precompile(
    Tuple{
        Core.kwftype(typeof(lazified_conditional_gradient)),
        NamedTuple{
            (
                :epsilon,
                :max_iteration,
                :print_iter,
                :trajectory,
                :verbose,
                :line_search,
                :memory_mode,
                :gradient,
            ),
            Tuple{
                Float64,
                Int64,
                Float64,
                Bool,
                Bool,
                Adaptive{Float64,Int64},
                InplaceEmphasis,
                SparseArrays.SparseMatrixCSC{Float64,Int64},
            },
        },
        typeof(lazified_conditional_gradient),
        Function,
        Function,
        NuclearNormLMO{Float64},
        RankOneMatrix{Float64,Vector{Float64},Vector{Float64}},
    },
)   # time: 0.36729497
Base.precompile(
    Tuple{
        Core.kwftype(typeof(away_frank_wolfe)),
        NamedTuple{
            (
                :epsilon,
                :max_iteration,
                :print_iter,
                :trajectory,
                :verbose,
                :lazy,
                :line_search,
                :memory_mode,
            ),
            Tuple{Float64,Int64,Float64,Bool,Bool,Bool,Adaptive{Float64,Int64},InplaceEmphasis},
        },
        typeof(away_frank_wolfe),
        Function,
        Function,
        NuclearNormLMO{Float64},
        RankOneMatrix{Float64,Vector{Float64},Vector{Float64}},
    },
)   # time: 0.828201
Base.precompile(
    Tuple{
        Core.kwftype(typeof(blended_conditional_gradient)),
        NamedTuple{
            (
                :epsilon,
                :max_iteration,
                :print_iter,
                :trajectory,
                :verbose,
                :line_search,
                :memory_mode,
            ),
            Tuple{Float64,Int64,Float64,Bool,Bool,Adaptive{Float64,Int64},InplaceEmphasis},
        },
        typeof(blended_conditional_gradient),
        Function,
        Function,
        NuclearNormLMO{Float64},
        RankOneMatrix{Float64,Vector{Float64},Vector{Float64}},
    },
)   # time: 1.6012237
Base.precompile(
    Tuple{
        Core.kwftype(typeof(lp_separation_oracle)),
        NamedTuple{(:inplace_loop, :force_fw_step),Tuple{Bool,Bool}},
        typeof(lp_separation_oracle),
        NuclearNormLMO{Float64},
        ActiveSet{RankOneMatrix{Float64,Vector{Float64},Vector{Float64}},Float64,Matrix{Float64}},
        Matrix{Float64},
        Float64,
        Float64,
    },
)   # time: 0.001352328
Base.precompile(
    Tuple{
        Core.kwftype(typeof(blended_pairwise_conditional_gradient)),
        NamedTuple{
            (
                :epsilon,
                :max_iteration,
                :print_iter,
                :trajectory,
                :verbose,
                :line_search,
                :memory_mode,
            ),
            Tuple{Float64,Int64,Float64,Bool,Bool,Adaptive{Float64,Int64},InplaceEmphasis},
        },
        typeof(blended_pairwise_conditional_gradient),
        Function,
        Function,
        NuclearNormLMO{Float64},
        RankOneMatrix{Float64,Vector{Float64},Vector{Float64}},
    },
)   # time: 0.34339795
Base.precompile(
    Tuple{
        typeof(active_set_update!),
        ActiveSet{ScaledHotVector{Float64},Float64,Vector{Float64}},
        Float64,
        ScaledHotVector{Float64},
        Bool,
        Nothing,
    },
)   # time: 0.114226826
Base.precompile(Tuple{Type{ActiveSet},Vector{Tuple{Float64,ScaledHotVector{Float64}}}})   # time: 0.047632866
Base.precompile(
    Tuple{
        typeof(perform_line_search),
        Adaptive{Float64,Int64},
        Int64,
        Function,
        Function,
        Vector{Float64},
        Vector{Float64},
        Vector{Float64},
        Float64,
        Vector{Float64},
    },
)   # time: 0.02716949
Base.precompile(
    Tuple{
        Core.kwftype(typeof(lazy_afw_step)),
        NamedTuple{(:sparsity_control,),Tuple{Float64}},
        typeof(lazy_afw_step),
        Vector{Float64},
        Vector{Float64},
        LpNormLMO{Float64,1},
        ActiveSet{ScaledHotVector{Float64},Float64,Vector{Float64}},
        Float64,
        Vector{Float64},
    },
)   # time: 0.009450513
Base.precompile(
    Tuple{
        typeof(active_set_update!),
        ActiveSet{ScaledHotVector{Float64},Float64,Vector{Float64}},
        Float64,
        ScaledHotVector{Float64},
        Bool,
        Int64,
    },
)   # time: 0.003349358
Base.precompile(
    Tuple{
        typeof(afw_step),
        Vector{Float64},
        Vector{Float64},
        LpNormLMO{Float64,1},
        ActiveSet{ScaledHotVector{Float64},Float64,Vector{Float64}},
        Vector{Float64},
    },
)   # time: 0.001109981
Base.precompile(
    Tuple{
        Core.kwftype(typeof(lp_separation_oracle)),
        NamedTuple{(:inplace_loop, :force_fw_step),Tuple{Bool,Bool}},
        typeof(lp_separation_oracle),
        LpNormLMO{Float64,1},
        ActiveSet{ScaledHotVector{Float64},Float64,Vector{Float64}},
        SparseVector{Float64,Int64},
        Float64,
        Float64,
    },
)   # time: 0.13437502
Base.precompile(
    Tuple{
        Core.kwftype(typeof(perform_line_search)),
        NamedTuple{(:should_upgrade,),Tuple{Val{true}}},
        typeof(perform_line_search),
        Adaptive{Float64,Int64},
        Int64,
        Function,
        Function,
        SparseVector{Float64,Int64},
        Vector{Float64},
        Vector{Float64},
        Float64,
        Vector{Float64},
    },
)   # time: 0.10003789
Base.precompile(
    Tuple{
        typeof(active_set_initialize!),
        ActiveSet{ScaledHotVector{Float64},Float64,Vector{Float64}},
        ScaledHotVector{Float64},
    },
)   # time: 0.02555044
Base.precompile(
    Tuple{
        typeof(perform_line_search),
        Adaptive{Float64,Int64},
        Int64,
        Function,
        Function,
        SparseVector{Float64,Int64},
        Vector{Float64},
        Vector{Float64},
        Float64,
        Vector{Float64},
    },
)   # time: 0.009839748
Base.precompile(Tuple{typeof(fast_dot),Vector{BigFloat},Vector{BigFloat}})   # time: 0.003462153
Base.precompile(
    Tuple{typeof(compute_extreme_point),LpNormLMO{Float64,1},SparseVector{Float64,Int64}},
)   # time: 0.00309479
Base.precompile(
    Tuple{
        Core.kwftype(typeof(active_set_cleanup!)),
        NamedTuple{(:weight_purge_threshold,),Tuple{Float64}},
        typeof(active_set_cleanup!),
        ActiveSet{ScaledHotVector{Float64},Float64,Vector{Float64}},
    },
)   # time: 0.001547255
Base.precompile(
    Tuple{
        typeof(print_callback),
        Tuple{String,String,Any,Any,Float64,Float64,Float64,Int64,Int64},
        String,
    },
)   # time: 0.001473014
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (:max_iteration, :line_search, :print_iter, :verbose, :memory_mode),
            Tuple{Int64,Shortstep{Rational{Int64}},Float64,Bool,OutplaceEmphasis},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        ProbabilitySimplexOracle{Rational{BigInt}},
        ScaledHotVector{Rational{BigInt}},
    },
)   # time: 0.141858
Base.precompile(Tuple{Type{Shortstep},Rational{Int64}})   # time: 0.00955542
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (:max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :trajectory),
            Tuple{Int64,Shortstep{Float64},Int64,InplaceEmphasis,Bool,Bool},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        ScaledBoundL1NormBall{Float64,1,Vector{Float64},Vector{Float64}},
        Vector{Float64},
    },
)   # time: 0.19644113
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (:max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :trajectory),
            Tuple{Int64,Shortstep{Float64},Int64,InplaceEmphasis,Bool,Bool},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        ScaledBoundLInfNormBall{Float64,1,Vector{Float64},Vector{Float64}},
        Vector{Float64},
    },
)   # time: 0.046453062
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (:max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :trajectory),
            Tuple{Int64,Shortstep{Float64},Float64,InplaceEmphasis,Bool,Bool},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        LpNormLMO{Float64,1},
        SparseVector{Float64,Int64},
    },
)   # time: 0.5265395
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (
                :max_iteration,
                :line_search,
                :print_iter,
                :memory_mode,
                :verbose,
                :trajectory,
                :momentum,
            ),
            Tuple{Int64,Shortstep{Float64},Float64,OutplaceEmphasis,Bool,Bool,Float64},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        LpNormLMO{Float64,1},
        SparseVector{Float64,Int64},
    },
)   # time: 0.10808421
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (:max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :trajectory),
            Tuple{Int64,Adaptive{Float64,Int64},Float64,InplaceEmphasis,Bool,Bool},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        LpNormLMO{Float64,1},
        SparseVector{Float64,Int64},
    },
)   # time: 0.053366497
Base.precompile(
    Tuple{
        Core.kwftype(typeof(frank_wolfe)),
        NamedTuple{
            (:max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :trajectory),
            Tuple{Int64,Agnostic{Float64},Float64,InplaceEmphasis,Bool,Bool},
        },
        typeof(frank_wolfe),
        Function,
        Function,
        LpNormLMO{Float64,1},
        SparseVector{Float64,Int64},
    },
)   # time: 0.06719333
