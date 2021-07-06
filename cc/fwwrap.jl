using FrankWolfe: LinearMinimizationOracle, LinearAlgebra, line_search_wrapper
using FrankWolfe
using Base: Int64, Float64, UInt64

###
# Accessing Oracle via C
###
struct LOPLinearMinimizationOracle <: LinearMinimizationOracle
    N::UInt64 # number of items
    DIM::UInt64 # dimension of problem, aka number of binary variables
    var_keys::Dict # indexes for binary vector assembly

    function LOPLinearMinimizationOracle(N)
        DIM = N * (N - 1)

        # create var_keys dict for ix2binvec function
        var_keys = Dict()
        ix = 1
        for i = 1 : N, j = 1 : N
            if i != j
                var_keys[(i, j)] = ix
                ix += 1
            end
        end

        return new(N, DIM, var_keys)
    end
end

function FrankWolfe.compute_extreme_point(lmo::LOPLinearMinimizationOracle, direction; kwargs...)
    v = zeros(UInt64, lmo.N)
    v_ix = Ref{UInt64}()
    v_obj = Ref{Float64}()

    # expand direction to full matrix
    full_direction = zeros(Float64, (lmo.N, lmo.N))
    for i in 1 : lmo.N, j in 1 : lmo.N
        if i != j
            full_direction[j, i] = direction[lmo.var_keys[(i, j)]]
        end
    end

    # call oracle via ccall 
    err = ccall(
        (:call_lop_oracle, "cc/libloporacle.so"), 
        Cvoid,
        (UInt64, Ptr{Float64}, Ptr{UInt64}, Ref{UInt64}, Ref{Float64}),
        lmo.N,
        # invert for col-major!
        full_direction',
        v,
        v_ix,
        v_obj)

    # convert vertex index to binary vector
    binvector = zeros(Float64, lmo.DIM)
    for ix_i = 1 : lmo.N
        i = v[ix_i] + 1

        for ix_j = (ix_i + 1) : lmo.N
            j = v[ix_j] + 1

            binvector[lmo.var_keys[(i, j)]] = 1.0
            binvector[lmo.var_keys[(j, i)]] = 0.0
        end
    end

    return binvector
end

###
# LOP solution via Frank-Wolfe
###

function f(x, xf)
    return 0.5 * LinearAlgebra.norm2(x - xf)^2
end

function df(x, xf)
    return (x - xf)
end


function LOP_solve(_N::Int64, xf::Vector{Float64}, x0::Vector{Float64}, _iters::Int64, away::Bool) :: Tuple{Vector{Float64}, Vector{Float64}}

    N::UInt64 = _N
    iters::UInt64 = _iters

    # create an LMO
    lmo = LOPLinearMinimizationOracle(N)

    function grad!(storage, x)
        @. storage = (x - xf)
    end

    # call ZIB's FrankWolfe implementation
    if away
        x, v, primal, dual_gap, trajectoryFW = FrankWolfe.away_frank_wolfe(
            x -> f(x, xf),
            grad!,
            lmo,
            x0,
            max_iteration = iters,
            epsilon = 1e-8,
            line_search = FrankWolfe.Adaptive(),
            print_iter = iters,
            verbose = true,
            emphasis=FrankWolfe.memory,
            lazy = true)
    else
        x, v, primal, dual_gap, trajectoryFW = FrankWolfe.frank_wolfe(
            x -> f(x, xf),
            grad!,
            lmo,
            x0,
            max_iteration = iters,
            epsilon = 1e-8,
            line_search = FrankWolfe.Adaptive(),
            print_iter = iters,
            verbose = true,
            emphasis=FrankWolfe.blas)
    end

    # report gaps
    println("Result: ", f(x, xf))

    println("Primal: ", primal)
    println("Dual gap: ", dual_gap)

    return -df(x, xf), x
end
