## Benchmark example
using FrankWolfe 
using Random
using Distributions
using LinearAlgebra
using Statistics
using SparseArrays

# The Optimal Experiment Design Problem consists of choosing a subset of experiments
# maximising the information gain.
# The Limit version of this problem (see below) is a continous version where the 
# number of allowed experiments is infinity.
# Thus, the solution can be interpreted as a probability distributions.
#
# min_x  Φ(A^T diag(x) A)
# s.t.   ∑ x_i = 1
#          x ≥ 0
#
# A denotes the Experiment Matrix. We generate it randomly.
# Φ is a function from the PD cone into R. 
# In our case, Φ is 
#   Trace(X^{-1})       (A-Optimal)
# and
#   -logdet(X)          (D-Optimal).

"""
    build_data(m)

seed - for the Random functions.
m    - number of experiments.
Build the experiment matrix A.
"""
function build_data(seed, m)
    Random.seed!(seed)

    #n = Int(floor(m/100))
    n = Int(floor(sqrt(m)))
    B = rand(m,n)
    B = B'*B
    @assert isposdef(B)
    D = MvNormal(randn(n),B)
    
    A = rand(D, m)'
    @assert rank(A) == n 

    A /= (n^(3/2))
        
    return A 
end

"""
Check if given point is in the domain of f, i.e. X = transpose(A) * diagm(x) * A 
positive definite.
"""
function build_domain_oracle(A)
    m, n = size(A)
    return function domain_oracle(x)
        S = findall(x-> !iszero(x),x)
        #@show rank(A[S,:]) == n
        return rank(A[S,:]) == n #&& sum(x .< 0) == 0 
    end
end

"""
Find n linearly independent rows of A to build the starting point.
"""
function linearly_independent_rows(A)
    S = []
    m, n = size(A)
    for i in 1:m
        S_i= vcat(S, i)
        if rank(A[S_i,:])==length(S_i)
            S=S_i
        end
        if length(S) == n # we only n linearly independent points
            return S
        end
    end 
    return S 
end

"""
Build start point used in Boscia in case of A-opt and D-opt.
The functions are self concordant and so not every point in the feasible region
is in the domain of f and grad!.
"""
function build_start_point(A)
    # Get n linearly independent rows of A
    m, n = size(A)
    S = linearly_independent_rows(A)
    @assert length(S) == n
    V = Vector{Float64}[]

    for i in S
        v = FrankWolfe.ScaledHotVector(1.0, i, m)
        push!(V, v)
    end

    x = SparseArrays.SparseVector(sum(V .* 1/n))
    active_set= FrankWolfe.ActiveSet(fill(1/n, n), V, x)

    return x, active_set, S
end

# A Optimal 
"""
Build function for the A-criterion. 
"""
function build_a_criterion(A; μ=0.0, build_safe=false)
    m, n = size(A) 
    a=m
    domain_oracle = build_domain_oracle(A)

    function f_a(x)
        #x = BigFloat.(x)
        X = transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X)
        U = cholesky(X)
        X_inv = U \ I
        return LinearAlgebra.tr(X_inv)/a 
    end

    function grad_a!(storage, x)
        #x .= BigFloat.(x)
        X = transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X*X)
        F = similar(X)
        #try 
            F = cholesky(X)
        #catch e
        #    @show minimum(eigvals(X)), maximum(eigvals(X))
        #    throw(e)
        #end
        for i in 1:length(x)
            storage[i] = LinearAlgebra.tr(- (F \ A[i,:]) * transpose(A[i,:]))/a
        end
        return storage #float.(storage) # in case of x .= BigFloat(x)
    end

    function f_a_safe(x)
        if !domain_oracle(x)
            return Inf
        end
        X = transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X)
        X_inv = LinearAlgebra.inv(X)
        return LinearAlgebra.tr(X_inv)/a 
    end

    function grad_a_safe!(storage, x)
        if !domain_oracle(x)
            return fill(Inf, length(x))        
        end
        #x = BigFloat.(x) # Setting can be useful for numerical tricky problems
        X = transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X*X)
        F = cholesky(X)
        for i in 1:length(x)
            storage[i] = LinearAlgebra.tr(- (F \ A[i,:]) * transpose(A[i,:]))/a
        end
        return storage #float.(storage) # in case of x .= BigFloat(x)
    end

    if build_safe
        return f_a_safe, grad_a_safe!
    end

    return f_a, grad_a!
end

# D Optimal
"""
Build function for the D-criterion.
"""
function build_d_criterion(A; μ =0.0, build_safe=false)
    m, n = size(A)
    a=m
    domain_oracle = build_domain_oracle(A)

    function f_d(x)
        X = transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X)
        return -log(det(X))/a
    end

    function grad_d!(storage, x)
        X = transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X= Symmetric(X)
        F = cholesky(X) 
        for i in 1:length(x)        
            storage[i] = 1/a * LinearAlgebra.tr(-(F \ A[i,:] )*transpose(A[i,:]))
        end
        # https://stackoverflow.com/questions/46417005/exclude-elements-of-array-based-on-index-julia
        return storage
    end

    function f_d_safe(x)
        if !domain_oracle(x)
            return Inf
        end
        X = transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X)
        return -log(det(X))/a
    end

    function grad_d_safe!(storage, x)
        if !domain_oracle(x)
            return fill(Inf, length(x))
        end
        X = transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X= Symmetric(X)
        F = cholesky(X) 
        for i in 1:length(x)        
            storage[i] = 1/a * LinearAlgebra.tr(-(F \ A[i,:] )*transpose(A[i,:]))
        end
        # https://stackoverflow.com/questions/46417005/exclude-elements-of-array-based-on-index-julia
        return storage
    end

    if build_safe
        return f_d_safe, grad_d_safe!
    end

    return f_d, grad_d!
end


function build_optimal_design(seed, m; criterion="A")
    A = build_data(seed, m)
    f, grad! = if criterion == "A"
        build_a_criterion(A)
    elseif criterion == "D"
        build_d_criterion(A)
    else
        error("Criterion not known!")
    end

    lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
    x0, active_set = build_start_point(A)
    domain_oracle = build_domain_oracle(A)

    return f, grad!, lmo, x0, active_set, domain_oracle, m
end

