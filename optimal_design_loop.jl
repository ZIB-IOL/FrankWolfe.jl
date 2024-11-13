# Showcases infinite loop for Lazy DICG on Optimal Design problem

using FrankWolfe
using LinearAlgebra
using StableRNGs
using Distributions
using SparseArrays

# Optimal Design functions
"""
    build_data

seed - for the Random Number Generator.
m    - number of experiments.
"""
function build_data(seed, m)
    # set up
    rng = StableRNG(seed)

    n = Int(floor(m/10))
    B = rand(rng, m,n)
    B = B'*B
    @assert isposdef(B)
    D = MvNormal(randn(rng, n),B)
    
    A = rand(rng, D, m)'
    @assert rank(A) == n 
        
    return A 
end

"""
Check if given point is in the domain of f, i.e. X = transpose(A) * diagm(x) * A 
positive definite.
"""
function build_domain_oracle(A, n)
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
    V = FrankWolfe.ScaledHotVector{Float64}[]

    for i in S
        v = FrankWolfe.ScaledHotVector(1.0, i, m)
        push!(V, v)
    end

    x = sum(V .* 1/n)
    x = convert(SparseArrays.SparseVector, x)
    active_set= FrankWolfe.ActiveSet(fill(1/n, n), V, x)

    return x, active_set, S
end

# A Optimal 
"""
Build function for the A-criterion. 
"""
function build_a_criterion(A; μ=0.0, build_safe=true)
    m, n = size(A) 
    a=m
    domain_oracle = build_domain_oracle(A, n)

    function f_a(x)
        X = transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X)
        U = cholesky(X)
        X_inv = U \ I
        return LinearAlgebra.tr(X_inv)/a 
    end

    function grad_a!(storage, x)
        X = Symmetric(X*X)
        F = cholesky(X)
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
function build_d_criterion(A; μ =0.0, build_safe=true)
    m, n = size(A)
    a=m
    domain_oracle = build_domain_oracle(A, n)

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

"""
Returns FW args for D-Criterion
"""
function build_d_opt(; n=100, seed=1234)
    A = build_data(seed, n)
    f, grad! = build_d_criterion(A)
    
    lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
    x0, _ = build_start_point(A)

    return f, grad!, lmo, x0
end

"""
Returns FW args for A-Criterion
"""
function build_a_opt(; n=100, seed=1234)
    A = build_data(seed, n)
    A = A
    f, grad! = build_a_criterion(A)

    lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
    x0, _ = build_start_point(A)

    return f, grad!, lmo, x0
end

println("D_Criterion")
f, grad!, lmo, x0 = build_d_opt(n=250, seed=95781326)
FrankWolfe.decomposition_invariant_conditional_gradient(f, grad!, lmo, copy(x0), verbose=true, max_iteration=2000, print_iter=100, lazy=true);

println("A-Criterion")
f, grad!, lmo, x0 = build_a_opt(n=250, seed=95781326)
FrankWolfe.decomposition_invariant_conditional_gradient(f, grad!, lmo, copy(x0), verbose=true, max_iteration=10000, print_iter=1000, lazy=true);

println("Fin.")
