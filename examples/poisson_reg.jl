using FrankWolfe
using Test
using Random
using SCIP
# using Statistics
using LinearAlgebra
using Distributions
import MathOptInterface
const MOI = MathOptInterface

# Poisson sparse regression

# For bug hunting:
seed = rand(UInt64)
@show seed
seed = 0xfe03ee83ca373eab   
Random.seed!(seed)

# min_{w, b, z} ∑_i exp(w x_i + b) - y_i (w x_i + b) + α norm(w)^2
# s.t. -N z_i <= w_i <= N z_i
# b ∈ [-N, N]
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p

# y_i    - data points, poisson distributed 
# X_i, b - coefficient for the linear estimation of the expected value of y_i
# w_i    - continuous variables
# z_i    - binary variables s.t. z_i = 0 => w_i = 0
# k      - max number of non zero entries in w

# In a poisson regression, we want to model count data.
# It is assumed that y_i is poisson distributed and that the log 
# of its expected value can be computed linearly.

n = 20
p = n

# underlying true weights
const ws = rand(Float64, p) 
# set 50 entries to 0
for _ in 1:20
    ws[rand(1:p)] = 0
end
const bs = rand(Float64) 
const Xs = randn(Float64, n, p) 
const ys = map(1:n) do idx
    a = dot(Xs[idx,:], ws) + bs
    rand(Distributions.Poisson(exp(a)))
end
Ns = 0.09

# TODO: document better

@testset "Sparse poisson regression" begin
    k = 10
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    w = MOI.add_variables(o, p)
    z = MOI.add_variables(o, p)
    b = MOI.add_variable(o)
    for i in 1:p
        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())
    end
    for i in 1:p
        MOI.add_constraint(o, Ns * z[i] + w[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, -Ns * z[i] + w[i], MOI.LessThan(0.0))
        # Indicator: z[i] = 1 => -N <= w[i] <= N
        #=gl = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),],
            [0.0, 0.0], )
        gg = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),],
            [0.0, 0.0], )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(Ns)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-Ns))) =#
    end
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    MOI.add_constraint(o, sum(z, init=0.0), MOI.GreaterThan(1.0))
    MOI.add_constraint(o, b, MOI.LessThan(Ns))
    MOI.add_constraint(o, b, MOI.GreaterThan(-Ns))

### enforce fixings

l = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
for i in 1:p
    MOI.add_constraint(o, z[i], MOI.EqualTo(l[i]))
end

###

    lmo = FrankWolfe.MathOptLMO(o)

    α = 1.3
    function f(θ)
        w = @view(θ[1:p])
        b = θ[end]
        s = sum(1:n) do i
            a = dot(w, Xs[:,i]) + b
            1/n * (exp(a) - ys[i] * a)
        end
        return s + α * norm(w)^2
    end
    function grad!(storage, θ)
        w = @view(θ[1:p])
        b = θ[end]
        storage[1:p] .= 2α .* w
        storage[p+1:2p] .= 0
        storage[end] = 0
        for i in 1:n
            xi = @view(Xs[:,i])
            a = dot(w, xi) + b
            storage[1:p] .+= 1/n * xi * exp(a)
            storage[1:p] .-= 1/n * ys[i] * xi
            storage[end] += 1/n * (exp(a) - ys[i])
        end
        storage ./= norm(storage)
        return storage
    end


v_indices = MOI.get(lmo.o, MOI.ListOfVariableIndices())
m = length(v_indices)

x00 = FrankWolfe.compute_extreme_point(lmo, zeros(m))
target_tolerance = 1e-5
max_iter = 100

x0 = deepcopy(x00)

x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=max_iter,
    epsilon=target_tolerance,
    line_search=FrankWolfe.Adaptive(),
    print_iter=max_iter / 10, #max_iter,
    memory_mode=FrankWolfe.InplaceEmphasis(),
#    memory_mode=FrankWolfe.OutplaceEmphasis(),
    verbose=true,
    trajectory=true,
    lazy=false,
)


FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=max_iter,
    print_iter=max_iter / 10,# max_iter,
    line_search=FrankWolfe.Goldenratio(),
    verbose=true,
)

end
