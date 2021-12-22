
using FrankWolfe
using LinearAlgebra
using Test

n = Int(1e3)
k = n

f(x) = dot(x, x)
function grad!(storage, x)
    @. storage = 2 * x
end

# pick feasible region
lmo = FrankWolfe.ProbabilitySimplexOracle{Rational{BigInt}}(1); # radius needs to be integer or rational

# compute some initial vertex
x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

x, v, primal, dual_gap0, trajectory = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Agnostic(),
    print_iter=k / 10,
    verbose=false,
    emphasis=FrankWolfe.blas,
)

xmem, vmem, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    copy(x0),
    max_iteration=k,
    line_search=FrankWolfe.Agnostic(),
    print_iter=k / 10,
    verbose=true,
    emphasis=FrankWolfe.memory,
)

@time xstep, _ = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2//1),
    print_iter=k / 10,
    verbose=true,
)

@test eltype(xmem) == eltype(xstep) == Rational{BigInt}
if !(eltype(xmem) <: Rational)
    @test eltype(xmem) == eltype(x)
end
 
@test xmem == x
@test abs(f(xstep) - f(x)) <= 1e-3
@test vmem == v
