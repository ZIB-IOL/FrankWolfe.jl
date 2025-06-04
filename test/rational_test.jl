
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
    memory_mode=FrankWolfe.OutplaceEmphasis(),
)

xmem, vmem, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    copy(x0),
    max_iteration=k,
    line_search=FrankWolfe.Agnostic(),
    print_iter=k / 10,
    verbose=false,
    memory_mode=FrankWolfe.InplaceEmphasis(),
)

xstep, _ = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2 // 1),
    print_iter=k / 10,
    verbose=false,
)

@test eltype(xmem) == eltype(xstep) == Rational{BigInt}
if !(eltype(xmem) <: Rational)
    @test eltype(xmem) == eltype(x)
end

@test xmem == x
@test abs(f(xstep) - f(x)) <= 1e-3
@test vmem == v

@testset "Testing rational variant" begin
    rhs = 1
    n = 40
    k = 1000

    xpi = rand(big(1):big(100), n)
    total = sum(xpi)
    xp = xpi .// total

    f(x) = norm(x - xp)^2
    function grad!(storage, x)
        @. storage = 2 * (x - xp)
    end

    lmo = FrankWolfe.ProbabilitySimplexOracle{Rational{BigInt}}(rhs)
    direction = rand(n)
    x0 = FrankWolfe.compute_extreme_point(lmo, direction)
    @test eltype(x0) == Rational{BigInt}

    x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
        f,
        grad!,
        lmo,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Agnostic(),
        print_iter=k / 10,
        memory_mode=FrankWolfe.OutplaceEmphasis(),
        verbose=false,
    )

    @test eltype(x0) == Rational{BigInt}

    x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
        f,
        grad!,
        lmo,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Agnostic(),
        print_iter=k / 10,
        memory_mode=FrankWolfe.InplaceEmphasis(),
        verbose=false,
    )
    @test eltype(x0) == eltype(x) == Rational{BigInt}
    @test f(x) <= 1e-4

    # very slow computation, explodes quickly
    x0 = collect(FrankWolfe.compute_extreme_point(lmo, direction))
    x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
        f,
        grad!,
        lmo,
        x0,
        max_iteration=15,
        line_search=FrankWolfe.Shortstep(2 // 1),
        print_iter=k / 100,
        memory_mode=FrankWolfe.InplaceEmphasis(),
        verbose=false,
    )

    x0 = FrankWolfe.compute_extreme_point(lmo, direction)
    x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
        f,
        grad!,
        lmo,
        x0,
        max_iteration=15,
        line_search=FrankWolfe.Shortstep(2 // 1),
        print_iter=k / 10,
        memory_mode=FrankWolfe.InplaceEmphasis(),
        verbose=false,
    )
    @test eltype(x) == Rational{BigInt}
end
