using FrankWolfe
using LinearAlgebra
using Test


# necessary for away step
function Base.convert(::Type{GenericArray{Float64, 1}}, v::FrankWolfe.ScaledHotVector{Float64})
    return GenericArray(collect(v))
end

@testset "GenericArray is maintained" begin
    n = Int(1e3)
    k = n

    f(x::GenericArray) = dot(x, x)
    function grad!(storage, x)
        @. storage = 2 * x
    end

    lmo = FrankWolfe.ProbabilitySimplexOracle(1)

    x0 = GenericArray(collect(FrankWolfe.compute_extreme_point(lmo, zeros(n))))

    x, v, primal, dual_gap0, trajectory = FrankWolfe.frank_wolfe(
        f,
        grad!,
        lmo,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Agnostic(),
        print_iter=k / 10,
        verbose=true,
        emphasis=FrankWolfe.memory,
    )

    @test f(x) < f(x0)
    @test x isa GenericArray

    x, v, primal, dual_gap0, trajectory = FrankWolfe.lazified_conditional_gradient(
        f,
        grad!,
        lmo,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Agnostic(),
        print_iter=k / 10,
        verbose=true,
        emphasis=FrankWolfe.memory,
        VType=FrankWolfe.ScaledHotVector{Float64},
    )

    @test f(x) < f(x0)
    @test x isa GenericArray
    
    @test_broken x, v, primal, dual_gap0, trajectory = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Agnostic(),
        print_iter=k / 10,
        verbose=true,
        emphasis=FrankWolfe.memory,
    )
end
