using FrankWolfe
using LinearAlgebra
using Test


# necessary for away step
function Base.convert(::Type{GenericArray{Float64,1}}, v::FrankWolfe.ScaledHotVector{Float64})
    return GenericArray(collect(v))
end

@testset "GenericArray is maintained                        " begin
    n = Int(1e3)
    k = n

    f_generic(x::GenericArray) = dot(x, x)
    function grad_generic!(storage, x)
        @. storage = 2 * x
    end

    lmo = FrankWolfe.ProbabilitySimplexOracle(1)

    x0 = GenericArray(collect(FrankWolfe.compute_extreme_point(lmo, zeros(n))))

    x, v, primal, dual_gap0, trajectory = FrankWolfe.frank_wolfe(
        f_generic,
        grad_generic!,
        lmo,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Agnostic(),
        print_iter=k / 10,
        verbose=true,
        memory_mode=FrankWolfe.InplaceEmphasis(),
    )

    @test f_generic(x) < f_generic(x0)
    @test x isa GenericArray

    x, v, primal, dual_gap0, trajectory = FrankWolfe.lazified_conditional_gradient(
        f_generic,
        grad_generic!,
        lmo,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Agnostic(),
        print_iter=k / 10,
        verbose=true,
        memory_mode=FrankWolfe.InplaceEmphasis(),
        VType=FrankWolfe.ScaledHotVector{Float64},
    )

    @test f_generic(x) < f_generic(x0)
    @test x isa GenericArray

    @test_broken x, v, primal, dual_gap0, trajectory = FrankWolfe.away_frank_wolfe(
        f_generic,
        grad_generic!,
        lmo,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Agnostic(),
        print_iter=k / 10,
        verbose=true,
        memory_mode=FrankWolfe.InplaceEmphasis(),
    )
end

@testset "Testing fast_equal for special types              " begin
    @testset "ScaledHotVector" begin
        a = FrankWolfe.ScaledHotVector(3.5, 3, 10)
        b = FrankWolfe.ScaledHotVector(3.5, 3, 11)
        @test !isequal(a, b)
        c = FrankWolfe.ScaledHotVector(3.5, 4, 10)
        @test !isequal(a, c)
        d = FrankWolfe.ScaledHotVector(3, 3, 10)
        @test !isequal(a, d)
        e = FrankWolfe.ScaledHotVector(3.0, 3, 10)
        @test isequal(d, e)
        f = FrankWolfe.ScaledHotVector(big(3.0), 3, 10)
        @test isequal(e, f)
        v = SparseArrays.spzeros(10)
        v[1:5] .= 3
        v2 = copy(v)
        copyto!(v, e)
        @test norm(v) ≈ 3
        @test norm(v[1:2]) ≈ 0
        copyto!(v2, collect(e))
        @test v == v2
    end
    @testset "RankOneMatrix" begin
        a = FrankWolfe.RankOneMatrix(ones(3), ones(4))
        b = FrankWolfe.RankOneMatrix(ones(3), 2 * ones(4))
        @test !isequal(a, b)
        @test isequal(2a, b)
        @test !isequal(2a, b')
        @test !isequal(2a, FrankWolfe.RankOneMatrix(2 * ones(4), ones(3)))
    end
end
