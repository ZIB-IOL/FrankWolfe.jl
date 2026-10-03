module Test_timeout

using Test
using FrankWolfe
using LinearAlgebra
using SparseArrays

@testset "Timing out" begin
    f(x) = norm(x)^2
    function grad!(storage, x)
        return storage .= 2x
    end
    lmo_prob = FrankWolfe.ProbabilitySimplexLMO(1)
    x0 = FrankWolfe.compute_extreme_point(lmo_prob, spzeros(5))
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Agnostic(),
            verbose=false,
        )[3] - 0.2,
    ) < 1.0e-5
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Agnostic(),
            verbose=false,
            gradient=collect(similar(x0)),
        )[3] - 0.2,
    ) < 1.0e-5
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Goldenratio(),
            verbose=false,
        )[3] - 0.2,
    ) < 1.0e-5
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Backtracking(),
            verbose=false,
        )[3] - 0.2,
    ) < 1.0e-5
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Nonconvex(),
            verbose=false,
        )[3] - 0.2,
    ) < 1.0e-2
    @test FrankWolfe.frank_wolfe(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=1000,
        line_search=FrankWolfe.Shortstep(2.0),
        verbose=false,
    )[3] ≈ 0.2
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Nonconvex(),
            verbose=false,
        )[3] - 0.2,
    ) < 1.0e-2
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Agnostic(),
            verbose=false,
            momentum=0.9,
        )[3] - 0.2,
    ) < 1.0e-3
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Agnostic(),
            verbose=false,
            momentum=0.5,
        )[3] - 0.2,
    ) < 1.0e-3
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Agnostic(),
            verbose=false,
            momentum=0.9,
            memory_mode=FrankWolfe.InplaceEmphasis(),
        )[3] - 0.2,
    ) < 1.0e-3
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.AdaptiveZerothOrder(L_est=100.0),
            verbose=false,
            momentum=0.9,
        )[3] - 0.2,
    ) < 1.0e-3
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Adaptive(L_est=100.0),
            verbose=false,
            momentum=0.9,
        )[3] - 0.2,
    ) < 1.0e-3
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Adaptive(L_est=100.0),
            verbose=false,
            momentum=0.5,
        )[3] - 0.2,
    ) < 1.0e-3
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.AdaptiveZerothOrder(L_est=100.0),
            verbose=false,
            momentum=0.5,
        )[3] - 0.2,
    ) < 1.0e-3
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Adaptive(L_est=100.0),
            verbose=false,
            momentum=0.9,
            memory_mode=FrankWolfe.InplaceEmphasis(),
        )[3] - 0.2,
    ) < 1.0e-3
end

end # module
