module Test_oddities

import FrankWolfe
using LinearAlgebra
using Test

@testset "Testing adaptive LS at optimality" begin
    f(x) = norm(x)^2
    function grad!(storage, x)
        return storage .= 2x
    end
    lmo = FrankWolfe.UnitSimplexLMO(1.0)
    x00 = FrankWolfe.compute_extreme_point(lmo, ones(5))

    x0 = copy(x00)
    @test abs(
        FrankWolfe.away_frank_wolfe(
            f,
            grad!,
            lmo,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Agnostic(),
            verbose=false,
        )[3],
    ) < 1.0e-10

    x0 = copy(x00)
    @test abs(
        FrankWolfe.away_frank_wolfe(
            f,
            grad!,
            lmo,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.AdaptiveZerothOrder(),
            verbose=false,
        )[3],
    ) < 1.0e-10
    @test abs(
        FrankWolfe.away_frank_wolfe(
            f,
            grad!,
            lmo,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Adaptive(),
            verbose=false,
        )[3],
    ) < 1.0e-10


    x0 = copy(x00)
    @test abs(
        FrankWolfe.away_frank_wolfe(
            f,
            grad!,
            lmo,
            x0,
            max_iteration=1000,
            lazy=true,
            line_search=FrankWolfe.AdaptiveZerothOrder(),
            verbose=false,
        )[3],
    ) < 1.0e-10
    @test abs(
        FrankWolfe.away_frank_wolfe(
            f,
            grad!,
            lmo,
            x0,
            max_iteration=1000,
            lazy=true,
            line_search=FrankWolfe.Adaptive(),
            verbose=false,
        )[3],
    ) < 1.0e-10

    x0 = copy(x00)
    @test abs(
        FrankWolfe.blended_conditional_gradient(
            f,
            grad!,
            lmo,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Adaptive(),
            verbose=false,
        )[3],
    ) < 1.0e-10


end

end # module
