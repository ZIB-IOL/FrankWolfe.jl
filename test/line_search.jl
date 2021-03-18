using Test
import FrankWolfe
using LinearAlgebra

@testset "Line Search methods" begin
    a = [-1.0, -1.0, -1.0]
    b = [1.0, 1.0, 1.0]
    function grad!(storage, x)
        return storage .= 2x
    end
    f(x) = norm(x)^2
    gradient = similar(a)
    grad!(gradient, a)
    @test FrankWolfe.backtrackingLS(f, gradient, a, a - b, 1.0) == (0.5, 1)
    @test abs(FrankWolfe.segment_search(f, grad!, a, a - b, 1.0)[1] - 0.5) < 0.0001

    (gamma0, M0) = @inferred FrankWolfe.adaptive_step_size(f, grad!, gradient, a, a - b, 2; eta=0.6, tau=1.1, gamma_max=1, upgrade_accuracy=false)
    (gammaB, MB) = @inferred FrankWolfe.adaptive_step_size(f, grad!, gradient, a, a - b, 2; eta=0.6, tau=1.1, gamma_max=1, upgrade_accuracy=true)
    @test gamma0 ≈ gammaB
    @test M0 ≈ MB
end
