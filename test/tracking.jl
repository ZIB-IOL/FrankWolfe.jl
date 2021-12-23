using FrankWolfe
using Test

f(x) = norm(x)^2
function grad!(storage, x)
    return storage .= 2x
end

x = zeros(6)
gradient = similar(x)
rhs = 10 * rand()
direction = zeros(6)
direction[1] = -1

@testset "TrackingGradient" begin
    tg = FrankWolfe.TrackingGradient(grad!)
    @test tg.counter == 0
    @test tg.grad! === grad!
    tg(gradient,direction)

end

@testset "TrackingObjective" begin
    to = FrankWolfe.TrackingObjective(f,0)
    @test to.counter == 0

end

@testset "TrackingLMO" begin
    tlmo_prob = FrankWolfe.TrackingLMO(FrankWolfe.ProbabilitySimplexOracle(rhs),0)
    @test tlmo_prob.counter == 0
    compute_extreme_point(tlmo_prob,direction)
    @test tlmo_prob.counter == 1
end

