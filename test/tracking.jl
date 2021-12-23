using FrankWolfe
using Test
using LinearAlgebra

using DoubleFloats
using DelimitedFiles
import FrankWolfe: ActiveSet


@testset "Tracking Testset" begin
    f(x) = norm(x)^2
    function grad!(storage, x)
        return storage .= 2x
    end
    
    x = zeros(6)
    gradient = similar(x)
    rhs = 10 * rand()
    lmo = FrankWolfe.ProbabilitySimplexOracle(rhs)
    direction = zeros(6)
    direction[1] = -1
    
    @testset "TrackingGradient" begin
        tgrad! = FrankWolfe.TrackingGradient(grad!)
        @test tgrad!.counter == 0
        @test tgrad!.grad! === grad!
        tgrad!(gradient,direction)
        @test tgrad!.counter == 1
    
    
    end
    
    @testset "TrackingObjective" begin
        tf = FrankWolfe.TrackingObjective(f,0)
        @test tf.counter == 0
        tf(x)
        @test tf.counter == 1
    
    end
    
    @testset "TrackingLMO" begin
        tlmo_prob = FrankWolfe.TrackingLMO(lmo,0)
        @test tlmo_prob.counter == 0
        @test tlmo_prob.lmo ===  lmo
        compute_extreme_point(tlmo_prob,direction)
        @test tlmo_prob.counter == 1
    end
end



SUITE = Dict()

SUITE["vanilla_fw"] = Dict()
SUITE["lazified_cd"] = Dict()
SUITE["blas_vs_memory"] = Dict()
SUITE["dense_structure"] = Dict()
SUITE["rational"] = Dict()
SUITE["multi_precision"] = Dict()
SUITE["stochastic_fw"] = Dict()
SUITE["away_step_fw"] = Dict()
SUITE["blended_cg"] = Dict()

# "Testing vanilla Frank-Wolfe with various step size and momentum strategies" begin
    f(x) = norm(x)^2
    tf = FrankWolfe.TrackingObjective(f,0)

    function grad!(storage, x)
        return storage .= 2x
    end
    tgrad! = FrankWolfe.TrackingGradient(grad!,0)
    
    lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1)
    tlmo_prob = FrankWolfe.TrackingLMO(lmo_prob)

    x0 = FrankWolfe.compute_extreme_point(tlmo_prob, zeros(5))
    # f_values, dual_values, function_calls, gradient_calls, lmo_calls, time_vec
    tracking_trajectory_callback = FrankWolfe.tracking_trajectory_callback

    SUITE["vanilla_fw"]["1"] = FrankWolfe.frank_wolfe(
            tf,
            tgrad!,
            tlmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Agnostic(),
            trajectory=true,
            callback=tracking_trajectory_callback,
            verbose=false,
    )

    x, v, primal, dual_gap, trajectory = SUITE["vanilla_fw"]["1"]

    #println(trajectory)


