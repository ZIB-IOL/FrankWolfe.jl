using FrankWolfe
using Test
using LinearAlgebra

import FrankWolfe: SimplexMatrix

include("lmo.jl")

@testset "Line Search methods" begin
    a = [-1.0,-1.0,-1.0]
    b = [1.0,1.0,1.0] 
    grad(x) = 2x 
    f(x) = norm(x)^2 
    @test FrankWolfe.backtrackingLS(f,grad,a,b) == (1, 0.5)
    @test abs(FrankWolfe.segmentSearch(f,grad,a,b)[2] - 0.5) < 0.0001
end

@testset "FrankWolfe.jl" begin
    @testset "Testing vanilla Frank-Wolfe with various step size strategies" begin
        f(x) = LinearAlgebra.norm(x)^2
        grad(x) = 2x;
        lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1)
        x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(5))
        @test abs(FrankWolfe.fw(f,grad,lmo_prob,x0,maxIt=1000,stepSize=FrankWolfe.agnostic,verbose=true)[3] - 0.2) < 1.0e-5
        @test abs(FrankWolfe.fw(f,grad,lmo_prob,x0,maxIt=1000,stepSize=FrankWolfe.goldenratio,verbose=true)[3] - 0.2) < 1.0e-5
        @test abs(FrankWolfe.fw(f,grad,lmo_prob,x0,maxIt=1000,stepSize=FrankWolfe.backtracking,verbose=true)[3] - 0.2) < 1.0e-5
        @test abs(FrankWolfe.fw(f,grad,lmo_prob,x0,maxIt=1000,stepSize=FrankWolfe.nonconvex,verbose=true)[3] - 0.2) < 1.0e-2
        @test FrankWolfe.fw(f,grad,lmo_prob,x0,maxIt=1000,stepSize=FrankWolfe.shortstep,L=2,verbose=true)[3] ≈ 0.2 
        @test abs(FrankWolfe.fw(f,grad,lmo_prob,x0,maxIt=1000,stepSize=FrankWolfe.nonconvex,verbose=true)[3] - 0.2) < 1.0e-2
    end
    @testset "Testing Lazified Conditional Gradients with various step size strategies" begin
        f(x) = LinearAlgebra.norm(x)^2
        grad(x) = 2x;
        lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1)
        x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(5))
        @test abs(FrankWolfe.lcg(f,grad,lmo_prob,x0,maxIt=1000,stepSize=FrankWolfe.goldenratio,verbose=true)[3] - 0.2) < 1.0e-5
        @test abs(FrankWolfe.lcg(f,grad,lmo_prob,x0,maxIt=1000,stepSize=FrankWolfe.backtracking,verbose=true)[3] - 0.2) < 1.0e-5
        @test abs(FrankWolfe.lcg(f,grad,lmo_prob,x0,maxIt=1000,stepSize=FrankWolfe.shortstep,L=2,verbose=true)[3] - 0.2) < 1.0e-5
    end
    @testset "Testing emphasis blas vs memory" begin
        n = Int(1e5);
        k = 100
        xpi = rand(n);
        total = sum(xpi);
        xp = xpi ./ total;        
        f(x) = LinearAlgebra.norm(x-xp)^2
        grad(x) = 2 * (x-xp)

        lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1.0);
        x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(n));

        x, v, primal, dualGap, trajectory = FrankWolfe.fw(f,grad,lmo_prob,x0,maxIt=k,
            stepSize=FrankWolfe.backtracking,printIt=k/10,verbose=true,emph=FrankWolfe.blas);
        
        @test x !== nothing 

        x, v, primal, dualGap, trajectory = FrankWolfe.fw(f,grad,lmo_prob,x0,maxIt=k,
            stepSize=FrankWolfe.backtracking,printIt=k/10,verbose=true,emph=FrankWolfe.memory);
        
        @test x !== nothing

    end
end
