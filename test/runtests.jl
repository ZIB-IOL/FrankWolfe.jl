using FrankWolfe
using Test
using LinearAlgebra

import FrankWolfe: SimplexMatrix

@testset "Simplex matrix type" begin
    s = SimplexMatrix{Float64}(3)
    sm = collect(s)
    @test sm == ones(1, 3)
    @test sm == ones(1, 3)
    v = rand(3)
    @test s * v ≈ sm * v
    m = rand(3, 5)
    @test sm * m ≈ s * m
    v2 = rand(5, 1)
    @test v2 * sm ≈ v2 * s
    # promotion test
    s2 = SimplexMatrix{Float32}(3)
    @test eltype(s2 * v) === Float64
    @test eltype(s2 * rand(Float32, 3)) === Float32
    @test eltype(s * rand(Float32, 3)) === Float64
end

@testset "Simplex LMOs projections" begin
    n = 6
    direction = zeros(6)
    rhs = 10 * rand()
    lmo_prob = FrankWolfe.ProbabilitySimplexOracle(rhs)
    lmo_unit = FrankWolfe.UnitSimplexOracle(rhs)
    @testset "Choosing improving direction" for idx in 1:n
        direction .= 0
        direction[idx] = -1
        res_point_prob = FrankWolfe.compute_extreme_point(lmo_prob, direction)
        res_point_unit = FrankWolfe.compute_extreme_point(lmo_unit, direction)
        for j in eachindex(res_point_prob)
            if j == idx
                @test res_point_prob[j] == res_point_unit[j] == rhs
            else
                @test res_point_prob[j] == res_point_unit[j] == 0
            end
        end
    end
    @testset "Choosing least-degrading direction" for idx in 1:n
        # all directions worsening, must pick idx
        direction .= 2
        direction[idx] = 1
        res_point_prob = FrankWolfe.compute_extreme_point(lmo_prob, direction)
        res_point_unit = FrankWolfe.compute_extreme_point(lmo_unit, direction)
        for j in eachindex(res_point_unit)
            @test res_point_unit[j] == 0
            if j == idx
                @test res_point_prob[j] == rhs
            else
                @test res_point_prob[j] == 0
            end
        end
    end
end

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

        lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1);
        x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(n));

        x, v, primal, dualGap, trajectory = FrankWolfe.fw(f,grad,lmo_prob,x0,maxIt=k,
            stepSize=FrankWolfe.backtracking,printIt=k/10,verbose=true,emph=FrankWolfe.blas);
        
        @test x !== nothing 

        x, v, primal, dualGap, trajectory = FrankWolfe.fw(f,grad,lmo_prob,x0,maxIt=k,
            stepSize=FrankWolfe.backtracking,printIt=k/10,verbose=true,emph=FrankWolfe.memory);
        
        @test x !== nothing

    end
end
