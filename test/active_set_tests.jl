using Test

using FrankWolfe
import FrankWolfe: ActiveSet

###########################################
# Testing extraction of lambdas and atoms
###########################################

@testset "Active sets" begin

    @testset "Constructors and eltypes" begin
        active_set = ActiveSet([
            (0.1, [1,2,3]),
            (0.9, [2,3,4]),
            (0.0, [5,6,7]),
        ])


        @test active_set.weights == [0.1, 0.9, 0.0]
        @test active_set.atoms == [[1, 2, 3], [2, 3, 4], [5, 6, 7]]
        @test eltype(active_set) === Tuple{Float64, Vector{Int}}

        @test FrankWolfe.active_set_validate(active_set) == true

        # providing the weight type converts the provided weights
        active_set2 = ActiveSet{Float64, Vector{Int}}([
            (0.1f, [1,2,3]),
            (0.9f, [2,3,4]),
            (0.0f, [5,6,7]),
        ])

        @test eltype(active_set) === eltype(active_set2)

        @test first(active_set) isa eltype(active_set)
        @test first(active_set2) isa eltype(active_set)
    end

    @testset "Presence and cleanup" begin
        active_set = ActiveSet([
            (0.1, [1,2,3]),
            (0.9, [2,3,4]),
            (0.0, [5,6,7]),
        ])
        @test FrankWolfe.find_atom(active_set, [5,6,7]) > 0
        FrankWolfe.active_set_cleanup!(active_set)
        @test FrankWolfe.find_atom(active_set, [5,6,7]) == -1
    end

    @testset "Renormalization and validation" begin
        active_set = ActiveSet([ 
            (0.1, [1,2,3]),
            (0.8, [2,3,4]),
            (0.0, [5,6,7]),
        ])
        @test !FrankWolfe.active_set_validate(active_set)
        FrankWolfe.active_set_renormalize!(active_set)
        @test FrankWolfe.active_set_validate(active_set)
    end

    @testset "addition of elements and manipulation" begin
        active_set = ActiveSet([ 
            (0.5, [1,2,3]),
            (0.5, [2,3,4]),
        ])
        FrankWolfe.active_set_update!(active_set, 1 / 2, [5,6,7])
        @test collect(active_set) == [(0.25, [1, 2, 3]), (0.25, [2, 3, 4]), (0.5, [5, 6, 7])]

        # element addition
        @test FrankWolfe.active_set_validate(active_set) == true
        # update existing element
        FrankWolfe.active_set_update!(active_set, 1.0 / 2, [5,6,7]) 
        @test collect(active_set) == [(0.125, [1, 2, 3]), (0.125, [2, 3, 4]), (0.75, [5, 6, 7])]
        @test FrankWolfe.active_set_validate(active_set) == true
    end

    @testset "Away step operations" begin
        active_set = ActiveSet([(0.125, [1, 2, 3]), (0.125, [2, 3, 4]), (0.75, [5, 6, 7])])

        lambda = FrankWolfe.weight_from_atom(active_set, [5, 6, 7])
        @test lambda == 0.75
        lambda_max = lambda / (1-lambda)
        FrankWolfe.active_set_update!(active_set, -lambda_max, [5,6,7]) 
        @test collect(active_set) == [(0.5, [1, 2, 3]), (0.5, [2, 3, 4])]
        
        ## intermediate away step
        lambda = FrankWolfe.weight_from_atom(active_set, [2, 3, 4])
        @test lambda == 0.5
        lambda_max = lambda / (1-lambda)
        FrankWolfe.active_set_update!(active_set, -lambda_max / 2, [2,3,4])
        @test collect(active_set) == [(0.75, [1, 2, 3]), (0.25, [2, 3, 4])]
        x = FrankWolfe.compute_active_set_iterate(active_set)
        @test x == [1.25, 2.25, 3.25]
        λ, a, i = FrankWolfe.active_set_argmin(active_set, [1.0, 1.0, 1.0])
        @test a == [1, 2, 3] && λ == 0.75 && i == 1
        λ, a, i = FrankWolfe.active_set_argmin(active_set, [-1.0, -1.0, -1.0])
        @test a == [2, 3, 4] && λ == 0.25 && i == 2
    end
end
