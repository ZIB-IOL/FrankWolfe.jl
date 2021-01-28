using Test

using FrankWolfe
import FrankWolfe: ActiveSet

@testset "Active sets" begin

    @testset "Constructors and eltypes" begin
        active_set = ActiveSet([(0.1, [1, 2, 3]), (0.9, [2, 3, 4]), (0.0, [5, 6, 7])])

        @test active_set.weights == [0.1, 0.9, 0.0]
        @test active_set.atoms == [[1, 2, 3], [2, 3, 4], [5, 6, 7]]
        @test eltype(active_set) === Tuple{Float64,Vector{Int}}
        @test FrankWolfe.active_set_validate(active_set)

        # providing the weight type converts the provided weights
        active_set2 = ActiveSet{Vector{Int},Float64}([
            (0.1f0, [1, 2, 3]),
            (0.9f0, [2, 3, 4]),
            (0.0f0, [5, 6, 7]),
        ])

        @test eltype(active_set) === eltype(active_set2)
        @test first(active_set) isa eltype(active_set)
        @test first(active_set2) isa eltype(active_set)
    end

    @testset "Presence and cleanup" begin
        active_set = ActiveSet([(0.1, [1, 2, 3]), (0.9, [2, 3, 4]), (0.0, [5, 6, 7])])
        @test FrankWolfe.find_atom(active_set, [5, 6, 7]) > 0
        FrankWolfe.active_set_cleanup!(active_set)
        @test FrankWolfe.find_atom(active_set, [5, 6, 7]) == -1
    end

    @testset "Renormalization and validation" begin
        active_set = ActiveSet([(0.1, [1, 2, 3]), (0.8, [2, 3, 4]), (0.0, [5, 6, 7])])
        @test !FrankWolfe.active_set_validate(active_set)
        FrankWolfe.active_set_renormalize!(active_set)
        @test FrankWolfe.active_set_validate(active_set)
    end

    @testset "addition of elements and manipulation" begin
        active_set = ActiveSet([(0.5, [1, 2, 3]), (0.5, [2, 3, 4])])
        FrankWolfe.active_set_update!(active_set, 1 / 2, [5, 6, 7])
        @test collect(active_set) == [(0.25, [1, 2, 3]), (0.25, [2, 3, 4]), (0.5, [5, 6, 7])]

        # element addition
        @test FrankWolfe.active_set_validate(active_set) == true
        # update existing element
        FrankWolfe.active_set_update!(active_set, 1.0 / 2, [5, 6, 7])
        @test collect(active_set) == [(0.125, [1, 2, 3]), (0.125, [2, 3, 4]), (0.75, [5, 6, 7])]
        @test FrankWolfe.active_set_validate(active_set) == true
    end

    @testset "Away step operations" begin
        active_set = ActiveSet([(0.125, [1, 2, 3]), (0.125, [2, 3, 4]), (0.75, [5, 6, 7])])

        lambda = FrankWolfe.weight_from_atom(active_set, [5, 6, 7])
        @test lambda == 0.75
        lambda_max = lambda / (1 - lambda)
        FrankWolfe.active_set_update!(active_set, -lambda_max, [5, 6, 7])
        @test collect(active_set) == [(0.5, [1, 2, 3]), (0.5, [2, 3, 4])]

        ## intermediate away step
        lambda = FrankWolfe.weight_from_atom(active_set, [2, 3, 4])
        @test lambda == 0.5
        lambda_max = lambda / (1 - lambda)
        FrankWolfe.active_set_update!(active_set, -lambda_max / 2, [2, 3, 4])
        @test collect(active_set) == [(0.75, [1, 2, 3]), (0.25, [2, 3, 4])]
        x = FrankWolfe.compute_active_set_iterate(active_set)
        @test x == [1.25, 2.25, 3.25]
        λ, a, i = FrankWolfe.active_set_argmin(active_set, [1.0, 1.0, 1.0])
        @test a == [1, 2, 3] && λ == 0.75 && i == 1
        λ, a, i = FrankWolfe.active_set_argmin(active_set, [-1.0, -1.0, -1.0])
        @test a == [2, 3, 4] && λ == 0.25 && i == 2
    end
end

@testset "Simplex gradient descent" begin
    # Gradient descent over a 2-D unit simplex
    # each atom is a vertex, direction points to [1,1]
    # note: integers for atom element types
    # |\ - -  + 
    # | \     |
    # |  \
    # |   \   |
    # |    \
    # |     \ |
    # |______\|

    active_set = ActiveSet([
        (0.5, [0, 0]), (0.5, [0, 1]), (0.0, [1, 0]),
    ])
    @test FrankWolfe.compute_active_set_iterate(active_set) ≈ [0, 0.5]
    f(x) = (x[1]-1)^2 + (x[2]-1)^2
    ∇f(x) = [2 * (x[1] - 1), 2 * (x[2] - 1)]
    gradient_dir = ∇f([0, 0.5])
    FrankWolfe.update_simplex_gradient_descent!(active_set, gradient_dir, f)
    @test length(active_set) == 2
    @test [1, 0] ∈ active_set.atoms
    @test [0, 1] ∈ active_set.atoms

    active_set2 = ActiveSet([
        (0.5, [0, 0]), (0.0, [0, 1]), (0.5, [1, 0]),
    ])
    @test FrankWolfe.compute_active_set_iterate(active_set2) ≈ [0.5, 0]
    gradient_dir = ∇f(FrankWolfe.compute_active_set_iterate(active_set2))
    FrankWolfe.update_simplex_gradient_descent!(active_set2, gradient_dir, f, L=4.0)
    @test length(active_set) == 2
    @test [1, 0] ∈ active_set.atoms
    @test [0, 1] ∈ active_set.atoms
    @test FrankWolfe.compute_active_set_iterate(active_set2) ≈ [0.5, 0.5]
    # updating again (at optimum) triggers the active set emptying
    for as in (active_set, active_set2)
        gradient_dir = ∇f(FrankWolfe.compute_active_set_iterate(as))
        FrankWolfe.update_simplex_gradient_descent!(as, gradient_dir, f)
        @test length(active_set) == 1
    end
end
