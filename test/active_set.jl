module Test_active_set

using Test

using FrankWolfe
import FrankWolfe: ActiveSet
using LinearAlgebra: dot, norm, I
using Random
using StableRNGs

rng = StableRNG(42)
Random.seed!(rng, 42)

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
        @test FrankWolfe.active_set_validate(active_set)
        # update existing element
        FrankWolfe.active_set_update!(active_set, 1.0 / 2, [5, 6, 7])
        @test collect(active_set) == [(0.125, [1, 2, 3]), (0.125, [2, 3, 4]), (0.75, [5, 6, 7])]
        @test FrankWolfe.active_set_validate(active_set)
    end

    @testset "active set isempty" begin
        active_set = ActiveSet([(1.0, [1, 2, 3])])
        @test !isempty(active_set)
        empty!(active_set)
        @test isempty(active_set)
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
        x = FrankWolfe.get_active_set_iterate(active_set)
        @test x == [1.25, 2.25, 3.25]
        λ, a, i = FrankWolfe.active_set_argmin(active_set, [1.0, 1.0, 1.0])
        @test a == [1, 2, 3] && λ == 0.75 && i == 1
        λ, a, i = FrankWolfe.active_set_argmin(active_set, [-1.0, -1.0, -1.0])
        @test a == [2, 3, 4] && λ == 0.25 && i == 2
    end

    @testset "Copying active sets" begin
        active_set = ActiveSet([(0.125, [1, 2, 3]), (0.125, [2, 3, 4]), (0.75, [5, 6, 7])])
        as_copy = copy(active_set)
        # copy is of same type
        @test as_copy isa ActiveSet{Vector{Int},Float64,Vector{Float64}}
        # copy fields are also copied, same value different location in memory
        @test as_copy.weights !== active_set.weights
        @test as_copy.weights == active_set.weights
        @test as_copy.x !== active_set.x
        @test as_copy.x == active_set.x
        @test as_copy.atoms !== active_set.atoms
        @test as_copy.atoms == active_set.atoms
        # Individual atoms are not copied
        @test as_copy.atoms[1] === active_set.atoms[1]
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

    active_set = ActiveSet([(0.5, [0, 0]), (0.5, [0, 1]), (0.0, [1, 0])])
    x = FrankWolfe.get_active_set_iterate(active_set)
    @test x ≈ [0, 0.5]
    f(x) = (x[1] - 1)^2 + (x[2] - 1)^2

    gradient = similar(x)
    function grad!(storage, x)
        return storage .= [2 * (x[1] - 1), 2 * (x[2] - 1)]
    end
    FrankWolfe.simplex_gradient_descent_over_convex_hull(
        f,
        grad!,
        gradient,
        active_set,
        1e-3,
        1,
        0.0,
        0,
        max_iteration=1000,
        callback=nothing,
    )
    FrankWolfe.active_set_cleanup!(active_set)
    @test length(active_set) == 2
    @test [1, 0] ∈ active_set.atoms
    @test [0, 1] ∈ active_set.atoms
    active_set2 = ActiveSet([(0.5, [0, 0]), (0.0, [0, 1]), (0.5, [1, 0])])
    x2 = FrankWolfe.get_active_set_iterate(active_set2)
    @test x2 ≈ [0.5, 0]
    FrankWolfe.simplex_gradient_descent_over_convex_hull(
        f,
        grad!,
        gradient,
        active_set2,
        1e-3,
        1,
        0.0,
        0,
        max_iteration=1000,
        callback=nothing,
    )
    @test length(active_set) == 2
    @test [1, 0] ∈ active_set.atoms
    @test [0, 1] ∈ active_set.atoms
    @test FrankWolfe.get_active_set_iterate(active_set2) ≈ [0.5, 0.5]
    # updating again (at optimum) triggers the active set emptying
    for as in (active_set, active_set2)
        x = FrankWolfe.get_active_set_iterate(as)
        number_of_steps = FrankWolfe.simplex_gradient_descent_over_convex_hull(
            f,
            grad!,
            gradient,
            as,
            1.0e-3,
            1,
            0.0,
            0,
            max_iteration=1000,
            callback=nothing,
        )
        @test number_of_steps == 0
    end
end

@testset "LP separation oracle" begin
    # Gradient descent over a L-inf ball of radius one
    # current active set contains 3 vertices
    # direction points to [1,1]
    # |\ - -  +
    # | \     |
    # |  \
    # |   \   |
    # |    \
    # |     \ |
    # |______\|

    active_set = ActiveSet([(0.6, [-1, -1]), (0.2, [0, 1]), (0.2, [1, 0])])
    f(x) = (x[1] - 1)^2 + (x[2] - 1)^2
    ∇f(x) = [2 * (x[1] - 1), 2 * (x[2] - 1)]
    lmo = FrankWolfe.LpNormBallLMO{Inf}(1)

    x = FrankWolfe.get_active_set_iterate(active_set)
    @test x ≈ [-0.4, -0.4]
    gradient_dir = ∇f(x)
    (y, _, y_index) = FrankWolfe.lp_separation_oracle(lmo, active_set, gradient_dir, 0.5, 1)
    @test y ∈ active_set.atoms
    # the oracle returns the position of the atom it took from the active set
    @test active_set.atoms[y_index] == y
    (y2, _, y2_index) =
        FrankWolfe.lp_separation_oracle(lmo, active_set, gradient_dir, 3 + dot(x, gradient_dir), 1)
    # found new vertex not in active set
    @test y2 ∉ active_set.atoms
    @test y2_index == -1

    # Criterion too high, no satisfactory point
    (y3, _, y3_index) = FrankWolfe.lp_separation_oracle(
        lmo,
        active_set,
        gradient_dir,
        norm(gradient_dir)^2 + dot(x, gradient_dir),
        1,
    )
    @test y3_index == FrankWolfe.find_atom(active_set, y3)

    # the LMO's vertex is already active: the oracle reports its position
    v_lmo = FrankWolfe.compute_extreme_point(lmo, gradient_dir)
    active_set_with_v = ActiveSet([(0.5, [-1.0, -1.0]), (0.3, [0.0, 1.0]), (0.2, Vector(v_lmo))])
    (y4, _, y4_index) =
        FrankWolfe.lp_separation_oracle(lmo, active_set_with_v, gradient_dir, 1e6, 1)
    @test y4_index == 3
    @test active_set_with_v.atoms[y4_index] == y4

    # a zero direction ties every atom and the LMO vertex is not the best atom:
    # the oracle falls back to the scan and still reports the right position
    active_set_reordered = ActiveSet([(0.3, [0.0, 1.0]), (0.5, [-1.0, -1.0]), (0.2, Vector(v_lmo))])
    (y5, _, y5_index) = FrankWolfe.lp_separation_oracle(lmo, active_set_reordered, zeros(2), 1e6, 1)
    @test y5_index == FrankWolfe.find_atom(active_set_reordered, y5)
    @test y5_index == 2
end

@testset "Argminmax" begin
    active_set = FrankWolfe.ActiveSet([(0.6, [-1, -1]), (0.2, [0, 1]), (0.2, [1, 0])])
    (λ_min, a_min, i_min, val, λ_max, a_max, i_max, valM, progress) =
        FrankWolfe.active_set_argminmax(active_set, [1, 1.5])
    @test i_min == 1
    @test i_max == 2
    @test_throws ErrorException FrankWolfe.active_set_argminmax(active_set, [NaN, NaN])
end

@testset "LPseparationWithScaledHotVector" begin
    v1 = FrankWolfe.ScaledHotVector(1, 1, 2)
    v2 = FrankWolfe.ScaledHotVector(1, 2, 2)
    v3 = FrankWolfe.ScaledHotVector(0, 2, 2)
    active_set = FrankWolfe.ActiveSet([(0.6, v1), (0.2, v2), (0.2, v3)])
    lmo = FrankWolfe.LpNormBallLMO{Float64,1}(1.0)
    direction = ones(2)
    min_gap = 0.5
    Ktolerance = 1.0
    FrankWolfe.lp_separation_oracle(
        lmo,
        active_set,
        direction,
        min_gap,
        Ktolerance;
        inplace_loop=true,
    )
end

@testset "ActiveSet for BigFloat" begin
    n = Int(1e2)
    lmo = FrankWolfe.LpNormBallLMO{BigFloat,1}(rand(rng))
    x0 = Vector(FrankWolfe.compute_extreme_point(lmo, zeros(n)))

    # add the first vertex to active set from initialization
    active_set = FrankWolfe.ActiveSet([(1.0, x0)])

    # ensure that ActiveSet is created correctly, tests a fix for a bug when x0 is a BigFloat
    @test length(FrankWolfe.ActiveSet([(1.0, x0)])) == 1
end

@testset "find_atom with argminmax result" begin
    atoms = [rand(rng, 6) for _ in 1:25]
    active_set = ActiveSet([(1 / 25, a) for a in atoms])
    direction = randn(rng, 6)
    _, v_local, v_local_loc, v_local_value, _, _, _, _, _ =
        FrankWolfe.active_set_argminmax(active_set, direction)

    # every active atom is found at its position, a copy of the best atom included
    for (i, a) in enumerate(atoms)
        @test FrankWolfe.find_atom(active_set, a, direction, v_local_loc, v_local_value) == i
    end
    @test FrankWolfe.find_atom(active_set, copy(v_local), direction, v_local_loc, v_local_value) ==
          v_local_loc

    # the LMO's vertex scores below the active set's minimum: certified absent
    lmo = FrankWolfe.LpNormBallLMO{Inf}(1.0)
    v = FrankWolfe.compute_extreme_point(lmo, direction)
    @test dot(v, direction) < v_local_value
    @test FrankWolfe.find_atom(active_set, v, direction, v_local_loc, v_local_value) == -1

    # random queries, present or not, agree with the scan
    for _ in 1:200
        query = rand(rng, Bool) ? atoms[rand(rng, 1:25)] : rand(rng, 6)
        @test FrankWolfe.find_atom(active_set, query, direction, v_local_loc, v_local_value) ==
              FrankWolfe.find_atom(active_set, query)
    end

    # a tie with an atom other than the best one falls back to the scan:
    # a zero direction makes every atom tie at zero
    zero_direction = zeros(6)
    @test FrankWolfe.find_atom(active_set, atoms[7], zero_direction, 1, 0.0) == 7
    @test FrankWolfe.find_atom(active_set, rand(rng, 6), zero_direction, 1, 0.0) == -1

    # a minimum that is no longer valid (index -1, value -Inf) also falls back
    @test FrankWolfe.find_atom(active_set, atoms[3], direction, -1, -Inf) == 3
    @test FrankWolfe.find_atom(active_set, rand(rng, 6), direction, -1, -Inf) == -1

    # types without the certificate ignore the minimum arguments and scan
    as_cached = FrankWolfe.ActiveSetQuadraticProductCaching(
        [(1 / 25, a) for a in atoms],
        Matrix{Float64}(LinearAlgebra.I, 6, 6),
        zeros(6),
    )
    @test FrankWolfe.find_atom(as_cached, atoms[7], direction, v_local_loc, Inf) == 7
    @test FrankWolfe.find_atom(as_cached, rand(rng, 6), direction, v_local_loc, Inf) == -1

    # end-to-end on sparse atoms (Birkhoff): the update paths must never store an atom twice
    n = 8
    xp = rand(rng, n, n)
    f(x) = dot(x - xp, x - xp)
    function grad!(storage, x)
        storage .= 2 .* (x .- xp)
        return storage
    end
    birkhoff = FrankWolfe.BirkhoffPolytopeLMO()
    x0 = FrankWolfe.compute_extreme_point(birkhoff, randn(rng, n, n))
    for run in (
        (f, g, l, x) -> FrankWolfe.pairwise_frank_wolfe(f, g, l, x; max_iteration=300, lazy=false),
        (f, g, l, x) -> FrankWolfe.pairwise_frank_wolfe(f, g, l, x; max_iteration=300, lazy=true),
        (f, g, l, x) ->
            FrankWolfe.blended_pairwise_conditional_gradient(f, g, l, x; max_iteration=300),
        (f, g, l, x) -> FrankWolfe.blended_conditional_gradient(f, g, l, x; max_iteration=300),
    )
        res = run(f, grad!, birkhoff, x0)
        @test allunique(res.active_set.atoms)
        @test FrankWolfe.active_set_validate(res.active_set)
        # weight landing on a wrong index would desynchronize x from the decomposition
        @test res.active_set.x ≈ sum(λ .* a for (λ, a) in res.active_set)
    end

    # an atom stored in another representation must be found, never certified
    # absent: its fresh dot can differ from the stored copy's by an ulp
    vertices = unique([FrankWolfe.compute_extreme_point(birkhoff, randn(rng, n, n)) for _ in 1:20])
    dense_set = ActiveSet([(1 / length(vertices), Matrix(v)) for v in vertices])
    for _ in 1:50
        dir = randn(rng, n, n)
        _, _, loc, val, _, _, _, _, _ = FrankWolfe.active_set_argminmax(dense_set, dir)
        @test FrankWolfe.find_atom(dense_set, vertices[loc], dir, loc, val) == loc
    end
end

end # module
