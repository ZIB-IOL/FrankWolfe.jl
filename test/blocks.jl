using Test
using Random
using LinearAlgebra
using FrankWolfe
using SparseArrays

@testset "Block array behaviour" begin
    arr = FrankWolfe.BlockVector([
        [
            1 3
            4 6.0
        ],
        [
            1 3 1
            4 1 2
            1 3 5.0
        ],
    ],)
    @test length(arr) == 4 + 9
    @test arr[end] == 5.0
    @test arr[1] == 1.0
    @test arr[5] == 1.0
    @test arr[4] == 6.0
    v1 = Vector(arr)
    arr2 = FrankWolfe.BlockVector([zeros(2, 2), ones(3, 3)],)
    copyto!(arr2, arr)
    v2 = Vector(arr2)
    @test v1 == v2
    v1[1] *= 2
    @test v1 != v2
    @test size(similar(arr)) == size(arr)
    @test typeof(similar(arr)) == typeof(arr)
    @test norm(arr) == norm(collect(arr))
    @test dot(arr, arr) == dot(collect(arr), collect(arr))
    
    arr3 = FrankWolfe.BlockVector([3 * ones(2, 2), ones(3, 3)],)
    @test dot(arr3, arr) ≈ dot(arr, arr3)
    @test dot(arr3, arr) ≈ dot(collect(arr3), collect(arr))
    arr4 = 2 * arr3
    @test arr4.blocks[1] == 6 * ones(2, 2)
    arr5 = FrankWolfe.BlockVector([6 * ones(2, 2), 2 * ones(3, 3)],)
    @test isequal(arr4, arr4)
    @test isequal(arr4, arr5)
    @test !isequal(arr3, arr4)
    @test !isequal(arr2, FrankWolfe.BlockVector([zeros(2, 2)]))

end

@testset "FrankWolfe array methods" begin
    mem = FrankWolfe.InplaceEmphasis()
    arr0 = FrankWolfe.BlockVector([
        [
            1 3
            4 6.0
        ],
        [
            1 3 1
            4 1 2
            1 3 5.0
        ],
    ],)
    arr1 = rand() * arr0
    d = similar(arr1)
    FrankWolfe.muladd_memory_mode(mem, d, arr1, arr0)
    dref = arr1 - arr0
    @test d == dref
    gamma = rand()
    arr2 = zero(arr0)
    FrankWolfe.muladd_memory_mode(mem, arr2, gamma, arr0)
    @test arr2 == -gamma * arr0
    arr3 = similar(arr0)
    FrankWolfe.muladd_memory_mode(mem, arr3, zero(arr0), gamma, arr0)
    @test arr3 == -gamma * arr0

    active_set = FrankWolfe.ActiveSet(
        [0.25, 0.75],
        [
            FrankWolfe.BlockVector([ones(100, 1), ones(30, 30)]),
            FrankWolfe.BlockVector([3 * ones(100, 1), zeros(30, 30)]),
        ],
        FrankWolfe.BlockVector([ones(100, 1), ones(30, 30)]),
    )
    FrankWolfe.compute_active_set_iterate!(active_set)
    x = active_set.x
    x_copy = 0 * x
    for (λi, ai) in active_set
        @. x_copy += λi * ai
    end
    @test norm(x_copy - x) ≤ 1e-12
end
