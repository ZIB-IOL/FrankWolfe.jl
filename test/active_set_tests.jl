using Test

using FrankWolfe

###########################################
# Testing extraction of lambdas and atoms
###########################################

@testset "Active set" begin 
    active_set = [ 
        [0.1, [1,2,3]],
        [0.9, [2,3,4]],
        [0.0, [5,6,7]]
    ]


    @test FrankWolfe.active_set_lambdas(active_set) == [0.1, 0.9, 0.0]

    @test FrankWolfe.active_set_atoms(active_set) == [[1, 2, 3], [2, 3, 4], [5, 6, 7]]

    @test FrankWolfe.active_set_validate(active_set) == true


    ###########################################
    # Testing presence and cleanup
    ###########################################

    active_set = [ 
        [0.1, [1,2,3]],
        [0.9, [2,3,4]],
        [0.0, [5,6,7]]
    ]

    @test FrankWolfe.active_set_atom_present(active_set, [5,6,7]) > 0

    FrankWolfe.active_set_cleanup!(active_set)

    @test FrankWolfe.active_set_atom_present(active_set, [5,6,7]) == -1


    ###########################################
    # Testing renormalization and validation
    ###########################################

    active_set = [ 
        [0.1, [1,2,3]],
        [0.8, [2,3,4]],
        [0.0, [5,6,7]]
    ]

    @test FrankWolfe.active_set_validate(active_set) == false

    FrankWolfe.active_set_renormalize!(active_set)

    @test FrankWolfe.active_set_validate(active_set) == true


    ###########################################
    # Testing addition of elements and manipulation
    ###########################################

    active_set = [ 
        [0.5, [1,2,3]],
        [0.5, [2,3,4]],
    ]

    # [0.0, [5,6,7]]

    # adding new element

    FrankWolfe.active_set_update!(active_set, 1 / 2, [5,6,7]) 

    @test active_set == [[0.25, [1, 2, 3]], [0.25, [2, 3, 4]], [0.5, [5, 6, 7]]]

    @test FrankWolfe.active_set_validate(active_set) == true

    # update existing element

    FrankWolfe.active_set_update!(active_set, 1.0 / 2, [5,6,7]) 

    @test active_set == [[0.125, [1, 2, 3]], [0.125, [2, 3, 4]], [0.75, [5, 6, 7]]]

    @test FrankWolfe.active_set_validate(active_set) == true

    # away step operations

    ## drop step first

    lambda = FrankWolfe.active_set_get_lambda_atom(active_set,[5, 6, 7])

    @test lambda == 0.75

    lambda_max = lambda / (1-lambda)

    FrankWolfe.active_set_update!(active_set, -lambda_max, [5,6,7]) 

    @test active_set == [[0.5, [1, 2, 3]], [0.5, [2, 3, 4]]]

    ## intermediate away step

    lambda = FrankWolfe.active_set_get_lambda_atom(active_set,[2, 3, 4])

    @test lambda == 0.5

    lambda_max = lambda / (1-lambda)

    FrankWolfe.active_set_update!(active_set, -lambda_max / 2, [2,3,4]) 

    @test active_set == [[0.75, [1, 2, 3]], [0.25, [2, 3, 4]]]

    x = FrankWolfe.active_set_return_iterate(active_set) 

    @test x == [1.25, 2.25, 3.25]

    a, lambda, i = FrankWolfe.active_set_argmin(active_set, [1.0, 1.0, 1.0])

    @test a == [1, 2, 3] && lambda == 0.75 && i == 1

    a, lambda, i = FrankWolfe.active_set_argmin(active_set, [-1.0, -1.0, -1.0])

    @test a == [2, 3, 4] && lambda == 0.25 && i == 2

end
