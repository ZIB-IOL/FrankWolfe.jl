include("activate.jl")

# using FrankWolfe: LinearAlgebra
# using ArgParse
include("fwwrap.jl")

###
# MAIN
###

# parse parameters
# s = ArgParseSettings()
# @add_arg_table s begin
#     "--away"
#         help = "Use Away-Step FrankWolfe"
#         action = :store_true
# end

# parsed_args = parse_args(ARGS, s)

N = 10
away = true  # parsed_args["away"]
iters = 100

# call Frank-Wolfe on LOP polytope to project a point
DIM = N * (N - 1)
xf = 1.2 * ones(Float64, DIM)
x0 = 0.5 * ones(Float64, N * (N - 1))
a, x = LOP_solve(N, xf, x0, iters, away)
