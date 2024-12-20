using FrankWolfe
using LinearAlgebra


for file in readdir(joinpath(@__DIR__, "/problems/"), join=true)
    if endswith(file, "jl")
        include(file)
    end
end

@enum LineSearchVariant begin
    LS_ONLY_SECANT = 1
    LS_SECANT_WITH_BACKTRACKING = 2
    LS_BACKTRACKING_AND_SECANT = 3
    LS_ADAPTIVE = 4
end

const linesearchvariant_string = (
    LS_ONLY_SECANT ="Only Secant",
    LS_SECANT_WITH_BACKTRACKING="Secant with Backtracking",
    LS_BACKTRACKING_AND_SECANT="Backtracking and Secant",
    LS_ADAPTIVE="Adaptive",
)

function solve_problems(seed, dimension, problem, ls_variant, write=true, verbose=true)
end