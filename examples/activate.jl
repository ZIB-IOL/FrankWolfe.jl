using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using FrankWolfe
Pkg.activate(@__DIR__)
Pkg.instantiate()
using ReverseDiff
