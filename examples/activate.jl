using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using FrankWolfe
using ProgressMeter
using Arpack
using Plots
Pkg.activate(@__DIR__)
Pkg.instantiate()
using ReverseDiff
