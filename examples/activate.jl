import Pkg
Pkg.activate(@__DIR__)

using TestEnv

Pkg.activate(joinpath(@__DIR__, ".."))
TestEnv.activate()

using FrankWolfe
using ProgressMeter
using Arpack
using Plots
using DoubleFloats
using ReverseDiff
using PlotThemes

include(joinpath(dirname(pathof(FrankWolfe)), "../examples/plot_utils.jl"))
