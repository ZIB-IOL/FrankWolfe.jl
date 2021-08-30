using TestEnv
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
TestEnv.activate()
using FrankWolfe
using ProgressMeter
using Arpack
using Plots
using DoubleFloats
using ReverseDiff
