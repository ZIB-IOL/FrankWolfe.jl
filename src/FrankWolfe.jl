module FrankWolfe

using LinearAlgebra
using Printf
using ProgressMeter
using TimerOutputs
using SparseArrays: spzeros, SparseVector
import SparseArrays
import Random

import MathOptInterface
const MOI = MathOptInterface

# for plotting -> keep here or move somewhere else?
using Plots

# for Birkhoff polytope LMO
import Hungarian

import Arpack
using DoubleFloats

export frank_wolfe, lazified_conditional_gradient, away_frank_wolfe
export blended_conditional_gradient, compute_extreme_point

include("defs.jl")
include("simplex_matrix.jl")

include("utils.jl")
include("oracles.jl")
include("simplex_oracles.jl")
include("norm_oracles.jl")
include("polytope_oracles.jl")
include("moi_oracle.jl")
include("function_gradient.jl")
include("active_set.jl")

include("blended_cg.jl")
include("afw.jl")
include("fw_algorithms.jl")

end
