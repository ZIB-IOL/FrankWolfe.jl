module FrankWolfe

using GenericSchur
using LinearAlgebra
using Printf
using ProgressMeter
using TimerOutputs
using SparseArrays: spzeros, SparseVector
import SparseArrays
import Random
using Setfield: @set

import MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities

# for Birkhoff polytope LMO
import Hungarian

import Arpack

export frank_wolfe, lazified_conditional_gradient, away_frank_wolfe
export blended_conditional_gradient, compute_extreme_point

include("abstract_oracles.jl")
include("defs.jl")
include("utils.jl")
include("linesearch.jl")
include("types.jl")
include("simplex_oracles.jl")
include("norm_oracles.jl")
include("polytope_oracles.jl")
include("moi_oracle.jl")
include("function_gradient.jl")
include("active_set.jl")
include("active_set_quadratic.jl")
include("active_set_quadratic_direct_solve.jl")
include("active_set_sparsifier.jl")

include("blended_cg.jl")
include("afw.jl")
include("fw_algorithms.jl")
include("block_oracles.jl")
include("block_coordinate_algorithms.jl")
include("alternating_methods.jl")
include("blended_pairwise.jl")
include("pairwise.jl")
include("tracking.jl")
include("callback.jl")

# collecting most common data types etc and precompile
# min version req set to 1.5 to prevent stalling of julia 1
@static if VERSION >= v"1.5"
    include("precompile.jl")
end

end
