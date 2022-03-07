module FrankWolfe

using GenericSchur
using LinearAlgebra
using Printf
using ProgressMeter
using TimerOutputs
using SparseArrays: spzeros, SparseVector
import SparseArrays
import Random

import MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities

# for Birkhoff polytope LMO
import Hungarian

import Arpack

# for tracking
import LibGit2

export frank_wolfe, lazified_conditional_gradient, away_frank_wolfe
export blended_conditional_gradient, compute_extreme_point

include("defs.jl")
include("utils.jl")
include("linesearch.jl")
include("types.jl")
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
include("pairwise.jl")
include("tracking.jl")

# collecting most common data types etc and precompile 
# min version req set to 1.5 to prevent stalling of julia 1
@static if VERSION >= v"1.5"   
    include("precompile.jl")
end

end
