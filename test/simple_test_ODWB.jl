############## A optimal design ######################################################################
# Problem described here: https://link.springer.com/article/10.1007/s11222-014-9476-y
# "A first-order algorithm for the A-optimal experimental design Problem: a mathematical programming approach"

# min 1/(trace(∑x_i v_iv_i^T))
# s.t. \sum x_i = s
#       lb ≤ x ≤ ub
#       x ∈ Z^m

# v_i ∈ R^n
# n - number of parameters
# m - number of possible experiments
# A = [v_1^T,.., v_m^T], so the rows of A correspond to the different experiments


################################ D optimal design ########################################################################
# Problem described here: https://arxiv.org/pdf/2302.07386.pdf
# "Branch-and-Bound for D-Optimality with fast local search and bound tightening"

# min log(1/(det(∑x_i v_iv_i^T)))
# s.t. \sum x_i = s
#       lb ≤ x ≤ ub
#       x ∈ Z^m

# v_i ∈ R^n
# n - number of parameters
# m - number of possible experiments
# A = [v_1^T,.., v_m^T], so the rows of A correspond to the different experiments

################################ D-fusion design ########################################################################
# Problem described here: https://arxiv.org/pdf/2302.07386.pdf
# "Branch-and-Bound for D-Optimality with fast local search and bound tightening"

# min log(1/(det(∑x_i v_iv_i^T)))
# s.t. \sum x_i = s
#       lb ≤ x ≤ ub
#       x ∈ Z^m

# v_i ∈ R^n
# n - number of parameters
# m - number of possible experiments
# A = [v_1^T,.., v_m^T], so the rows of A correspond to the different experiments

using Boscia
using FrankWolfe
using Bonobo
using Random
using SCIP
using JuMP
using Hypatia
import Hypatia.Cones: vec_length, vec_copyto!, svec_length, svec_side
import Hypatia.Cones: smat_to_svec!, svec_to_smat!
const Cones = Hypatia.Cones
using Pajarito
using PajaritoExtras # https://github.com/chriscoey/PajaritoExtras.jl
using HiGHS
using LinearAlgebra
using Statistics
using Distributions
import MathOptInterface
using Printf
using Dates
using Test
using ProfileView
using DataFrames
using CSV
const MOI = MathOptInterface
const MOIU = MOI.Utilities

import MathOptSetDistances
const MOD = MathOptSetDistances

include("utilities.jl")
include("opt_design_boscia.jl")
include("scip_oa.jl")
include("opt_design_scip.jl")
include("spectral_functions_JuMP.jl")
include("opt_design_pajarito.jl")
include("opt_design_custom_BB.jl")

seed = rand(UInt64)
#seed = 0xad2cba3722a98b62
@show seed
Random.seed!(seed)

dimensions = [20,30]
facs = [10,4]
time_limit = 300
verbose = false

m = 20
k = 4
n = Int(floor(m/k))

A, _, N, ub, _ = build_data(seed, m, n, false, false; scaling_C=false)
ub = ones(m)
o = SCIP.Optimizer()
MOI.set(o, MOI.Silent(), true)
lmo, x = build_lmo(o, m, N, ub)
blmo = build_blmo(m, N, ub)
branching_strategy = Bonobo.MOST_INFEASIBLE()
heu = Boscia.Heuristic()

result = 0.0
domain_oracle = build_domain_oracle(A, n)
f, grad! = build_a_criterion(A, false, μ=1e-4, build_safe=true, long_run=false)
_, active_set, S = build_start_point2(A, m, n, N, ub)
z = greedy_incumbent(A, m, n, N, ub)
#x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=10, active_set=active_set, domain_oracle=domain_oracle, start_solution=z, dual_tightening=false, global_dual_tightening=false, lazy_tolerance=2.0, branching_strategy=branching_strategy, use_shadow_set=false, custom_heuristics=[heu]) 

function Boscia.is_decomposition_invariant_oracle_simple(sblmo::Boscia.ProbabilitySimplexSimpleBLMO)
    return true  
end

function Boscia.bounded_compute_inface_extreme_point(sblmo::Boscia.ProbabilitySimplexSimpleBLMO, d, x, lb, ub, int_vars; kwargs...)
    a = zeros(length(d))
    indices = collect(1:length(d))
    perm = sortperm(d)

    # The lower bounds always have to be met. 
    a[int_vars] = lb

    for i in indices[perm]
        if x[i] !== 0.0
            if i in int_vars
                idx = findfirst(x -> x == i, int_vars)
                a[i] += min(ub[idx] - lb[idx], sblmo.N - sum(a))
            else
                a[i] += sblmo.N - sum(a)
            end
        end
    end
    return a
end

function Boscia.bounded_dicg_maximum_step(sblmo::Boscia.ProbabilitySimplexSimpleBLMO, direction, x, lb, ub, int_vars; kwargs...)
    gamma_max = one(eltype(direction))
    @inbounds for idx in eachindex(x)
        di = direction[idx]
        if di < 0
            gamma_max = min(gamma_max, ub[idx]-x[idx])
        elseif di > 0
            gamma_max = min(gamma_max, x[idx]-lb[idx])
        end
    end
    return gamma_max
end


ProfileView.@profview x, _, result = Boscia.solve(f, grad!, blmo; verbose=true, time_limit=time_limit, active_set=active_set, domain_oracle=domain_oracle, start_solution=z, dual_tightening=false, global_dual_tightening=false, lazy_tolerance=2.0, branching_strategy=branching_strategy, use_shadow_set=false, custom_heuristics=[heu])  
#x, _, result = Boscia.solve(f, grad!, blmo; verbose=true, time_limit=time_limit, active_set=active_set, domain_oracle=domain_oracle, start_solution=z, dual_tightening=false, global_dual_tightening=false, lazy_tolerance=2.0, branching_strategy=branching_strategy, use_shadow_set=false, custom_heuristics=[heu], variant=Boscia.DICG(),lazy=false)