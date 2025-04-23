"""

@article{carderera2020second,
  title={Second-order conditional gradient sliding},
  author={Carderera, Alejandro and Pokutta, Sebastian},
  journal={arXiv preprint arXiv:2002.08907},
  year={2020},
  URL={https://arxiv.org/abs/2002.08907}
}

@article{doi:10.1137/140992382,
author = {Lan, Guanghui and Zhou, Yi},
title = {Conditional Gradient Sliding for Convex Optimization},
journal = {SIAM Journal on Optimization},
volume = {26},
number = {2},
pages = {1379-1409},
year = {2016},
doi = {10.1137/140992382},
URL = {https://doi.org/10.1137/140992382},
eprint = {   
        https://doi.org/10.1137/140992382   

}


"""

using FrankWolfe
import FrankWolfe: ActiveSet


"""
Supertype for parameters of the condiditional gradient descent.
    All MomentumStepsize must implement `compute_CnGD_parameters(parameters_rule::ConstantCnGDParameters,...)`
    and return a tuple of parameters `param` for conditional_gradient_descent!
    See Algorithm 1 in https://doi.org/10.1137/140992382.
"""
abstract type CnGDParameters end

function compute_CnGD_parameters end


"""
    ConstantCnGDParameters <:CnGDParameters 
    Constant parameters eta and beta for condiditional gradient step.
"""
struct ConstantCnGDParameters <: CnGDParameters
    threshold::Real
    regularization::Real
end

function compute_CnGD_parameters(parameters_rule::ConstantCnGDParameters, state)
    return parameters_rule.threshold, parameters_rule.regularization
end

"""
    OpenLoopCnGDParameters <:CnGDParameters  
    Fixed trajectory only depending on the iteration t of parameters eta and beta for condiditional gradient step.
"""
struct OpenLoopCnGDParameters <: CnGDParameters
    threshold_trajectory
    regularization_trajectory
end

"""
    LZCnGDParameters <:CnGDParameters  
    Fixed trajectory for the threshold η_t = threshold_multiplier/t and
    the regularization β_t = regularization_multiplier/t
"""
struct LZCnGDParameters{T} <: CnGDParameters where T<:Real
    threshold_multiplier::T
    regularization_mutliplier::T
end

function compute_CnGD_parameters(parameters_rule::LZCnGDParameters{T}, state) where T
    return T(parameters_rule.threshold_multiplier/state.t), T(parameters_rule.regularization_mutliplier/state.t)
end


"""
Supertype for stepsize gamma.
    All MomentumStepsize must implement `compute_momentum_stepsize(stepsize_rule::ConstantMomentumStepsize,...)`
    and return a stepsize.
    See Algorithm 1 in https://doi.org/10.1137/140992382.
"""
abstract type MomentumStepsize end

function compute_momentum_stepsize end

"""
    ConstantMomentumStepsize <: MomentumStepsize 
    Constant stepsize gamma for momentum update.
"""

struct ConstantMomentumStepsize <: MomentumStepsize
    stepsize::Real
end

function compute_momentum_stepsize(stepsize_rule::ConstantMomentumStepsize, state)
    return stepsize_rule.stepsize
end

"""
    OpenloopMomentStepsize <: MomentumStepsize
    Fixed trajectory only depending on the iteration t of stepsizes gamma for momentum update.
"""
struct OpenloopMomentStepsize <: MomentumStepsize
    stepsize_trajectory
end

function compute_momentum_stepsize(stepsize_rule::OpenloopMomentStepsize, state)
    return stepsize_rule.stepsize_trajectory(state.t)
end

"""
    LZMomentStepsize{T} <: MomentumStepsize
    Fixed momentum trajectory of γ_t = mutliplier/(t+1).
"""
struct LZMomentStepsize{T} <: MomentumStepsize where T<:Real
    multipier::T
end

LZMomentStepsize() = LZMomentStepsize{Float64}(2.0)

function compute_momentum_stepsize(stepsize_rule::LZMomentStepsize{T}, state) where T
    return stepsize_rule.multipier/T(state.t + 1)
end


"""
Supertype for condiditional gradient oracles.

All CndGOs must implement `conditional_gradient_descent!(x, cndgrado::CndGO, ...)`
and return the status of the conditional gradient descent.
"""
abstract type CndGO end

function conditional_gradient_descent! end

@enum CndGStatus begin
    CndGS_UNSOLVED = 1
    CndGS_SOLVED = 2
    CndGS_TIMELIMIT = 3
    CndGS_ITERLIMIT = 4
end

mutable struct CndGState
    status::CndGStatus
end

"""
    LanZhouProcedure <: CndGO

Procedure to compute a condiditional gradient step. A linear minimization oracle `lmo` is required.
See Algorithm 1 in https://doi.org/10.1137/140992382.
"""
struct LanZhouProcedure <: CndGO
    timeout::Real
    max_iteration::Int
    lmo::LinearMinimizationOracle
end


function _cgd_value_function(g::Vector{T}, u::Vector{T}, 
                            β::T, ut::Vector{T}, vt
                    )  where T<:Real
    
    return fast_dot(g,ut) - fast_dot(g,vt) + β *(fast_dot(ut,ut) + fast_dot(u,vt) - fast_dot(u,ut) - fast_dot(ut,vt))
end

function conditional_gradient_descent!(
    x::Vector{T},
    cndgrado::LanZhouProcedure,
    gradient::Vector{T};
    params::Tuple{T,T}
) where T<:Real

    t = 0
    time_start = time_ns()
    tot_time = time_start
    ut = copy(x)
    vt = collect(x)
    eta, beta = params

    while t < cndgrado.max_iteration && tot_time < cndgrado.timeout
        time_at_loop = time_ns()
        tot_time = (time_at_loop - time_start) / 1e9
        t += 1

        if t == 1
            vt = compute_extreme_point(cndgrado.lmo, gradient + beta * (ut - x))
        else
            vt = compute_extreme_point(cndgrado.lmo, gradient + beta * (ut - x), v=vt)
        end
        V = _cgd_value_function(gradient, x, beta, ut, vt)
        if V <= eta
            break
        end
        alpha = V/(beta * (fast_dot(vt,vt) + fast_dot(ut,ut) -2*fast_dot(ut,vt)))
        alpha = min(1, alpha)
        @. ut = (1-alpha) * ut + alpha * vt
    end

    x .= ut
    if tot_time >= cndgrado.timeout
        return CndGS_TIMELIMIT
    end

    if t >= cndgrado.max_iteration
        return CndGS_ITERLIMIT
    end

    return CndGS_SOLVED

end



"""
    conditional_gradient_sliding(grad!, cndgrado,  momentum_stepsize, cndgrad_params, stop_rule, x0)

Simplest form of the conditional gradient sliding algorithm. See Algorithm 1 in https://doi.org/10.1137/140992382.
Returns a tuple `(x, v, primal, dual_gap, traj_data)` with:
- `x` final iterate
- `y` last momentum
- `z` last lookahead
- `t` number of iterations
- `tot_time` total time
- `traj_data` vector of trajectory information.
"""
function conditional_gradient_sliding(
    grad!,
    cndgrado,
    x0;
    momentum_stepsize::MomentumStepsize,
    cndgrad_params::CnGDParameters,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    traj_data=[],
    timeout=Inf,
)

    

    x = copy(x0)
    y = copy(x0)
    z = collect(x0)
    gradient = copy(x0)
    gamma = zero(x0[1])

    t = 1
    cndG_state = CndGState(CndGS_UNSOLVED)   
    state = (
            t=t,
            tot_time=0.,
            x=x,
            y=y,
            z=z,
            gamma=0.,
            cnGDescent_params = (0., 0.),
            gradient=gradient,
            cndG_status=cndG_state.status,
            
        )

    time_start = time_ns()
    tot_time = time_start
    while t ≤ max_iteration && (tot_time - time_start) ≤ timeout 
        time_at_loop = time_ns()
        tot_time = (time_at_loop - time_start) / 1e9
        cndG_state.status = CndGS_UNSOLVED

        gamma = compute_momentum_stepsize(momentum_stepsize, state)
        @. z = (1-gamma) * y + gamma * x
        grad!(gradient, z)
        cnGDescent_params  = compute_CnGD_parameters(cndgrad_params, state)
        cndG_state.status = conditional_gradient_descent!(x,cndgrado, gradient; params = cnGDescent_params )
        @. y = (1-gamma) * y + gamma * x

        state = (
            t=t,
            tot_time=tot_time,
            x=x,
            y=y,
            z=z,
            gamma=gamma,
            cnGDescent_params = cnGDescent_params,
            gradient=gradient,
            cndG_status=cndG_state.status,
        )
        if trajectory
            push!(traj_data,state)
        end

        if verbose && mod(t, print_iter) == 0
            print("It. ", t, " Tot.Time ", tot_time, " CndG status ", cndG_state.status)
        end

        t += 1
    end

    if verbose && timeout < Inf && tot_time ≥ timeout
        @info "Time limit reached"
    end

    if verbose && t ≥ max_iteration
        @info "Iteration limit reached"
    end


    return state
end

#################################################################################################################
#################################################################################################################
#SOCGS  Second-order conditional gradient sliding, Carderera, Alejandro and Pokutta, Sebastian, arXiv preprint arXiv:2002.08907
#################################################################################################################
#################################################################################################################

@enum CGSStepsize begin
    CGS_FW_STEP = 1
    CGS_PVM_STEP = 2
end

""" TOWRITE
"""
abstract type LowerBoundEstimator end
function compute_pvm_threshold end

""" TOWRITE
"""
struct LowerBoundFiniteSteps{LMO<:LinearMinimizationOracle} <: LowerBoundEstimator 
    f
    grad!
    lmo::LMO
    corrective_step::CorrectiveStep
    max_iter::Int
end
function compute_pvm_threshold(lb_estimator::LowerBoundFiniteSteps,
                                x,
                                primal::Real,
                                gradient;
                                line_search::LineSearchMethod)
    _, _, primal_finite_steps, _, _, _ =  corrective_frank_wolfe(
                            lb_estimator.f,
                            lb_estimator.grad!,
                            lb_estimator.lmo,
                            lb_estimator.corrective_step,
                            ActiveSet([(one(x[1]),x)]);
                            line_search = line_search,
                            max_iteration= lb_estimator.max_iter ,
                            gradient = gradient
    )
    return (primal - primal_finite_steps)^4/(fast_dot(gradient,gradient)^2)
end
#function conditional_gradient_step! end

function second_order_conditional_gradient_sliding(
    f,
    grad!,
    build_quadratic_approximation!, 
    fw_step::CorrectiveStep, 
    lmo_fw::LinearMinimizationOracle,
    pvm_step::CorrectiveStep, 
    lmo_pvm::LinearMinimizationOracle,
    x0;
    lb_estimator::LowerBoundEstimator,
    line_search_fw::LineSearchMethod=Adaptive(),
    line_search_pvm::LineSearchMethod=Adaptive(),
    line_search_LB_estimator::LineSearchMethod=Adaptive(),
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    traj_data=[],
    timeout=Inf,
)

    active_set_fw = ActiveSet([(one(x0[1]),x0)])
    active_set_pvm = ActiveSet([(one(x0[1]),x0)]) 
    gradient= collect(x0)
    Hx = collect(x0)
    quadratic_term_storage = collect(x0)
    x_fw = get_active_set_iterate(active_set_fw)
    x_pvm = get_active_set_iterate(active_set_pvm)
    x = x_pvm    
    
    dual_gap_fw = Inf
    primal_fw = Inf
    primal_pvm = Inf
    primal = f(x0)

    t = 0
    step_type = CGS_FW_STEP
    time_start = time_ns()
    tot_time = time_start

    state = (
            t= 0,
            tot_time = tot_time,
            primal = primal,
            dual_fw = Inf,
            dual_gap_fw = Inf,
            x = x0,
            step_type = CGS_FW_STEP,
            )    
    while t ≤ max_iteration && (tot_time - time_start) ≤ timeout 
        time_at_loop = time_ns()
        tot_time = (time_at_loop - time_start) / 1e9

        #computing gradient 
        grad!(gradient, x)
        #building quadratic approximation (problem dependent)
        quadratic_term_function!, Hx = build_quadratic_approximation!(Hx,x,gradient,primal)
        constant_term = primal - FrankWolfe.fast_dot(gradient,x) + 0.5 * FrankWolfe.fast_dot(Hx,x)
        function f_quad_approx(p)
            return 0.5*quadratic_term_function!(quadratic_term_storage,p) + FrankWolfe.fast_dot(gradient,p) - FrankWolfe.fast_dot(Hx,p) + constant_term
        end
        function grad_quad_approx!(storage,p)
            storage .=  p + gradient - Hx
        end

        epsilon = compute_pvm_threshold(lb_estimator,x,primal,gradient; line_search = line_search_LB_estimator)   
        #H-projection (pvm)
        x_pvm, _ , _, _, _ = corrective_frank_wolfe(
            f_quad_approx,
            grad_quad_approx!,
            lmo_pvm,
            pvm_step,
            active_set_pvm;
            line_search=line_search_pvm,
            epsilon=epsilon,
            gradient = gradient
        )
        primal_pvm = f(x_pvm)
        #Fw corrective step
        x_fw, _ , primal_fw, _, _=corrective_frank_wolfe(
                f,
                grad!,
                lmo_fw,
                fw_step,
                active_set_fw;
                line_search=line_search_fw,
                max_iteration=1,
                gradient = gradient
            )
          
        
        if primal_pvm >= primal_fw 
            copyto!(active_set_pvm,active_set_fw)
            primal = primal_fw
            x = x_fw
            step_type = CGS_FW_STEP
        else
            primal = primal_pvm
            x = x_pvm
            step_type = CGS_PVM_STEP
        end
        
        t += 1
        state = (
                t= t,
                primal = primal,
                dual_fw = primal_fw - dual_gap_fw,
                dual_gap_fw = dual_gap_fw,
                tot_time = tot_time,          
                x = x,
                epsilon = epsilon,
                step_type = step_type,
            )
        if trajectory
            push!(
                traj_data,
                state
            )
        end

        if verbose && mod(t, print_iter) == 0
            println("It. ", t, " Tot.Time ", tot_time," ", step_type)
        end

        
    end

    if verbose && timeout < Inf && tot_time ≥ timeout
        @info "Time limit reached"
    end

    if verbose && t ≥ max_iteration
        @info "Iteration limit reached"
    end


    return (x=x, primal=primal, dual_gap_fw = dual_gap_fw,
             t = t, tot_time = tot_time,traj_data=traj_data)
end