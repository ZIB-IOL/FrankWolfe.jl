using FrankWolfe
import MathOptInterface as MOI

import FrankWolfe: ActiveSet
import FrankWolfe: LinearMinimizationOracle
import FrankWolfe: CorrectiveStep
import FrankWolfe: LineSearchMethod
import FrankWolfe: get_active_set_iterate
import FrankWolfe: corrective_frank_wolfe
import FrankWolfe: fast_dot
import FrankWolfe: AbstractActiveSet
import FrankWolfe: Adaptive
import FrankWolfe: Secant

#################################################################################################################
#################################################################################################################
#SOCGS  Second-order conditional gradient sliding, Carderera, Alejandro and Pokutta, Sebastian, arXiv preprint arXiv:2002.08907
#################################################################################################################
#################################################################################################################

@enum CGSStepsize begin
    CGS_FW_STEP = 1
    CGS_PVM_STEP = 2
end

""" 
    LowerBoundEstimator

Supertype for lower bounds estimator for the PVM step in the SOCGS algorithm.

All LowerBoundEstimators must implement `compute_pvm_threshold`.
"""
abstract type LowerBoundEstimator end
"""
    compute_pvm_threshold(lb_estimator::LowerBoundEstimator,...)
 
Returns the threshold ε used as a stopping threshold on the dual gap for the PVM step in SOCGS algortihm.
See https://arxiv.org/abs/2002.08907 for details.
"""
function compute_pvm_threshold end

""" 
    LowerBoundFiniteSteps

    Lower bound estimator for PVM threshold which performs a finite number `LowerBoundFiniteSteps.max_iter` of
    corrective steps `LowerBoundFiniteSteps.corrective_step` to estimate the PVM threshold.
"""
struct LowerBoundFiniteSteps{LS<:LineSearchMethod,R<:Real} <: LowerBoundEstimator 
    corrective_step::CorrectiveStep
    max_iter::Int
    line_search::LS
    min_threshold::R
end

LowerBoundFiniteSteps(f,grad!,
                    lmo,
                    corrective_step,
                    max_iter,
                    line_search
                    ) = LowerBoundFiniteSteps(f,grad!,lmo, corrective_step, max_iter,line_search, 1e-4)

""" 
    compute_pvm_threshold(lb_estimator::LowerBoundFiniteSteps,f,grad!,lmo,x,primal,gradient)

    Performs `lb_estimator.max_iter` number of corrective step `lb_estimator.corrective_step` 
    from the current iterate `x`.
    The found lower bound `f(x) - f(new_x)` is plugged in the formula for the PVM threshold ε found in https://arxiv.org/abs/2002.08907.
    If ε is too small, we return `lb_estimator.min_threshold` instead.
"""
function compute_pvm_threshold(lb_estimator::LowerBoundFiniteSteps,f,grad!,lmo,
                                x,
                                primal::Real,
                                gradient,
                                dual_gap
                                )
    _, _, primal_finite_steps, _, _, _ =  corrective_frank_wolfe(
                            f,
                            grad!,
                            lmo,
                            lb_estimator.corrective_step,
                            ActiveSet([(one(x[1]),x)]);
                            line_search = lb_estimator.line_search,
                            max_iteration= lb_estimator.max_iter -1 , #-1 because recompute_last_vertex = true
    )
    return max((primal - primal_finite_steps)^4/(fast_dot(gradient,gradient)^2),lb_estimator.min_threshold)
end


""" 
    LowerBoundLSmoothness

    Lower bound estimator for PVM threshold which uses a L-smoothness formula 
    to estimate the PVM threshold.
    The L-smoothness parameter must be provided in `LowerBoundLSmoothness.L`.
"""
struct LowerBoundLSmoothness{LS<:LineSearchMethod,R<:Real} <: LowerBoundEstimator 
    corrective_step::CorrectiveStep
    max_iter::Int
    line_search::LS
    min_threshold::R
    L::R
end

"""
    compute_pvm_threshold(lb_estimator::LowerBoundLSmoothness,f,grad!,lmo,x)

    Computes a lowerbound using a L-smoothness formula.
    To do so, a `lb_estimator.max_iter` number of `lb_estimator.corrective_step` are performed.
    See https://arxiv.org/abs/2311.05313 for smoothness formulae.
    If the threshold is too small, we return `lb_estimator.min_threshold` instead.
"""
function compute_pvm_threshold(lb_estimator::LowerBoundLSmoothness,f,grad!,lmo,
    x,
    primal::Real,
    gradient,
    dual_gap
    )

    function make_linear_search_stepsize_callback(traj_data::Vector)
        return function callback_with_trajectory(state, args...)
            if state.step_type !== FrankWolfe.ST_LAST || state.step_type !== FrankWolfe.ST_POSTPROCESS
                push!(traj_data, state.gamma)
            end
            return true
        end
    end
        gamma_traj = []    
        _, v, primal_finite_steps, dual_gap, _, _ =  corrective_frank_wolfe(
            f,
            grad!,
            lmo,
            lb_estimator.corrective_step,
            ActiveSet([(one(x[1]),x)]);
            line_search = lb_estimator.line_search,
            max_iteration= lb_estimator.max_iter -1 , #-1 because recompute_last_vertex = true
            callback = make_linear_search_stepsize_callback(gamma_traj)
    )
    L = lb_estimator.L
    gamma = gamma_traj[end]
    norm2_v_x = FrankWolfe.fast_dot(v,v) - 2.0 * FrankWolfe.fast_dot(v,x) + FrankWolfe.fast_dot(x,x)
    return max(gamma* dual_gap - 0.5*L * gamma^2 * norm2_v_x,lb_estimator.min_threshold)
end


""" 
    LowerBoundKnown
    
    Type used when the optimal value of the objective function is known.
    If so, the lower bound estimation is exact.
"""
struct LowerBoundKnown{R<:Real} <: LowerBoundEstimator 
    known_optimal_sol::R
    min_threshold::R
end
LowerBoundKnown(known_optimal_sol) = LowerBoundKnown(known_optimal_sol, 1e-7)

"""
    compute_pvm_threshold(lb_estimator::LowerBoundKnown, primal, gradient)
    Returns the PVM threshold ε from the formula for found in https://arxiv.org/abs/2002.08907, 
    using the exact lower bound given by `primal - lb_estimator.known_optimal_sol`.
    If the threshold ε is too small, we return `lb_estimator.min_threshold` instead.
"""
function compute_pvm_threshold(lb_estimator::LowerBoundKnown,f,grad!,lmo,
    x,
    primal::Real,
    gradient,
    dual_gap
    )
    return max( (primal - lb_estimator.known_optimal_sol)^4/(fast_dot(gradient,gradient)^2), lb_estimator.min_threshold)
end

""" 
    LowerBoundKnown
    
    Type used when a constant PVM threshold is used.
"""
struct LowerBoundConstant{R<:Real} <: LowerBoundEstimator 
    lowerbound_value::R
end 

""" 
    compute_pvm_threshold(lb_estimator::LowerBoundConstant)
    
    Returns the constant threshold `lb_estimator.lowerbound_value`.
"""
function compute_pvm_threshold(lb_estimator::LowerBoundConstant,f,grad!,lmo,
    x,
    primal::Real,
    gradient,
    dual_gap
    )
    return lb_estimator.lowerbound_value
end

""" 
    LowerBoundDualGapSquared

    Returns a PVM threshold ε as a quadratic function of the current dual gap.
"""
struct LowerBoundDualGapSquared{R<:Real} <: LowerBoundEstimator 
    factor::R
    default_threshold::R
end 

""" 
    compute_pvm_threshold(lb_estimator::LowerBoundDualGapSquared,dual_gap)

    Returns a PVM threshold ε given by `lb_estimator.factor* dual_gap^2`.
    If the threshold ε is too small, we return `lb_estimator.min_threshold` instead.
"""
function compute_pvm_threshold(lb_estimator::LowerBoundDualGapSquared,f,grad!,lmo,
    x,
    primal::Real,
    gradient,
    dual_gap
    )
    if dual_gap < Inf
        return lb_estimator.factor* dual_gap^2
    else
        return lb_estimator.default_threshold
    end
end

""" 
    LowerBoundDualGapLinear

    Returns a PVM threshold ε as a linear function of the current dual gap.
"""
struct LowerBoundDualGapLinear{R<:Real} <: LowerBoundEstimator 
    factor::R
    default_threshold::R
end 

""" 
    compute_pvm_threshold(lb_estimator::LowerBoundDualGapLinear,dual_gap)

    Returns a PVM threshold ε given by `lb_estimator.factor * dual_gap`.
    If the threshold ε is too small, we return `lb_estimator.min_threshold` instead.
"""
function compute_pvm_threshold(lb_estimator::LowerBoundDualGapLinear,f,grad!,lmo,
    x,
    primal::Real,
    gradient,
    dual_gap
    )
    if dual_gap < Inf
        return lb_estimator.factor * dual_gap
    else
        return lb_estimator.default_threshold
    end
end

###########################################################################

"""
    make_pvm_callback(callback, traj_data::Vector)
    
    Building simple callback function to gather run data from pvm steps.
"""
function make_pvm_callback(callback, traj_data::Vector)
    return function callback_with_trajectory(state, args...)
        if state.step_type !== FrankWolfe.ST_LAST || state.step_type !==  FrankWolfe.ST_POSTPROCESS
            push!(traj_data, (pvm_t = state.t, pvm_time = state.time ) )
        end
        if callback === nothing
            return true
        end
        return callback(state, args...)
    end
end


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
    line_search_fw::LineSearchMethod=Secant(),
    line_search_pvm::LineSearchMethod=Secant(),
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    verbose_pvm = false,
    traj_data=[],
    pvm_traj_data=[],
    timeout=Inf,
    pvm_max_iteration,
    #
    pvm_trajectory = false,
    lazy_pvm =false,
    #
    do_cgs= false
)


    active_set_fw = ActiveSet([(one(x0[1]),x0)])
    active_set_pvm = ActiveSet([(one(x0[1]),x0)]) 
    gradient= collect(x0)
    Hx = collect(x0)
    x_fw = get_active_set_iterate(active_set_fw)
    x_pvm = get_active_set_iterate(active_set_pvm)
    x = x_pvm    
    dual_gap_fw = Inf
    socgs_fw_gap = Inf
    primal_fw = Inf
    primal_pvm = Inf
    primal = f(x0)

    t = 0
    step_type = CGS_FW_STEP
    time_start = time_ns()
    tot_time = 0.0
    lmo_pvm = FrankWolfe.TrackingLMO(lmo_pvm) #wrapper to track lmos call in pvm

    if trajectory
        state = (
                t= t,
                primal = primal,
                dual_fw = Inf,
                dual_gap_fw = Inf,
                tot_time = tot_time,          
                x = x,
                epsilon = Inf,
                step_type = step_type,
                grad_time = 0.0,
                quad_time = 0.0,
                thresh_time = 0.0,
                pvm_t = 0,
                pvm_tot_time = 0.0,  
                primal_eval_time = 0.0,   
                fw_time = 0.0,         
                copyto_time = 0.0,
                traj_data_pvm = [],
                cumul_lmo_calls_pvm = 0
            )
        push!(
            traj_data,
            state
        )
    end


    function relative_gap_stop_condition(::Any,primal_value)
        return true
    end

    while t < max_iteration && tot_time ≤ timeout && relative_gap_stop_condition(lb_estimator,primal)
        #pvm_traj_data
        pvm_traj_data = []
        pvm_callback = make_pvm_callback(nothing, pvm_traj_data)

        #computing gradient 
        grad_time_start = time_ns()
        grad!(gradient, x)
        grad_time = (time_ns() - grad_time_start ) / 1e9

        #dual gap of fw
        v = FrankWolfe.compute_extreme_point(lmo_fw, gradient)
        socgs_fw_gap = dot(gradient,x-v)
        #building quadratic approximation (problem dependent) 
        build_quad_time_start = time_ns()

        #quadratic_term_function!, gradient_corrector!, Hx = build_quadratic_approximation!(Hx,x,gradient,primal)
        f_quad_approx, grad_quad_approx! = build_quadratic_approximation!(Hx,x,gradient,primal,t+1)
        build_quad_tot_time = (time_ns() - build_quad_time_start) / 1e9
              
        #compute threshold for pvm
        threshold_time_start = time_ns()
        epsilon = compute_pvm_threshold(lb_estimator,f,grad!,lmo_fw,x,primal,gradient,state.dual_gap_fw) 
        threshold_tot_time = (time_ns() - threshold_time_start) / 1e9
        #H-projection (pvm)
        #pvm
        x_pvm, _ , _,_, traj_data_pvm, _ = corrective_frank_wolfe(
                f_quad_approx,
                grad_quad_approx!,
                lmo_pvm,
                pvm_step,
                active_set_pvm;
                line_search=line_search_pvm,
                epsilon= epsilon, 
                callback = pvm_callback,
                trajectory= pvm_trajectory,
                verbose = verbose_pvm, 
                max_iteration = pvm_max_iteration ,
            )
        
        #primal evaluation
        primal_eval_time_start = time_ns()
        primal_pvm = f(x_pvm)
        primal_eval_time = (time_ns() - primal_eval_time_start) / 1e9

        #Fw corrective step
        fw_step_time_start = time_ns()
        x_fw, _ , primal_fw, dual_gap_fw , _=corrective_frank_wolfe(
                f,
                grad!,
                lmo_fw,
                fw_step,
                active_set_fw;
                line_search=line_search_fw,
                max_iteration=0, #0 because recompute_last_vertex = true
                gradient = gradient
            )
        fw_step_tot_time = (time_ns() - fw_step_time_start) / 1e9

        #copying active sets
        copyto_time_start = time_ns()
        if (!do_cgs) && primal_pvm > primal_fw
            copyto!(active_set_pvm,active_set_fw)
            primal = primal_fw
            x = x_fw
            step_type = CGS_FW_STEP
        else
            primal = primal_pvm
            x = x_pvm
            step_type = CGS_PVM_STEP
        end
        copyto_tot_time = (time_ns() - copyto_time_start) / 1e9

        tot_time = (time_ns() - time_start) / 1e9
        state = (
                t= t,
                primal = primal,
                dual_fw = primal_fw - socgs_fw_gap,
                dual_gap_fw = socgs_fw_gap,
                tot_time = tot_time,          
                x = x,
                epsilon = epsilon,
                step_type = step_type,
                grad_time = grad_time,
                quad_time = build_quad_tot_time,
                thresh_time = threshold_tot_time,
                pvm_t = pvm_traj_data[end].pvm_t,
                pvm_tot_time = pvm_traj_data[end].pvm_time,  
                primal_eval_time = primal_eval_time,   
                fw_time = fw_step_tot_time,         
                copyto_time = copyto_tot_time,
                traj_data_pvm = traj_data_pvm,
                cumul_lmo_calls_pvm = lmo_pvm.counter
            )
        t += 1
        if trajectory
            push!(
                traj_data,
                state
            )
        end

        if verbose && mod(t, print_iter) == 0
            rel_gap_string =""
            if isa(lb_estimator,LowerBoundKnown)
                rel_gap = abs(primal - lb_estimator.known_optimal_sol )/ (1e-8 + abs(lb_estimator.known_optimal_sol))
                rel_gap_string = " Rel.Gap "*string(rel_gap)
            end

            println("It. ", t, " Tot.Time ", tot_time," ", step_type,rel_gap_string)
        end

        
    end

    if verbose && timeout < Inf && tot_time ≥ timeout
        @info "Time limit reached"
    end

    if verbose && t ≥ max_iteration
        @info "Iteration limit reached"
    end


    return (x=x, primal=primal, dual_gap_fw = socgs_fw_gap,
             t = t, tot_time = tot_time,traj_data=traj_data)
end