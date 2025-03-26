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


"""
Supertype for condiditional gradient oracles.

All CndGOs must implement `conditional_gradient_descent(cndgrado::CndGO; ...; state::CndGState )`
and return a vector `ut` and update the value of `state::`.
"""

abstract type CndGO end

function conditional_gradient_descent end

@enum CndGState begin
    CndGS_UNSOLVED = 1
    CndGS_SOLVED = 2
    CndGS_TIMELIMIT = 3
    CndGS_ITERLIMIT = 4
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
    
function conditional_gradient_descent(cndgrado::LanZhouProcedure, 
                                    gradient,
                                    x,
                                    beta,
                                    eta; 
                                    state 
                                )

    function value_function(g,u,β,ut,vt)
        return sum( (g + β * (ut - u)) .* (ut - vt) )
    end

    t = 0
    time_start = time_ns()
    tot_time = time_start
    ut = copy(x)

    while t< cndgrado.max_iteration && tot_time < cndgrado.timeout
        time_at_loop = time_ns()
        tot_time = (time_at_loop - time_start) / 1e9
        t +=1

        if t == 1
            vt = compute_extreme_point(lmo, gradient + beta * (ut - x)) 
        else
            vt = compute_extreme_point(lmo, gradient + beta * (ut - x), v=vt)
        end
        V = value_function(gradient,x,beta,ut,vt)
        if V <= eta break end
        alpha = V/(beta * sum((vt - ut).^2))
        alpha = min(1,alpha)
        ut .= (1-alpha)* ut + alpha * vt 
    end


    if tot_time >= cndgrado.timeout
        state = CndGS_TIMELIMIT 
        return ut
    end
    
    if t >= cndgrado.max_iteration
        state = CndGS_ITERLIMIT
        return ut
    end

    state = CndGS_SOLVED 
    return ut

end


"""
Supertype for stop conditions of Gradient Sliding method. 
    All GSStopCondition must implement `stop_condition(stop_rule::IterStop,...)`
    and return a boolean value which is true when the stop condition is satisfied.
"""
abstract type GSStopCondition end

function stop_condition end

"""
    IterStop<: GSStopCondition
    Stop condition based on a maximum number of iteration.
"""
struct IterStop<: GSStopCondition 
    maxIter<:Int
end

function stop_condition(stop_rule::IterStop,
                        t
                    )
        return t > stop_rule.maxIter
end


"""
Supertype for stepsize gamma.
    All MomentumStepsize must implement `compute_momentum_stepsize(stepsize_rule::ConstantMStepsize,...)`
    and return a stepsize.
    See Algorithm 1 in https://doi.org/10.1137/140992382.
"""
abstract type MomentumStepsize end

function compute_momentum_stepsize end

"""
    ConstantMStepsize <: MomentumStepsize 
    Constant stepsize gamma for momentum update.
"""

struct ConstantMStepsize <: MomentumStepsize 
    stepsize<:Real
end

function compute_momentum_stepsize(stepsize_rule::ConstantMStepsize,
                                t
                                )
    return stepsize_rule.stepsize
end

"""
    FixedMStepsize <: MomentumStepsize
    Fixed trajectory only depending on the iteration t of stepsizes gamma for momentum update.
"""
struct FixedMStepsize <: MomentumStepsize 
    stepsize_trajectory
end

function compute_momentum_stepsize(stepsize_rule::FixedMStepsize,
                                t
                                )                                
    return stepsize_rule.stepsize_trajectory(t)
end

"""
Supertype for parameters eta and beta of the condiditional gradient descent.
    All MomentumStepsize must implement `compute_CnGD_parameters(parameters_rule::ConstantCnGDParameters,...)`
    and return a threshold and a regularization parameters.
    See Algorithm 1 in https://doi.org/10.1137/140992382.
"""

abstract type CnGDParameters end

function compute_CnGD_parameters end

"""
    ConstantCnGDParameters <:CnGDParameters 
    Constant parameters eta and beta for condiditional gradient step.
"""

struct ConstantCnGDParameters <:CnGDParameters 
    threshold<:Real
    regularization<:Real
end

function compute_CnGD_parameters(parameters_rule::ConstantCnGDParameters,
                                t
                                )
    return parameters_rule.threshold, parameters_rule.regularization
end

"""
    FixedCnGDParameters <:CnGDParameters  
    Fixed trajectory only depending on the iteration t of parameters eta and beta for condiditional gradient step.
"""
struct FixedCnGDParameters <:CnGDParameters 
    threshold_trajectory
    regularization_trajectory
end

function compute_CnGD_parameters(stepsize_rule::FixedCnGDParameters,
                                t
                                )                                
    return parameters_rule.threshold_trajectory(t), parameters_rule.regularization_trajectory(t)
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
    momentum_stepsize::MomentumStepsize,
    cndgrad_params::CnGDParameters ,
    stop_rule::GSStopCondition,
    x0;
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    traj_data=[],
    timeout=Inf
)

    time_start = time_ns()
    tot_time = time_start
    
    x = copy(x0)
    y = copy(x0)
    z = copy(x0)
    gradient = copy(x0)
    gamma = zero(x0[1])
    eta = zero(x0[1])
    beta = zero(x0[1])

    t = 0
    gamma = compute_momentum_stepsize(momentum_stepsize,t)
    cndG_state = nothing

    while t ≤ max_iteration && (tot_time - time_start) < timeout && !stop_condition(stop_rule,t)

        time_at_loop = time_ns()
        tot_time = (time_at_loop - time_start) / 1e9
        t += 1
        cndG_state = CndGS_UNSOLVED 
        
        z .= (1-gamma) * y + gamma * x 
        grad!(gradient,z)        
        eta, beta =  compute_CnGD_parameters(cndgrad_params,t)        
        x .= conditional_gradient_descent(cndgrado,
                                            gradient,
                                            x,
                                            beta,
                                            eta; 
                                            state = cndG_state)
        y .= (1-gamma) * y + gamma * x         

        gamma = compute_momentum_stepsize(momentum_stepsize,t)

        if trajectory
            push!(traj_data,
                    (
                        t=t,
                        tot_time=tot_time,
                        x=x,
                        y=y,
                        z=z,
                        gamma=gamma,
                        eta=eta,
                        beta=beta,
                        gradient=gradient,
                        cndG_state=cndG_state
                    )
            )
        end
    
        if verbose
           if  mod(state.t, print_iter) == 0
                print("It. ", t," Tot.Time ",tot_time, " CndG state ", cndG_state)
           end
        end
    end

    if timeout < Inf
        if tot_time ≥ timeout
            if verbose
                @info "Time limit reached"
            end
        end
    end
    
    if t ≥  max_iteration 
        if verbose
            @info "Iteration limit reached"
        end
    end
    


return (x=x, y=y,z=z, t=t, tot_time = tot_time, traj_data=traj_data)
end