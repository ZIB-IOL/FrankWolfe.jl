struct QCMNPStep <: CorrectiveStep 
    A::H # Hessian matrix
    b::BT # linear term
end

function prepare_corrective_step(
    corrective_step::QCMNPStep,
    f,
    grad!,
    gradient,
    active_set,
    t,
    lmo,
    primal,
    phi_value,
)
    return false
end

function run_corrective_step(
    corrective_step::QCMNPStep,
    f,
    grad!,
    gradient,
    x,
    v,
    dual_gap,
    active_set,
    t,
    lmo,
    line_search,
    linesearch_workspace,
    primal,
    phi_value,
    tot_time,
    callback,
    renorm_interval,
    memory_mode,
    epsilon,
    d,
)

    





    return x, v, phi_value, dual_gap, should_fw_step, should_continue
end

function _truncate_weights(weights::Vector{R}, old_weights::Vector{R}) where {R}

    indices_to_remove = Int[]

    if all(>=(-10eps()), new_weights)
        return indices_to_remove, weights 
    end

    # ratio test - identify which coordinate hit zero first
    tau_min = 1.0
    set_indices_zero = BitSet()
    for idx in eachindex(λ)
        if weights[idx] < old_weights[idx]
            tau = old_weights[idx] / (old_weights[idx] - weights[idx])
            if abs(tau - tau_min) ≤ 2weight_purge_threshold_default(typeof(tau))
                push!(set_indices_zero, idx)
            elseif tau < tau_min
                tau_min = tau
                empty!(set_indices_zero)
                push!(set_indices_zero, idx)
            end
        end
    end
    @assert length(set_indices_zero) >= 1
    weights = (1-tau_min) * old_weights + tau_min * weights
    weights[set_indices_zero] .= 0
    @assert all(>=(-2weight_purge_threshold_default(eltype(weights))), weights) "All weights must be between nonnegative: $(minimum(weights))"
    @assert isapprox(sum(weights), 1.0) "The sum of weights must be approximately 1"
    indices_to_remove = Int[]
    new_weights = R[]
    for idx in eachindex(λ)
        weight_value = weights[idx] # using new lambdas
        if weight_value <= eps()
            push!(indices_to_remove, idx)
        else
            push!(new_weights, weight_value)
        end
    end

    return indices_to_remove, new_weights
end
