"""
    QCMNPStep{H, BT}

A corrective step structure for the Quadratic Correction Minimum Norm Problem (QCMNP).

# Fields
- `A::H`: Hessian matrix or operator representing the quadratic term.
- `b::BT`: Linear term in the objective.
- `ls_solve::Function`: In-place solver for the linear system (x, A_mat, b).

This type is used to encapsulate the data and solver required for performing minimum-norm projections or corrective steps when working with quadratic correction methods.
"""


struct QuadraticLSCorrection{H, BT} <: CorrectiveStep 
    A::H # Hessian matrix
    b::BT # linear term
    ls_solve::Function
    mnp::Bool
end

function QuadraticLSCorrection(A::H, b::BT) where {H, BT}
    function ls_solve(x, M, rhs; kwargs...)
        x .= M \ rhs
        return true
    end
    return QuadraticLSCorrection{H, BT}(A, b, ls_solve, false)
end

function QuadraticLSCorrection(A::H, b::BT, mnp::Bool) where {H, BT}
    function ls_solve(x, M, rhs; kwargs...)
        x .= M \ rhs
        return true
    end
    return QuadraticLSCorrection{H, BT}(A, b, ls_solve, mnp)
end

# Note: The 4-argument constructor is automatically provided by the struct definition
# No need to explicitly define it here to avoid method overwriting during precompilation


function prepare_corrective_step(
    corrective_step::QuadraticLSCorrection,
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
    corrective_step::QuadraticLSCorrection,
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

    # Computes the minimizer for a quadratic function f(x) = xᵗAx + bᵗx over the affine hull over the atoms of a given active set
    # by solving the non-symmetric linear system:
    # Wᵗ A V λ == -Wᵗ b
    # V has columns vi (atoms of the active set)
    # W has columns vi - v1

    nv = length(active_set)

    # Pre-allocate arrays
    A_mat = Matrix{Float64}(undef, nv, nv)
    r_vec = Vector{Float64}(undef, nv)
    μ = Vector{Float64}(undef, nv)

    A_mat[1,:] .= 1.0
    r_vec[1] = 1.0
    if active_set isa ActiveSetQuadraticProductCaching
        # dots_A is a lower triangular matrix

        d1 = active_set.dots_A[1]
        for i in 2:nv
            di = active_set.dots_A[i]
            for j in 1:i
                val = di[j] - d1[j]
                A_mat[i, j] = val
                if i != j && j != 1
                    A_mat[j, i] = val
                end
            end
            r_vec[i] = active_set.dots_b[1] - active_set.dots_b[i]
        end
    else
        temp1 = similar(active_set.atoms[1])
        d1 = A * active_set.atoms[1]
        for i in 2:nv
            di = mul!(temp1, A, active_set.atoms[i])
            for j in 1:i
                val = dot(di, active_set.atoms[j]) - dot(d1, active_set.atoms[j])
                A_mat[i, j] = val
                if i != j && j != 1
                    A_mat[j, i] = val
                end
            end
            r_vec[i] = dot(b, active_set.atoms[1]) - dot(b, active_set.atoms[i])
        end
    end

    # Solve system - μ are the bary-centric coordinates of the/an affine minizer
    converged = corrective_step.ls_solve(μ, A_mat, r_vec; active_set=active_set)

    if converged

        # Perform pullback to the convex hull with a ratio test (minimum-norm point)
        if corrective_step.mnp
            indices_to_remove, new_weights = _truncate_weights(μ, active_set.weights)
        else
            if all(>=(-10eps()), μ)
                indices_to_remove = Int[]
                new_weights = μ
            else
                return x, v, phi_value, dual_gap, false, true
            end
        end

        # update active set
        deleteat!(active_set, indices_to_remove)
        @assert length(active_set) == length(new_weights)
        update_weights!(active_set, new_weights)
        active_set_cleanup!(active_set)
        active_set_renormalize!(active_set)
        x = compute_active_set_iterate!(active_set)
    end

    return x, v, phi_value, dual_gap, false, true
end

function _truncate_weights(weights::Vector{R}, old_weights::Vector{R}) where {R}

    indices_to_remove = Int[]

    if all(>=(-10eps()), weights)
        return indices_to_remove, weights 
    end

    # ratio test - identify which coordinate hit zero first
    tau_min = 1.0
    set_indices_zero = BitSet()
    for idx in eachindex(weights)
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
    return _purge_weights(weights)
end



"""
    QCLPStep (Quadratic corrections LP)
    This step attempts to find the optimal weights (for the current active set) for a quadratic objective with hessian ´A´ and linear term ´b´ through linear programming.
    The LP is infeasible, if no minimizer over the affine hull of the atoms lie in the convex hull. In this case the method returns the current iterate unchanged.
    If ´relax´ is true, we relax the non-negativity constraint and use the MNP approach as in QC-MNP.
    The main difference to QCMNPStep is that it is still LP-based, but this allows to choose the best affine minimizer (in case it is not unique).
"""
struct QuadraticLPCorrection{H, LT, OT<:MOI.AbstractOptimizer} <: CorrectiveStep
    A::H # Hessian matrix
    b::LT # linear term
    optimizer::OT
    mnp::Bool
end

function QuadraticLPCorrection(A::H, b::LT) where {H, LT}
    optimizer = MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true))
    return QuadraticLPCorrection{H, LT, typeof(optimizer)}(A, b, optimizer, false)
end

function QuadraticLPCorrection(A::H, b::LT, optimizer::OT) where {H, LT, OT<:MOI.AbstractOptimizer}
    return QuadraticLPCorrection{H, LT, OT}(A, b, optimizer, false)
end

function QuadraticLPCorrection(A::H, b::LT, mnp::Bool) where {H, LT}
    optimizer = MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true))
    return QuadraticLPCorrection{H, LT, typeof(optimizer)}(A, b, optimizer, mnp)
end

# Note: The 4-argument constructor is automatically provided by the struct definition
# No need to explicitly define it here to avoid method overwriting during precompilation

function prepare_corrective_step(
    corrective_step::QuadraticLPCorrection{H, LT, OT},
    f,
    grad!,
    gradient,
    active_set,
    t,
    lmo,
    primal,
    phi_value,
) where {H, LT, OT<:MOI.AbstractOptimizer}
    return false
end

function run_corrective_step(
    step::QuadraticLPCorrection{H, LT, OT},
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
) where {H, LT, OT<:MOI.AbstractOptimizer}

    nv = length(active_set)
    o = step.optimizer
    MOI.empty!(o)
    λ = MOI.add_variables(o, nv)
    sum_of_variables = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, λ), 0.0)
    MOI.add_constraint(o, sum_of_variables, MOI.EqualTo(1.0))

    if step.mnp
        β = MOI.add_variable(o)
        for j = 1:nv
            MOI.add_constraint(o,MOI.ScalarAffineFunction{Float64}([MOI.ScalarAffineTerm(1, λ[j]), MOI.ScalarAffineTerm(active_set.weights[j], β)], 0.0), MOI.GreaterThan(0.0)) 
        end
    else
        MOI.add_constraint.(o, λ, MOI.GreaterThan(0.0))
    end

    # Get scaling factor for active set partial caching
    if active_set isa ActiveSetQuadraticPartialCaching
        c = active_set.λ[]
    else
        c = 1.0
    end

    # Wᵗ A V λ == -Wᵗ b
    # V has columns vi
    # W has columns vi - v1
    for i in 2:nv
        lhs = MOI.ScalarAffineFunction{Float64}([], 0.0)
        Base.sizehint!(lhs.terms, nv)
        if active_set isa
        Union{ActiveSetQuadraticProductCaching,ActiveSetQuadraticPartialCaching}
            # dots_A is a lower triangular matrix
            for j in 1:i
                push!(
                    lhs.terms,
                    MOI.ScalarAffineTerm(
                        c * (active_set.dots_A[i][j] - active_set.dots_A[j][1]),
                        λ[j],
                    ),
                )
            end
            for j in i+1:nv
                push!(
                    lhs.terms,
                    MOI.ScalarAffineTerm(
                        c * (active_set.dots_A[j][i] - active_set.dots_A[j][1]),
                        λ[j],
                    ),
                )
            end
            if active_set isa ActiveSetQuadraticProductCaching
                rhs = active_set.dots_b[1] - active_set.dots_b[i]
            else
                # ActiveSetQuadraticPartialCaching doesn't have a b field, use step.b
                rhs = dot(active_set.atoms[1], step.b) - dot(active_set.atoms[i], step.b)
            end
        else
            # replaces direct sum because of MOI and MutableArithmetic slow sums
            for j in 1:nv
                push!(
                    lhs.terms,
                    _compute_quadratic_constraint_term(
                        active_set.atoms[i],
                        active_set.atoms[1],
                        step.A,
                        active_set.atoms[j],
                        λ[j],
                    ),
                )
            end
            rhs = dot(active_set.atoms[1], step.b) - dot(active_set.atoms[i], step.b)
        end
        MOI.add_constraint(o, lhs, MOI.EqualTo{Float64}(rhs))
    end

    if step.mnp
        MOI.set(
           o,
           MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
           MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], β), 0.0),
       )
    else
        MOI.set(o, MOI.ObjectiveFunction{typeof(sum_of_variables)}(), sum_of_variables)
    end
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(o)
    if MOI.get(o, MOI.TerminationStatus()) ∉ (MOI.OPTIMAL, MOI.FEASIBLE_POINT, MOI.ALMOST_OPTIMAL)
        return x, v, phi_value, dual_gap, false, true
    end

    # Compute new weights and which atoms to drop
    indices_to_remove, new_weights = _purge_weights(MOI.get.(o, MOI.VariablePrimal(), λ))
    # Update active set
    deleteat!(active_set, indices_to_remove)
    @assert length(active_set) == length(new_weights)
    update_weights!(active_set, new_weights)
    active_set_cleanup!(active_set)
    active_set_renormalize!(active_set)
    x = compute_active_set_iterate!(active_set)

    return x, v, phi_value, dual_gap, true, true
end

#### Helper function that are already contained in the active set module

# function _compute_quadratic_constraint_term(atom1, atom0, A::AbstractMatrix, atom2, λ)
#     return MOI.ScalarAffineTerm(fast_dot(atom1, A, atom2) - fast_dot(atom0, A, atom2), λ)
# end

# function _compute_quadratic_constraint_term(
#     atom1,
#     atom0,
#     A::Union{Identity,LinearAlgebra.UniformScaling},
#     atom2,
#     λ,
# )
#     return MOI.ScalarAffineTerm(A.λ * (dot(atom1, atom2) - dot(atom0, atom2)), λ)
# end


function _purge_weights(weights::AbstractArray{R}) where {R}
    indices_to_remove = BitSet()
    new_weights = R[]
    eps = 2 * weight_purge_threshold_default(R)
    for (idx, weight) in enumerate(weights)
        if weight <= eps
            push!(indices_to_remove, idx)
        else
            push!(new_weights, weight)
        end
    end
    return indices_to_remove, new_weights
end