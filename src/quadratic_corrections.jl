
"""
    ScheduledStep(base_step, special_step, scheduler=LogScheduler(start_time=2, scaling_factor=2.0, max_interval=1000); lazy=true, lazy_tolerance=2.0)

Hybrid corrective step used by [`corrective_frank_wolfe`](@ref): run `base_step`
by default and replace it by `special_step` according to `scheduler`.

The schedule counts **new atoms** via the increase in active-set size since the
previous iteration. The counter is updated every iteration, including Frank-Wolfe
steps, but the scheduler is queried only when a corrective step is due.

`scheduler` may be a [`LogScheduler`](@ref) or a callable
`(atom_counter, t, active_set) -> Bool`.
"""
struct ScheduledStep{S<:FrankWolfe.CorrectiveStep,T<:FrankWolfe.CorrectiveStep,SF} <: FrankWolfe.CorrectiveStep
    base_step::S
    special_step::T
    scheduler::SF
    lazy::Bool
    lazy_tolerance::Float64
    atom_counter::Base.RefValue{Int}
    prev_length::Base.RefValue{Int}
end

make_default_scheduler(start_time::Int, scaling_factor::Float64, max_interval::Int) =
    LogScheduler(; start_time=start_time, scaling_factor=scaling_factor, max_interval=max_interval)

function ScheduledStep(
    base_step::S,
    special_step::T,
    scheduler=make_default_scheduler(2, 2.0, 1000);
    lazy::Bool=true,
    lazy_tolerance::Float64=2.0,
) where {S<:FrankWolfe.CorrectiveStep,T<:FrankWolfe.CorrectiveStep}
    return ScheduledStep{S,T,typeof(scheduler)}(
        base_step,
        special_step,
        scheduler,
        lazy,
        lazy_tolerance,
        Ref(0),
        Ref(-1),
    )
end

function should_run_scheduled_step(scheduler::LogScheduler, atom_counter, t, active_set)
    if atom_counter - scheduler.last_solve_counter[] >= scheduler.current_interval[]
        scheduler.last_solve_counter[] = atom_counter
        scheduler.current_interval[] = min(
            round(Int, scheduler.scaling_factor * scheduler.current_interval[]),
            scheduler.max_interval,
        )
        return true
    end
    return false
end

function should_run_scheduled_step(scheduler, atom_counter, t, active_set)
    return scheduler(atom_counter, t, active_set)
end

function _update_scheduled_atom_counter!(step::ScheduledStep, active_set)
    n = length(active_set)
    prev = step.prev_length[]

    # initialize the prev counter to the initial active set size without increasing the atom counter of the scheduler
    if prev < 0
        step.prev_length[] = n
        return
    end

    if n > prev
        step.atom_counter[] += n - prev
    end
    step.prev_length[] = n
    return
end

function FrankWolfe.prepare_corrective_step(
    step::ScheduledStep,
    f,
    grad!,
    gradient,
    active_set,
    t,
    lmo,
    primal,
    phi_value,
)
    # update the atom counter to count the new atoms added since the last iteration
    _update_scheduled_atom_counter!(step, active_set)
    return !step.lazy
end

function FrankWolfe.run_corrective_step(
    step::ScheduledStep,
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

    _, v_local, v_loc, _, a_lambda, a, a_loc, _, _ = FrankWolfe.active_set_argminmax(active_set, gradient)
    grad_dot_x = dot(gradient, x)
    grad_dot_a = dot(gradient, a)
    grad_dot_local_fw_vertex = dot(gradient, v_local)
    local_gap = grad_dot_a - grad_dot_local_fw_vertex

    if local_gap >= max(phi_value / step.lazy_tolerance, epsilon)

        if should_run_scheduled_step(step.scheduler, step.atom_counter[], t, active_set)
            old_len = length(active_set)
            old_weights = hasproperty(active_set, :weights) ? copy(active_set.weights) : nothing
            x_s, v_s, phi_s, gap_s, _should_fw_step, should_continue = FrankWolfe.run_corrective_step(
                step.special_step,
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

            success = (length(active_set) != old_len) ||
                (old_weights !== nothing &&
                length(active_set.weights) == length(old_weights) &&
                !all(active_set.weights .== old_weights))

            if success # Special step was successful, return
                return x_s, v_s, phi_s, gap_s, false, should_continue
            end
        end

        # Special step was unsuccessful, perform fallback step
        return FrankWolfe.run_corrective_step(
            step.base_step,
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
    else
        if step.lazy
            v = FrankWolfe.compute_extreme_point(lmo, gradient)
            dual_gap = grad_dot_x - dot(gradient, v)
        end
        if dual_gap ≥ max(epsilon, phi_value / step.lazy_tolerance)
            should_fw_step = true
        else
            should_fw_step = false
            phi_value = min(dual_gap, phi_value / 2)
        end
        return x, v, phi_value, dual_gap, should_fw_step, true
    end
end

"""
    QuadraticLSCorrection(A, b, mnp=true)

Quadratic correction step used by [`corrective_frank_wolfe`](@ref)
implementings the linear-system-based variant from Halbey, Rakotomandimby,
Besançon, Designolle, Pokutta (2025),
[Efficient Quadratic Corrections for Frank-Wolfe Algorithms](https://arxiv.org/abs/2506.02635),
Algorithm 6 (**QC-MNP**).
This method solves the affine-minimization linear system over the
current active set `S`. For a convex quadratic
``f(x) = \\frac12 \\langle x, A x \\rangle + \\langle b, x \\rangle + c``,
the affine minimizer over ``\\operatorname{aff}(S)`` is obtained from the
symmetric reduced system (Remark 3 / Eq. 10 in the paper)
``W^\\top A W \\mu = -W^\\top (A w + b)``.

- If `mnp=false`, the affine minimizer is accepted only when all barycentric
  coordinates are nonnegative (a fully-corrective step), otherwise the correction
  fails.
- If `mnp=true`, a single Wolfe ratio test pulls the affine minimizer back onto
  ``\\operatorname{conv}(S)`` and drops at least one atom.

# Arguments
- `A`: Hessian (or linear operator) of the quadratic term.
- `b`: Linear term, so that ``\\nabla f(x) = A x + b``.
- `mnp`: whether to apply the Minimum-Norm Point ratio test.

The optional field `ls_solve` is an in-place solver `(x, M, rhs; kwargs...) -> Bool`
for the reduced system of size `|S|-1`. The default uses `M \\ rhs`.
"""

struct QuadraticLSCorrection{H,BT} <: CorrectiveStep
    A::H # Hessian matrix
    b::BT # linear term
    ls_solve::Function
    mnp::Bool
end

function QuadraticLSCorrection(A::H, b::BT, mnp::Bool=true) where {H,BT}
    function ls_solve(x, M, rhs; kwargs...)
        x .= M \ rhs
        return true
    end
    return QuadraticLSCorrection{H,BT}(A, b, ls_solve, mnp)
end

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

    # Affine minimizer over aff(S) via the symmetric reduced system (paper Remark 3 / eq. 10):
    #   Wᵀ A W μ = -Wᵀ (A w + b)
    # W has columns vᵢ - w with w = atoms[1], μ has length |S|-1.
    # Then λ_w = 1 - 1ᵀμ and λ_{vᵢ} = μᵢ.

    nv = length(active_set)

    if nv <= 1
        return x, v, phi_value, dual_gap, false, true
    end

    nμ = nv - 1
    A_mat = Matrix{Float64}(undef, nμ, nμ)
    r_vec = Vector{Float64}(undef, nμ)
    μ_red = Vector{Float64}(undef, nμ)

    if active_set isa ActiveSetQuadraticProductCaching
        dA11 = active_set.dots_A[1][1]
        db1 = active_set.dots_b[1]
        for i in 2:nv
            ii = i - 1
            for j in 2:i
                jj = j - 1
                # (vᵢ - w)ᵀ A (vⱼ - w)
                val =
                    active_set.dots_A[i][j] - active_set.dots_A[j][1] - active_set.dots_A[i][1] +
                        dA11
                A_mat[ii, jj] = val
                A_mat[jj, ii] = val
            end
            r_vec[ii] = -active_set.dots_A[i][1] + dA11 - active_set.dots_b[i] + db1
        end
    else
        A = corrective_step.A
        b = corrective_step.b
        w = active_set.atoms[1]
        d1 = A * w
        dw_storage = similar(w)
        for i in 2:nv
            ii = i - 1
            di = mul!(dw_storage, A, active_set.atoms[i])
            di .-= d1
            for j in 2:nv
                A_mat[ii, j-1] = dot(di, active_set.atoms[j]) - dot(di, w)
            end
            r_vec[ii] = -dot(di, w) - dot(b, active_set.atoms[i]) + dot(b, w)
        end
    end

    # μ_red are barycentric coordinates of all atoms except the anchor w
    converged = corrective_step.ls_solve(μ_red, Symmetric(A_mat), r_vec; active_set=active_set)

    if converged
        # Compute barycentric coordinates from reduced coordinates µ_red
        μ = Vector{Float64}(undef, nv)
        μ[2:nv] .= μ_red
        μ[1] = 1 - sum(μ_red)

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
    for idx in set_indices_zero
        weights[idx] = 0
    end
    @assert all(>=(-2weight_purge_threshold_default(eltype(weights))), weights) "All weights must be between nonnegative: $(minimum(weights))"
    @assert isapprox(sum(weights), 1.0) "The sum of weights must be approximately 1"
    return _purge_weights(weights)
end



"""
    QuadraticLPCorrection(A, b, optimizer, mnp=false)

Quadratic correction step used by [`corrective_frank_wolfe`](@ref)
implementings the LP-based variant from Halbey, Rakotomandimby,
Besançon, Designolle, Pokutta (2025),
[Efficient Quadratic Corrections for Frank-Wolfe Algorithms](https://arxiv.org/abs/2506.02635).
This method encodes affine minimality over the current active set
as linear equalities and solves the resulting LP.

- If `mnp=false`, this is **QC-LP** (Algorithm 5): the affine-minimality
  equalities together with ``\\lambda \\ge 0`` and ``\\sum \\lambda = 1``.
  The LP is feasible only when an affine minimizer lies in ``\\operatorname{conv}(S)``.
  Otherwise the correction fails.
  Unlike [`QuadraticLSCorrection`](@ref) with `mnp=false`, this method is guaranteed to select an affine minimizer inside the convex hull, if one exists.
- If `mnp=true`, this is the LP form of **QC-MNP** (Algorithm 8): minimize
  ``\\beta \\ge 0`` subject to ``\\lambda + \\beta \\lambda(x) \\ge 0`` and the
  same affine-minimality equalities. Unlike [`QuadraticLSCorrection`](@ref),
  this selects the affine minimizer that allows the largest feasible step when
  the affine minimizer is not unique.

# Arguments
- `A`: Hessian (or linear operator) of the quadratic term.
- `b`: Linear term, so that ``\\nabla f(x) = A x + b``.
- `mnp`: whether to use the QC-MNP LP instead of QC-LP.
- `optimizer`: a MathOptInterface optimizer used to solve the LP.
"""
struct QuadraticLPCorrection{H,LT,OT<:MOI.AbstractOptimizer} <: CorrectiveStep
    A::H # Hessian matrix
    b::LT # linear term
    optimizer::OT
    mnp::Bool
end

function QuadraticLPCorrection(A::H, b::LT) where {H,LT}
    optimizer = MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true))
    return QuadraticLPCorrection{H,LT,typeof(optimizer)}(A, b, optimizer, false)
end

function QuadraticLPCorrection(A::H, b::LT, optimizer::OT) where {H,LT,OT<:MOI.AbstractOptimizer}
    return QuadraticLPCorrection{H,LT,OT}(A, b, optimizer, false)
end

function QuadraticLPCorrection(
    A::H,
    b::LT,
    mnp::Bool,
    optimizer::MOI.AbstractOptimizer=MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
) where {H,LT}
    return QuadraticLPCorrection(A, b, optimizer, mnp)
end

function prepare_corrective_step(
    corrective_step::QuadraticLPCorrection{H,LT,OT},
    f,
    grad!,
    gradient,
    active_set,
    t,
    lmo,
    primal,
    phi_value,
) where {H,LT,OT<:MOI.AbstractOptimizer}
    return false
end

function run_corrective_step(
    step::QuadraticLPCorrection{H,LT,OT},
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
) where {H,LT,OT<:MOI.AbstractOptimizer}

    nv = length(active_set)
    o = step.optimizer
    MOI.empty!(o)
    λ = MOI.add_variables(o, nv)
    sum_of_variables = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, λ), 0.0)
    MOI.add_constraint(o, sum_of_variables, MOI.EqualTo(1.0))

    if step.mnp
        β = MOI.add_variable(o)
        MOI.add_constraint(o, β, MOI.GreaterThan(0.0))
        for j = 1:nv
            MOI.add_constraint(o, MOI.ScalarAffineFunction{Float64}([MOI.ScalarAffineTerm(1, λ[j]), MOI.ScalarAffineTerm(active_set.weights[j], β)], 0.0), MOI.GreaterThan(0.0))
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
            for j in (i+1):nv
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
                    _compute_quadratic_constraint(
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
    λ_vals = MOI.get.(o, MOI.VariablePrimal(), λ)
    if step.mnp
        β_val = MOI.get(o, MOI.VariablePrimal(), β)
        if β_val > 10 * eps(typeof(β_val))
            τ = 1 / (β_val + 1)
            λ_vals = τ .* λ_vals .+ (1 - τ) .* active_set.weights
        end
    end
    indices_to_remove, new_weights = _purge_weights(λ_vals)

    # Update active set
    deleteat!(active_set, indices_to_remove)
    @assert length(active_set) == length(new_weights)
    update_weights!(active_set, new_weights)
    active_set_cleanup!(active_set)
    active_set_renormalize!(active_set)
    x = compute_active_set_iterate!(active_set)

    return x, v, phi_value, dual_gap, true, true
end

function _compute_quadratic_constraint(atom1, atom0, A::AbstractMatrix, atom2, λ)
    return MOI.ScalarAffineTerm(fast_dot(atom1, A, atom2) - fast_dot(atom0, A, atom2), λ)
end

function _compute_quadratic_constraint(
    atom1,
    atom0,
    A::Union{Identity,LinearAlgebra.UniformScaling},
    atom2,
    λ,
)
    return MOI.ScalarAffineTerm(A.λ * (dot(atom1, atom2) - dot(atom0, atom2)), λ)
end


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