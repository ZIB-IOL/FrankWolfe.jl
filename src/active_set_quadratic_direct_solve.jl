abstract type AffineMinSolver end

function compute_affine_min end

"""
    ActiveSetQuadraticLinearSolve{AT, R, IT}

Represents an active set of extreme vertices collected in a FW algorithm,
along with their coefficients `(λ_i, a_i)`.
`R` is the type of the `λ_i`, `AT` is the type of the atoms `a_i`.
The iterate `x = ∑λ_i a_i` is stored in x with type `IT`.
The objective function is assumed to be of the form `f(x)=½⟨x,Ax⟩+⟨b,x⟩+c`
so that the gradient is `∇f(x)=Ax+b`.

This active set stores an inner `active_set` that keeps track of the current set of vertices and convex decomposition.
It therefore delegates all update, deletion, and addition operations to this inner active set.
The `weight`, `atoms`, and `x` fields should only be accessed to read and are effectively the same objects as those in the inner active set.
The flag `wolfe_step` determines whether to use a Wolfe step from the min-norm point algorithm or the normal direct solve.
The Wolfe step solves the auxiliary subproblem over the affine hull of the current active set (instead of the convex hull).

The structure also contains a scheduler struct which is called with the `should_solve_lp` function.
To define a new frequency at which the LP should be solved, one can define another scheduler struct and implement the corresponding method.
"""
struct ActiveSetQuadraticLinearSolve{
    AT,
    R<:Real,
    IT,
    H,
    BT,
    OT<:MOI.AbstractOptimizer,
    AS<:AbstractActiveSet,
    SF,
} <: AbstractActiveSet{AT,R,IT}
    weights::Vector{R}
    atoms::Vector{AT}
    x::IT
    A::H # Hessian matrix
    b::BT # linear term
    active_set::AS
    lp_optimizer::OT
    wolfe_step::Union{Bool,AffineMinSolver} # If bool use QC-MNP or QC-LP with Highs, if AffineMinSolver use QC-MNP with custom linear solver.
    scheduler::SF
    counter::Base.RefValue{Int}
end

"""
    ActiveSetQuadraticLinearSolve(tuple_values::Vector{Tuple{R,AT}}, grad!::Function, lp_optimizer)

Creates an ActiveSetQuadraticLinearSolve by computing the Hessian and linear term from `grad!`.
"""
function ActiveSetQuadraticLinearSolve(
    tuple_values::Vector{Tuple{R,AT}},
    grad!::Function,
    lp_optimizer;
    scheduler=LogScheduler(),
    wolfe_step=false,
) where {AT,R}
    A, b = detect_quadratic_function(grad!, tuple_values[1][2])
    return ActiveSetQuadraticLinearSolve(
        tuple_values,
        A,
        b,
        lp_optimizer,
        scheduler=scheduler,
        wolfe_step=wolfe_step,
    )
end

"""
    ActiveSetQuadraticLinearSolve(tuple_values::Vector{Tuple{R,AT}}, A, b, lp_optimizer)

Creates an `ActiveSetQuadraticLinearSolve` from the given Hessian `A`, linear term `b` and `lp_optimizer` by creating an inner `ActiveSetQuadraticProductCaching` active set.
"""
function ActiveSetQuadraticLinearSolve(
    tuple_values::Vector{Tuple{R,AT}},
    A::H,
    b,
    lp_optimizer;
    scheduler=LogScheduler(),
    wolfe_step=false,
) where {AT,R,H}
    inner_as = ActiveSetQuadraticProductCaching(tuple_values, A, b)
    return ActiveSetQuadraticLinearSolve(
        inner_as.weights,
        inner_as.atoms,
        inner_as.x,
        inner_as.A,
        inner_as.b,
        inner_as,
        lp_optimizer,
        wolfe_step,
        scheduler,
        Ref(0),
    )
end

function ActiveSetQuadraticLinearSolve(
    inner_as::AbstractActiveSet,
    A,
    b,
    lp_optimizer;
    scheduler=LogScheduler(),
    wolfe_step=false,
)
    as = ActiveSetQuadraticLinearSolve(
        inner_as.weights,
        inner_as.atoms,
        inner_as.x,
        A,
        b,
        inner_as,
        lp_optimizer,
        wolfe_step,
        scheduler,
        Ref(0),
    )
    compute_active_set_iterate!(as)
    return as
end

function ActiveSetQuadraticLinearSolve(
    inner_as::AbstractActiveSet,
    A::LinearAlgebra.UniformScaling,
    b,
    lp_optimizer;
    scheduler=LogScheduler(),
    wolfe_step=false,
)
    as = ActiveSetQuadraticLinearSolve(
        inner_as.weights,
        inner_as.atoms,
        inner_as.x,
        A,
        b,
        inner_as,
        lp_optimizer,
        wolfe_step,
        scheduler,
        Ref(0),
    )
    compute_active_set_iterate!(as)
    return as
end

function ActiveSetQuadraticLinearSolve(
    inner_as::AbstractActiveSet,
    grad!::Function,
    lp_optimizer;
    scheduler=LogScheduler(),
    wolfe_step=false,
)
    A, b = detect_quadratic_function(grad!, inner_as.atoms[1])
    return ActiveSetQuadraticLinearSolve(
        inner_as,
        A,
        b,
        lp_optimizer;
        scheduler=scheduler,
        wolfe_step=wolfe_step,
    )
end

function ActiveSetQuadraticLinearSolve{AT,R}(
    tuple_values::Vector{<:Tuple{<:Number,<:Any}},
    grad!::Function,
    lp_optimizer;
    scheduler=LogScheduler(),
) where {AT,R}
    A, b = detect_quadratic_function(grad!, tuple_values[1][2])
    return ActiveSetQuadraticLinearSolve{AT,R}(
        tuple_values,
        A,
        b,
        lp_optimizer;
        scheduler=scheduler,
    )
end

function ActiveSetQuadraticLinearSolve{AT,R}(
    tuple_values::Vector{<:Tuple{<:Number,<:Any}},
    A::H,
    b,
    lp_optimizer;
    scheduler=LogScheduler(),
) where {AT,R,H}
    inner_as = ActiveSetQuadraticProductCaching{AT,R}(tuple_values, A, b)
    as = ActiveSetQuadraticLinearSolve{AT,R,typeof(x),H}(
        inner_as.weights,
        inner_as.atoms,
        inner_as.x,
        A,
        b,
        inner_as,
        lp_optimizer,
        scheduler,
        Ref(0),
    )
    compute_active_set_iterate!(as)
    return as
end

function ActiveSetQuadraticLinearSolve(
    tuple_values::Vector{Tuple{R,AT}},
    A::UniformScaling,
    b,
    lp_optimizer;
    scheduler=LogScheduler(),
) where {AT,R}
    return ActiveSetQuadraticLinearSolve(
        tuple_values,
        Identity(A.λ),
        b,
        lp_optimizer;
        scheduler=scheduler,
    )
end
function ActiveSetQuadraticLinearSolve{AT,R}(
    tuple_values::Vector{<:Tuple{<:Number,<:Any}},
    A::UniformScaling,
    b,
    lp_optimizer;
    scheduler=LogScheduler(),
) where {AT,R}
    return ActiveSetQuadraticLinearSolve{AT,R}(
        tuple_values,
        Identity(A.λ),
        b,
        lp_optimizer;
        scheduler=scheduler,
    )
end

# all mutating functions are delegated to the inner active set

Base.push!(as::ActiveSetQuadraticLinearSolve, tuple) = push!(as.active_set, tuple)

Base.deleteat!(as::ActiveSetQuadraticLinearSolve, idx::Int) = deleteat!(as.active_set, idx)

Base.empty!(as::ActiveSetQuadraticLinearSolve) = empty!(as.active_set)

function active_set_update!(
    as::ActiveSetQuadraticLinearSolve{AT,R},
    lambda,
    atom,
    renorm=true,
    idx=nothing;
    weight_purge_threshold=weight_purge_threshold_default(R),
    add_dropped_vertices=false,
    vertex_storage=nothing,
) where {AT,R}
    if idx === nothing
        idx = find_atom(as, atom)
    end
    active_set_update!(
        as.active_set,
        lambda,
        atom,
        renorm,
        idx;
        weight_purge_threshold=weight_purge_threshold,
        add_dropped_vertices=add_dropped_vertices,
        vertex_storage=vertex_storage,
    )
    # new atom introduced, we can solve the auxiliary LP
    if idx < 0
        as.counter[] += 1
        if should_solve_lp(as, as.scheduler)
            solve_quadratic_activeset_lp!(as)
        end
    end
    return as
end

active_set_renormalize!(as::ActiveSetQuadraticLinearSolve) = active_set_renormalize!(as.active_set)

function active_set_argmin(as::ActiveSetQuadraticLinearSolve, direction)
    return active_set_argmin(as.active_set, direction)
end

function active_set_argminmax(as::ActiveSetQuadraticLinearSolve, direction; Φ=0.5)
    return active_set_argminmax(as.active_set, direction; Φ=Φ)
end

struct SymmetricAffineMinSolver <: AffineMinSolver
    solve::Function
end

struct NonSymmetricAffineMinSolver <: AffineMinSolver
    solve::Function
end

function compute_affine_min(as::ActiveSetQuadraticLinearSolve{AT,R,IT,H}, solver::SymmetricAffineMinSolver) where {AT,R,IT,H}

    nv = length(as)
    n_reduced = nv - 1

    # Pre-allocate arrays
    A_mat = Matrix{Float64}(undef, n_reduced, n_reduced)
    r_vec = Vector{Float64}(undef, n_reduced)
    μ = zeros(eltype(as.weights[1]), n_reduced)

    if as.active_set isa ActiveSetQuadraticProductCaching

        for i in 2:nv
            for j in 2:i
                val = as.active_set.dots_A[i][j] - as.active_set.dots_A[j][1] - as.active_set.dots_A[i][1] + as.active_set.dots_A[1][1]
                A_mat[i-1, j-1] = val
                if i != j
                    A_mat[j-1, i-1] = val
                end
            end
        end
        for i in 2:nv
            r_vec[i-1] = - as.active_set.dots_A[i][1] + as.active_set.dots_A[1][1] - as.active_set.dots_b[i] + as.active_set.dots_b[1]
        end
    else

        temp1 = similar(as.atoms[1])
        temp2 = similar(as.atoms[1])
        temp3 = similar(as.atoms[1])
        temp4 = similar(as.atoms[1])
        
        # Cache the first atom and its transformation
        atom1 = as.atoms[1]
        A_atom1 = mul!(temp3, as.A, atom1)
        
        # Construct A_mat and rhs more efficiently
        for i in 2:nv
            # This is fast because it computes (as.atoms[i] - atom1) only once and reuses the result.
            diff_i = copyto!(temp1, as.atoms[i])
            @. diff_i -= atom1
            
            # Calculate A * (as.atoms[i] - atom1) once for reuse
            A_diff_i = mul!(temp2, as.A, diff_i)
            
            # Fill rhs
            r_vec[i-1] = -dot(diff_i, A_atom1 + as.b)
            
            # Fill A_mat
            for j in 2:i
                # Calculate (as.atoms[j] - atom1) separately
                diff_j = mul!(temp3, I, as.atoms[j])
                @. diff_j -= atom1
                
                # Calculate A * (as.atoms[j] - atom1) separately
                A_diff_j = mul!(temp4, as.A, diff_j)
                
                # Compute dot product
                A_mat[i-1, j-1] = dot(diff_i, A_diff_j)
                if i != j
                    A_mat[j-1, i-1] = A_mat[i-1, j-1]  # Exploit symmetry
                end
            end
        end
    end    
    
    # Solve system
    converged = solver.solve(μ, A_mat, r_vec)
    s = sum(μ)

    # If linear solving failed, return the original weights
    if converged && abs(s) < 1e7
       return [1 - s; μ]
    else
        return as.weights
    end
end

function compute_affine_min(as::ActiveSetQuadraticLinearSolve{AT,R,IT,H}, solver::NonSymmetricAffineMinSolver) where {AT,R,IT,H}

    nv = length(as)

    # Pre-allocate arrays
    A_mat = Matrix{Float64}(undef, nv, nv)
    r_vec = Vector{Float64}(undef, nv)
    μ = Vector{Float64}(undef, nv)

    A_mat[1,:] .= 1.0
    r_vec[1] = 1.0
    if as.active_set isa ActiveSetQuadraticProductCaching
        # dots_A is a lower triangular matrix

        d1 = as.active_set.dots_A[1]
        for i in 2:nv
            di = as.active_set.dots_A[i]
            for j in 1:i
                val = di[j] - d1[j]
                A_mat[i, j] = val
                if i != j && j != 1
                    A_mat[j, i] = val
                end
            end
            r_vec[i] = as.active_set.dots_b[1] - as.active_set.dots_b[i]
        end
    else
        temp1 = similar(as.atoms[1])
        d1 = as.A * as.atoms[1]
        for i in 2:nv
            di = mul!(temp1, as.A, as.atoms[i])
            for j in 1:i
                val = dot(di, as.atoms[j]) - dot(d1, as.atoms[j])
                A_mat[i, j] = val
                if i != j && j != 1
                    A_mat[j, i] = val
                end
            end
            r_vec[i] = dot(as.b, as.atoms[1]) - dot(as.b, as.atoms[i])
        end
    end

    # Solve system
    converged = solver.solve(μ, A_mat, r_vec)
    @info "sum(μ) = $(sum(μ))"
    if converged && abs(sum(μ)) < 1e7
        return μ
    else
        return as.weights
    end
end



# generic quadratic with quadratic information provided

"""
    solve_quadratic_activeset_lp!(as::ActiveSetQuadraticLinearSolve{AT, R, IT, H}))

Solves the auxiliary LP over the current active set.
The method is specialized by type `H` of the Hessian matrix `A`.
"""
function solve_quadratic_activeset_lp!(
    as::ActiveSetQuadraticLinearSolve{AT,R,IT,H},
) where {AT,R,IT,H}
    nv = length(as)

    if as.wolfe_step isa AffineMinSolver
        affine_min_weights = compute_affine_min(as, as.wolfe_step)
        indices_to_remove, new_weights = _compute_new_weights_wolfe_step(affine_min_weights, R, as.weights)
    elseif as.wolfe_step isa Bool
        o = as.lp_optimizer
        MOI.empty!(o)
        λ = MOI.add_variables(o, nv)
        # λ ≥ 0, ∑ λ == 1
        if !as.wolfe_step
            MOI.add_constraint.(o, λ, MOI.GreaterThan(0.0))
        end
        sum_of_variables = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, λ), 0.0)
        MOI.add_constraint(o, sum_of_variables, MOI.EqualTo(1.0))
        # Wᵗ A V λ == -Wᵗ b
        # V has columns vi
        # W has columns vi - v1
        for i in 2:nv
            lhs = MOI.ScalarAffineFunction{Float64}([], 0.0)
            Base.sizehint!(lhs.terms, nv)
            if as.active_set isa ActiveSetQuadraticProductCaching
                # dots_A is a lower triangular matrix
                for j in 1:i
                    push!(
                        lhs.terms,
                        MOI.ScalarAffineTerm(
                            as.active_set.dots_A[i][j] - as.active_set.dots_A[j][1],
                            λ[j],
                        ),
                    )
                end
                for j in i+1:nv
                    push!(
                        lhs.terms,
                        MOI.ScalarAffineTerm(
                            as.active_set.dots_A[j][i] - as.active_set.dots_A[j][1],
                            λ[j],
                        ),
                    )
                end
                rhs = as.active_set.dots_b[1] - as.active_set.dots_b[i]
            else
                # replaces direct sum because of MOI and MutableArithmetic slow sums
                for j in 1:nv
                    push!(
                        lhs.terms,
                        _compute_quadratic_constraint_term(
                            as.atoms[i],
                            as.atoms[1],
                            as.A,
                            as.atoms[j],
                            λ[j],
                        ),
                    )
                end
                rhs = dot(as.atoms[1], as.b) - dot(as.atoms[i], as.b)
            end
            MOI.add_constraint(o, lhs, MOI.EqualTo{Float64}(rhs))
        end
        MOI.set(o, MOI.ObjectiveFunction{typeof(sum_of_variables)}(), sum_of_variables)
        MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.optimize!(o)
        if MOI.get(o, MOI.TerminationStatus()) ∉ (MOI.OPTIMAL, MOI.FEASIBLE_POINT, MOI.ALMOST_OPTIMAL)
            return as
        end
        indices_to_remove, new_weights = if as.wolfe_step isa AffineMinSolver
            _compute_new_weights_wolfe_step(MOI.get.(o, MOI.VariablePrimal(), λ), R, as.weights)
        else
            _compute_new_weights_direct_solve(MOI.get.(o, MOI.VariablePrimal(), λ), R)
        end
    end
    deleteat!(as.active_set, indices_to_remove)
    @assert length(as) == length(new_weights)
    update_weights!(as.active_set, new_weights)
    active_set_cleanup!(as)
    active_set_renormalize!(as)
    compute_active_set_iterate!(as)
    return as
end

function _compute_new_weights_direct_solve(wolfe_weights, ::Type{R}) where {R}
    indices_to_remove = BitSet()
    new_weights = R[]
    for (idx, weight_value) in enumerate(wolfe_weights)
        if weight_value <= 2 * weight_purge_threshold_default(typeof(weight_value))
            push!(indices_to_remove, idx)
        else
            push!(new_weights, weight_value)
        end
    end
    return indices_to_remove, new_weights
end

function _compute_new_weights_wolfe_step(
    wolfe_weights,
    ::Type{R},
    old_weights,
) where {R}
    # all nonnegative -> use non-wolfe procedure
    if all(>=(-10eps()), wolfe_weights)
        return _compute_new_weights_direct_solve(wolfe_weights, R)
    end
    # ratio test to know which coordinate would hit zero first
    tau_min = 1.0
    set_indices_zero = BitSet()
    for idx in eachindex(wolfe_weights)
        if wolfe_weights[idx] < old_weights[idx]
            tau = old_weights[idx] / (old_weights[idx] - wolfe_weights[idx])
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
    new_lambdas =
        [(1 - tau_min) * old_weights[idx] + tau_min * wolfe_weights[idx] for idx in eachindex(wolfe_weights)]
    for idx in set_indices_zero
        new_lambdas[idx] = 0
    end
    @assert all(>=(-2weight_purge_threshold_default(eltype(new_lambdas))), new_lambdas) "All new_lambdas must be between nonnegative $(minimum(new_lambdas))"
    @assert isapprox(sum(new_lambdas), 1.0) "The sum of new_lambdas must be approximately 1, but is $(sum(new_lambdas))"
    indices_to_remove = Int[]
    new_weights = R[]
    for idx in eachindex(wolfe_weights)
        weight_value = new_lambdas[idx] # using new lambdas
        if weight_value <= eps()
            push!(indices_to_remove, idx)
        else
            push!(new_weights, weight_value)
        end
    end
    return indices_to_remove, new_weights
end

function _compute_quadratic_constraint_term(atom1, atom0, A::AbstractMatrix, atom2, λ)
    return MOI.ScalarAffineTerm(fast_dot(atom1, A, atom2) - fast_dot(atom0, A, atom2), λ)
end

function _compute_quadratic_constraint_term(
    atom1,
    atom0,
    A::Union{Identity,LinearAlgebra.UniformScaling},
    atom2,
    λ,
)
    return MOI.ScalarAffineTerm(A.λ * (fast_dot(atom1, atom2) - fast_dot(atom0, atom2)), λ)
end

struct LogScheduler{T}
    start_time::Int
    scaling_factor::T
    max_interval::Int
    current_interval::Base.RefValue{Int}
    last_solve_counter::Base.RefValue{Int}
end

LogScheduler(; start_time=20, scaling_factor=1.5, max_interval=1000) =
    LogScheduler(start_time, scaling_factor, max_interval, Ref(start_time), Ref(0))

function should_solve_lp(as::ActiveSetQuadraticLinearSolve, scheduler::LogScheduler)
    if as.counter[] - scheduler.last_solve_counter[] >= scheduler.current_interval[]
        scheduler.last_solve_counter[] = as.counter[]
        scheduler.current_interval[] = min(
            round(Int, scheduler.scaling_factor * scheduler.current_interval[]),
            scheduler.max_interval,
        )
        return true
    end
    return false
end
