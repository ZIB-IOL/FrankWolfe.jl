
##############################
### memory_mode macro
##############################

macro memory_mode(memory_mode, ex)
    return esc(quote
        if $memory_mode isa InplaceEmphasis
            @. $ex
        else
            $ex
        end
    end)
end

"""
    muladd_memory_mode(memory_mode::MemoryEmphasis, d, x, v)

Performs `d = x - v` in-place or not depending on MemoryEmphasis
"""
function muladd_memory_mode(memory_mode::MemoryEmphasis, d, x, v)
    @memory_mode(memory_mode, d = x - v)
end

"""
    (memory_mode::MemoryEmphasis, x, gamma::Real, d)

Performs `x = x - gamma * d` in-place or not depending on MemoryEmphasis
"""
function muladd_memory_mode(memory_mode::MemoryEmphasis, x, gamma::Real, d)
    @memory_mode(memory_mode, x = x - gamma * d)
end

"""
    (memory_mode::MemoryEmphasis, storage, x, gamma::Real, d)

Performs `storage = x - gamma * d` in-place or not depending on MemoryEmphasis
"""
function muladd_memory_mode(memory_mode::MemoryEmphasis, storage, x, gamma::Real, d)
    @memory_mode(memory_mode, storage = x - gamma * d)
end

##############################################################
# simple benchmark of elementary costs of oracles and
# critical components
##############################################################

function benchmark_oracles(f, grad!, x_gen, lmo; k=100, nocache=true)
    x = x_gen()
    sv = sizeof(x) / 1024^2
    println("\nSize of single atom ($(eltype(x))): $sv MB\n")
    to = TimerOutput()
    @showprogress 1 "Testing f... " for i in 1:k
        x = x_gen()
        @timeit to "f" temp = f(x)
    end
    @showprogress 1 "Testing grad... " for i in 1:k
        x = x_gen()
        temp = similar(x)
        @timeit to "grad" grad!(temp, x)
    end
    @showprogress 1 "Testing lmo... " for i in 1:k
        x = x_gen()
        @timeit to "lmo" temp = compute_extreme_point(lmo, x)
    end
    @showprogress 1 "Testing dual gap... " for i in 1:k
        x = x_gen()
        gradient = collect(x)
        grad!(gradient, x)
        v = compute_extreme_point(lmo, gradient)
        @timeit to "dual gap" begin
            dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
        end
    end
    @showprogress 1 "Testing update... (Emphasis: OutplaceEmphasis) " for i in 1:k
        x = x_gen()
        gradient = collect(x)
        grad!(gradient, x)
        v = compute_extreme_point(lmo, gradient)
        gamma = 1 / 2
        @timeit to "update (OutplaceEmphasis)" @memory_mode(
            OutplaceEmphasis(),
            x = (1 - gamma) * x + gamma * v
        )
    end
    @showprogress 1 "Testing update... (Emphasis: InplaceEmphasis) " for i in 1:k
        x = x_gen()
        gradient = collect(x)
        grad!(gradient, x)
        v = compute_extreme_point(lmo, gradient)
        gamma = 1 / 2
        # TODO: to be updated to broadcast version once data structure ScaledHotVector allows for it
        @timeit to "update (InplaceEmphasis)" @memory_mode(
            InplaceEmphasis(),
            x = (1 - gamma) * x + gamma * v
        )
    end
    if !nocache
        @showprogress 1 "Testing caching 100 points... " for i in 1:k
            @timeit to "caching 100 points" begin
                cache = [gen_x() for _ in 1:100]
                x = gen_x()
                gradient = collect(x)
                grad!(gradient, x)
                v = compute_extreme_point(lmo, gradient)
                gamma = 1 / 2
                test = (x -> fast_dot(x, gradient)).(cache)
                v = cache[argmin(test)]
                val = v in cache
            end
        end
    end
    print_timer(to)
    return nothing
end

"""
    _unsafe_equal(a, b)

Like `isequal` on arrays but without the checks. Assumes a and b have the same axes.
"""
function _unsafe_equal(a::Array, b::Array)
    if a === b
        return true
    end
    @inbounds for idx in eachindex(a)
        if a[idx] != b[idx]
            return false
        end
    end
    return true
end

_unsafe_equal(a, b) = isequal(a, b)

function _unsafe_equal(a::SparseArrays.AbstractSparseArray, b::SparseArrays.AbstractSparseArray)
    return a == b
end

fast_dot(A, B) = dot(A, B)

fast_dot(B::SparseArrays.SparseMatrixCSC, A::Matrix) = conj(fast_dot(A, B))

function fast_dot(A::Matrix{T1}, B::SparseArrays.SparseMatrixCSC{T2}) where {T1,T2}
    T = promote_type(T1, T2)
    (m, n) = size(A)
    if (m, n) != size(B)
        throw(DimensionMismatch("Size mismatch"))
    end
    s = zero(T)
    if m * n == 0
        return s
    end
    rows = SparseArrays.rowvals(B)
    vals = SparseArrays.nonzeros(B)
    @inbounds for j in 1:n
        for ridx in SparseArrays.nzrange(B, j)
            i = rows[ridx]
            v = vals[ridx]
            s += v * conj(A[i, j])
        end
    end
    return s
end

fast_dot(a, Q, b) = dot(a, Q, b)

function fast_dot(a::SparseArrays.AbstractSparseVector{<:Real}, Q::Diagonal{<:Real}, b::AbstractVector{<:Real})
    if a === b
        return _fast_quadratic_form_symmetric(a, Q)
    end
    d = Q.diag
    nzvals = SparseArrays.nonzeros(a)
    nzinds = SparseArrays.nonzeroinds(a)
    return sum(eachindex(nzvals); init=zero(eltype(a))) do nzidx
        nzvals[nzidx] * d[nzinds[nzidx]] * b[nzinds[nzidx]]
    end
end

function fast_dot(a::SparseArrays.AbstractSparseVector{<:Real}, Q::Diagonal{<:Real}, b::SparseArrays.AbstractSparseVector{<:Real})
    if a === b
        return _fast_quadratic_form_symmetric(a, Q)
    end
    n = length(a)
    if length(b) != n
        throw(
            DimensionMismatch("Vector a has a length $n but b has a length $(length(b))")
        )
    end
    anzind = SparseArrays.nonzeroinds(a)
    bnzind = SparseArrays.nonzeroinds(b)
    anzval = SparseArrays.nonzeros(a)
    bnzval = SparseArrays.nonzeros(b)
    s = zero(Base.promote_eltype(a, Q, b))

    if isempty(anzind) || isempty(bnzind)
        return s
    end

    a_idx = 1
    b_idx = 1
    a_idx_last = length(anzind)
    b_idx_last = length(bnzind)

    # go through the nonzero indices of a and b simultaneously
    @inbounds while a_idx <= a_idx_last && b_idx <= b_idx_last
        ia = anzind[a_idx]
        ib = bnzind[b_idx]
        if ia == ib
            s += dot(anzval[a_idx], Q.diag[ia], bnzval[b_idx])
            a_idx += 1
            b_idx += 1
        elseif ia < ib
            a_idx += 1
        else
            b_idx += 1
        end
    end
    return s
end


function _fast_quadratic_form_symmetric(a, Q)
    d = Q.diag
    if length(d) != length(a)
        throw(DimensionMismatch())
    end
    nzvals = SparseArrays.nonzeros(a)
    nzinds = SparseArrays.nonzeroinds(a)
    s = zero(Base.promote_eltype(a, Q))
    @inbounds for nzidx in eachindex(nzvals)
        s += nzvals[nzidx]^2 * d[nzinds[nzidx]]
    end
    return s
end

"""
    trajectory_callback(storage)

Callback pushing the state at each iteration to the passed storage.
The state data is only the 5 first fields, usually:
`(t,primal,dual,dual_gap,time)`
"""
function trajectory_callback(storage)
    return function push_trajectory!(data, args...)
        return push!(storage, callback_state(data))
    end
end

"""
    momentum_iterate(iter::MomentumIterator) -> ρ

Method to implement for a type `MomentumIterator`.
Returns the next momentum value `ρ` and updates the iterator internal state.
"""
function momentum_iterate end

"""
    ExpMomentumIterator{T}

Iterator for the momentum used in the variant of Stochastic Frank-Wolfe.
Momentum coefficients are the values of the iterator:
`ρ_t = 1 - num / (offset + t)^exp`

The state corresponds to the iteration count.

Source:
Stochastic Conditional Gradient Methods: From Convex Minimization to Submodular Maximization
Aryan Mokhtari, Hamed Hassani, Amin Karbasi, JMLR 2020.
"""
mutable struct ExpMomentumIterator{T}
    exp::T
    num::T
    offset::T
    iter::Int
end

ExpMomentumIterator() = ExpMomentumIterator(2 / 3, 4.0, 8.0, 0)

function momentum_iterate(em::ExpMomentumIterator)
    em.iter += 1
    return 1 - em.num / (em.offset + em.iter)^(em.exp)
end

"""
    ConstantMomentumIterator{T}

Iterator for momentum with a fixed damping value, always return the value and a dummy state.
"""
struct ConstantMomentumIterator{T}
    v::T
end

momentum_iterate(em::ConstantMomentumIterator) = em.v

# batch sizes

"""
    batchsize_iterate(iter::BatchSizeIterator) -> b

Method to implement for a batch size iterator of type `BatchSizeIterator`.
Calling `batchsize_iterate` returns the next batch size and typically update the internal state of `iter`.
"""
function batchsize_iterate end

"""
    ConstantBatchIterator(batch_size)

Batch iterator always returning a constant batch size.
"""
struct ConstantBatchIterator
    batch_size::Int
end

batchsize_iterate(cbi::ConstantBatchIterator) = cbi.batch_size

"""
    IncrementBatchIterator(starting_batch_size, max_batch_size, [increment = 1])

Batch size starting at starting_batch_size and incrementing by `increment` at every iteration.
"""
mutable struct IncrementBatchIterator
    starting_batch_size::Int
    max_batch_size::Int
    increment::Int
    iter::Int
    maxreached::Bool
end

function IncrementBatchIterator(starting_batch_size::Int, max_batch_size::Int, increment::Int)
    return IncrementBatchIterator(starting_batch_size, max_batch_size, increment, 0, false)
end

function IncrementBatchIterator(starting_batch_size::Int, max_batch_size::Int)
    return IncrementBatchIterator(starting_batch_size, max_batch_size, 1, 0, false)
end

function batchsize_iterate(ibi::IncrementBatchIterator)
    if ibi.maxreached
        return ibi.max_batch_size
    end
    new_size = ibi.starting_batch_size + ibi.iter * ibi.increment
    ibi.iter += 1
    if new_size > ibi.max_batch_size
        ibi.maxreached = true
        return ibi.max_batch_size
    end
    return new_size
end

"""
Vertex storage to store dropped vertices or find a suitable direction in lazy settings.
The algorithm will look for at most `return_kth` suitable atoms before returning the best.
See [Extra-lazification with a vertex storage](@ref) for usage.

A vertex storage can be any type that implements two operations:
1. `Base.push!(storage, atom)` to add an atom to the storage.
Note that it is the storage type responsibility to ensure uniqueness of the atoms present.
2. `storage_find_argmin_vertex(storage, direction, lazy_threshold) -> (found, vertex)`
returning whether a vertex with sufficient progress was found and the vertex.
It is up to the storage to remove vertices (or not) when they have been picked up.
"""
struct DeletedVertexStorage{AT}
    storage::Vector{AT}
    return_kth::Int
end

DeletedVertexStorage(storage::Vector) = DeletedVertexStorage(storage, 1)
DeletedVertexStorage{AT}() where {AT} = DeletedVertexStorage(AT[])

function Base.push!(vertex_storage::DeletedVertexStorage{AT}, atom::AT) where {AT}
    # do not push duplicates
    if !any(v -> _unsafe_equal(atom, v), vertex_storage.storage)
        push!(vertex_storage.storage, atom)
    end
    return vertex_storage
end

Base.length(storage::DeletedVertexStorage) = length(storage.storage)

"""
Give the vertex `v` in the storage that minimizes `s = direction ⋅ v` and whether `s` achieves
`s ≤ lazy_threshold`.
"""
function storage_find_argmin_vertex(vertex_storage::DeletedVertexStorage, direction, lazy_threshold)
    if isempty(vertex_storage.storage)
        return (false, nothing)
    end
    best_idx = 1
    best_val = lazy_threshold
    found_good = false
    counter = 0
    for (idx, atom) in enumerate(vertex_storage.storage)
        s = dot(direction, atom)
        if s < best_val
            counter += 1
            best_val = s
            found_good = true
            best_idx = idx
            if counter ≥ vertex_storage.return_kth
                return (found_good, vertex_storage.storage[best_idx])
            end
        end
    end
    return (found_good, vertex_storage.storage[best_idx])
end

# temporary fix because argmin is broken on julia 1.8
argmin_(v) = argmin(v)
function argmin_(v::SparseArrays.SparseVector{T}) where {T}
    if isempty(v.nzind)
        return 1
    end
    idx = -1
    val = T(Inf)
    for s_idx in eachindex(v.nzind)
        if v.nzval[s_idx] < val
            val = v.nzval[s_idx]
            idx = s_idx
        end
    end
    # if min value is already negative or the indices were all checked
    if val < 0 || length(v.nzind) == length(v)
        return v.nzind[idx]
    end
    # otherwise, find the first zero
    for idx in eachindex(v)
        if idx ∉ v.nzind
            return idx
        end
    end
    error("unreachable")
end

function weight_purge_threshold_default(::Type{T}) where {T<:AbstractFloat}
    return sqrt(eps(T) * Base.rtoldefault(T)) # around 1e-12 for Float64
end
weight_purge_threshold_default(::Type{T}) where {T<:Number} = Base.rtoldefault(T)



# Direct solvers and simplex based sparsification


#### 
# Leave for now - potentially faster for the norm projection case
#### 

"""
    direct_solve(active_set, grad!, lp_solver)

Directly solve the optimization problem for the given active set using a linear programming solver.

This function performs the following steps:
1. Computes the gradient at zero.
2. Calculates x0 as -1/2 * grad(0).
3. Generates a matrix A with columns being the points in the active set.
4. Sets up and solves a linear programming problem using the provided LP solver.

The LP problem is formulated to find optimal weights λ for the atoms in the active set,
subject to the constraints that the weights sum to 1 and are non-negative.

# Arguments
- `active_set`: The current active set of atoms.
- `grad!`: A function to compute the gradient.
- `lp_solver`: A linear programming solver to use for the optimization.

# Returns
A new ActiveSet with updated weights based on the LP solution.

# Note
This function is particularly useful for problems where direct solving is more efficient
than iterative methods. It can only used for functions of the form f(x) = \norm{x-x_0}_2^2

# Usage
- Set `squadratic = true` when calling `blended_pairwise_conditional_gradient`
- Can be used together with `ActiveSetQuadratic`
"""
function direct_solve(active_set, grad!, lp_solver)
    x0 = get_active_set_iterate(active_set)
    # Compute gradient at 0
    zero_x = zero(x0)
    zero_grad = similar(x0)
    grad!(zero_grad, zero_x)
    
    # 1. Compute x0 = -1/2 * grad(0)
    x0 = -0.5 * zero_grad

    # 2. Generate matrix A with columns being the points in the active set
    if isempty(active_set.atoms)
        A = zeros(length(x0), 0)
    else
        # A = hcat(getindex.(active_set, 2)...) // this leads to stack overflow
        A = zeros(eltype(x0), length(x0), length(active_set))
        for (i, atom) in enumerate(active_set)
            A[:, i] = atom[2]
        end
    end
    # println(active_set)
    # println(A)
    # println(x0)

    # 3. Setup and solve the system using the provided LP solver
    n = size(A, 2)
    model = MOI.instantiate(lp_solver)
    MOI.set(model, MOI.Silent(), true)

    # Define variables
    λ = MOI.add_variables(model, n)
    for i in 1:n
        MOI.add_constraint(model, λ[i], MOI.GreaterThan(0.0))
    end

    # Add sum constraint
    sum_λ = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), λ), 0.0)
    MOI.add_constraint(model, sum_λ, MOI.EqualTo(1.0))

    # Add constraint A'A λ = A'x0
    Q = A'A
    b = A'x0
    # println(Q)
    # println(b)
    for i in 1:size(Q, 1)
        terms = MOI.ScalarAffineTerm{Float64}[]
        for j in 1:size(Q, 2)
            if !iszero(Q[i,j])
                push!(terms, MOI.ScalarAffineTerm(Q[i,j], λ[j]))
            end
        end
        constraint = MOI.ScalarAffineFunction(terms, 0.0)
        MOI.add_constraint(model, constraint, MOI.EqualTo(b[i]))
    end

    # Set a dummy objective (minimize sum of λ)
    dummy_objective = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), λ), 0.0)
    MOI.set(model, MOI.ObjectiveFunction{typeof(dummy_objective)}(), dummy_objective)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(model)

    # 4. Check if solve was feasible and update active set weights
    if MOI.get(model, MOI.TerminationStatus()) in [MOI.OPTIMAL, MOI.FEASIBLE_POINT]
        # @info "Direct solve successful"
        # λ_values = MOI.get.(model, MOI.VariablePrimal(), λ) # leads to stack overflow
        λ_values = Vector{Float64}(undef, n)
        for i in 1:n
            λ_values[i] = MOI.get(model, MOI.VariablePrimal(), λ[i])
        end
        new_weights = λ_values
        return ActiveSet([(new_weights[i], active_set.atoms[i]) for i in 1:n])
    else
#        @info "Direct solve failed"
        return active_set
    end
end

"""
    direct_solve_gq(active_set, grad!, lp_solver)

Perform a direct solve for a quadratic problem over the active set by solving the 
optimality system A'G λ = 0 with the provided LP solver. Here A is the matrix of atoms
and G is the matrix of gradients of the atoms in the active set.

# Arguments
- `active_set`: The current active set containing atoms and their weights.
- `grad!`: A function to compute the gradient for a given atom.
- `lp_solver`: The linear programming solver to use for the optimization.

# Returns
- A new `ActiveSet` with updated weights if the solve is successful.
- The original `active_set` if the solve fails.

# Details
1. Retrieves the current iterate from the active set.
2. Generates matrices A and G, where A contains the atoms and G contains their gradients.
3. Sets up a linear program for the optimization problem.
4. Solves the system and updates the active set weights if successful using the provided LP solver.

Note: This function is specifically designed for quadratic problems and uses the
optimality condition A'G λ = 0, where A is the matrix of atoms and G is the matrix of
their gradients.

# Usage
- Set `squadratic = true` when calling `blended_pairwise_conditional_gradient' 
- Can we used together with ActiveSetQuadratic
"""
function direct_solve_gq(active_set, grad!, lp_solver)
    # 1. Get current iterate // artifact only used for size and type of A
    x0 = get_active_set_iterate(active_set)
    
    # 2. Generate matrix A with columns being the gradients over the atoms in the active set
    if isempty(active_set)
        A = zeros(length(x0), 0)
        G = zeros(length(x0), 0)
    else
        A = zeros(eltype(x0), length(x0), length(active_set))
        G = zeros(eltype(x0), length(x0), length(active_set))
        grad_storage = similar(x0)
        for (i, atom) in enumerate(active_set)
            grad!(grad_storage, atom[2])
            A[:, i] = atom[2]
            G[:, i] = grad_storage
        end
    end


    # @info "Solving system with gradients"

    # 3. Setup and solve the system using the provided LP solver
    n = size(A, 2)
    model = MOI.instantiate(lp_solver)
    MOI.set(model, MOI.Silent(), true)

    # Define variables
    λ = MOI.add_variables(model, n)
    for i in 1:n
        MOI.add_constraint(model, λ[i], MOI.GreaterThan(0.0))
    end

    # Add sum constraint
    sum_λ = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), λ), 0.0)
    MOI.add_constraint(model, sum_λ, MOI.EqualTo(1.0))

    # Add constraint A'G λ = 0
    Q = A'G
    for i in 1:size(Q, 1)
        terms = MOI.ScalarAffineTerm{Float64}[]
        for j in 1:size(Q, 2)
            if !iszero(Q[i,j])
                push!(terms, MOI.ScalarAffineTerm(Q[i,j], λ[j]))
            end
        end
        constraint = MOI.ScalarAffineFunction(terms, 0.0)
        MOI.add_constraint(model, constraint, MOI.EqualTo(0.0))
    end
    # @info "Direct solve problem setup complete"
    # Set a dummy objective (minimize sum of λ)
    dummy_objective = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), λ), 0.0)
    MOI.set(model, MOI.ObjectiveFunction{typeof(dummy_objective)}(), dummy_objective)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # @info "Direct solve problem setup + objective set"
    MOI.optimize!(model)
    # @info "Direct solve problem solved"
    # 4. Check if solve was feasible and update active set weights
    if MOI.get(model, MOI.TerminationStatus()) in [MOI.OPTIMAL, MOI.FEASIBLE_POINT]
        # @info "Direct solve successful"
        # λ_values = MOI.get.(model, MOI.VariablePrimal(), λ) # seems to lead to stack overflow // using explicit loop instead // does not seem to fix the problem
        λ_values = Vector{Float64}(undef, n)
        for i in 1:n
            λ_values[i] = MOI.get(model, MOI.VariablePrimal(), λ[i])
        end
        new_weights = λ_values
        return ActiveSet([(new_weights[i], active_set.atoms[i]) for i in 1:n])
    else
        # @info "Direct solve failed"
        return active_set
    end
end



"""
    sparsify_iterate(active_set, lp_solver)

Sparsify the current iterate of the active set using linear programming.

This function attempts to find a sparser representation of the current iterate
by solving a linear program. It aims to reduce the number of non-zero weights
in the active set while maintaining the same iterate.

# Arguments
- `active_set`: The current ActiveSet object containing atoms and their weights.
- `lp_solver`: A linear programming solver compatible with MathOptInterface.

# Returns
- A new ActiveSet object with potentially fewer non-zero weights, representing
  the same iterate as the input active set.

# Details
The function performs the following steps:
1. Extracts the current iterate from the active set.
2. Constructs a matrix A where each column is an atom from the active set.
3. Sets up and solves a linear program to find new weights that:
   - Are non-negative
   - Sum to 1
   - Reproduce the current iterate when multiplied with the atoms
4. If a feasible solution is found, creates a new ActiveSet with the new weights.
   Otherwise, returns the original active set.

Usage: 
- Set `sparsify = true` when calling `blended_pairwise_conditional_gradient`
"""
function sparsify_iterate(active_set, lp_solver)
    # 1. Get current iterate
    x0 = get_active_set_iterate(active_set)

    # 2. Generate matrix A with columns being the points in the active set
    if isempty(active_set.atoms)
        A = zeros(length(x0), 0)
    else
        # A = hcat(getindex.(active_set, 2)...) // this leads to stack overflow
        A = zeros(eltype(x0), length(x0), length(active_set))
        for (i, atom) in enumerate(active_set)
            A[:, i] = atom[2]
        end
    end
    # @info "Sparsification matrix A generated"

    # 3. Setup and solve the system using the provided LP solver
    n = size(A, 2)
    model = MOI.instantiate(lp_solver)
    MOI.set(model, MOI.Silent(), true)

    # Define variables
    λ = MOI.add_variables(model, n)
    for i in 1:n
        MOI.add_constraint(model, λ[i], MOI.GreaterThan(0.0))
    end

    # Add sum constraint
    sum_λ = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), λ), 0.0)
    MOI.add_constraint(model, sum_λ, MOI.EqualTo(1.0))

    # Add constraint A λ = x0
    Q = A
    b = x0
    for i in 1:size(Q, 1)
        terms = MOI.ScalarAffineTerm{Float64}[]
        for j in 1:size(Q, 2)
            if !iszero(Q[i,j])
                push!(terms, MOI.ScalarAffineTerm(Q[i,j], λ[j]))
            end
        end
        constraint = MOI.ScalarAffineFunction(terms, 0.0)
        MOI.add_constraint(model, constraint, MOI.EqualTo(b[i]))
    end
    # @info "Sparsification problem setup complete"
    # Set a dummy objective (minimize sum of λ)
    dummy_objective = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), λ), 0.0)
    MOI.set(model, MOI.ObjectiveFunction{typeof(dummy_objective)}(), dummy_objective)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # @info "Sparsification problem setup + objective set"
    MOI.optimize!(model)
    # @info "Sparsification problem solved"
    # 4. Check if solve was feasible and update active set weights
    if MOI.get(model, MOI.TerminationStatus()) in [MOI.OPTIMAL, MOI.FEASIBLE_POINT]
        # @info "Sparsification successful"
        # λ_values = MOI.get.(model, MOI.VariablePrimal(), λ) # seems to lead to stack overflow // using explicit loop instead // does not seem to fix the problem
        λ_values = Vector{Float64}(undef, n)
        for i in 1:n
            λ_values[i] = MOI.get(model, MOI.VariablePrimal(), λ[i])
        end
        new_weights = λ_values
        return ActiveSet([(new_weights[i], active_set.atoms[i]) for i in 1:n])
    else
        # @info "Sparsification failed"
        return active_set
    end
end