
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

"""
Given an array `array`, `NegatingArray` represents `-1 * array` lazily.
"""
struct NegatingArray{T, N, AT <: AbstractArray{T,N}} <: AbstractArray{T, N}
    array::AT
    function NegatingArray(array::AT) where {T, N, AT <: AbstractArray{T,N}}
        return new{T, N, AT}(array)
    end
end

Base.size(a::NegatingArray) = Base.size(a.array)
Base.getindex(a::NegatingArray, idxs...) = -Base.getindex(a.array, idxs...)

LinearAlgebra.dot(a1::NegatingArray, a2::NegatingArray) = dot(a1.array, a2.array)
LinearAlgebra.dot(a1::NegatingArray{T1, N}, a2::AbstractArray{T2, N}) where {T1, T2, N} = -dot(a1.array, a2)
LinearAlgebra.dot(a1::AbstractArray{T1, N}, a2::NegatingArray{T2, N}) where {T1, T2, N} = -dot(a1, a2.array)
Base.sum(a::NegatingArray) = -sum(a.array)

function weight_purge_threshold_default(::Type{T}) where {T<:AbstractFloat}
    return sqrt(eps(T) * Base.rtoldefault(T)) # around 1e-12 for Float64
end
weight_purge_threshold_default(::Type{T}) where {T<:Number} = Base.rtoldefault(T)

"""
    Computes the linear minimizer in the direction on the PrecomputedSet.
"""
function pre_computed_set_argminmax(pre_computed_set, direction)
    val = Inf
    valM = -Inf
    idx = -1
    idxM = -1
    for i in eachindex(pre_computed_set)
        temp_val = fast_dot(pre_computed_set[i], direction)
        if temp_val < val
            val = temp_val
            idx = i
        end
        if valM < temp_val
            valM = temp_val
            idxM = i
        end
    end
    if idx == -1 || idxM == -1
        error("Infinite minimum $val or maximum $valM in the PrecomputedSet. Does the gradient contain invalid (NaN / Inf) entries?")
    end
    v_local = pre_computed_set[idx]
    a_local = pre_computed_set[idxM]
    return (v_local, idx, val, a_local, idxM, valM)
end
