
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
        gradient = similar(x)
        grad!(gradient, x)
        v = compute_extreme_point(lmo, gradient)
        @timeit to "dual gap" begin
            dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
        end
    end
    @showprogress 1 "Testing update... (Emphasis: OutplaceEmphasis) " for i in 1:k
        x = x_gen()
        gradient = similar(x)
        grad!(gradient, x)
        v = compute_extreme_point(lmo, gradient)
        gamma = 1 / 2
        @timeit to "update (OutplaceEmphasis)" @memory_mode(OutplaceEmphasis(), x = (1 - gamma) * x + gamma * v)
    end
    @showprogress 1 "Testing update... (memory_mode: InplaceEmphasis) " for i in 1:k
        x = x_gen()
        gradient = similar(x)
        grad!(gradient, x)
        v = compute_extreme_point(lmo, gradient)
        gamma = 1 / 2
        # TODO: to be updated to broadcast version once data structure ScaledHotVector allows for it
        @timeit to "update (memory_mode)" @memory_mode(InplaceEmphasis(), x = (1 - gamma) * x + gamma * v)
    end
    if !nocache
        @showprogress 1 "Testing caching 100 points... " for i in 1:k
            @timeit to "caching 100 points" begin
                cache = [gen_x() for _ in 1:100]
                x = gen_x()
                gradient = similar(x)
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
function _unsafe_equal(a::AbstractArray, b::AbstractArray)
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
    return function push_trajectory!(data)
        return push!(storage, Tuple(data)[1:5])
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

ExpMomentumIterator() = ExpMomentumIterator(2/3, 4.0, 8.0, 0)

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
