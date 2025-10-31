
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
            dual_gap = dot(gradient, x) - dot(gradient, v)
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
                test = (x -> dot(gradient, x)).(cache)
                v = cache[argmin(test)]
                val = v in cache
            end
        end
    end
    print_timer(to)
    return nothing
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
