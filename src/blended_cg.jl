
"""
    simplex_gradient_descent(active_set::ActiveSet, direction, f)

Performs a Simplex Gradient Descent step and modifies `active_set`.

Algorithm reference and notation taken from:
Blended Conditional Gradients:The Unconditioning of Conditional Gradients
http://proceedings.mlr.press/v97/braun19a/braun19a.pdf
"""
function update_simplex_gradient_descent!(active_set::ActiveSet, direction, f, L=nothing)
    linesearch_method = L === nothing ? backtracking : shortstep
    c = [dot(direction, a) for a in active_set]
    k = length(active_set)
    csum = sum(c)
    c .-= (csum / k)
    ActiveSet
    # name change to stay consistent with the paper
    d = c
    if norm(c) <= 1e-5
        # reset x and S
        return
    end
    η = eltype(d)(Inf)
    remove_idx = -1
    @inbounds for idx in eachindex(d)
        if d[idx] ≥ 0
            η = min(
                η,
                active_set.weights[idx] / d[idx]
            )
        end
    end
    η = max(0, η)
    # TODO do not materialize previous point.
    x = compute_active_set_iterate(active_set)
    @. active_set.weights -= η * d
    if !active_set_validate(active_set)
        error("""
        Eta computation error?
        $η\n$d\nactive_set.weights
        """)
    end
    f_y = f(compute_active_set_iterate(active_set))
    if f(x) ≥ f_y
        active_set_cleanup!(active_set)
        @assert(length(active_set) == length(x) - 1)
        return active_set
    end
    # TODO move η between x and y till opt

    return active_set
end
