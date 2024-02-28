abstract type LineSearchMethod end

"""
Computes step size: `l/(l + t)` at iteration `t`, given `l > 0`.

Using `l ≥ 4` is advised only for strongly convex sets, see:
> Acceleration of Frank-Wolfe Algorithms with Open-Loop Step-Sizes, Wirth, Kerdreux, Pokutta, (2023), https://arxiv.org/abs/2205.12838

Fixing l = -1, results in the step size gamma_t = (2 + log(t+1)) / (t + 2 + log(t+1))
# S. Pokutta "The Frank-Wolfe algorith: a short introduction" (2023), https://arxiv.org/abs/2311.05313
"""
struct GeneralizedAgnostic{T<:Real, F<:Function} <: LineSearchMethod
    l::F
end

function GeneralizedAgnostic()
    l(x) = 2 + log(x + 1)
    return GeneralizedAgnostic{Float64, typeof(l)}(l)
end


function perform_line_search(
    ls::GeneralizedAgnostic{T, F},
    t,
    f,
    g!,
    gradient,
    x,
    d,
    gamma_max,
    workspace,
    memory_mode::MemoryEmphasis,
) where {T, F}
    return
        T(ls.l(t) / (t + ls.l(t)))
    end
end
