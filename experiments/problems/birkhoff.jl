using FrankWolfe
using LinearAlgebra
using Random

function build_birkhoff_problem(seed, n)
    Random.seed!(seed)
    xpi = rand(n, n)
    # total = sum(xpi)
    xp = xpi # / total
    normxp2 = dot(xp, xp)

    # better for memory consumption as we do coordinate-wise ops
    function cf(x)
        return (normxp2 - 2dot(x, xp) + dot(x, x)) / n^2
    end

    function cgrad!(storage, x)
        return @. storage = 2 * (x - xp) / n^2
    end

    # BirkhoffPolytopeLMO via Hungarian Method
    lmo = FrankWolfe.BirkhoffPolytopeLMO()

    # initial direction for first vertex
    direction_mat = randn(n, n)
    x0 = FrankWolfe.compute_extreme_point(lmo, direction_mat)
    active_set = FrankWolfe.ActiveSet([(1.0, x0)])

    return cf, cgrad!, lmo, x0, active_set, x -> true, n^2

end

