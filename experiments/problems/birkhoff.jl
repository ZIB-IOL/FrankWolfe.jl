using FrankWolfe
using LinearAlgebra
using Random

function build_birkhoff_problem(seed, n)
    Random.seed!(seed)
    xpi = rand(n, n)
    # total = sum(xpi)
    const xp = xpi # / total
    const normxp2 = dot(xp, xp)

    # better for memory consumption as we do coordinate-wise ops
    function cf(x, xp, normxp2)
        return (normxp2 - 2dot(x, xp) + dot(x, x)) / n^2
    end

    function cgrad!(storage, x, xp)
        return @. storage = 2 * (x - xp) / n^2
    end

    # BirkhoffPolytopeLMO via Hungarian Method
    lmo = FrankWolfe.BirkhoffPolytopeLMO()

    # initial direction for first vertex
    direction_mat = randn(n, n)
    x0 = FrankWolfe.compute_extreme_point(lmo, direction_mat)

    return cf, cgrad!, lmo, x0, [], x -> true

end

