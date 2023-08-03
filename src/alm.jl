"""
alm(bc_algo, f, grad!, lmos, x0; ...)

Alternating Linear Mnimizations Frank-Wolfe algorithm.
Returns a tuple `(x, v, primal, dual_gap, infeas, traj_data)` with:
- `x` cartesian product of final iterates
- `v` cartesian product of last vertices of the LMOs
- `primal` primal value `f(x)`
- `dual_gap` final Frank-Wolfe gap
- `infeas` sum of squared, pairwise distances between iterates 
- `traj_data` vector of trajectory information.
"""
function alternating_linear_minimization(
    bc_algo::BlockCoordinateMethod,
    f,
    grad!,
    lmos,
    x0;
    lambda=1.0,
)

    l = length(lmos)
    ndim = ndims(x0) + 1 # New product dimension
    prod_lmo = ProductLMO(lmos)
    x0_bc = cat(compute_extreme_point(prod_lmo, tuple([x0 for i in 1:l]...))..., dims=ndim)

    function grad_bc!(storage, x)
        gradf = similar(storage)
        for i in 1:l
            grad!(selectdim(gradf, ndim, i), selectdim(x, ndim, i))
        end
        t = lambda * 2.0 * (l * x .- sum(x, dims=ndim))
        @. storage = gradf + t
    end

    function f_bc(x)
        return sum([f(selectdim(x, ndim, i)) for i in 1:l]) +
               lambda * sum([
            fast_dot(
                selectdim(x, ndim, i) - selectdim(x, ndim, j),
                selectdim(x, ndim, i) - selectdim(x, ndim, j),
            ) for i in 1:l for j in 1:i-1
        ])
    end

    x, v, primal, dual_gap, infeas, traj_data =
        perform_bc_updates(bc_algo, f_bc, grad_bc!, prod_lmo, x0_bc)


    return x, v, primal, dual_gap, infeas, traj_data
end
