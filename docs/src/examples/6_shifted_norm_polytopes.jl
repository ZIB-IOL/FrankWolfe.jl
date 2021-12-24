using FrankWolfe
using LinearAlgebra
using LaTeXStrings
using Plots

# # FrankWolfe for scaled, shifted ``\ell^1`` and ``\ell^{\infty}`` norm balls

# In this example, we run the vanilla FrankWolfe algorithm on a scaled and shifted ``\ell^1`` and ``\ell^{\infty}`` norm ball, using the `ScaledBoundL1NormBall`
# and `ScaledBoundLInfNormBall` LMOs. We shift both onto the point ``(1,0)`` and then scale them by a factor of ``2`` along the x-axis. We project the point ``(2,1)`` onto the polytopes.

n = 2

k = 1000

xp = [2.0,1.0]

f(x) = norm(x-xp)^2

function grad!(storage,x)
    @. storage = 2 * (x - xp)
    return nothing
end

lower = [-1.0,-1.0]
upper = [3.0,1.0]

l1 = FrankWolfe.ScaledBoundL1NormBall(lower, upper)

linf = FrankWolfe.ScaledBoundLInfNormBall(lower, upper)

x1 = FrankWolfe.compute_extreme_point(l1, zeros(n))
gradient = collect(x1)

x_l1, v_1, primal_1, dual_gap_1, trajectory_1 = FrankWolfe.frank_wolfe(
    f,
    grad!,
    l1,
    collect(copy(x1)),
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    print_iter=50,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
);

println("\nFinal solution: ", x_l1)

x2 = FrankWolfe.compute_extreme_point(linf, zeros(n))
gradient = collect(x2)

x_linf, v_2, primal_2, dual_gap_2, trajectory_2 = FrankWolfe.frank_wolfe(
    f,
    grad!,
    linf,
    collect(copy(x2)),
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    print_iter=50,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
);

println("\nFinal solution: ", x_linf)


# We plot the polytopes alongside the solutions from above:

xcoord1 = [1,3,1,-1,1]
ycoord1 = [-1,0,1,0,-1]

xcoord2 = [3,3,-1,-1,3]
ycoord2 = [-1,1,1,-1,-1]

plot(xcoord1, ycoord1, title = "Visualization of scaled shifted norm balls", lw = 2, label = L"\ell^1 \textrm{ norm}")
plot!(xcoord2, ycoord2, lw = 2, label = L"\ell^{\infty} \textrm{ norm}")
plot!([x_l1[1]], [x_l1[2]], seriestype = :scatter, lw = 5, color = "blue", label = L"\ell^1 \textrm{ solution}")
plot!([x_linf[1]], [x_linf[2]], seriestype = :scatter, lw = 5, color = "orange", label = L"\ell^{\infty} \textrm{ solution}", legend = :bottomleft)
