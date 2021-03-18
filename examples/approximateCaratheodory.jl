
using FrankWolfe
using LinearAlgebra

n = Int(1e2);
k = n

f(x) = dot(x, x)
function grad!(storage, x)
    @. storage = 2 * x
end

# pick feasible region
lmo = FrankWolfe.ProbabilitySimplexOracle{Rational{BigInt}}(1); # radius needs to be integer or rational


# compute some initial vertex
x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

# verify that the output is rational
println("Output type of LMO: ", eltype(x0))

# benchmarking Oracles
FrankWolfe.benchmark_oracles(f, grad!, () -> rand(n), lmo; k=100)

# the algorithm runs in rational arithmetic even if the gradients and the function itself are not rational
# this is because we replace the descent direction by the directions of the LMO are rational

@time x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.agnostic,
    print_iter=k / 10,
    verbose=true,
    emphasis=FrankWolfe.blas
);

@time xmem, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.agnostic,
    print_iter=k / 10,
    verbose=true,
    emphasis=FrankWolfe.memory,
);


println("\nOutput type of solution (blas): ", eltype(x))
println("Output type of solution (memory): ", eltype(xmem))

# you can even run everything in rational arithmetic using the shortstep rule
# NOTE: in this case the gradient computation has to be rational as well

# compute some initial vertex
x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

@time x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.rationalshortstep,
    L=2,
    print_iter=k / 10,
    verbose=true,
    emphasis=FrankWolfe.memory
);

println("\nOutput type of solution: ", eltype(x))

println("\nNote: the last step where we exactly close the gap. This is not an error. ")
fract = 1 // n
println(
    "We have *exactly* computed the optimal solution with with the $fract * (1, ..., 1) vector.\n",
)
println("x = $x")
