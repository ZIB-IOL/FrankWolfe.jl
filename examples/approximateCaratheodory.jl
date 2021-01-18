
using FrankWolfe
using LinearAlgebra

n = Int(1e2);
k = n 

const f(x) = dot(x,x)
const grad(x) = 2 * (x) 

# pick feasible region
lmo = FrankWolfe.ProbabilitySimplexOracle{Rational{BigInt}}(1); # radius needs to be integer or rational

# compute some initial vertex
x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

# verify that the output is really rational
println("Output type of LMO: ", eltype(x0))

# benchmarking Oracles
FrankWolfe.benchmark_oracles(f,grad,lmo,n;k=100,T=Float64)

# the algorithm runs in rational arithmetic even if the gradients and the function itself are not rational
# this is because we replace the descent direction by the directions of the LMO are rational

@time x, v, primal, dualGap, trajectory = FrankWolfe.fw(f,grad,lmo,x0,maxIt=k,
    stepSize=FrankWolfe.agnostic,printIt=k/10,verbose=true);

println("\nOutput type of solution: ", eltype(x))

# you can even run everything in rational arithmetic using the shortstep rule
# NOTE: in this case the gradient computation has to be rational as well

@time x, v, primal, dualGap, trajectory = FrankWolfe.fw(f,grad,lmo,x0,maxIt=k,
    stepSize=FrankWolfe.rationalshortstep,L=2,printIt=k/10,verbose=true);

println("\nOutput type of solution: ", eltype(x))

println("\nNote: the last step where we exactly close the gap. This is not an error. ")
fract = 1//n
println("We have *exactly* computed the optimal solution with with the $fract * (1, ..., 1) vector.\n")
println("x = $x")
