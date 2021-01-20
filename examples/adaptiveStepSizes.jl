import FrankWolfe
import LinearAlgebra


n = Int(1e5)
k = 1000

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

f(x) = LinearAlgebra.norm(x - xp)^2
grad(x) = 2 * (x - xp)

# better for memory consumption as we do coordinate-wise ops

function cf(x, xp)
    return @. LinearAlgebra.norm(x - xp)^2
end

function cgrad(x, xp)
    return @. 2 * (x - xp)
end

lmo = FrankWolfe.KSparseLMO(40, 1);
x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

FrankWolfe.benchmark_oracles(x -> cf(x, xp), x -> cgrad(x, xp), lmo, n; k=100, T=Float64)

println("\n==> Short Step rule - if you know L.\n")

@time x, v, primal, dualGap, trajectory = FrankWolfe.fw(
    f,
    grad,
    lmo,
    x0,
    maxIt=k,
    step_size=FrankWolfe.shortstep,
    L=2,
    printIt=k / 10,
    emph=FrankWolfe.memory,
    verbose=true,
);

println("\n==> Adaptive if you do not know L.\n")

@time x, v, primal, dualGap, trajectory = FrankWolfe.fw(
    f,
    grad,
    lmo,
    x0,
    maxIt=k,
    step_size=FrankWolfe.adaptive,
    L=100,
    printIt=k / 10,
    emph=FrankWolfe.memory,
    verbose=true,
);

println("\n==> Agnostic if function is too expensive for adaptive.\n")

@time x, v, primal, dualGap, trajectory = FrankWolfe.fw(
    f,
    grad,
    lmo,
    x0,
    maxIt=k,
    step_size=FrankWolfe.agnostic,
    printIt=k / 10,
    emph=FrankWolfe.memory,
    verbose=true,
);
