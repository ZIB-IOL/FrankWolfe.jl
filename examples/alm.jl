using FrankWolfe

n = Int(1e4)

xpi = rand(1:100, n)
total = sum(xpi)
xp = xpi .// total

f(x) = FrankWolfe.fast_dot(x - xp, x - xp)

function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

vertices = 2 * rand(100, n) .- 1

lmo_nb = FrankWolfe.ScaledBoundL1NormBall(-ones(n), ones(n))
lmo_ball = FrankWolfe.KNormBallLMO(5, 1.0)
lmo_sparse = FrankWolfe.KSparseLMO(100, 1.0)
lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1.0)

lmo_pairs = [(lmo_prob, lmo_sparse), (lmo_prob, lmo_ball), (lmo_prob, lmo_nb), (lmo_ball, lmo_nb)]

for pair in lmo_pairs

    @time FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        f,
        grad!,
        pair,
        zeros(n);
        lambda=1.0,
        verbose=true,
    )
end
