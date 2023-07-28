using FrankWolfe
using ProgressMeter
using Arpack
using DoubleFloats
using ReverseDiff
using LinearAlgebra

n = Int(1e4)

xpi = rand(1:100, n)
total = sum(xpi)
xp = xpi .// total

f(x) = norm(x - xp)^2

function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

vertices = 2*rand(100, n) .- 1

lmoNB = FrankWolfe.ScaledBoundL1NormBall(-ones(n), ones(n))
lmo_ball = FrankWolfe.KNormBallLMO(5, 1.0)
lmo_sparse = FrankWolfe.KSparseLMO(100, 1.0)
lmoProb = FrankWolfe.ProbabilitySimplexOracle(1.0)
lmoConv = FrankWolfe.ConvHull(vertices)

lmo_pairs = [
    (lmoProb, lmo_sparse),
    (lmoProb, lmo_ball),
    (lmoProb, lmoNB),
    (lmo_ball, lmoNB),
    (lmo_ball, lmoConv)
]

for pair in lmo_pairs

    @time FrankWolfe.alm(
        FrankWolfe.BCFW(
            verbose=true,
        ),
        f,
        grad!,
        pair,
        zeros(n),
        lambda=1.0,
    );
end 