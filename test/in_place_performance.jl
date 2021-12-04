using TimerOutputs
using LinearAlgebra

to = TimerOutput()

f(x) = norm(x)^2
function grad!(storage, x)
    return storage .= 2x
end

lmos = [FrankWolfe.ProbabilitySimplexOracle(1),
        FrankWolfe.ProbabilitySimplexOracle(1),
        FrankWolfe.ProbabilitySimplexOracle(1),
        FrankWolfe.ScaledBoundLInfNormBall(-ones(10), 2 * ones(10)),
        FrankWolfe.ScaledBoundLInfNormBall(-2 * ones(10), 2 * ones(10)),
        FrankWolfe.ScaledBoundL1NormBall(-ones(10), 2 * ones(10)),
        FrankWolfe.ScaledBoundL1NormBall(-2 * ones(10), 2 * ones(10)),
        FrankWolfe.KNormBallLMO(5, 2 * pi),
        FrankWolfe.KNormBallLMO(10, 2 * pi)
        ]
d =    [zeros(5),
        zeros(50),
        zeros(200),
        zeros(10),
        zeros(10),
        zeros(10),
        zeros(10),
        zeros(10),
        zeros(10)]

for i in 1:length(d)
        FrankWolfe.frank_wolfe(f, grad!, lmos[i], FrankWolfe.compute_extreme_point(lmos[i], d[i]))
        @timeit to "Iteration " * string(i) FrankWolfe.frank_wolfe(f, grad!, lmos[i], FrankWolfe.compute_extreme_point(lmos[i], d[i]))
end
