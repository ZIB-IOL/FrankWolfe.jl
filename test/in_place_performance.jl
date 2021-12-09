using TimerOutputs
using LinearAlgebra

to = TimerOutput()

f(x) = norm(x)^2
function grad!(storage, x)
    return storage .= 2x
end

lmos = [
        FrankWolfe.ScaledBoundLInfNormBall(-ones(10), 2 * ones(10)),
        FrankWolfe.ScaledBoundLInfNormBall(-2 * ones(10), 2 * ones(10)),
        FrankWolfe.ScaledBoundLInfNormBall(-ones(100), 2 * ones(100)),
        FrankWolfe.ScaledBoundLInfNormBall(-2 * ones(100), 2 * ones(100)),
        FrankWolfe.ScaledBoundL1NormBall(-ones(10), 2 * ones(10)),
        FrankWolfe.ScaledBoundL1NormBall(-2 * ones(10), 2 * ones(10)),
        FrankWolfe.KNormBallLMO(5, 5.0),
        FrankWolfe.KNormBallLMO(10, 5.0),
        FrankWolfe.KNormBallLMO(10, 5.0),
        FrankWolfe.LpNormLMO{1.5}(5.0),
        FrankWolfe.LpNormLMO{1.5}(50.0),
        FrankWolfe.LpNormLMO{1.5}(50.0),
        FrankWolfe.LpNormLMO{2}(5.0),
        FrankWolfe.LpNormLMO{2}(50.0),
        FrankWolfe.LpNormLMO{2}(50.0),
        FrankWolfe.LpNormLMO{Inf}(5.0),
        FrankWolfe.LpNormLMO{Inf}(50.0),
        FrankWolfe.LpNormLMO{Inf}(50.0),
        ]
d =    [
        zeros(10),
        zeros(10),
        zeros(100),
        zeros(100),
        zeros(10),
        zeros(10),
        zeros(10),
        zeros(10),
        zeros(100),
        zeros(10),
        zeros(10),
        zeros(100),
        zeros(10),
        zeros(10),
        zeros(100),
        zeros(10),
        zeros(10),
        zeros(100),
        ]

for i in 1:length(d)
        FrankWolfe.frank_wolfe(f, grad!, lmos[i], FrankWolfe.compute_extreme_point(lmos[i], d[i]))
        @timeit to "Iteration " * string(i) FrankWolfe.frank_wolfe(f, grad!, lmos[i], FrankWolfe.compute_extreme_point(lmos[i], d[i]))
end
