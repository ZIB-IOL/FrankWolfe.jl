module FrankWolfeKrylovKitExt

import FrankWolfe
import KrylovKit

function FrankWolfe.compute_extreme_point(
    lmo::NuclearNormBallLMO{TL,FrankWolfe.KrylovKitBackend},
    direction::AbstractMatrix{TD};
    kwargs...,
) where {TL,TD}
    T = promote_type(TD, TL)
    svd_res = KrylovKit.svdsolve(direction, 1, tol=lmo.backend.tol, maxiter=lmo.backend.maxiter)
    u = -lmo.radius * svd_res[2][1]
    return RankOneMatrix(u::Vector{T}, svd_res[3][1])
end

end
