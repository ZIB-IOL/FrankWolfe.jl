module FrankWolfeKrylovKitExt

import FrankWolfe
import KrylovKit
using LinearAlgebra

function FrankWolfe.compute_extreme_point(
    lmo::FrankWolfe.NuclearNormBallLMO{TL,<:FrankWolfe.KrylovKitBackend},
    direction::AbstractMatrix{TD};
    kwargs...,
) where {TL,TD}
    T = promote_type(TD, TL)
    # handle zero matrix to avoid LAPACK exceptions
    if norm(direction) <= lmo.backend.tol
        return FrankWolfe.RankOneMatrix(fill(T(lmo.radius) / sqrt(length(direction)), size(direction, 1)), ones(T, size(direction, 2)))
    end
    svd_res = KrylovKit.svdsolve(direction, 1, tol=lmo.backend.tol, maxiter=lmo.backend.maxiter)
    u = -lmo.radius * svd_res[2][1]
    return FrankWolfe.RankOneMatrix(u::Vector{T}, svd_res[3][1] * one(T))
end

end
