module FrankWolfeCudaExt

import CUDA
import FrankWolfe

function FrankWolfe.compute_extreme_point(
    lmo::Union{FrankWolfe.UnitHyperSimplexLMO,FrankWolfe.HyperSimplexLMO},
    direction::CUDA.CuVector{T};
    v=nothing,
    kwargs...,
) where {T}
    K = FrankWolfe._compute_k_hypersimplex(lmo, direction)
    K_indices = sort!(sortperm(direction)[1:K])
    return CUDA.CUSPARSE.CuSparseVector(
        K_indices,
        CUDA.fill(T(lmo.radius), K),
        length(direction),
    )
end


end
