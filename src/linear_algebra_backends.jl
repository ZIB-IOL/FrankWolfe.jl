
"""
    LinearAlgebraBackend

Defines the backend to use to perform linear algebra subroutines for the spectral LMOs,
including singular value decompositions, eigenvalue decompositions, etc.
"""
abstract type LinearAlgebraBackend end

struct ArpackBackend <: LinearAlgebraBackend
    tol::Float64
    maxiter::Int
end

ArpackBackend() = ArpackBackend(1e-8, 500)

"""
    KrylovKitBackend

LinearAlgebraBackend based on KrylovKit.
The methods are implemented in a package extension only and available only if KrylovKit is loaded.
"""
struct KrylovKitBackend{T} <: FrankWolfe.LinearAlgebraBackend
    tol::T
    maxiter::Int
end

KrylovKitBackend() = KrylovKitBackend(1e-8, 500)