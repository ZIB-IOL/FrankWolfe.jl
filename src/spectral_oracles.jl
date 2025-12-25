
"""
    NuclearNormBallLMO{T,LinearAlgebraBackend,NT}(radius)

LMO over matrices that have a nuclear norm less than `radius`.
The LMO returns the best rank-one approximation matrix with singular value `radius`, computed with Arpack.
`T` is the type of the radius. 
`LinearAlgebraBackend` is a symbol indicating the linear algebra backend.
`NT` is the type of the backend storage.
The backends implemented in FrankWolfe or extensions use named tuples, external packages can define their own type.
"""
struct NuclearNormBallLMO{T,LinearAlgebraBackend,NT} <: LinearMinimizationOracle
    radius::T
    backend::NT
end

_default_linearalgebra_backend_params(::Val{:Arpack}) = (tol=1e-8, maxiter=400)
_default_linearalgebra_backend_params(::Val{:StdLib}) = (;)

NuclearNormBallLMO{T}() where {T} = NuclearNormBallLMO{T,:Arpack}(one(T), _default_linearalgebra_backend_params(Val{:Arpack}()))
NuclearNormBallLMO() = NuclearNormBallLMO{Float64,:Arpack}(1.0, _default_linearalgebra_backend_params(Val{:Arpack}()))

function NuclearNormBallLMO(radius::T) where {T}
    backend_params = _default_linearalgebra_backend_params(Val{:Arpack}())
    NuclearNormBallLMO{T, :Arpack, typeof(backend_params)}(radius, backend_params)
end

function NuclearNormBallLMO{T,LinearAlgebraBackend}(radius, backend::NT) where {T, NT, LinearAlgebraBackend}
    return NuclearNormBallLMO{T,LinearAlgebraBackend,NT}(T(radius), backend)
end

function compute_extreme_point(
    lmo::NuclearNormBallLMO{TL,:Arpack},
    direction::AbstractMatrix{TD};
    kwargs...,
) where {TL,TD}
    T = promote_type(TD, TL)
    Z = Arpack.svds(direction, nsv=1, tol=lmo.backend.tol, maxiter=lmo.backend.maxiter)[1]
    u = -lmo.radius * view(Z.U, :)
    return RankOneMatrix(u::Vector{T}, Z.V[:]::Vector{T})
end

function convert_mathopt(
    lmo::NuclearNormBallLMO,
    optimizer::OT;
    row_dimension::Integer,
    col_dimension::Integer,
    use_modify=true::Bool,
    kwargs...,
) where {OT}
    MOI.empty!(optimizer)
    x = MOI.add_variables(optimizer, row_dimension * col_dimension)
    (t, _) = MOI.add_constrained_variable(optimizer, MOI.LessThan(lmo.radius))
    MOI.add_constraint(optimizer, [t; x], MOI.NormNuclearCone(row_dimension, col_dimension))
    return MathOptLMO(optimizer, use_modify)
end

"""
    SpectraplexLMO{T,M,LinearAlgebraBackend}(radius::T, gradient_container::M, ensure_symmetry::Bool=true)

Feasible set
```
{X ∈ 𝕊_n^+, trace(X) == radius}
```
`gradient_container` is used to store the symmetrized negative direction.
`ensure_symmetry` indicates whether the linear function is made symmetric before computing the eigenvector.
`LinearAlgebraBackend` is a symbol indicating the backend to use
"""
struct SpectraplexLMO{T,M,LinearAlgebraBackend,NT} <: LinearMinimizationOracle
    radius::T
    gradient_container::M
    ensure_symmetry::Bool
    backend::NT
end

function SpectraplexLMO(
    radius::T,
    side_dimension::Int,
    ensure_symmetry::Bool=true,
    maxiter::Int=500,
    tol=1e-8,
) where {T}
    backend_params = (; tol, maxiter)
    return SpectraplexLMO{T, Matrix{T}, :Arpack, typeof(backend_params)}(
        radius,
        Matrix{T}(undef, side_dimension, side_dimension),
        ensure_symmetry,
        backend_params,
    )
end

function SpectraplexLMO(
    radius::Integer,
    side_dimension::Int,
    ensure_symmetry::Bool=true,
    maxiter::Int=500,
    tol=1e-8,
)
    return SpectraplexLMO(
        float(radius),
        side_dimension,
        ensure_symmetry,
        maxiter,
        tol,
    )
end

function compute_extreme_point(
    lmo::SpectraplexLMO{T,<:Any,:Arpack},
    direction::M;
    v=nothing,
    kwargs...,
) where {T,M<:AbstractMatrix}
    lmo.gradient_container .= direction
    if !(M <: Union{LinearAlgebra.Symmetric,LinearAlgebra.Diagonal,LinearAlgebra.UniformScaling}) &&
       lmo.ensure_symmetry
        # make gradient symmetric
        @. lmo.gradient_container += direction'
    end
    lmo.gradient_container .*= -1

    _, evec = Arpack.eigs(
        lmo.gradient_container;
        nev=1,
        which=:LR,
        maxiter=lmo.backend.maxiter,
        tol=lmo.backend.tol,
    )
    # type annotation because of Arpack instability
    unit_vec::Vector{T} = vec(evec)
    # scaling by sqrt(radius) so that x x^T has spectral norm radius while using a single vector
    unit_vec .*= sqrt(lmo.radius)
    return FrankWolfe.RankOneMatrix(unit_vec, unit_vec)
end

"""
    UnitSpectrahedronLMO{T,M,LAB}(radius::T, gradient_container::M, ensure_symmetry::Bool, backend::LAB)

Feasible set of PSD matrices with bounded trace:
```
{X ∈ 𝕊_n^+, trace(X) ≤ radius}
```
`gradient_container` is used to store the symmetrized negative direction.
`ensure_symmetry` indicates whether the linear function is made symmetric before computing the eigenvector.
"""
struct UnitSpectrahedronLMO{T,M,LinearAlgebraBackend,NT} <: LinearMinimizationOracle
    radius::T
    gradient_container::M
    ensure_symmetry::Bool
    backend::NT
end

function UnitSpectrahedronLMO(
    radius::T,
    side_dimension::Int,
    ensure_symmetry::Bool=true,
) where {T}
    backend_params = _default_linearalgebra_backend_params(Val{:Arpack}())
    return UnitSpectrahedronLMO{T,Matrix{T},:Arpack, typeof(backend_params)}(
        radius,
        Matrix{T}(undef, side_dimension, side_dimension),
        ensure_symmetry,
        backend_params,
    )
end

UnitSpectrahedronLMO(radius::Integer, side_dimension::Int) =
    UnitSpectrahedronLMO(float(radius), side_dimension)

function compute_extreme_point(
    lmo::UnitSpectrahedronLMO{T,<:Any,:Arpack},
    direction::M;
    v=nothing,
    kwargs...,
) where {T,M<:AbstractMatrix}
    lmo.gradient_container .= direction
    if !(M <: Union{LinearAlgebra.Symmetric,LinearAlgebra.Diagonal,LinearAlgebra.UniformScaling}) &&
       lmo.ensure_symmetry
        # make gradient symmetric
        @. lmo.gradient_container += direction'
    end
    lmo.gradient_container .*= -1

    e_val::Vector{T}, evec::Matrix{T} = Arpack.eigs(
        lmo.gradient_container;
        nev=1,
        which=:LR,
        maxiter=lmo.backend.maxiter,
        tol=lmo.backend.tol,
    )
    # type annotation because of Arpack instability
    unit_vec::Vector{T} = vec(evec)
    if e_val[1] < 0
        # return a zero rank-one matrix
        unit_vec .*= 0
    else
        # scaling by sqrt(radius) so that x x^T has spectral norm radius while using a single vector
        unit_vec .*= sqrt(lmo.radius)
    end
    return FrankWolfe.RankOneMatrix(unit_vec, unit_vec)
end

function convert_mathopt(
    lmo::Union{SpectraplexLMO{T},UnitSpectrahedronLMO{T}},
    optimizer::OT;
    side_dimension::Integer,
    use_modify::Bool=true,
    kwargs...,
) where {T,OT}
    MOI.empty!(optimizer)
    X = MOI.add_variables(optimizer, side_dimension * side_dimension)
    MOI.add_constraint(optimizer, X, MOI.PositiveSemidefiniteConeSquare(side_dimension))
    sum_diag_terms = MOI.ScalarAffineFunction{T}([], zero(T))
    # collect diagonal terms of the matrix
    for i in 1:side_dimension
        push!(sum_diag_terms.terms, MOI.ScalarAffineTerm(one(T), X[i+side_dimension*(i-1)]))
    end
    constraint_set = if lmo isa SpectraplexLMO
        MOI.EqualTo(lmo.radius)
    else
        MOI.LessThan(lmo.radius)
    end
    MOI.add_constraint(optimizer, sum_diag_terms, constraint_set)
    return MathOptLMO(optimizer, use_modify)
end

"""
    FantopeLMO(k::Int)

Spectrahedron defined on square symmetric matrices as:
`F_k = {X : 0 ≼ X ≼ I_n, tr(X) = k}` or equivalently as:
`F_k = conv{VVᵀ, VᵀV = I_k}`

Source: [V.Q. Vu, J. Cho, J. Lei, K. Rohe](https://papers.nips.cc/paper_files/paper/2013/hash/81e5f81db77c596492e6f1a5a792ed53-Abstract.html)
[Dattorro, Convex optimization & euclidean distance geometry, 2.3.2](https://web.stanford.edu/group/SOL/Books/0976401304.pdf).
"""
struct FantopeLMO{LinearAlgebraBackend,NT} <: FrankWolfe.LinearMinimizationOracle
    k::Int
    backend::NT
end

FantopeLMO(k::Int) = FantopeLMO{:StdLib,@NamedTuple{}}(k, (;))

function compute_extreme_point(
    lmo::FantopeLMO{:StdLib},
    direction::AbstractMatrix{T};
    kwargs...,
) where {T}
    @assert issymmetric(direction)
    n = size(direction, 1)
    eigen_info = eigen(direction)
    eigen_info.values .= 1
    if 1 <= lmo.k < n
        eigen_info.values[(lmo.k+1):end] .= 0
    end
    return eigen_info.vectors * Diagonal(eigen_info.values) * eigen_info.vectors'
end

function compute_extreme_point(lmo::FantopeLMO, direction::AbstractVector; kwargs...)
    n = isqrt(length(direction))
    V = compute_extreme_point(lmo, reshape(direction, n, n))
    return vec(V)
end

function convert_mathopt(
    lmo::FantopeLMO,
    optimizer::OT;
    side_dimension::Integer,
    use_modify::Bool=true,
) where {OT<:MOI.AbstractOptimizer}
    MOI.empty!(optimizer)
    X = MOI.add_variables(optimizer, side_dimension * side_dimension)
    MOI.add_constraint(optimizer, X, MOI.PositiveSemidefiniteConeSquare(side_dimension))
    sum_diag_terms = MOI.ScalarAffineFunction{Float64}([], 0.0)
    # collect diagonal terms of the matrix
    for i in 1:side_dimension
        push!(sum_diag_terms.terms, MOI.ScalarAffineTerm(1.0, X[i+side_dimension*(i-1)]))
    end
    # trace constraint
    MOI.add_constraint(optimizer, sum_diag_terms, MOI.EqualTo(1.0 * lmo.k))
    MOI.add_constraint(
        optimizer,
        MOI.VectorAffineFunction(
            vec(
                Matrix(1.0I, side_dimension, side_dimension) -
                1.0 * reshape(X, side_dimension, side_dimension),
            ),
        ),
        MOI.PositiveSemidefiniteConeSquare(side_dimension),
    )
    return MathOptLMO(optimizer, use_modify)
end
