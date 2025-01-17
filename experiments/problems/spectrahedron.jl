# # Spectrahedron
#
# This example shows an optimization problem over the spectraplex:
# ```math
# S = \{X \in \mathbb{S}_+^n, Tr(X) = 1\}
# ```
# with $\mathbb{S}_+^n$ the set of positive semidefinite matrices.
# Linear optimization with symmetric objective $D$ over the spetraplex consists in computing the
# leading eigenvector of $D$.
#
# The package also exposes `UnitSpectrahedronLMO` which corresponds to the feasible set:
# ```math
# S_u = \{X \in \mathbb{S}_+^n, Tr(X) \leq 1\}
# ```

using FrankWolfe
using LinearAlgebra
using Random
using SparseArrays

# The objective function will be the symmetric squared distance to a set of known or observed entries $Y_{ij}$ of the matrix.
# ```math
# f(X) = \sum_{(i,j) \in L} 1/2 (X_{ij} - Y_{ij})^2
# ```

# ## Setting up the input data, objective, and gradient

# Dimension, number of iterations and number of known entries:

function build_spectrahedron(seed, n)
    Random.seed!(seed)
    n_entries = Int(floor(2*n/3))

    entry_indices = unique!([minmax(rand(1:n, 2)...) for _ in 1:n_entries])
    entry_values = randn(length(entry_indices))
    function f(X)
        r = zero(eltype(X))
        for (idx, (i, j)) in enumerate(entry_indices)
            r += 1 / 2 * (X[i, j] - entry_values[idx])^2
            r += 1 / 2 * (X[j, i] - entry_values[idx])^2
        end
        return r / length(entry_values)
    end
    
    function grad!(storage, X)
        storage .= 0
        for (idx, (i, j)) in enumerate(entry_indices)
            storage[i, j] += (X[i, j] - entry_values[idx])
            storage[j, i] += (X[j, i] - entry_values[idx])
        end
        return storage ./= length(entry_values)
    end


    lmo = FrankWolfe.SpectraplexLMO(1.0, n, false)
    x0 = FrankWolfe.compute_extreme_point(lmo, spzeros(n, n))
    active_set = FrankWolfe.ActiveSet([(1.0, x0)])

    return f, grad!, lmo, x0, active_set, x -> true, n^2
end

