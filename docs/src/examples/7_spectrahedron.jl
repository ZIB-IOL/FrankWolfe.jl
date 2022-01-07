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
n = 1500
k = 5000
n_entries = 1000

Random.seed!(41)

const entry_indices = unique!([minmax(rand(1:n, 2)...) for _ in 1:n_entries])
const entry_values = randn(length(entry_indices))

function f(X)
    r = zero(eltype(X))
    for (idx, (i, j)) in enumerate(entry_indices)
        r += 1/2 * (X[i,j] - entry_values[idx])^2
        r += 1/2 * (X[j,i] - entry_values[idx])^2
    end
    return r / length(entry_values)
end

function grad!(storage, X)
    storage .= 0
    for (idx, (i, j)) in enumerate(entry_indices)
        storage[i,j] += (X[i,j] - entry_values[idx])
        storage[j,i] += (X[j,i] - entry_values[idx])
    end
    storage ./= length(entry_values)
end

# Note that the `ensure_symmetry = false` argument to `SpectraplexLMO`.
# It skips an additional step making the used direction symmetric.
# It is not necessary when the gradient is a `LinearAlgebra.Symmetric` (or more rarely a `LinearAlgebra.Diagonal` or `LinearAlgebra.UniformScaling`).

const lmo = FrankWolfe.SpectraplexLMO(1.0, n, false)
const x0 = FrankWolfe.compute_extreme_point(lmo, spzeros(n, n))

target_tolerance = 1e-8;

#src the following two lines are used only to precompile the functions
FrankWolfe.frank_wolfe(f, grad!, lmo, x0, max_iteration=2, line_search=FrankWolfe.MonotonousStepSize()) #src
FrankWolfe.lazified_conditional_gradient(f, grad!, lmo, x0, max_iteration=2, line_search=FrankWolfe.MonotonousStepSize()) #src

# ## Running standard and lazified Frank-Wolfe

Xfinal, Vfinal, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.MonotonousStepSize(),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
    epsilon=target_tolerance,
)

Xfinal, Vfinal, primal, dual_gap, trajectory_lazy = FrankWolfe.lazified_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.MonotonousStepSize(),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
    epsilon=target_tolerance,
);

# ## Plotting the resulting trajectories

data = [trajectory, trajectory_lazy]
label = ["FW", "LCG"]
plot_trajectories(data, label, xscalelog=true)
