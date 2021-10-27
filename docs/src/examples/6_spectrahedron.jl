# # Spectrahedron

# This example shows an example optimizing over the spectraplex:
# ```math
# S = \{X \in \mathbb{S}_+^n, Tr(X) = 1\}
# ```

using FrankWolfe
using LinearAlgebra
using Random
using SparseArrays

n = 500
k = 10000
n_entries = 50

Random.seed!(41)

const entry_indices = unique!([minmax(rand(1:n, 2)...) for _ in 1:n_entries])
const entry_values = randn(length(entry_indices))

function f(X)
    r = zero(eltype(X))
    for (idx, (i, j)) in enumerate(entry_indices)
        r += 1 / 2 * (X[i,j] - entry_values[idx])^2
        r += 1 / 2 * (X[j,i] - entry_values[idx])^2
    end
    return r
end

function grad!(storage, X)
    storage .= 0
    for (idx, (i, j)) in enumerate(entry_indices)
        storage[i,j] += (X[i,j] - entry_values[idx])
        storage[i,j] += (X[j,i] - entry_values[idx])
    end
end

const lmo = FrankWolfe.SpectraplexLMO(1.0, n)
const x0 = FrankWolfe.compute_extreme_point(lmo, spzeros(n, n))

target_tolerance = 1e-6

#src the following two lines are used only to precompile the functions
FrankWolfe.frank_wolfe(f, grad!, lmo, x0, max_iteration=2, line_search=FrankWolfe.MonotonousStepSize()) #src
FrankWolfe.lazified_conditional_gradient(f, grad!, lmo, x0, max_iteration=2, line_search=FrankWolfe.MonotonousStepSize()) #src

# Running standard and lazified Frank-Wolfe

Xfinal, Vfinal, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.MonotonousStepSize(),
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
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
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true,
    epsilon=target_tolerance,
)

data = [trajectory, trajectory_lazy]
label = ["FW", "LCG"]
FrankWolfe.plot_trajectories(data, label, xscalelog=true)
