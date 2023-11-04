# # Extra-lazification

# Sometimes the Frank-Wolfe algorithm will be run multiple times
# with slightly different settings under which vertices collected
# in a previous run are still valid.

# The extra-lazification feature can be used for this purpose.
# It consists of a storage that can collect dropped vertices during a run,
# and the ability to use these vertices in another run, when they are not part
# of the current active set.
# The vertices that are part of the active set do not need to be duplicated in the extra-lazification storage.
# The extra-vertices can be used instead of calling the LMO when it is a relatively expensive operation.

using FrankWolfe
using Test
using LinearAlgebra

# We will use a parameterized objective function ``1/2 \|x - c\|^2``
# over the unit simplex.

const n = 100
const center0 = 5.0 .+ 3 * rand(n)
f(x) = 0.5 * norm(x .- center0)^2
function grad!(storage, x)
    return storage .= x .- center0
end

# The `TrackingLMO` will let us count how many real calls to the LMO are performed
# by a single run of the algorithm.

lmo = FrankWolfe.UnitSimplexOracle(4.3)
tlmo = FrankWolfe.TrackingLMO(lmo)
x0 = FrankWolfe.compute_extreme_point(lmo, randn(n));

# ## Adding a vertex storage

# `FrankWolfe` offers a simple `FrankWolfe.DeletedVertexStorage` storage type
# which has as parameter `return_kth`, the number of good directions to find before returning the best.
# `return_kth` larger than the number of vertices means that the best-aligned vertex will be found.
# `return_kth = 1` means the first acceptable vertex (with the specified threhsold) is returned.
#
# See [FrankWolfe.DeletedVertexStorage](@ref)

vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(x0)[], 5)
tlmo.counter = 0

results = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    tlmo,
    x0,
    max_iteration=4000,
    verbose=true,
    lazy=true,
    epsilon=1e-5,
    add_dropped_vertices=true,
    extra_vertex_storage=vertex_storage,
)

# The counter indicates the number of initial calls to the LMO.
# We will now construct different objective functions based on new centers,
# call the BPCG algorithm while accumulating vertices in the storage,
# in addition to warm-starting with the active set of the previous iteration.
# This allows for a "double-warmstarted" algorithm, reducing the number of LMO
# calls from one problem to the next.

active_set = results[end]
tlmo.counter

for iter in 1:10
    center = 5.0 .+ 3 * rand(n)
    f_i(x) = 0.5 * norm(x .- center)^2
    function grad_i!(storage, x)
        return storage .= x .- center
    end
    tlmo.counter = 0
    FrankWolfe.blended_pairwise_conditional_gradient(
        f_i,
        grad_i!,
        tlmo,
        active_set,
        max_iteration=4000,
        lazy=true,
        epsilon=1e-5,
        add_dropped_vertices=true,
        use_extra_vertex_storage=true,
        extra_vertex_storage=vertex_storage,
        verbose=false,
    )
    @info "Number of LMO calls in iter $iter: $(tlmo.counter)"
    @info "Vertex storage size: $(length(vertex_storage.storage))"
end
