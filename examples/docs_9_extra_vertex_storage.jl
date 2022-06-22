# # Extra-lazification

using FrankWolfe
using Test
using LinearAlgebra
using FrankWolfe: ActiveSet

const center0 = 5.0 .+ 3 * rand(n)
f(x) = 0.5 * norm(x .- center0)^2
function grad!(storage, x)
    return storage .= x .- center0
end

lmo = FrankWolfe.UnitSimplexOracle(4.3)
tlmo = FrankWolfe.TrackingLMO(lmo)

n = 100
x0 = FrankWolfe.compute_extreme_point(lmo, randn(n))

results = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    tlmo,
    x0,
    max_iteration=4000,
    verbose=true,
    lazy=true,
    epsilon=1e-5,
)

active_set = results[end];

# The counter for the number of LMO calls in the first run.
tlmo.counter

# ## Warm-starting on a similar problem
# We change the point projected to the polytope at every iteration to a different one
# and reuse the active set of the previous iteration.
for iter in 1:10
    center = 5.0 .+ 3 * rand(n)
    f_i(x) = 0.5 * norm(x .- center)^2
    @info "Distance of solution to new center: $(f_i(active_set.x))"
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
    )
    @info "Number of LMO calls in iter $iter: $(tlmo.counter)"
end

# Adding a vertex storage

vertex_storage = Vector{typeof(x0)}()
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

active_set = results[end]
tlmo.counter

for iter in 1:10
    center = 5.0 .+ 3 * rand(n)
    f_i(x) = 0.5 * norm(x .- center)^2
    @info "Distance of solution to new center: $(f_i(active_set.x))"
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
        verbose=true,
    )
    @info "Number of LMO calls in iter $iter: $(tlmo.counter)"
end
