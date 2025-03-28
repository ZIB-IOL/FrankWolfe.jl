using FrankWolfe


#Returns a method of half the squared euclidean distance from a fixed point b.
function quadratic_objective(b::Vector{<:Real})
    return p -> 0.5 * sum(abs2,p .- b)
end


#FOO for euclidean distance from a fixed point b
function foo_euclideandistance(b::Vector{<:Real})
    function grad!(storage,p)
        storage .= p .- b
    end
    return grad!
end


n = 10
T = 50
b = rand(n)
lmo = FrankWolfe.ProbabilitySimplexOracle(1.)
cndgrad_params = FrankWolfe.FixedLZParameters(t -> 1/t, t -> 1000/(T*t) )
cndgrado = FrankWolfe.LanZhouProcedure(Inf,10000,lmo)
grad! = foo_euclideandistance(b)
momentum_stepsize = FrankWolfe.FixedMStepsize(t -> 2/(t+1))
stop_rule = FrankWolfe.IterStop(T)
x0 = zeros(n)
x0[1] = 1

res = FrankWolfe.conditional_gradient_sliding(
    grad!,
    cndgrado,
    momentum_stepsize,
    cndgrad_params,
    stop_rule,
    x0;
    max_iteration=10000,
    print_iter=1000,
    trajectory=true,
    verbose=true,
    traj_data=[],
    timeout=Inf
);


#https://proceedings.mlr.press/v48/martins16
function project_simplex(z, r::Real = 1.)
    z_sorted = sort(z,rev=true)
    z_cumul = sum(z)
    n = length(z)
    tau = zero(r)
    for k in n:-1:1
        if z_sorted[k]  - (z_cumul - r)/k > zero(r)
            tau = (z_cumul - r)/k
            break
        end
        z_cumul -= z_sorted[k]
     end
    z_proj = max.(z .- tau, zero(r))
    return z_proj
end

proj = project_simplex(b);

f=  quadratic_objective(b)
fw_proj ,_ = frank_wolfe(f,grad!,lmo,x0; verbose = false);

@info "rel dist1 proj VS GS" sum(abs.(res.x -proj))/sum(abs.(proj))
@info "rel dist1 fw_proj VS GS" sum(abs.(res.x - fw_proj))/sum(abs.(fw_proj))