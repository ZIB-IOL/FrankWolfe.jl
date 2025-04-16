using FrankWolfe
using ProfileView

#Returns a method of half the squared euclidean distance from a fixed point b.
function quadratic_objective(b)
    sqnormb = dot(b,b)
    return p -> 0.5 * dot(p,p) - dot(p,b) + 0.5 * sqnormb 
end


#FOO for euclidean distance from a fixed point b
function foo_euclideandistance(b)
    function grad!(storage,p)
        storage .= p .- b
    end
    return grad!
end


n = 100000
T = 50.0
b = rand(n)
lmo = FrankWolfe.ProbabilitySimplexOracle(1.)
cndgrad_params = FrankWolfe.LZCnGDParameters(2.0,T)
cndgrado = FrankWolfe.LanZhouProcedure(Inf,10000,lmo)
grad! = foo_euclideandistance(b)
momentum_stepsize = FrankWolfe.LZMomentStepsize()
x0 = zeros(n)
x0[1] = 1

#ProfileView.@profview 
res = FrankWolfe.conditional_gradient_sliding(
    grad!,
    cndgrado,
    x0;
    momentum_stepsize = momentum_stepsize,
    cndgrad_params = cndgrad_params,
    max_iteration=T,
    print_iter=1000,
    trajectory=true,
    verbose=true,
    traj_data=[],
    timeout=Inf
);


#=
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
=#