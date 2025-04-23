using FrankWolfe
using LinearAlgebra
using ProfileView

include("plot_utils.jl")


#Projection simplex
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


n = 100
b = rand(n)

f = quadratic_objective(b)
grad! = foo_euclideandistance(b)


function build_quadratic_approximation!(Hx,x,gradient,fx)
    Hx .= x
    return (storage,p)-> FrankWolfe.fast_dot(p,p), Hx
end



lmo = FrankWolfe.ProbabilitySimplexOracle(1.)
pvm_step = FrankWolfe.AwayStep(false)
fw_step = FrankWolfe.AwayStep(false)
x0 = zeros(n)
N = 50
x0[1] = 1

lb_estimator = FrankWolfe.LowerBoundFiniteSteps(f,grad!,lmo,fw_step,1)
res_0 = FrankWolfe.second_order_conditional_gradient_sliding(
    f,
    grad!,
    build_quadratic_approximation!,
    fw_step,
    lmo,
    pvm_step,
    lmo,
    x0;
    lb_estimator = lb_estimator,
    max_iteration=N,
    print_iter=1,
    trajectory=true,
    verbose=true,
    traj_data=[],
    timeout=Inf
);

lb_estimator = FrankWolfe.LowerBoundFiniteSteps(f,grad!,lmo,fw_step,1)
res_1 = FrankWolfe.second_order_conditional_gradient_sliding(
    f,
    grad!,
    build_quadratic_approximation!,
    fw_step,
    lmo,
    pvm_step,
    lmo,
    x0;
    lb_estimator = lb_estimator,
    max_iteration=N,
    print_iter=1,
    trajectory=true,
    verbose=true,
    traj_data=[],
    timeout=Inf
);

lb_estimator = FrankWolfe.LowerBoundFiniteSteps(f,grad!,lmo,fw_step,2)
res_2 = FrankWolfe.second_order_conditional_gradient_sliding(
    f,
    grad!,
    build_quadratic_approximation!,
    fw_step,
    lmo,
    pvm_step,
    lmo,
    x0;
    lb_estimator = lb_estimator,
    max_iteration=N,
    print_iter=1,
    trajectory=true,
    verbose=true,
    traj_data=[],
    timeout=Inf
);


lb_estimator = FrankWolfe.LowerBoundFiniteSteps(f,grad!,lmo,fw_step,3)
res_3 = FrankWolfe.second_order_conditional_gradient_sliding(
    f,
    grad!,
    build_quadratic_approximation!,
    fw_step,
    lmo,
    pvm_step,
    lmo,
    x0;
    lb_estimator = lb_estimator,
    max_iteration=N,
    print_iter=1,
    trajectory=true,
    verbose=true,
    traj_data=[],
    timeout=Inf
);


lb_estimator = FrankWolfe.LowerBoundFiniteSteps(f,grad!,lmo,fw_step,10)
res_10 = FrankWolfe.second_order_conditional_gradient_sliding(
    f,
    grad!,
    build_quadratic_approximation!,
    fw_step,
    lmo,
    pvm_step,
    lmo,
    x0;
    lb_estimator = lb_estimator,
    max_iteration=N,
    print_iter=1,
    trajectory=true,
    verbose=true,
    traj_data=[],
    timeout=Inf,
    line_search_fw=FrankWolfe.Adaptive(verbose=false),
    line_search_pvm=FrankWolfe.Adaptive(verbose=false),
    line_search_LB_estimator=FrankWolfe.Adaptive(verbose=false),
);


#data = [res_1.traj_data, res_2.traj_data, res_3.traj_data, res_10.traj_data]
#label = ["Est. 1 St." "Est. 2 St." "Est. 3 St." "Est. 10 St."]

#plot_trajectories(data, label, filename ="examples/figs/socgs_proj_simplex.pdf")


#comparison of result with other projection on simplex algorithm
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


#@info "rel dist1 proj VS GS" sum(abs.(res.x -proj))/sum(abs.(proj))
=#


#################################################################
###Sparse coding over the Birkhoff Polytope
#################################################################

function sqnorm_matrix(Y,Z)
    sqnorm_y = norm(Y)^2
    _,m = size(Y)
    return X -> 
            begin
            Xz = similar(@view Z[:,1])
            res = sqnorm_y 
                for i in 1:m
                    y = @view Y[:,i]
                    z = @view Z[:,i]
                    Xz .= X*z ;
                    res += FrankWolfe.fast_dot(Xz,Xz) - 2* FrankWolfe.fast_dot(Xz,y) 
                end
            return res
            end
end

function foo_grad_sqnorm_matrix(Y,Z)
    _,m = size(Y)
    function grad!(storage,X)
        storage .= zero(storage[1])
        for i in 1:m
            y = @view Y[:,i]
            z = @view Z[:,i] 
            @. storage = storage - 2*y + 2*X*z*z'
        end
    end
        
    return grad!
end


function quadratic_approximation_builder(Y,Z)
    
        n,m = size(Y)
        function build_quadratic_approximation!(HX,X,gradient,fX)
            HX .= zero(HX[1,1])
            for j in 1:m
                z = @view Z[:,j]
                @. HX = HX + X * z * z'           
            end
            function quadratic_term_function!(storage,P)
                res = zero(P[1,1])
                for j in 1:m
                    z = @view Z[:,j]
                    storage .= P*z 
                    res += FrankWolfe.fast_dot(storage,storage)
                end
            
                return res
            end
            return quadratic_term_function!, HX
        end
end

n = 20
m = 400
B = randn(n,n)
Z = rand(n,m)
Y = similar(Z)
for j in 1:m
    Y[:,j] = B * @view Z[:,j]
end

f = sqnorm_matrix(Y,Z)
grad! = foo_grad_sqnorm_matrix(Y,Z)

lmo = FrankWolfe.BirkhoffPolytopeLMO()
pvm_step = FrankWolfe.AwayStep(false)
fw_step = FrankWolfe.AwayStep(false)
lb_estimator = FrankWolfe.LowerBoundFiniteSteps(f,grad!,lmo,fw_step,1)
N = 50

x0 = diagm(ones(n))

res = FrankWolfe.second_order_conditional_gradient_sliding(
    f,
    grad!,
    quadratic_approximation_builder(Y,Z),
    fw_step,
    lmo,
    pvm_step,
    lmo,
    x0;
    lb_estimator = lb_estimator,
    max_iteration=N,
    print_iter=1,
    trajectory=true,
    verbose=true,
    traj_data=[],
    timeout=Inf,
    line_search_fw=FrankWolfe.Adaptive(verbose=false),
    line_search_pvm=FrankWolfe.Adaptive(verbose=false),
    line_search_LB_estimator=FrankWolfe.Adaptive(verbose=false)
);

#=

#################################################################
###Structured Logistic Regression over l1 unit ball
#################################################################
function build_logistic_loss(y,Z)
    n,m = size(Z)
    function logistic_loss(x)
        res = 0.5 * FrankWolfe.fast_dot(x,x)
        for j in 1:m 
            z = @view Z[:,j]
            res += log(1 + exp(-y[j] * FrankWolfe.fast_dot(x,z))  )
        end
        res /= m
        return res
    end
    return logistic_loss   
end

function build_logistic_grad(y,Z)
    n,m = size(Z)
    function logistic_grad!(storage,x)
        storage .= x
        for j in 1:m 
            z = @view Z[:,j]
            @. storage = storage - y[j] *z / (1 + exp(y[j] * FrankWolfe.fast_dot(x,z)))
        end
        storage /= m
    end   
    return logistic_grad! 
end


function quadratic_approximation_builder(Z)    
    n,m = size(Z)
    function build_quadratic_approximation!(Hx,x,gradient,fx)
        Hx .= x
        for j in 1:m
            z = @view Z[:,j]
            dot_x_z = FrankWolfe.fast_dot(x,z)
            denom = (1 + exp(FrankWolfe.fast_dot(x,z))) * (1 + exp(-FrankWolfe.fast_dot(x,z)))
            @. Hx = Hx + dot_x_z/denom * z
        end
        Hx /= m
        function quadratic_term_function!(storage,p)
            res = FrankWolfe.fast_dot(p,p)
            for j in 1:m
                z = @view Z[:,j]
                dot_p_z = FrankWolfe.fast_dot(x,z)                
                denom = (1 + exp(FrankWolfe.fast_dot(p,z))) * (1 + exp(-FrankWolfe.fast_dot(p,z)))
                res += dot_p_z^2/denom
            end
            res /= m
            return res
        end
        return quadratic_term_function!, Hx
    end
end


n = 100
m = 120

y = rand([-1,1],m)
Z = rand(n,m)


f = build_logistic_loss(y,Z)
grad! = build_logistic_grad(y,Z)

lmo = FrankWolfe.LpNormLMO{1}(1.0)
pvm_step = FrankWolfe.AwayStep(false)
fw_step = FrankWolfe.AwayStep(false)
lb_estimator = FrankWolfe.LowerBoundFiniteSteps(f,grad!,lmo,fw_step,1)
N = 50

x0 = zeros(n)
x0[1] = one(x0[1])

res = FrankWolfe.second_order_conditional_gradient_sliding(
    f,
    grad!,
    quadratic_approximation_builder(Z),
    fw_step,
    lmo,
    pvm_step,
    lmo,
    x0;
    lb_estimator = lb_estimator,
    max_iteration=N,
    print_iter=1,
    trajectory=true,
    verbose=true,
    traj_data=[],
    timeout=Inf,
    line_search_fw=FrankWolfe.Adaptive(verbose=false),
    line_search_pvm=FrankWolfe.Adaptive(verbose=false),
    line_search_LB_estimator=FrankWolfe.Adaptive(verbose=false)
);
=#