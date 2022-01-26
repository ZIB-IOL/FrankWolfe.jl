# # Visualization of Frank-Wolfe running on a 2-dimensional polytope

# This example provides an intuitive view of the Frank-Wolfe algorithm
# by running it on a polyhedral set with a quadratic function.
# The Linear Minimization Oracle (LMO) corresponds to a call to a generic simplex solver from `MathOptInterface.jl` (MOI).

# ## Import and setup

# We first import the necessary packages, including Polyhedra to visualize the feasible set.

using LinearAlgebra
using FrankWolfe

import MathOptInterface
const MOI = MathOptInterface
using GLPK

using Polyhedra
using Plots

# We can then define the objective function,
# here the squared distance to a point in the place, and its in-place gradient.

n = 2
y = [3.2, 0.5]

function f(x)
    return 1 / 2 * norm(x - y)^2
end
function grad!(storage, x)
    @. storage = x - y
end

# ## Custom callback
#
# FrankWolfe.jl lets users define custom callbacks to record information about each iteration.
# In that case, the callback will copy the current iterate `x`, the current vertex `v`, and the current step size `gamma`
# to an array thanks to a closure.
# We then declare the array and the callback over this array.
# Each iteration will then push to this array.

function build_callback(trajectory_arr)
    return function callback(state)
        return push!(trajectory_arr, (copy(state.x), copy(state.v), state.gamma))
    end
end

iterates_information_vector = []
callback = build_callback(iterates_information_vector)

# ## Creating the Linear Minimization Oracle
# The LMO is defined as a call to a linear optimization solver, each iteration resets the objective and calls the solver.
# The linear constraints must be defined only once at the beginning and remain identical along iterations.
# We use here MathOptInterface directly but the constraints could also be defined with JuMP or Convex.jl.

o = GLPK.Optimizer()
x = MOI.add_variables(o, n)

## −x + y ≤ 2
c1 = MOI.add_constraint(
    o,
    -1.0x[1] + x[2],
    MOI.LessThan(2.0),
)

## x + 2 y ≤ 4
c2 = MOI.add_constraint(
    o,
    x[1] + 2.0x[2],
    MOI.LessThan(4.0),
)
        
## −2 x − y ≤ 1
c3 = MOI.add_constraint(
    o,
    -2.0x[1] - x[2],
    MOI.LessThan(1.0),
)
    
## x − 2 y ≤ 2
c4 = MOI.add_constraint(
    o,
    x[1] - 2.0x[2],
    MOI.LessThan(2.0),
)
    
## x ≤ 2
c5 = MOI.add_constraint(
    o,
    x[1] + 0.0x[2],
    MOI.LessThan(2.0),
)

# The LMO is then built by wrapping the current MOI optimizer

lmo_moi = FrankWolfe.MathOptLMO(o)

# ## Calling Frank-Wolfe
# We can now compute an initial starting point from any direction
# and call the Frank-Wolfe algorithm.
# Note that we copy `x0` before passing it to the algorithm because it is modified in-place by `frank_wolfe`.

x0 = FrankWolfe.compute_extreme_point(lmo_moi, zeros(n))

xfinal, vfinal, primal_value, dual_gap, traj_data = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo_moi,
    copy(x0),
    line_search = FrankWolfe.Adaptive(),
    max_iteration = 10,
    epsilon=1e-8,
    callback=callback,
    verbose=true,
    print_iter=1,
)

# We now collect the iterates and vertices across iterations.

iterates = Vector{Vector{Float64}}()
push!(iterates, x0)
vertices = Vector{Vector{Float64}}()
for s in iterates_information_vector
    push!(iterates, s[1])
    push!(vertices, s[2])
end

# ## Plotting the algorithm run

# We define another method for `f` adapted to plot its contours.

function f(x1, x2)
    x = [x1, x2]
    return f(x)
end

xlist = collect(range(-1, 3, step = 0.2))
ylist = collect(range(-1, 3, step = 0.2))

X = repeat(reshape(xlist, 1, :), length(ylist), 1)
Y = repeat(ylist, 1, length(xlist))

# The feasible space is represented using Polyhedra.

h = HalfSpace([-1, 1], 2) ∩
    HalfSpace([1, 2], 4) ∩
    HalfSpace([-2, -1], 1) ∩
    HalfSpace([1, -2], 2) ∩
    HalfSpace([1, 0], 2)

p = polyhedron(h)

p1 = contour(xlist, ylist, f, fill = true, line_smoothing = 0.85)
plot(p1, opacity = 0.5)
plot!(p, ratio = :equal, opacity = 0.5, label = "feasible region", framestyle = :zerolines, legend = true, color=:blue);

# Finally, we add all iterates and vertices to the plot.
colors = ["gold", "purple", "darkorange2", "firebrick3"]
iterates = unique!(iterates)
for i in 1:3
    scatter!([iterates[i][1]], [iterates[i][2]], label = string("x_", i - 1), markersize = 6, color = colors[i])
end
scatter!([last(iterates)[1]], [last(iterates)[2]], label = string("x_", length(iterates) - 1), markersize = 6, color = last(colors))

# plot chosen vertices
scatter!([vertices[1][1]], [vertices[1][2]], m = :diamond, markersize = 6, color = colors[1], label = "v_1")
scatter!([vertices[2][1]], [vertices[2][2]], m = :diamond, markersize = 6, color = colors[2], label = "v_2", legend = :outerleft, colorbar = true)
