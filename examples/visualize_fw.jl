using FrankWolfe
using LinearAlgebra
using GLPK
using JuMP
const MOI = JuMP.MOI
using Polyhedra
using Plots
pyplot()

n = 2 # dimension
y = [3.2, 0.5]

function f(x)
    return 1 / 2 * norm(x - y)^2
end

# callback 
function build_callback(trajectory_arr)
    return function callback(state)
        return push!(trajectory_arr, deepcopy(Tuple(state)[6:8])) # x, vertex, gamma
    end
end

# gradient
function grad!(storage, x)
    @. storage = x - y
end


### create a MathOptInterface Optimizer and build the linear constraints
o = GLPK.Optimizer()
x = MOI.add_variables(o, n)

# −x + y ≤ 2
MOI.add_constraint(
    o,
    (-1.0x[1] + x[2]),
    MOI.LessThan(2.0),
)

# x + 2 y ≤ 4
MOI.add_constraint(
    o,
    (x[1] + 2.0x[2]),
    MOI.LessThan(4.0),
)

# −2 x − y ≤ 1
MOI.add_constraint(
    o,
    (-2.0x[1] - x[2]),
    MOI.LessThan(1.0),
)

# x − 2 y ≤ 2
MOI.add_constraint(
    o,
    (x[1] - 2.0x[2]),
    MOI.LessThan(2.0),
)

# x ≤ 2
MOI.add_constraint(
    o,
    (x[1] + 0.0x[2]),
    MOI.LessThan(2.0),
)


### build lmo and call frank_wolfe
status_mathopt = []
callback = build_callback(status_mathopt)

lmo_moi = FrankWolfe.MathOptLMO(o)

x0 = FrankWolfe.compute_extreme_point(lmo_moi, zeros(n))

iterates = []
push!(iterates, copy(x0))
vertices = []

xfinal, vfinal, primal_value, dual_gap, traj_data = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo_moi,
    collect(copy(x0)),
    line_search = FrankWolfe.Adaptive(),
    max_iteration = 10,
    epsilon = 10^-8,
    callback = callback
)

for s in status_mathopt
    push!(iterates, s[1])
    push!(vertices, s[2])
end

### plot
# plot contours of function
function f(x1, x2)
    x = [x1, x2]
    return 1 / 2 * norm(x - y)^2
end

xlist = collect(range(-1, 3, step = 0.2))
ylist = collect(range(-1, 3, step = 0.2))

X = repeat(reshape(xlist, 1, :), length(ylist), 1)
Y = repeat(ylist, 1, length(xlist))

p1 = contour(xlist, ylist, f, fill = true, line_smoothing = 0.85)
plot(p1, opacity = 0.7)

# plot feasible region
h = HalfSpace([-1, 1], 2) ∩ HalfSpace([1, 2], 4) ∩ HalfSpace([-2, -1], 1) ∩ HalfSpace([1, -2], 2) ∩ HalfSpace([1, 0], 2)
p = polyhedron(h)
plot!(p, ratio = :equal, opacity = 0.5, label = "feasible region", framestyle = :zerolines, legend = true)

# plot iterates x 
colors = ["gold", "purple", "darkorange2", "firebrick3"]
iterates = unique(iterates)
for i in 1:3
    scatter!([iterates[i][1]], [iterates[i][2]], label = string("x_", i - 1), markersize = 6, color = colors[i])
end
scatter!([last(iterates)[1]], [last(iterates)[2]], label = string("x_", length(iterates) - 1), markersize = 6, color = last(colors))

# plot chosen vertices
scatter!([vertices[1][1]], [vertices[1][2]], m = :diamond, markersize = 6, color = colors[1], label = "v_1")
scatter!([vertices[2][1]], [vertices[2][2]], m = :diamond, markersize = 6, color = colors[2], label = "v_2", legend = :outerleft, colorbar = true)

savefig("MathOpt.png")
