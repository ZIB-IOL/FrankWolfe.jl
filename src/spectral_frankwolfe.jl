
"""
Implements the spectral Frank-Wolfe algorithm from:
A Linearly Convergent Frank-Wolfe-type Method for Smooth Convex Minimization over the Spectrahedron, Garber, 2025.
"""
function spectral_frankwolfe(
    f,
    grad!,
    X0;
    line_search::LineSearchMethod=Secant(),
    max_iterations=1000,
    linesearch_workspace=nothing,
    memory_mode::MemoryEmphasis=InplaceEmphasis(),
    epsilon=1e-7,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    gradient=nothing,
    callback=nothing,
    traj_data=[],
    timeout=Inf,
    renorm_interval=1000,
    weight_purge_threshold=weight_purge_threshold_default(eltype(X0)),
    extra_vertex_storage=nothing,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    recompute_last_vertex=true,
    )
    format_string = "%6s %13s %14e %14e %14e %14e %14e \n"
    headers = ("Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec")
    function format_state(state, args...)
        rep = (
            steptype_string[Symbol(state.step_type)],
            string(state.t),
            Float64(state.primal),
            Float64(state.primal - state.dual_gap),
            Float64(state.dual_gap),
            state.time,
            state.t / state.time,
        )
        return rep
    end

    if trajectory
        callback = make_trajectory_callback(callback, traj_data)
    end
    if verbose
        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    t = 0
    x = X0
    primal = convert(eltype(X0), Inf)
    step_type = ST_REGULAR
    time_start = time_ns()

    d = similar(x)

    if gradient === nothing
        gradient = collect(x)
    end
    

    if verbose
        println(
            "\nSpectral Frank-Wolfe Algorithm with $(nameof(typeof(line_search))) line search.",
        )
        NumType = eltype(X0)
        println("MEMORY_MODE: $memory_mode STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iterations TYPE: $NumType")
        grad_type = typeof(gradient)
        println("GRADIENTTYPE: $grad_type")
    end

    grad!(gradient, x)
    # the default LMO
    lmo = SpectraplexLMO(one(eltype(X0)), size(X0, 1))
    v = compute_extreme_point(lmo, gradient)
    
    if linesearch_workspace === nothing
        linesearch_workspace = build_linesearch_workspace(line_search, x, gradient)
    end

    while t <= max_iteration && phi >= max(epsilon, eps(epsilon))
        # managing time limit
        time_at_loop = time_ns()
        if t == 0
            time_start = time_at_loop
        end
        # time is measured at beginning of loop for consistency throughout all algorithms
        tot_time = (time_at_loop - time_start) / 1e9

        if timeout < Inf
            if tot_time ≥ timeout
                if verbose
                    @info "Time limit reached"
                end
                break
            end
        end

        #####################
        t += 1

        primal = f(x)
        if t > 1
            grad!(gradient, x)
        end

    end
end

"""
Returns the eigenvector and eigenvalue corresponding to the away step.
"""
function spectral_away_step(x, gram_matrix, gradient, regularizer=1e-2)
    T = eltype(x)
    # TODO optimize
    M = gram_matrix * gradient * gram_matrix + (1 + regularizer) * norm(gradient) * gram_matrix
    e_val::Vector{T}, evec::Matrix{T} = Arpack.eigs(M, nev=1, which=:LM)
    # type annotation because of Arpack instability
    unit_vec::Vector{T} = vec(evec)
    return unit_vec, e_val[1]
end

function random_away_vector(x, gram_matrix)
    z = randn(size(x, 1))
    u = gram_matrix * z
    u ./= norm(u)
    return u
end

function frankwolfe_vertex(x, step_size, away_vector, gradient, L_smoothness)
    T = eltype(x)
    # TODO optimize
    M = L_smoothness * step_size * away_vector * away_vector' - gradient
    _, evec::Matrix{T} = Arpack.eigs(M, nev=1, which=:LM)
    unit_vec::Vector{T} = vec(evec)
    return unit_vec
end
