#!/usr/bin/env bats

# export the julia binary as $JULIA

@test "adaptive_step_sizes.jl" {
    $JULIA adaptive_step_sizes.jl
}

@test "approximate_caratheodory.jl" {
    $JULIA approximate_caratheodory.jl
}

@test "away_step_cg.jl" {
    $JULIA away_step_cg.jl
}

@test "birkhoff_polytope.jl" {
    $JULIA birkhoff_polytope.jl
}

@test "blended_cg.jl" {
    $JULIA blended_cg.jl
}

@test "large_scale.jl" {
    $JULIA large_scale.jl
}

@test "lazy_away_step_cg-sparsity.jl" {
    $JULIA lazy_away_step_cg-sparsity.jl
}

@test "lazy_away_step_cg.jl" {
    $JULIA lazy_away_step_cg.jl
}

@test "lcg_cache_size.jl" {
    $JULIA lcg_cache_size.jl
}

@test "lcg_expensive_LMO.jl" {
    $JULIA lcg_expensive_LMO.jl
}

@test "linear_regression.jl" {
    $JULIA linear_regression.jl
}

@test "lowerbound.jl" {
    $JULIA lowerbound.jl
}

@test "moi_optimizer.jl" {
    $JULIA moi_optimizer.jl
}

@test "movielens.jl" {
    $JULIA movielens.jl
}

@test "nonconvex_lasso.jl" {
    $JULIA nonconvex_lasso.jl
}

@test "nuclear_norm.jl" {
    $JULIA nuclear_norm.jl
}

@test "polynomials.jl" {
    $JULIA polynomials.jl
}

@test "trajectory_comparison.jl" {
    $JULIA trajectory_comparison.jl
}
