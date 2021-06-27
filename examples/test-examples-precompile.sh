#!/usr/bin/env bats

# export the julia binary as ./genPreCompile.sh

@test "adaptive_step_sizes.jl" {
    ./genPreCompile.sh adaptive_step_sizes.jl
}

@test "approximate_caratheodory.jl" {
    ./genPreCompile.sh approximate_caratheodory.jl
}

@test "away_step_cg.jl" {
    ./genPreCompile.sh away_step_cg.jl
}

@test "birkhoff_polytope.jl" {
    ./genPreCompile.sh birkhoff_polytope.jl
}

@test "blended_cg.jl" {
    ./genPreCompile.sh blended_cg.jl
}

@test "large_scale.jl" {
    ./genPreCompile.sh large_scale.jl
}

@test "lazy_away_step_cg-sparsity.jl" {
    ./genPreCompile.sh lazy_away_step_cg-sparsity.jl
}

@test "lazy_away_step_cg.jl" {
    ./genPreCompile.sh lazy_away_step_cg.jl
}

@test "lcg_cache_size.jl" {
    ./genPreCompile.sh lcg_cache_size.jl
}

@test "lcg_expensive_LMO.jl" {
    ./genPreCompile.sh lcg_expensive_LMO.jl
}

@test "linear_regression.jl" {
    ./genPreCompile.sh linear_regression.jl
}

@test "lowerbound.jl" {
    ./genPreCompile.sh lowerbound.jl
}

@test "moi_optimizer.jl" {
    ./genPreCompile.sh moi_optimizer.jl
}

@test "movielens.jl" {
    ./genPreCompile.sh movielens.jl
}

@test "nonconvex_lasso.jl" {
    ./genPreCompile.sh nonconvex_lasso.jl
}

@test "nuclear_norm.jl" {
    ./genPreCompile.sh nuclear_norm.jl
}

@test "polynomials.jl" {
    ./genPreCompile.sh polynomials.jl
}

@test "trajectory_comparison.jl" {
    ./genPreCompile.sh trajectory_comparison.jl
}
