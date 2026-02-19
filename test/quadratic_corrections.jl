module Test_quadratic_corrections

using FrankWolfe
using LinearAlgebra
using Random
using Test
import HiGHS
import MathOptInterface as MOI
using StableRNGs

n_features = 1000
n_samples = 200
K = 10
τ = 1.0
max_iter = 1000
target_tolerance = 1e-6

s = 42
rng = StableRNG(s)
Random.seed!(rng, s)

X = randn(rng, n_samples, n_features)
y = randn(rng, n_samples)
# Normalize for better conditioning
X .= X ./ sqrt(n_samples)

# Linear regression: f(β) = (1/2)‖Xβ - y‖² = (1/2)β'X'Xβ - y'Xβ + const
# Hessian A = X'X, gradient = X'Xβ - X'y, so b = -X'y
hessian = X' * X
linear_term = -X' * y

f(β) = 0.5 * norm(X * β - y)^2
function grad!(storage, β)
    mul!(storage, X', X * β)
    storage .+= linear_term
end

lmo = FrankWolfe.KSparseLMO(K, τ)
x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n_features))

common_kw = (;
    max_iteration=max_iter,
    epsilon=target_tolerance,
    verbose=false,
    trajectory=true,
)

function build_callback(trajectory_arr)
    return function callback(state, active_set, args...)
        return push!(trajectory_arr, (FrankWolfe.callback_state(state)..., length(active_set)))
    end
end

@testset "Quadratic Corrections - Linear Regression" begin
    @testset "QC-LS without MNP" begin
        as_ls_no_mnp = FrankWolfe.ActiveSetQuadraticProductCaching(
            [(1.0, copy(x0))],
            hessian,
            linear_term,
        )
        step_ls_no_mnp = FrankWolfe.ScheduledStep(
            FrankWolfe.BlendedPairwiseStep(true),
            FrankWolfe.QuadraticLSCorrection(hessian, linear_term, false),
        )
        trajectory_qc_ls_no_mnp = []
        result_qc_ls_no_mnp = FrankWolfe.corrective_frank_wolfe(
            f,
            grad!,
            lmo,
            step_ls_no_mnp,
            as_ls_no_mnp;
            common_kw...,
            callback=build_callback(trajectory_qc_ls_no_mnp),
        )

        # Test that dual gap is sufficiently small
        @test result_qc_ls_no_mnp.dual_gap < target_tolerance
    end

    @testset "QC-LS with MNP" begin
        as_ls_mnp = FrankWolfe.ActiveSetQuadraticProductCaching(
            [(1.0, copy(x0))],
            hessian,
            linear_term,
        )
        step_ls_mnp = FrankWolfe.ScheduledStep(
            FrankWolfe.BlendedPairwiseStep(true),
            FrankWolfe.QuadraticLSCorrection(hessian, linear_term, true),
        )
        trajectory_qc_ls_mnp = []
        result_qc_ls_mnp = FrankWolfe.corrective_frank_wolfe(
            f,
            grad!,
            lmo,
            step_ls_mnp,
            as_ls_mnp;
            common_kw...,
            callback=build_callback(trajectory_qc_ls_mnp),
        )

        # Test that dual gap is sufficiently small
        @test result_qc_ls_mnp.dual_gap < target_tolerance
    end

    @testset "QC-LP without MNP" begin
        as_lp_no_mnp = FrankWolfe.ActiveSetQuadraticProductCaching(
            [(1.0, copy(x0))],
            hessian,
            linear_term,
        )
        step_lp_no_mnp = FrankWolfe.ScheduledStep(
            FrankWolfe.BlendedPairwiseStep(true),
            FrankWolfe.QuadraticLPCorrection(
                hessian,
                linear_term,
                MOI.instantiate(
                    MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true),
                ),
                false,
            ),
        )
        trajectory_qc_lp_no_mnp = []
        result_qc_lp_no_mnp = FrankWolfe.corrective_frank_wolfe(
            f,
            grad!,
            lmo,
            step_lp_no_mnp,
            as_lp_no_mnp;
            common_kw...,
            callback=build_callback(trajectory_qc_lp_no_mnp),
        )

        # Test that dual gap is sufficiently small
        @test result_qc_lp_no_mnp.dual_gap < target_tolerance
    end

    @testset "QC-LP with MNP" begin
        as_lp_mnp = FrankWolfe.ActiveSetQuadraticProductCaching(
            [(1.0, copy(x0))],
            hessian,
            linear_term,
        )
        step_lp_mnp = FrankWolfe.ScheduledStep(
            FrankWolfe.BlendedPairwiseStep(true),
            FrankWolfe.QuadraticLPCorrection(
                hessian,
                linear_term,
                MOI.instantiate(
                    MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true),
                ),
                true,
            ),
        )
        trajectory_qc_lp_mnp = []
        result_qc_lp_mnp = FrankWolfe.corrective_frank_wolfe(
            f,
            grad!,
            lmo,
            step_lp_mnp,
            as_lp_mnp;
            common_kw...,
            callback=build_callback(trajectory_qc_lp_mnp),
        )

        # Test that dual gap is sufficiently small
        @test result_qc_lp_mnp.dual_gap < target_tolerance
    end
end

end # module
