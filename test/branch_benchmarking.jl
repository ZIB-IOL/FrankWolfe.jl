using FrankWolfe
using Test
using LinearAlgebra
using DelimitedFiles
using SparseArrays
using LibGit2
import FrankWolfe: ActiveSet


function get_head_shastring()
    repo_dir = pwd()
    repo = LibGit2.GitRepo(repo_dir)
    commit_head = LibGit2.peel(LibGit2.GitCommit,LibGit2.head(repo))
    shastring_head = LibGit2.GitHash(commit_head)
    return shastring_head
end

function run_benchmark()
    f(x) = norm(x)^2

    function grad!(storage, x)
        @. storage = 2x
    end

    lmo = FrankWolfe.ProbabilitySimplexOracle(1)

    tf = FrankWolfe.TrackingObjective(f,0)
    tgrad! = FrankWolfe.TrackingGradient(grad!,0)
    tlmo = FrankWolfe.TrackingLMO(lmo)

    x0 = FrankWolfe.compute_extreme_point(tlmo, spzeros(1000))
    storage = []
    callback = FrankWolfe.tracking_trajectory_callback(storage)

    x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
        tf,
        tgrad!,
        tlmo,
        x0,
        line_search=FrankWolfe.Agnostic(),
        max_iteration=5000,
        trajectory=true,
        callback=callback,
        verbose=true,
    )
    return trajectory
end

suite=Dict()

dir_base = pwd()
repo_base = LibGit2.GitRepo(dir_base)
commit_base = LibGit2.peel(LibGit2.GitCommit,LibGit2.head(repo_base))
shastring_base = string(LibGit2.GitHash(commit_base))

suite[shastring_base]=run_benchmark()

shastring_branch = "1bd668459b93d39d6e1c68dd2ce8d362470dae76"



LibGit2.transact(repo_base) do rb
    branch_base = try LibGit2.branch(rb) catch err; nothing end
    try
        LibGit2.checkout!(rb,shastring_branch)
        suite[shastring_branch]=run_benchmark()
    catch err
        rethrow(err)
    finally
        if branch_base !== nothing
            LibGit2.branch!(rb, branch_base)
        else
            LibGit2.checkout!(rb, branch_base)
        end
    end
end

println(suite[shastring_base][1])