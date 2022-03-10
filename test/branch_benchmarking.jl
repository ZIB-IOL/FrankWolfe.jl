using FrankWolfe
using Test
using LinearAlgebra
using DelimitedFiles
using SparseArrays
using LibGit2
import FrankWolfe: ActiveSet

function get_include(dir)
    path = joinpath(dir,"test/benchmarking_suite.jl")
    function run_include()
        include(path)
        run_benchmark_latest = Base.invokelatest(run_benchmark)
        return run_benchmark_latest
    end
end

@testset "Branch Benchmarking" begin

    suite=Dict()
    dir_base = pwd()
    run_include = get_include(dir_base)

    repo_base = LibGit2.GitRepo(dir_base)
    commit_base = LibGit2.peel(LibGit2.GitCommit,LibGit2.head(repo_base))
    shastring_base = string(LibGit2.GitHash(commit_base))

    suite[shastring_base] = run_include()

    # branch where the iteration count of the benchmarking_suite.jl frank_wolfe call was halved

    commit_branch = LibGit2.GitObject(repo_base,"benchmarking-mirror")
    
    shastring_branch = string(LibGit2.GitHash(commit_branch))

    suite[shastring_branch] = FrankWolfe.withcommit(run_include, repo_base,shastring_branch)

    @test suite[shastring_base][end][end] ==  5002
    @test suite[shastring_branch][end][end] == 2502

end

