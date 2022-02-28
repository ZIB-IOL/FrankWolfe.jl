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
        return run_benchmark()
    end
end



suite=Dict()
dir_base = pwd()
run_include = get_include(dir_base)

repo_base = LibGit2.GitRepo(dir_base)
commit_base = LibGit2.peel(LibGit2.GitCommit,LibGit2.head(repo_base))
shastring_base = string(LibGit2.GitHash(commit_base))

suite[shastring_base] = run_include()


shastring_branch = "2d262639d02bc1a6bb5d1ed286160a0c96b0f5cc"

# function, grad! and lmo counters
# println(suite[shastring_base][end])

suite[shastring_branch] = FrankWolfe.withcommit(run_include, repo_base,shastring_branch)

# @test suite[shastring_base][end][end] =  2502
# @test suite[shastring_branch][end][end] = 5002



