using FrankWolfe
using Test
using LinearAlgebra
using DelimitedFiles
using SparseArrays
using LibGit2
import FrankWolfe: ActiveSet
using PkgBenchmark


function get_include(dir)
    path = joinpath(dir,"test/benchmarking_suite.jl")
    function run_include()
        include(path)
        println(read(path,String))
        return run_benchmark()
    end
end



# function get_head_shastring()
#     repo_dir = pwd()
#     repo = LibGit2.GitRepo(repo_dir)
#     commit_head = LibGit2.peel(LibGit2.GitCommit,LibGit2.head(repo))
#     shastring_head = LibGit2.GitHash(commit_head)
#     return shastring_head
# end

suite=Dict()
dir_base = pwd()
run_include = get_include(dir_base)

repo_base = LibGit2.GitRepo(dir_base)
commit_base = LibGit2.peel(LibGit2.GitCommit,LibGit2.head(repo_base))
shastring_base = string(LibGit2.GitHash(commit_base))

run_include()


shastring_branch = "2d262639d02bc1a6bb5d1ed286160a0c96b0f5cc"

# function, grad! and lmo counters
# println(suite[shastring_base][end])

suite[shastring_branch] = withcommit(run_include, repo_base,shastring_branch)

# println(suite[shastring_base][end][end])
# println(suite[shastring_branch][end][end])