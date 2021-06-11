using Documenter, FrankWolfe
using SparseArrays
using LinearAlgebra

makedocs(
    modules=[FrankWolfe],
    sitename="FrankWolfe.jl",
    format=Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages=["Home"=>"index.md","Examples"=>"examples2.md","Reference"=>"reference.md",
            "Index"=>"indexlist.md"])
