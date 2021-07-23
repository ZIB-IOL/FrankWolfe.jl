using Documenter, FrankWolfe
using SparseArrays
using LinearAlgebra

ENV["GKSwstype"] = "100"

makedocs(
    modules=[FrankWolfe],
    sitename="FrankWolfe.jl",
    format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true"),
    pages=[
        "Home" => "index.md",
        "Examples" => "examples.md",
        "References" => "reference.md",
        "Index" => "indexlist.md",
    ],
)

deploydocs(repo="github.com/ZIB-IOL/FrankWolfe.jl.git", push_preview=true)
