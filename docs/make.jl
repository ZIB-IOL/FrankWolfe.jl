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
        "Examples" => [
            "Comparison with MathOptInterface on a Probability Simplex" => "example1.md",
            "Polynomial Regression" => "example2.md",
            "Matrix Completion" => "example3.md",
            "Exact Optimization with Rational Arithmetic" => "example4.md",
        ],
        "References" => "reference.md",
        "Index" => "indexlist.md",
    ],
)

deploydocs(repo="github.com/ZIB-IOL/FrankWolfe.jl.git", push_preview=true)
