using Documenter, FrankWolfe
using SparseArrays
using LinearAlgebra

using Literate, Test

EXAMPLE_DIR = joinpath(dirname(@__DIR__), "examples")
DOCS_EXAMPLE_DIR = joinpath(@__DIR__, "src", "examples")
DOCS_REFERENCE_DIR = joinpath(@__DIR__, "src", "reference")

function file_list(dir, extension)
    return filter(file -> endswith(file, extension), sort(readdir(dir)))
end

function literate_directory(jl_dir, md_dir)
    for filename in file_list(md_dir, ".md")
        filepath = joinpath(md_dir, filename)
        rm(filepath)
    end
    for filename in file_list(jl_dir, ".jl")
        filepath = joinpath(jl_dir, filename)
        # `include` the file to test it before `#src` lines are removed. It is
        # in a testset to isolate local variables between files.
        if startswith(filename, "docs")
            Literate.markdown(
                filepath, md_dir; documenter=true, flavor=Literate.DocumenterFlavor()
            )
        end
    end
    return nothing
end

literate_directory(EXAMPLE_DIR, DOCS_EXAMPLE_DIR)

ENV["GKSwstype"] = "100"

generated_path = joinpath(@__DIR__, "src")
base_url = "https://github.com/ZIB-IOL/FrankWolfe.jl/blob/master/"
isdir(generated_path) || mkdir(generated_path)

open(joinpath(generated_path, "contributing.md"), "w") do io
    # Point to source license file
    println(
        io,
        """
        ```@meta
        EditURL = "$(base_url)CONTRIBUTING.md"
        ```
        """,
    )
    # Write the contents out below the meta block
    for line in eachline(joinpath(dirname(@__DIR__), "CONTRIBUTING.md"))
        println(io, line)
    end
end

open(joinpath(generated_path, "index.md"), "w") do io
    # Point to source license file
    println(
        io,
        """
        ```@meta
        EditURL = "$(base_url)README.md"
        ```
        """,
    )
    # Write the contents out below the meta block
    for line in eachline(joinpath(dirname(@__DIR__), "README.md"))
        println(io, line)
    end
end

makedocs(;
    modules=[FrankWolfe],
    sitename="FrankWolfe.jl",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true", collapselevel=1),
    pages=[
        "Home" => "index.md",
        "How does it work?" => "basics.md",
        "Advanced features" => "advanced.md",
        "Examples" => [joinpath("examples", f) for f in file_list(DOCS_EXAMPLE_DIR, ".md")],
        "API reference" =>
            [joinpath("reference", f) for f in file_list(DOCS_REFERENCE_DIR, ".md")],
        "Contributing" => "contributing.md",
    ],
)

deploydocs(; repo="github.com/ZIB-IOL/FrankWolfe.jl.git", push_preview=true)
