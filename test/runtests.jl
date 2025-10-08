using Test

SKIP_FILES = ["runtests.jl", "benchmark.jl", "benchmark_cache.jl"]

filter_fn(f) = endswith(f, ".jl") && !(f in SKIP_FILES)

@testset verbose = true "FrankWolfe.jl test suite" begin
    @testset "$root" for (root, dirs, files) in walkdir(@__DIR__)
        @testset "$file" for file in filter(filter_fn, files)
            include(joinpath(root, file))
        end
    end
end
