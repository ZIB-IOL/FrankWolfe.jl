using Aqua
using FrankWolfe
using Test

@testset "Aqua.jl" begin
    Aqua.test_all(
        FrankWolfe;
        # You can customize which tests to run and their options
        # ambiguities=(exclude=[SomePackage.some_function], broken=true),
        # stale_deps=(ignore=[:SomePackage],),
        # deps_compat=(ignore=[:SomeOtherPackage],),
        # piracies=false,
    )
end
 