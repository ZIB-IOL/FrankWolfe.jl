module Test_aqua

using Aqua
using FrankWolfe
using Test
using LinearAlgebra

@testset "Aqua.jl" begin
    Aqua.test_all(
        FrankWolfe;
        # You can customize which tests to run and their options
        unbound_args=false,
        ambiguities=(exclude=[Base.:*],),
        # stale_deps=(ignore=[:SomePackage],),
        # deps_compat=(ignore=[:SomeOtherPackage],),
        # piracies=false,
    )
end

end # module
