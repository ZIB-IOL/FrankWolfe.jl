import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using TestEnv

Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()

TestEnv.activate()
