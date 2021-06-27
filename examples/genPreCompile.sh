#!/bin/bash
export JULIA=/home/spokutta/Downloads/julia/julia-1.6.1/bin/julia

$JULIA --trace-compile=precompile_fw.jl $1
cat precompile_fw.jl | grep "FrankWolfe." | grep -v "#" | grep -v ":callback" | grep -v "GLPK" | grep -v "MathOptInterface" | grep -v "StochasticObjective" | grep -v "Main.callback"  | grep -v "Main.f" | grep -v "Main.grad!" >> precompile_fw_clean.jl
cat precompile_fw_clean.jl | sort | uniq > precompile_fw_clean_uniq.jl
