function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(MathOptInterface.add_constraint),Optimizer,MathOptInterface.ScalarAffineFunction{Float64},Union{MathOptInterface.EqualTo{Float64}, MathOptInterface.GreaterThan{Float64}, MathOptInterface.LessThan{Float64}}})   # time: 0.00684748
    Base.precompile(Tuple{typeof(MathOptInterface.set),Optimizer,MathOptInterface.ObjectiveFunction{F<:MathOptInterface.ScalarAffineFunction{Core.Float64}},F<:MathOptInterface.ScalarAffineFunction{Core.Float64}})   # time: 0.006374317
end
