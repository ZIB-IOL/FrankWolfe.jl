function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(unbridged_function),LazyBridgeOptimizer,MathOptInterface.AbstractScalarFunction})   # time: 0.7096968
    Base.precompile(Tuple{typeof(node),LazyBridgeOptimizer,Type})   # time: 0.13811369
    Base.precompile(Tuple{typeof(bridge_index),Graph,ConstraintNode})   # time: 0.044580877
    Base.precompile(Tuple{typeof(_functionize_bridge),Vector{Any},Type{MathOptInterface.Bridges.Constraint.VectorFunctionizeBridge}})   # time: 0.009250539
    Base.precompile(Tuple{typeof(add_constraint_node),Graph})   # time: 0.006662106
    Base.precompile(Tuple{typeof(_functionize_bridge),Vector{Any},Type{MathOptInterface.Bridges.Objective.FunctionizeBridge}})   # time: 0.006550288
    Base.precompile(Tuple{typeof(add_objective_node),Graph})   # time: 0.003297667
    Base.precompile(Tuple{typeof(add_variable_node),Graph})   # time: 0.003117756
end
