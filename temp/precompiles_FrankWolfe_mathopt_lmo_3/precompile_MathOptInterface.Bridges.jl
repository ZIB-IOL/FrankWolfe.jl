function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(unbridged_function),LazyBridgeOptimizer,MathOptInterface.AbstractScalarFunction})   # time: 0.31605217
    Base.precompile(Tuple{typeof(node),LazyBridgeOptimizer,Type})   # time: 0.05343488
    Base.precompile(Tuple{typeof(bridge_index),Graph,ConstraintNode})   # time: 0.026072258
    Base.precompile(Tuple{typeof(_functionize_bridge),Vector{Any},Type{MathOptInterface.Bridges.Objective.FunctionizeBridge}})   # time: 0.00353472
    Base.precompile(Tuple{typeof(_functionize_bridge),Vector{Any},Type{MathOptInterface.Bridges.Constraint.VectorFunctionizeBridge}})   # time: 0.003471233
    Base.precompile(Tuple{typeof(add_constraint_node),Graph})   # time: 0.002223452
    Base.precompile(Tuple{typeof(add_objective_node),Graph})   # time: 0.001812303
    Base.precompile(Tuple{typeof(add_variable_node),Graph})   # time: 0.001625338
end
