function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(unbridged_function),LazyBridgeOptimizer,MathOptInterface.AbstractScalarFunction})   # time: 3.07497
    Base.precompile(Tuple{typeof(bridging_cost),Graph,VariableNode})   # time: 0.040493276
    Base.precompile(Tuple{typeof(_functionize_bridge),Vector{Any},Type{MathOptInterface.Bridges.Constraint.VectorFunctionizeBridge}})   # time: 0.009407265
    Base.precompile(Tuple{typeof(_functionize_bridge),Vector{Any},Type{MathOptInterface.Bridges.Constraint.ScalarFunctionizeBridge}})   # time: 0.005661457
    Base.precompile(Tuple{typeof(add_variable_node),Graph})   # time: 0.004823927
    Base.precompile(Tuple{typeof(_functionize_bridge),Vector{Any},Type{MathOptInterface.Bridges.Objective.FunctionizeBridge}})   # time: 0.004625653
    Base.precompile(Tuple{typeof(add_objective_node),Graph})   # time: 0.002584093
    Base.precompile(Tuple{typeof(add_edge),Graph,ConstraintNode,Edge})   # time: 0.001791471
    isdefined(MathOptInterface.Bridges, Symbol("#39#40")) && Base.precompile(Tuple{getfield(MathOptInterface.Bridges, Symbol("#39#40")),Nothing})   # time: 0.001777993
    Base.precompile(Tuple{typeof(bridged_function),LazyBridgeOptimizer,Any})   # time: 0.001685454
    Base.precompile(Tuple{typeof(add_edge),Graph,ObjectiveNode,ObjectiveEdge})   # time: 0.001559598
    Base.precompile(Tuple{Type{Edge},Int64,Any,Any})   # time: 0.001273466
end
