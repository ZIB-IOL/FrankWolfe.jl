
"""
    MathOptLMO{OT <: MOI.Optimizer} <: LinearMinimizationOracle

Linear minimization oracle with feasible space defined through a MathOptInterface.Optimizer.
The oracle call sets the direction and reruns the optimizer.

The `direction` vector has to be set in the same order of variables as the `MOI.ListOfVariableIndices()` getter.

The Boolean `use_modify` determines if the objective in`compute_extreme_point` is updated with
`MOI.modify(o, ::MOI.ObjectiveFunction, ::MOI.ScalarCoefficientChange)` or with `MOI.set(o, ::MOI.ObjectiveFunction, f)`.
`use_modify = true` decreases the runtime and memory allocation for models created as an optimizer object and defined directly
with MathOptInterface. `use_modify = false` should be used for CachingOptimizers.
"""
struct MathOptLMO{OT<:MOI.AbstractOptimizer} <: LinearMinimizationOracle
    o::OT
    use_modify::Bool
    function MathOptLMO(o, use_modify=true)
        MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        return new{typeof(o)}(o, use_modify)
    end
end

function compute_extreme_point(
    lmo::MathOptLMO{OT},
    direction::AbstractVector{T};
    kwargs...,
) where {OT,T<:Real}
    variables = MOI.get(lmo.o, MOI.ListOfVariableIndices())
    if lmo.use_modify
        for i in eachindex(variables)
            MOI.modify(
                lmo.o,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                MOI.ScalarCoefficientChange(variables[i], direction[i]),
            )
        end
    else
        terms = [MOI.ScalarAffineTerm(d, v) for (d, v) in zip(direction, variables)]
        obj = MOI.ScalarAffineFunction(terms, zero(T))
        MOI.set(lmo.o, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    end
    return _optimize_and_return(lmo, variables)
end

function compute_extreme_point(
    lmo::MathOptLMO{OT},
    direction::AbstractMatrix{T};
    kwargs...,
) where {OT,T<:Real}
    n = size(direction, 1)
    v = compute_extreme_point(lmo, vec(direction))
    return reshape(v, n, n)
end

"""
    
"""

is_decomposition_invariant_oracle(::MathOptLMO) = true

function compute_inface_extreme_point(lmo::MathOptLMO{OT}, direction, x; kwargs...) where {OT}
    lmo2 = copy(lmo.o)
    variables = MOI.get(lmo, MOI.ListOfVariableIndices())
    terms = [MOI.ScalarAffineTerm(d, v) for (d, v) in zip(direction, variables)]
    obj = MOI.ScalarAffineFunction(terms, zero(T))
    MOI.set(lmo, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    for (F, S) in MOI.get(opt, MOI.ListOfConstraintTypesPresent())
        valvar(f) = x[f.value]
        const_list = MOI.get(opt, MOI.ListOfConstraintIndices{F,S}())
        for c_idx in const_list
            if !(S <: MOI.ZeroOne)
                func = MOI.get(opt, MOI.ConstraintFunction(), c_idx)
                val = MOIU.eval_variables(valvar, func)
                set = MOI.get(opt, MOI.ConstraintSet(), c_idx)
                # @debug("Constraint: $(F)-$(S) $(func) = $(val) in $(set)")
                if ( S <: MOI.GreaterThan)
                    if set.lower === val
                        idx = MOI.add_constraint(opt, func, MOI.EqualTo(val))
                    elseif ( S <: MOI.LessThan)
                        if set.upper === val
                            idx = MOI.add_constraint(opt, func, MOI.EqualTo(val)) 
                        end
                    end
                end  
            end
        end
    end
    MOI.optimize!(lmo2)
    return MOI.get(lmo2, MOI.VariablePrimal(), variables)
end

function dicg_maximum_step(lmo::MathOptLMO{OT}, x, direction; exactness=40) where {OT}
    gamma_max = 0.0
    gamma = 1.0
    while(exactness != 0)
        flag, _ = is_constraints_feasible(lmo, x+gamma*direction)
        if flag
            if gamma === 1.0
                return gamma
            else
                gamma_max = max(gamma, gamma_max)
                gamma = (1+gamma_max) / 2
                exactness -= 1
            end
        end
        if !flag
            gamma = (gamma+gamma_max) / 2
            exactness -= 1
        end
    end
    return gamma_max
end


function Base.copy(lmo::MathOptLMO{OT}; ensure_identity=true) where {OT}
    opt = OT() # creates the empty optimizer
    index_map = MOI.copy_to(opt, lmo.o)
    if ensure_identity
        for (src_idx, des_idx) in index_map.var_map
            if src_idx != des_idx
                error("Mapping of variables is not identity")
            end
        end
    end
    return MathOptLMO(opt)
end

function Base.copy(
    lmo::MathOptLMO{OT};
    ensure_identity=true,
) where {OTI,OT<:MOIU.CachingOptimizer{OTI}}
    opt = MOIU.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        OTI(),
    )
    index_map = MOI.copy_to(opt, lmo.o)
    if ensure_identity
        for (src_idx, des_idx) in index_map.var_map
            if src_idx != des_idx
                error("Mapping of variables is not identity")
            end
        end
    end
    return MathOptLMO(opt)
end


function compute_extreme_point(
    lmo::MathOptLMO{OT},
    direction::AbstractVector{MOI.ScalarAffineTerm{T}};
    kwargs...,
) where {OT,T}
    if lmo.use_modify
        for d in direction
            MOI.modify(
                lmo.o,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                MOI.ScalarCoefficientChange(d.variable, d.coefficient),
            )
        end

        variables = MOI.get(lmo.o, MOI.ListOfVariableIndices())
        variables_to_zero = setdiff(variables, [dir.variable for dir in direction])

        terms = [
            MOI.ScalarAffineTerm(d, v) for
            (d, v) in zip(zeros(length(variables_to_zero)), variables_to_zero)
        ]

        for t in terms
            MOI.modify(
                lmo.o,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
                MOI.ScalarCoefficientChange(t.variable, t.coefficient),
            )
        end
    else
        variables = [d.variable for d in direction]
        obj = MOI.ScalarAffineFunction(direction, zero(T))
        MOI.set(lmo.o, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    end
    return _optimize_and_return(lmo, variables)
end

function _optimize_and_return(lmo, variables)
    MOI.optimize!(lmo.o)
    term_st = MOI.get(lmo.o, MOI.TerminationStatus())
    if term_st ∉ (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.SLOW_PROGRESS)
        @error "Unexpected termination: $term_st"
        return MOI.get.(lmo.o, MOI.VariablePrimal(), variables)
    end
    return MOI.get.(lmo.o, MOI.VariablePrimal(), variables)
end

"""
    convert_mathopt(lmo::LMO, optimizer::OT; kwargs...) -> MathOptLMO{OT}

Converts the given LMO to its equivalent MathOptInterface representation using `optimizer`.
Must be implemented by LMOs.
"""
function convert_mathopt end
