@enum LineSearchVariant begin
    LS_ONLY_SECANT = 1
    LS_SECANT_WITH_BACKTRACKING = 2
    LS_BACKTRACKING_AND_SECANT = 3
    LS_ADAPTIVE = 4
    LS_ADAPTIVE_AND_SECANT = 5
    LS_ADAPTIVE_ZERO_AND_SECANT = 6
    LS_SECANT_3 = 7
    LS_SECANT_5 = 8
    LS_SECANT_7 = 9
    LS_SECANT_12 = 10
    LS_MONOTONIC = 11
    LS_AGNOSTIC = 12
    LS_ADAPTIVE_ZERO = 13
    LS_BACKTRACKING = 14
    LS_GOLDEN_RATIO = 15
end

const linesearchvariant_string = (
    LS_ONLY_SECANT ="Only_Secant",
    LS_SECANT_WITH_BACKTRACKING="Secant_with_Backtracking",
    LS_BACKTRACKING_AND_SECANT="Backtracking_and_Secant",
    LS_ADAPTIVE="Adaptive",
    LS_ADAPTIVE_AND_SECANT="Adaptive_and_Secant",
    LS_ADAPTIVE_ZERO_AND_SECANT="Adaptive_Zero_and_Secant",
    LS_SECANT_3="Secant_3",
    LS_SECANT_5="Secant_5",
    LS_SECANT_7="Secant_7",
    LS_SECANT_12="Secant_12",
    LS_MONOTONIC="Monotonic",
    LS_AGNOSTIC="Agnostic",
    LS_ADAPTIVE_ZERO="Adaptive_Zero",
    LS_BACKTRACKING="Backtracking",
    LS_GOLDEN_RATIO="Golden_Ratio",
)

function get_dimensions(problem)
    dimensions = if problem in ["IllConditionedQuadratic"]
        collect(500:500:5000)
    elseif problem in ["OEDP_A", "OEDP_D"]
        collect(100:100:1000)
    elseif problem in ["Birkhoff", "Nuclear", "QuadraticProbSimplex"]
        collect(50:50:300).^2
    elseif problem == "Portfolio"
        [800, 1200, 1500]
    else
        collect(100:100:1000).^2
    end
    return dimensions
end

function is_type_secant(ls::LineSearchVariant)
    return ls in [LS_ONLY_SECANT, LS_SECANT_12, LS_SECANT_3, LS_SECANT_5, LS_SECANT_7, LS_SECANT_WITH_BACKTRACKING]
end

function geom_shifted_mean(xs; shift=big"1.0", dash=false)
    a = length(xs)  
    n= 0
    prod = 1.0  
    if a != 0 
        for xi in xs
            if xi != Inf 
                prod = prod*(xi+shift)  
                n += 1
            end
        end
        return Float64(prod^(1/n) - shift)
    end
    return dash ? "-" : Inf
end

function custom_mean(group)
    sum = 0.0
    n = 0
    dash = false

    if isempty(group)
        return Inf
    end
    for element in group
        if element == "-"
            dash = true
            continue
        end
        if element != Inf 
            if typeof(element) == String7 || typeof(element) == String3
                element = parse(Float64, element)
            end
            sum += element
            n += 1
        end
    end
    if n == 0
        return dash ? "-" : Inf
    end
    return sum/n
end

function geo_standard_deviation(xs, mean)
    a = length(xs)  
   # @show xs
    n= 0
    sum = 0.0
    if a != 0 
        for xi in xs
            if xi != Inf 
                sum = log(xi / mean)^2
                n += 1
            end
        end
        return exp(sum / n)
    end
    return Inf
    #return exp(sum((log.(group ./ mean)).^2) / length(group))
end