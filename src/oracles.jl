
# simple unitSimplexLMO
# TODO:
# - not optimized

function unitSimplexLMO(grad;r=1)
    n = length(grad)
    v = zeros(n)
    aux = argmin(grad)
    if grad[aux] < 0.0
        v[aux] = 1.0
    end
    return v*r
end

# simple probabilitySimplexLMO
# TODO:
# - not optimized

function probabilitySimplexLMO(grad;r=1)
    n = length(grad)
    v = zeros(n)
    aux = argmin(grad)
    v[aux] = 1.0
    return v*r
end

