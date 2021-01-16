# benchmark for tuple VS vector cache

using BenchmarkTools
using FrankWolfe

for n = (10, 100, 1000)
    @info "n = $n"
    direction = zeros(n)
    rhs = 10 * rand()
    lmo_unit = FrankWolfe.UnitSimplexOracle(rhs)
    lmo_veccached = FrankWolfe.VectorCacheLMO(lmo_unit)
    res_vec = @benchmark begin
        for idx in 1:$n
            $direction .= 0
            $direction[idx] = -1
            res_point_cached_vec = FrankWolfe.compute_extreme_point($lmo_veccached, $direction, threshold=0)
            _ = length($lmo_veccached)
        end
        empty!($lmo_veccached)
    end
    res_multi = map(1:10) do N
        lmo_multicached = FrankWolfe.MultiCacheLMO{N}(lmo_unit)
        @benchmark begin
            for idx in 1:$n
                $direction .= 0
                $direction[idx] = -1
                res_point_cached_vec = FrankWolfe.compute_extreme_point($lmo_multicached, $direction, threshold=0)
                _ = length($lmo_multicached)
            end
            empty!($lmo_multicached)
        end
    end
    @info "Vec benchmark"
    display(res_vec)
    for N in 1:10
        @info "Tuple benchmark $N"
        display(res_multi[N])
    end
end
