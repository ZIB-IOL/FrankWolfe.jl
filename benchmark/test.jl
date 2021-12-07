using PkgBenchmark
using BenchmarkTools
# A = zeros(3);

# # each evaluation will modify A
# b = PkgBenchmark.@benchmarkable fill!($A, rand())

# run(b, samples = 1)

# println(typeof(A))
# println(A)

BenchmarkTools.ratio(1,2)


BenchmarkTools.judge(1,2)
