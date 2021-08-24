# # Exact Optimization with Rational Arithmetic

# This example can be found in section 4.3 [in the paper](https://arxiv.org/pdf/2104.06675.pdf).
# The package allows for exact optimization with rational arithmetic. For this, it suffices to set up the LMO
# to be rational and choose an appropriate step-size rule as detailed below. For the LMOs included in the
# package, this simply means initializing the radius with a rational-compatible element type, e.g., `1`, rather
# than a floating-point number, e.g., `1.0`. Given that numerators and denominators can become quite large in
# rational arithmetic, it is strongly advised to base the used rationals on extended-precision integer types such
# as `BigInt`, i.e., we use `Rational{BigInt}`. For the probability simplex LMO with a rational radius of `1`,
# the LMO would be created as follows:

# ```
# lmo = FrankWolfe.ProbabilitySimplexOracle{Rational{BigInt}}(1)
# ```

# As mentioned before, the second requirement ensuring that the computation runs in rational arithmetic is
# a rational-compatible step-size rule. The most basic step-size rule compatible with rational optimization is
# the agnostic step-size rule with ``\gamma_t = 2/(2 + t)``. With this step-size rule, the gradient does not even need to
# be rational as long as the atom computed by the LMO is of a rational type. Assuming these requirements are
# met, all iterates and the computed solution will then be rational:


n = 100
x = fill(big(1)//100, n)
## equivalent to { 1/100 }^100


# Another possible step-size rule is `rationalshortstep` which computes the step size by minimizing the
# smoothness inequality as ``\gamma_t=\frac{\langle \nabla f(x_t),x_t-v_t\rangle}{2L||x_t-v_t||^2}``. However, as this step size depends on an upper bound on the
# Lipschitz constant ``L`` as well as the inner product with the gradient ``\nabla f(x_t)``, both have to be of a rational type.
