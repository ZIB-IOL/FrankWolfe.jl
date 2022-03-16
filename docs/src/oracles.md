# Linear Minimization Oracles

## What are they?

The Linear Minimization Oracle (LMO) is a key component, which is called at each iteration of the FW algorithm. Given a direction $d$, it returns the optimal vertex of the feasible set:

```math
v \in \arg \min_{x\in \mathcal{C}} \langle d,x \rangle.
```

To be used by the algorithms provided here, an LMO must be a subtype of [`FrankWolfe.LinearMinimizationOracle`](@ref) and implement the following method:

```julia
compute_extreme_point(lmo::LMO, direction; kwargs...) -> v
```

This method should minimize $v \mapsto \langle d, v \rangle$ over the set $\mathcal{C}$ defined by the LMO. Note that this means the set $\mathcal{C}$ doesn't have to be defined explicitly: all we need is to be able to minimize a linear function over it, even if the minimization procedure is a black box.

## Available LMOs

If you don't want to define your LMO manually, several common implementations are available out-of-the-box:

- Simplices: unit simplex, probability simplex
- Balls in various norms
- Polytopes: K-sparse, Birkhoff

You can use an oracle defined via a Linear Programming solver (e.g. `SCIP` or `HiGHS`) with `MathOptInferface`: see [`FrankWolfe.MathOptLMO`](@ref).

Finally, we provide wrappers to combine oracles easily, for example in a product.

See [Combettes, Pokutta (2021)](https://arxiv.org/abs/2101.10040) for references on most LMOs implemented in the package and their comparison with projection operators.
