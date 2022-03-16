# Algorithms

The package features several variants of Frank-Wolfe that share the same basic API.

Most of the algorithms listed below also have a lazified version: see [Braun, Pokutta, Zink (2016)](https://arxiv.org/abs/1610.05120).

## Standard Frank-Wolfe (FW)

It is implemented in the [`frank_wolfe`](@ref) function.

See [Jaggi (2013)](http://proceedings.mlr.press/v28/jaggi13.html) for an overview.

This algorithm works both for convex and non-convex functions (use step size rule `FrankWolfe.Nonconvex()` in the second case).

## Away-step Frank-Wolfe (AFW)

It is implemented in the [`away_frank_wolfe`](@ref) function.

See [Lacoste-Julien, Jaggi (2015)](https://arxiv.org/abs/1511.05932) for an overview.

## Stochastic Frank-Wolfe (SFW)

It is implemented in the [`FrankWolfe.stochastic_frank_wolfe`](@ref) function.

## Blended Conditional Gradients (BCG)

It is implemented in the [`blended_conditional_gradient`](@ref) function, with a built-in stability feature that temporarily increases accuracy.

See [Braun, Pokutta, Tu, Wright (2018)](https://arxiv.org/abs/1805.07311).

## Blended Pairwise Conditional Gradients

It is implemented in the [`FrankWolfe.blended_pairwise_conditional_gradient`](@ref) function, with a minor [modification](https://hackmd.io/@spokutta/B14MTMsLF) to improve sparsity.

See [Tsuji, Tanaka, Pokutta (2021)](https://arxiv.org/abs/2110.12650)

## Comparison

The following table compares the characteristics of the algorithms presented in the package:

| Algorithm | Progress/Iteration | Time/Iteration | Sparsity | Numerical Stability | Active Set | Lazifiable |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **FW** | Low | Low | Low | High | No | Yes |
| **AFW** | Medium | Medium-High | Medium | Medium-High | Yes | Yes |
| **BCG** | High | Medium-High | High | Medium | Yes | By design |
| **SFW** | Low | Low | Low | High | No | No |

While the standard Frank-Wolfe algorithm can only move _towards_ extreme points of the compact convex set $\mathcal{C}$, Away-step Frank-Wolfe can move _away_ from them. The following figure from our paper illustrates this behaviour:

![FW vs AFW](./fw_vs_afw.PNG).

Both algorithms minimize a quadratic function (whose contour lines are depicted) over a simple polytope (the black square). When the minimizer lies on a face, the standard Frank-Wolfe algorithm zig-zags towards the solution, while its Away-step variant converges more quickly.
