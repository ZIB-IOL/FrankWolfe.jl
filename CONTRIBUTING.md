# Contributing to FrankWolfe

First, thanks for taking the time to contribute.
Contributions in any form, such as documentation, bug fix, examples or algorithms, are appreciated and welcome.

We list below some guidelines to help you contribute to the package.

#### Table of Contents

- [Contributing to `FrankWolfe`](#Contributing-to-FrankWolfe)
  - [Community Standards](#Community-Standards)
  - [Where can I get an overview](#Where-can-I-get-an-overview)
  - [I just have a question](#I-just-have-a-question)
  - [How can I file an issue](#How-can-I-file-an-issue)
  - [How can I contribute](#How-can-I-contribute)
    - [Improve the documentation](#Improve-the-documentation)
    - [Provide a new example or test](#Provide-a-new-example-or-test)
    - [Provide a new feature](#Provide-a-new-feature)
    - [Code style](#Code-style)

## Community Standards

Interactions on this repository must follow the Julia [Community Standards](https://julialang.org/community/standards/) including Pull Requests and issues.

## Where can I get an overview

Check out the [paper](https://arxiv.org/abs/2104.06675) presenting the package
for a high-level overview of the feature and algorithms and
the [documentation](https://zib-iol.github.io/FrankWolfe.jl/dev/) for more details.

## I just have a question

If your question is related to Julia, its syntax or tooling, the best places to get help will be tied to the Julia community,
see [the Julia community page](https://julialang.org/community/) for a number of communication channels.

For now, the best way to ask a question is to reach out to [Mathieu Besançon](https://github/matbesancon) or [Sebastian Pokutta](github.com/pokutta).
You can also ask your question on [discourse.julialang.org](https://discourse.julialang.org) in the optimization topic or on the Julia Slack
on `#mathematical-optimization`, see [the Julia community page](https://julialang.org/community/) to gain access.

## How can I file an issue

If you found a bug or want to propose a feature, we track our issues within the [GitHub repository](https://github.com/ZIB-IOL/FrankWolfe.jl/issues).
Once opened, you can edit the issue or add new comments to continue the conversation.

If you encounter a bug, send the stack trace (the lines appearing after the error occurred containing some source files)
and ideally a Minimal Working Example (MWE), a small program that reproduces the bug.

## How can I contribute

Contributing to the repository will likely be made in a Pull Request (PR).
You will need to:
1. Fork the repository
2. Clone it on your machine to perform the changes
3. Create a branch for your modifications, based on the branch you want to merge on (typically master)
4. Push to this branch on your fork
5. The GitHub web interface will then automatically suggest opening a PR onto the original repository.

See the GitHub [guide to creating PRs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) for more help on workflows using Git and GitHub.

A PR should do a single thing to reduce the amount of code that must be reviewed.
Do not run the formatter on the whole repository except if your PR is specifically about formatting.

### Improve the documentation

The documentation can be improved by changing the files in `docs/src`,
for example to add a section in the documentation, expand a paragraph or add a plot.
The documentation attached to a given type of function can be modified in the source files directly, it appears above the thing you try to document
with three double quotations mark like this:
```julia
"""
This explains what the function `f` does, it supports markdown.
"""
function f(x)
    # ...
end
```

### Provide a new example or test

If you fix a bug, one would typically expect to add a test that validates that the bug is gone.
A test would be added in a file in the `test/` folder, for which the entry point is `runtests.jl`.

The `examples/` folder features several examples covering different problem settings and algorithms.
The examples are expected to run with the same environment and dependencies as the tests using
[TestEnv](https://github.com/JuliaTesting/TestEnv.jl).
If the example is lightweight enough, it can be added to the `docs/src/examples/` folder which generates
pages for the documentation based on Literate.jl.

### Provide a new feature

Contributions bringing new features are also welcome.
If the feature is likely to impact performance, some benchmarks should be run with `BenchmarkTools` on several
of the examples to assert the effect at different problem sizes.
If the feature should only be active in some cases, a keyword should be added to the main algorithms to support it.

Some typical features to implement are:
1. A new Linear Minimization Oracle (LMO)
2. A new step size
3. A new algorithm (less frequent) following the same API.

### Code style

We try to follow the [Julia documentation guidelines](https://docs.julialang.org/en/v1/manual/documentation/).
We run [`JuliaFormatter.jl`](https://github.com/domluna/JuliaFormatter.jl) on the repo in the way set in the `.JuliaFormatter.toml` file, which enforces a number of conventions.

This contribution guide was inspired by [ColPrac](https://github.com/SciML/ColPrac) and the one in [Manopt.jl](https://github.com/JuliaManifolds/Manopt.jl).
