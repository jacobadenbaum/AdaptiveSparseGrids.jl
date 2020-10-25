# AdaptiveSparseGrids.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jacobadenbaum.github.io/AdaptiveSparseGrids.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jacobadenbaum.github.io/AdaptiveSparseGrids.jl/dev)
[![Build Status](https://github.com/jacobadenbaum/AdaptiveSparseGrids.jl/workflows/CI/badge.svg)](https://github.com/jacobadenbaum/AdaptiveSparseGrids.jl/actions)
[![Coverage](https://codecov.io/gh/jacobadenbaum/AdaptiveSparseGrids.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jacobadenbaum/AdaptiveSparseGrids.jl)

This repository provides an implementation of Linear Interpolation via sparse
adaptive grids in Julia.  See Ma and Zabras (2009) or Brumm and Scheidegger
(2017) for more details on the mathematics.

Basic construction/usage:
```julia
using AdaptiveSparseGrids

# Bounds
lb  = zeros(2)
ub  = ones(2)

# True function to approximate (in practice, this function is costly to
# evaluate)
f(x) = 1/(sum(xv^2 for xv in x) + 0.3)


# Construct our approximation (this will evaluate f at the needed points, using
# all available threads)
fun = AdaptiveSparseGrid(f, lb, ub,
                         max_depth = 10,    # The maximum depth of the tree of
                                            # basis elements
                         tol = 1e-3)        # Add nodes when min(abs(alpha/f(x)),
                                            # abs(alpha)) < tol

# Evaluating fun
x = rand(2)
fun(x)          # returns value of fun at x
fun(x, 1)       # returns value of fun[1] at x (if f: R^n -> R^m with m > 1)

# Check how many basis elements we used (dimension of the approximation in
# function space)
length(fun.nodes)
```
