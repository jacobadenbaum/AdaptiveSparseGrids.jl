# AdaptiveSparseGrids.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jacobadenbaum.github.io/AdaptiveSparseGrids.jl/dev)
[![Build Status](https://github.com/jacobadenbaum/AdaptiveSparseGrids.jl/workflows/CI/badge.svg)](https://github.com/jacobadenbaum/AdaptiveSparseGrids.jl/actions)
[![codecov](https://codecov.io/gh/jacobadenbaum/AdaptiveSparseGrids.jl/branch/main/graph/badge.svg?token=IZoJg3QPPo)](undefined)

This repository provides an implementation of Linear Interpolation via sparse
adaptive grids (using hierarchical linear basis functions) in Julia.  
This interpolation method allows one to approximate high
dimensional functions to a high degree of accuracy with grid point requirements that grow
with a polynomial in the dimension (rather than exponentially).  In practice, this can be used
to approximate very high dimensional functions and integrals. 

The main drawback is that evaluating the interpolant is more costly than with standard interpolation methods, and 
grows with the dimension of the problem.  In many applications, this cost can be worth paying, since what one loses 
in more costly interpolation calls, one gains in needing to evaluate the (costly) function that is being approximated
at _far_ fewer grid points.  

See Ma and Zabras (2009) or Brumm and Scheidegger (2017) for more details on the mathematics. 

## Basic construction/usage:
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
                         max_depth = 10,    # The maximum depth of basis elements in 
                                            # each dimension
                         tol = 1e-3)        # Add nodes when 
                                            # min(abs(alpha/f(x)), abs(alpha)) < tol

# Evaluating fun
x = rand(2)
fun(x)          # returns value of fun at x
fun(x, 1)       # returns value of fun[1] at x (if f: R^n -> R^m with m > 1)

# Check how many basis elements we used (dimension of the approximation in
# function space)
length(fun)
```

## Functions can return named arguments
The return type of the functions can be named tuples.  You can reference the fieldnames later when accessing the results!

```julia
lb  = zeros(2)
ub  = ones(2)
fun = AdaptiveSparseGrid(lb, ub) do (x, y)
  (a = 1/( abs(0.5 - x^2 - y^2) + 0.3), 
   b = sin(x) * cos(y))
end

# Evaluate the function
x = [0.1, 0.2]
fun(x)          # returns (a = 1.3328486358686777, b = 0.09784904745121431)
fun(x, :a)      # returns 1.3328486358686775
fun(x).a        # returns 1.3328486358686775
```

## Integrating out over a dimension
You can also integrate out a dimension from your approximation
```julia
lb  = zeros(2)
ub  = [2pi, pi/4]
d   = 2 
fun = AdaptiveIntegral((x,y) -> sin(x) * cos(y), lb, ub, d)   # Integrates out the dth

# Evaluate the integral
fun(pi/2,1)     # returns 0.7070484622124219 (truth is sqrt(2)/2)
fun(pi/4,1)     # returns 0.4999225237591045 (truth is 0.5)
```
Specifying `d` as collection of dimensions will integrate out over all of them.  
I haven't yet implemented integrating over anything other than the full domain
in each of the specified dimensions, but that wouldn't be too hard to do.  

## Project Status
This repository is a work in progress.  API changes may come without warning (although I will obviously try not to break things where possible).  The exported API is quite simple, and so my guess is that there won't be many changes, although internals may get shifted around as needed.  
