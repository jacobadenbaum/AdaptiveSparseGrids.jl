module AdaptiveSparseGrids

using StaticArrays
using Parameters
using LinearAlgebra

import LinearAlgebra: norm
import Base: show
export AdaptiveSparseGrid, AdaptiveIntegral

################################################################################
#################### 1D Basis Function Stuff ###################################
################################################################################

function m(i::Int)
    i >  1 && return 2 << (i-2)
    i == 1 && return 1
    throw(ArgumentError("i must be a nonnegative integer.  You passed $i"))
end

function Y(i::Int,j::Int)
    # Recover the dimension of the ith layer
    mi = m(i)

    # Let's check the indices -- j must be less than m(i)
    if !( j <= mi )
        msg = "j must be less than m(i).  You passed j=$j, which is larger than m($i) = $mi"
        throw(ArgumentError(msg))
    end

    # Return the indices
    mi > 2  && return j/(mi)
    mi == 1 && return 0.5
    mi == 2 && j == 0   && return 0.0
    mi == 2 && j == 2   && return 1.0

    throw(ArgumentError("j must be positive.  You passed j=$j"))
end

h(i) = 1/m(i)

function ϕ(x)
    -1 <= x <= 1 && return 1 - abs(x)
    return 0.0
end

function Dϕ(x)
    0  <= x <= 1 && return -1
    -1 <= x <  0 && return 1
    return 0
end


function ϕ(i,j,x)
    i > 2               && return ϕ(x*m(i) - j)
    i == 2 && j == 0    && return 0 <= x <= 0.5 ?  1 - 2 * x : 0.0
    i == 2 && j == 2    && return 0.5 <= x <= 1 ?  2 * x - 1 : 0.0
    i == 1              && return 1.0
    throw(error("i=$i, j=$j are not valid indices"))
end

ϕ(i::Int, j::Int) = x -> ϕ(i,j,x)

function Dϕ(i,j,x)
    i == 1              && return 0.0
    i > 2               && return (mi = m(i); Dϕ(x*mi - j) * mi)
    i == 2 && j == 0    && return 0 <= x <= 0.5 ?  - 2.0  : 0.0
    i == 2 && j == 2    && return 0.5 <= x <= 1 ?    2.0  : 0.0
    throw(error("i=$i, j=$j are not valid indices"))
end


function leftchild(i,j)
    i > 2               && return (i+1, 2j - 1)
    i == 2 && j == 0    && return (i+1, -1)
    i == 2 && j == 2    && return (i+1, 3)
    i == 1              && return (i+1, 0)
    throw(error("i=$i, j=$j are not valid indices"))
end

function rightchild(i,j)
    i > 2               && return (i+1, 2j + 1)
    i == 2  && j == 0   && return (i+1, 1)
    i == 2  && j == 2   && return (i+1, -1)
    i == 1              && return (i+1, 2)
    throw(error("i=$i, j=$j are not valid indices"))
end


################################################################################
#################### Nodes #####################################################
################################################################################
KTuple{N,T} = Union{NTuple{N,T},
                    NamedTuple{S,V} where {S, V <: NTuple{N,T}}} where {N,T}
KTuple{N}   = KTuple{N,T} where T

mutable struct Node{D,L,K,T<:KTuple{K}}
    parent::Int
    children::MMatrix{D, 2, Int, L}
    α::T
    x::SVector{D, Float64}
    fx::T
    l::NTuple{D, Int}
    i::NTuple{D, Int}
    depth::Int
end

getx(n::Node) = n.x

getzero(t::T) where {T <: KTuple}                   = T(zero(tv) for tv in t)
getzero(TT::Type{K}) where {N,T, K <: KTuple{N,T}}  = TT(zero(T) for i in 1:N)
getzero(t::Vector)                                  = Tuple(zeros(size(t)))
getzero(t::Number)                                  = (zero(t),)

function Node(parent, children, α, l::NTuple{N,Int}, i::NTuple{N,Int}, depth) where N
    x  = SVector{N,Float64}(Y.(l, i))
    fx = getzero(α)
    return Node(parent, children, α, x, fx, l, i, depth)
end

function leftchild(idx::Int, p::Node{D,L,K}, d) where {D, L, K}
    # Compute child along the dth dimension
    lc, ic = leftchild(p.l[d], p.i[d])

    # When l[d] == 2, there's the possibility that there is no left child
    ic == -1 && return nothing

    return Node(idx,
                MMatrix{D,2,Int}(zeros(Int, D, 2)),
                getzero(p.α),
                (p.l[1:d-1]..., lc, p.l[d+1:end]...),
                (p.i[1:d-1]..., ic, p.i[d+1:end]...),
                p.depth + 1)
end

function rightchild(idx::Int, p::Node{D,L,K}, d) where {D, L, K}
    # Compute child along the dth dimension
    lc, ic = rightchild(p.l[d], p.i[d])

    # When l[d] == 2, there's the possibility that there is no right child
    ic == -1 && return nothing

    return Node(idx,
                MMatrix{D,2,Int}(zeros(Int, D, 2)),
                getzero(p.α),
                (p.l[1:d-1]..., lc, p.l[d+1:end]...),
                (p.i[1:d-1]..., ic, p.i[d+1:end]...),
                p.depth + 1)
end

ϕ(p::Node, x, d) = ϕ(p.l[d], p.i[d], x[d])

function ϕ(p::Node{D,L,K}, x) where {D,L,K}
    u = 1.0
    for d in 1:D
        ud = ϕ(p, x, d)
        if ud > 0
            u *= ud
        else
            return 0.0
        end
    end
    return u
end

################################################################################
#################### Function Representation ###################################
################################################################################

mutable struct AdaptiveSparseGrid{N, K, L, T} <: Function
    nodes::Vector{Node{N, L, K, T}}
    bounds::SMatrix{N, 2, Float64, L}
    depth::Int
    max_depth::Int
end

getT(::AdaptiveSparseGrid{N,K,L,T}) where {N,K,L,T} = T

dims(fun::AdaptiveSparseGrid{N,K,L,T}) where {N,K,L,T} = (N, K)
dims(fun, i) = dims(fun)[i]

function AdaptiveSparseGrid(f::Function, lb, ub; tol = 1e-3, max_depth = 10)
    N  = length(lb)
    @assert N == length(ub)

    # Evaluate the function once to get the output dimensions/types
    fx = f((lb .+ ub)./2)

    # Make the initial node
    head = Node(0,
                MMatrix{N, 2}(zeros(Int, N,2)),
                getzero(fx),
                Tuple(1 for i in 1:N),
                Tuple(1 for i in 1:N),
                1)
    nodes = [head]

    # Bounds
    bounds = SMatrix{N, 2}(hcat(lb, ub))

    # Construct the approximation, and then fit it
    fun = AdaptiveSparseGrid(nodes, bounds, 1, max_depth)
    fit!(f, fun, tol = tol)

    return fun
end

function Base.show(io::IO, fun::AdaptiveSparseGrid)
    N, K = dims(fun)
    println(io, "Sparse Adaptive Function Representation: R^$N → R^$K")
    println(io, "    nodes: $(fun.nodes |> length)")
    println(io, "    depth: $(fun.max_depth)")
    println(io, "    domain: $(fun.bounds)")
end

################################################################################
#################### Evaluating the Interpolant ################################
################################################################################

@generated function takeT(::Type{T}, x) where {N, T <: KTuple{N}}
    ex = Expr(:tuple, Tuple( :(x[$i]) for i in 1:N)...)
    return quote
        T($ex)
    end
end

function (fun::AdaptiveSparseGrid{N,1,L,T})(x) where {N,L,T}
    return evaluate(fun, scale(fun, x), 1)
end

function (fun::AdaptiveSparseGrid{N,1,L,T})(x...) where {N,L,T}
    return evaluate(fun, scale(fun, x), 1)
end

function (fun::AdaptiveSparseGrid{N,1,L,T})(x, s::Symbol) where {N,L,T}
    return evaluate(fun, scale(fun, x), s)
end

function (fun::AdaptiveSparseGrid{N,1,L,T})(x, s::Int) where {N,L,T}
    return evaluate(fun, scale(fun, x), s)
end

function (fun::AdaptiveSparseGrid)(x)
    return takeT(getT(fun), evaluate(fun, scale(fun, x)))
end

function (fun::AdaptiveSparseGrid)(x, k::Int)
    return evaluate(fun, scale(fun, x), k)
end

getkeys(::Type{NamedTuple{K,T}}) where {K,T} = K

function (fun::AdaptiveSparseGrid)(x, s::Symbol)
    T = getT(fun)
    T <: NamedTuple || throw(KeyError(s))

    k = findfirst(isequal(s), getkeys(getT(fun)))
    return evaluate(fun, scale(fun, x), k)
end

function rescale(fun::AdaptiveSparseGrid, x)
    N, K            = dims(fun)
    @unpack bounds  = fun
    return bounds[:, 1] .+ x .* (bounds[:,2] - bounds[:,1])
end

function scale(fun::AdaptiveSparseGrid, x)
    N, K            = dims(fun)
    @unpack bounds  = fun

    # Check the bounds
    for d in 1:N
        bounds[d,1] <= x[d] <= bounds[d,2] || throw(ArgumentError("$x is out of bounds"))
    end

    return (x .- bounds[:,1]) ./ (bounds[:,2] .- bounds[:,1])
end

function evaluate(fun::AdaptiveSparseGrid, x)
    K = dims(fun, 2)
    y = MVector{K}(zeros(K))
    evaluate!(y, fun, x)
end

evaluate(fun::AdaptiveSparseGrid, x, k)         = evaluate_recursive!(makework(fun,x),    fun, 1, 1, x, k)
evaluate!(y, fun::AdaptiveSparseGrid, x)        = evaluate_recursive!(y, makework(fun,x), fun, 1, 1, x)
evaluate!(y, wrk, fun::AdaptiveSparseGrid, x)   = evaluate_recursive!(y, wrk, fun, 1, 1, x)

function makework(fun, x)
    L = dims(fun,1) + fun.max_depth + 1
    T = promote_type(Float64, eltype(x))
    return ones(T, L)
end

function evaluate_recursive!(y, wrk, fun::AdaptiveSparseGrid, idx::Int, dimshift, x)
    # Dimensions of domain/codomain
    N, K = dims(fun)

    # Get the node that we're working on now
    @inbounds node  = fun.nodes[idx]
    @inbounds depth = node.depth

    # We have stored the basis function evaluations for every dimension except
    # dimshift -- move that dimension to the end (for storage, so we can put it
    # back later)
    wrk[N+depth]    = wrk[dimshift]
    wrk[dimshift]   = ϕ(node, x, dimshift)

    # Compute the product across all the dimensions
    u = prod(1:N) do d
        @inbounds wrk[d]
    end

    # Add in the the contribution of this node to the running sum
    @inbounds @simd for k in 1:K
        y[k] += u * node.α[k]
    end

    # If the contribution of this node is nonzero (i.e, x lies in the support of
    # this basis function), then we continue checking all of it's children
    if u > 0
        for d in 1:N
            kd = childsplit(node, x, d)
            kd == 0 && continue

            child = node.children[d,kd]
            if child  > 0
                evaluate_recursive!(y, wrk, fun, child, d, x)
            end
        end
    end

    # We have to clean up the work buffer now (we want all entries with index <
    # N + depth put back the way we found them)
    @inbounds wrk[dimshift] = wrk[N + depth]

    return y
end

function childsplit(n::Node, x, d; inclusive=false)
    if n.x[d] > x[d] || inclusive && n.x[d] == x[d]
        return 1
    elseif n.x[d] < x[d]
        return 2
    else
        return 0
    end
end

get(x::KTuple, i::Int)      = x[i]
get(x::KTuple, s::Symbol)   = getproperty(x, s)

function evaluate_recursive!(wrk, fun::AdaptiveSparseGrid, idx::Int, dimshift, x, k)
    # Dimensions of domain/codomain
    N, K = dims(fun)

    # Get the node that we're working on now
    node = @inbounds fun.nodes[idx]
    depth = node.depth

    # We have stored the basis function evaluations for every dimension except
    # dimshift -- move that dimension to the end (for storage, so we can put it
    # back later)
    @inbounds wrk[N+depth]    = wrk[dimshift]
    @inbounds wrk[dimshift]   = ϕ(node, x, dimshift)

    # Compute the product across all the dimensions
    @inbounds u = prod(1:N) do d
        wrk[d]
    end

    # Add in the the contribution of this node to the running sum
    y = u * get(node.α, k)

    # If the contribution of this node is nonzero (i.e, x lies in the support of
    # this basis function), then we continue checking all of it's children
    if u > 0
        for d in 1:N
            kd = childsplit(node, x, d)
            kd == 0 && continue

            child = node.children[d,kd]
            if child  > 0
                y += evaluate_recursive!(wrk, fun, child, d, x, k)
            end
        end
    end

    # We have to clean up the work buffer now (we want all entries with index <
    # N + depth put back the way we found them)
    @inbounds wrk[dimshift] = wrk[N + depth]

    return y
end

################################################################################
#################### Fitting the Interpolant ###################################
################################################################################

function fit!(f, fun::AdaptiveSparseGrid; kwargs...)
    # We need to evaluate f on the base node
    train!(f, fun, fun.nodes)

    while fun.depth < fun.max_depth
        refinegrid!(f, fun; kwargs...)
    end
    return fun
end

err(node::Node,d=:) = norm(err.(Tuple(node.α)[d], Tuple(node.fx)[d]), Inf)
err(a, f) = abs(a) / max(abs(f), 1)

# err(node::Node) = norm(node.α, Inf)

"""
Proceeds in 4 steps
    1) Obtain the list of possible child nodes to be created
    2) Remove duplicates
    3) Evalute the function
    4) Insert the points into the grid
"""
function refinegrid!(f, fun::AdaptiveSparseGrid; kwargs...)
    # Get the list of possible child nodes
    children = procreate!(fun; kwargs...)

    # Delete duplicates (each node should only be the child of a single parent)
    sort!(children, by = n -> (id(n)..., n.parent))
    unique!(id, children)

    # Compute the gain for each child
    train!(f, fun, children)

    # Insert the new children into the main function
    drive_to_college!(fun, children; kwargs...)

    # Increment the depth counter
    fun.depth += 1
end

id(n::Node) = (n.l..., n.i...)

function procreate!(fun; tol = 1e-3)
    # Dimensions of function (Domain -> Codomain)
    N, K = dims(fun)

    # Get the list of possible child nodes
    TN       = eltype(fun.nodes)
    children = Vector{TN}(undef, 0)
    for (idx, node) in enumerate(fun.nodes)
        if node.depth == fun.depth
            # Check the error at this node -- if it's low enough, we can stop
            # refining in this area.
            #
            # Note: We insist on refining up to at least the 3rd layer to make
            # sure that we don't stop prematurely
            node.depth > 5 && err(node) < tol && continue

            # Add in the children -- this should be a separate function
            for d in 1:N
                addchildren!(children, idx, node, d)
            end

        end
    end
    return children
end

function train!(f, fun::AdaptiveSparseGrid{N,K,L,T}, children) where {N,K,L,T}
    # Evaluate the function and compute the gain for each of the children
    # Note: This should be done in parallel, since this is where all of the hard
    # work (computing function evaluations) happens
    @sync for child in children
        Threads.@spawn begin
            x           = getx(child)
            child.fx    = T(f(rescale(fun, x)))
            ux          = evaluate(fun, x)
            child.α     = T(Tuple(child.fx) .- ux)
        end
    end
    return
end

"""
This function inserts the children into the list of nodes, and sets up the
parent/child linkages that allow the child to be used in function evaluations
"""
function drive_to_college!(fun, children; tol=1e-3)
    for child in children

        push!(fun.nodes, child)

        # Update its parent (so that we can find it later)
        idx                     = length(fun.nodes)
        pid                     = child.parent
        parent                  = fun.nodes[pid]
        dd                      = whichchild(parent, child)
        parent.children[dd...]  = idx
    end
end

function addchildren!(children, idx, node, d)
    # Add the left child
    child = leftchild(idx, node, d)
    if !isnothing(child)
        push!(children, child)
    end

    # Add the right child
    child = rightchild(idx, node, d)
    if !isnothing(child)
        push!(children, child)
    end
end

function whichchild(parent, child)
    # Which dimension is different
    Δl = child.l .- parent.l
    sum(Δl) == 1 || throw(error("Something has gone horribly wrong"))
    d  = findfirst(isequal(1), Δl)

    # Did we split up or down?
    if child.i[d] == leftchild(parent.l[d], parent.i[d])[2]
        return (d, 1)
    elseif child.i[d] == rightchild(parent.l[d], parent.i[d])[2]
        return (d, 2)
    else
        @show parent, child
        throw(error("Something has gone horribly wrong!"))
    end

end

################################################################################
#################### Integration ###############################################
################################################################################

struct AdaptiveIntegral{T<:AdaptiveSparseGrid}
    fun::T
    dims::Set{Int}
    idims::Vector{Int}
end

function AdaptiveIntegral(fun::AdaptiveSparseGrid, dd)
    # Check the dimensions
    N   = dims(fun,1)
    all(d -> d <= N, dd) || begin
        msg = "Invalid integration dimensions $dims with $(N)D integrand"
        throw(ArgumentError(msg))
    end

    sdims = Set(dd)
    idims = setdiff(1:N, sdims)

    # Let's make sure the dimensions are unique and sorted
    return AdaptiveIntegral(fun, sdims, idims)
end


function AdaptiveIntegral(f::Function, lb, ub, dd; kwargs...)
    fun = AdaptiveSparseGrid(f, lb, ub; kwargs...)
    return AdaptiveIntegral(fun, dd)
end

function (int::AdaptiveIntegral)(x)
    # Compute the integral
    v  = integrate(int, scale(int, intx(int, x)))
    bd = int.fun.bounds

    # Compute Scaling Factor
    s = 1.0
    for d in int.dims
        s *= (bd[d,2] - bd[d,1])
    end

    return v * s
end

(int::AdaptiveIntegral)(x, k) = int(x)[k]

function (int::AdaptiveIntegral)()
    if dims(int.fun, 1) == length(int.dims)
        return int(Float64[])
    else
        throw(ArgumentError("You must specify a point to evaluate the integral at"))
    end
end

function intx(int::AdaptiveIntegral, x)
    N  = dims(int.fun, 1)
    T  = promote_type(eltype(x), Float64)
    xx = zeros(T, N)

    i = 0
    for d in 1:N
        in(d, int.dims) && continue
        i += 1
        xx[d] = x[i]
    end
    return xx
end

function integrate(int::AdaptiveIntegral, x)
    T    = promote_type(eltype(x), Float64)
    y    = zeros(T, dims(int.fun, 2))
    wrk  = makework(int.fun, x)
    return integrate_recursive!(y, wrk, int, 1, 1, x)
end

function scale(int::AdaptiveIntegral, x)
    N, K            = dims(int.fun)
    @unpack bounds  = int.fun

    for d in int.idims
        bounds[d,1] <= x[d] <= bounds[d,2] || throw(ArgumentError("$x is out of bounds"))
    end

    return (x .- bounds[:,1]) ./ (bounds[:,2] .- bounds[:,1])
end


function integrate_recursive!(y, wrk, int::AdaptiveIntegral, idx::Int, dimshift, x)
    # Dimensions of domain/codomain
    fun  = int.fun
    N, K = dims(fun)

    # Get the node that we're working on now
    @inbounds node  = fun.nodes[idx]
    @inbounds depth = node.depth

    # We have stored the basis function evaluations for every dimension except
    # dimshift -- move that dimension to the end (for storage, so we can put it
    # back later)
    wrk[N+depth]    = wrk[dimshift]
    wrk[dimshift]   = in(dimshift, int.dims) ?
                        I(node,dimshift)     :
                        ϕ(node, x, dimshift)
    # Compute the product across all the dimensions
    u = prod(1:N) do d
        @inbounds wrk[d]
    end

    # Add in the the contribution of this node to the running sum
    @inbounds @simd for k in 1:K
        y[k] += u * node.α[k]
    end

    # If the contribution of this node is nonzero (i.e, x lies in the support of
    # this basis function), then we continue checking all of it's children
    if u > 0
        for d in 1:N
            # Are we considering in an integration dimension
            dd = in(d, int.dims)

            # If not, we can compute which side to split along
            if !dd
                kd = childsplit(node, x, d)
                kd == 0 && continue
            end

            # Along the integration dimension, we will always follow all the
            # splits. But in a non-integration dimension, we can just follow the
            # binary search tree
            for split in 1:2
                !dd && kd != split && continue

                child = node.children[d, split]
                if child  > 0
                    integrate_recursive!(y, wrk, int, child, d, x)
                end
            end
        end
    end

    # We have to clean up the work buffer now (we want all entries with index <
    # N + depth put back the way we found them)
    @inbounds wrk[dimshift] = wrk[N + depth]

    return y
end

function I(l)
    l >  2 && return 1/(2 << (l-2))
    l == 2 && return 1/4
    l == 1 && return 1.0
end

I(n::Node, d) = I(n.l[d])
################################################################################
##################### Helper Utilities #########################################
################################################################################

function norm(f1::T, f2::T, p=Inf; dim = :) where {T <: AdaptiveSparseGrid}
    # Get the set of evaluation points (union of both functions)
    Xs = vcat(rescale.(f1, getx.(f1.nodes)),
              rescale.(f2, getx.(f2.nodes))) |> sort! |> unique!

    chx = Channel{eltype(Xs)}(Threads.nthreads()) do c
        for x in Xs
            put!(c,x)
        end
    end

    # Split up the work among the threads
    accum = zeros(Threads.nthreads())
    @sync for thread in 1:Threads.nthreads()
        Threads.@spawn begin
            id = Threads.threadid()
            for x in chx
                d = diff(f1, f2, x, dim)
                if isinf(p)
                    accum[id] = max(accum[id], reduce(max, d))
                else
                    accum[id] += mapreduce(x->x^p, +, d)
                end
            end
        end
    end

    # Accumulate the results from each thread, and raise to the power of 1/p (or
    # take the max if it's the Inf norm)
    if isinf(p)
        return reduce(max, accum)
    else
        return reduce(+, accum)^(1/p)
    end
end

function diff(f1, f2, x, dim=:)
    f1v = f1(x) |> Tuple
    f2v = f2(x) |> Tuple

    return rel_err.(f1v[dim], f2v[dim])
end

rel_err(v1,v2) = abs(v1 - v2)/max(min(abs(v1), abs(v2)), 1.0)

function getx(fun::AdaptiveSparseGrid)
    return rescale.(fun, getx.(fun.nodes))
end

getα(fun::AdaptiveSparseGrid)  = [n.α  for n in fun.nodes]
getf(fun::AdaptiveSparseGrid)  = [n.fx for n in fun.nodes]
nodes(fun::AdaptiveSparseGrid) = fun.nodes

Base.length(fun::AdaptiveSparseGrid) = length(fun.nodes)

export getx, getα, getf, nodes

end
