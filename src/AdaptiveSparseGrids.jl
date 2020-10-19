module AdaptiveSparseGrids

using StaticArrays
using Parameters
using LinearAlgebra

import LinearAlgebra: norm

export AdaptiveSparseGrid, fit!

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
struct Node{D,L,K}
    parent::Int
    children::MMatrix{D, 2, Int, L}
    α::MVector{K, Float64}
    x::SVector{D, Float64}
    fx::MVector{K, Float64}
    l::NTuple{D, Int}
    i::NTuple{D, Int}
    depth::Int
end

getx(n::Node) = n.x

function Node(parent, children, α, l::NTuple{N,Int}, i::NTuple{N,Int}, depth) where N
    x  = SVector{N,Float64}(Y.(l, i))
    K  = length(α)
    fx = MVector{K,Float64}(zeros(K))
    return Node(parent, children, α, x, fx, l, i, depth)
end

function leftchild(idx::Int, p::Node{D,L,K}, d) where {D, L, K}
    # Compute child along the dth dimension
    lc, ic = leftchild(p.l[d], p.i[d])

    # When l[d] == 2, there's the possibility that there is no left child
    ic == -1 && return nothing

    return Node(idx,
                MMatrix{D,2,Int}(zeros(Int, D, 2)),
                MVector{K,Float64}(zeros(Float64, K)),
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
                MVector{K,Float64}(zeros(Float64, K)),
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

mutable struct AdaptiveSparseGrid{N, K, L} <: Function
    nodes::Vector{Node{N, L, K}}
    bounds::SMatrix{N, 2, Float64, L}
    depth::Int
    max_depth::Int
end

function AdaptiveSparseGrid(K, lb, ub; max_depth = 10)
    N = length(lb)
    @assert N == length(ub)

    # Make the initial node
    head = Node(0,
                MMatrix{N, 2}(zeros(Int, N,2)),
                MVector{K}(zeros(K)),
                Tuple(1 for i in 1:N),
                Tuple(1 for i in 1:N),
                1)
    nodes = [head]

    # Bounds
    bounds = SMatrix{N, 2}(hcat(lb, ub))

    return AdaptiveSparseGrid(nodes, bounds, 1, max_depth)
end

function AdaptiveSparseGrid(f::Function, lb, ub; tol = 1e-3, kwargs...)
    N  = length(lb)
    @assert N == length(ub)
    fx = f((lb .+ ub)./2)
    K  = length(fx)

    fun = AdaptiveSparseGrid(K, lb, ub; kwargs...)
    fit!(f, fun, tol = tol)
end

################################################################################
#################### Evaluating the Interpolant ################################
################################################################################

function (fun::AdaptiveSparseGrid)(x)
    return evaluate(fun, scale(fun, x))
end

function (fun::AdaptiveSparseGrid)(x, k)
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

dims(fun::AdaptiveSparseGrid{N,K,L}) where {N,K,L} = (N, K)
dims(fun, i) = dims(fun)[i]

evaluate(fun::AdaptiveSparseGrid, x)            = evaluate!(zeros(dims(fun, 2)), fun, x)
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
    y  .+= u .* node.α

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
    y = u * node.α[k]

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

err(node::Node) = norm(min.(abs.(node.α), abs.(node.α ./ node.fx)), Inf)
# err(node::Node) = norm(node.α, Inf)

"""
Proceeds in 4 steps
    1) Obtain the list of possible child nodes to be created
    2) Remove duplicates
    3) Evalute the function
    4) Insert the points into the grid
"""
function refinegrid!(f, fun::AdaptiveSparseGrid; kwargs...)
    # Dimensions of function (Domain -> Codomain)
    N, K = dims(fun)

    # Get the list of possible child nodes
    children = procreate!(fun; kwargs...)

    # Delete duplicates (each node should only be the child of a single parent)
    sort!(children, by = n -> (id(n)..., n.parent))
    unique!(id, children)

    # Compute the gain for each child
    train!(f, fun, children)

    # Insert the new children into the main function
    drive_to_college!(fun, children)

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
            err(node) < tol && node.depth > 2 && continue

            # Add in the children -- this should be a separate function
            for d in 1:N
                addchildren!(children, idx, node, d)
            end

        end
    end
    return children
end

function train!(f, fun::AdaptiveSparseGrid, children)
    # Evaluate the function and compute the gain for each of the children
    # Note: This should be done in parallel, since this is where all of the hard
    # work (computing function evaluations) happens
    @sync for child in children
        Threads.@spawn begin
            x           = getx(child)
            child.fx   .= f(rescale(fun, x))
            ux          = evaluate(fun, x)
            child.α    .= child.fx .- ux
        end
    end
    return
end

"""
This function inserts the children into the list of nodes, and sets up the
parent/child linkages that allow the child to be used in function evaluations
"""
function drive_to_college!(fun, children)
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
#################### Gradients #################################################
################################################################################
#
# function ∇(fun::AdaptiveSparseGrid, x, k)::Vector{Float64}
#     N, K = dims(fun)
#     J    = zeros(N)
#     wrk  = zeros(N)
#
#     return recursive_jacobian!(J, wrk, fun, 1, x, k)
# end
#
# function recursive_jacobian!(J, u, fun::AdaptiveSparseGrid, idx::Int, x, k)
#     # Dimensions of domain/codomain
#     N, K = dims(fun)
#
#     # Get the node that we're working on now
#     node = fun.nodes[idx]
#
#     # Evaluate x against this node's basis function
#     for d in 1:N
#         u[d] = ϕ(node.l[d], node.i[d], x[d])
#     end
#     uu = prod(u)
#
#     if uu > 0
#         # Add in this node's contribution to the jacobian
#         for d in 1:N
#             if uu > 0
#                 # Most of the time, we can do this the fast way
#                 dϕdx = uu/u[d] * Dϕ(node.l[d],node.i[d], x[d])
#             else
#                 # Handle the special case with uu == 0
#                 # Technically, the derivative isn't defined here.  But we're
#                 # going to construct things using the "right side" derivative,
#                 # since otherwise the gradient looks really weird at the node
#                 # points
#                 # dϕdx = prod(1:N) do k
#                 #     k != d && u[k]
#                 #     k == d && Dϕ(node.l[d], node.i[d], x[d])
#                 # end
#                 dϕdx = 0.0
#             end
#             J[d] += dϕdx * node.α[k]
#         end
#     end
#
#     # If the contribution of this node is nonzero (i.e, x lies in the support of
#     # this basis function), then we continue checking all of it's children
#     for d in 1:N
#         kd = childsplit(node, x, d, inclusive=true)
#         kd == 0 && continue
#
#         child = node.children[d,kd]
#         if child  > 0
#             recursive_jacobian!(J, u, fun, child, x, k)
#         end
#     end
#
#     return J
# end

################################################################################
##################### Helper Utilities #########################################
################################################################################

function norm(f1::T, f2::T, args...) where {T <: AdaptiveSparseGrid}
    # Get the set of evaluation points (union of both functions)
    Xs = vcat(rescale.(Ref(f1), getx.(f1.nodes)),
              rescale.(Ref(f2), getx.(f2.nodes))) |> unique

    return norm( (diff(f1, f2, x) for x in Xs), args...)
end

function diff(f1, f2, x)
    f1v = f1(x)
    f2v = f2(x)
    return min.(abs.(f1v.-f2v),                           # Absolute Error
                abs.(f1v.-f2v)./max.(abs.(f1v),abs.(f2v)) # Relative Error
               )
end

end
