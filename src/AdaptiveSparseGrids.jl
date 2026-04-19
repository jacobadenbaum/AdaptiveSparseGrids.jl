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

# Unchecked hot-path variant. Only valid for i >= 1; used by ϕ / I on
# grid nodes where `l[d]` is always >= 1. Dropping the throw branch
# lets LLVM eliminate the GC-frame setup inside evaluate_recursive.
@inline _m(i::Int) = i > 1 ? (2 << (i-2)) : 1

function Y(i::Int,j::Int)
    # Recover the dimension of the ith layer
    mi = m(i)

    # Let's check the indices -- j must be less than m(i)
    if !( 0 <= j <= mi )
        msg = """
        j must be in {0, 1, ...,  m(i)}.  You passed j=$j
        """
        throw(ArgumentError(msg))
    end

    # Return the indices
    mi > 1  && return j/(mi)
    mi == 1 && return 0.5

    throw(ArgumentError("j must be positive.  You passed j=$j"))
end

h(i) = 1/m(i)

function ϕ(x::Real)
    -1 <= x <= 1 && return 1 - abs(x)
    return zero(x)
end

function Dϕ(x::Real)
    0  <= x <= 1 && return -1
    -1 <= x <  0 && return 1
    return zero(x)
end


@inline function ϕ(i, j, x)
    # No throw path: LLVM would otherwise emit a GC frame for the
    # ArgumentError string interpolation even though the branch is
    # unreachable with well-formed (l, i) on the grid. Only call
    # with a valid (i, j) pair; this is internal.
    if i > 2
        return ϕ(x*_m(i) - j)
    elseif i == 2
        j == 0 && return 0.0 <= x <= 0.5 ? 1 - 2*x : 0.0
        j == 2 && return 0.5 <= x <= 1   ? 2*x - 1 : 0.0
        return ϕ(2*x - 1)            # j == 1
    else                             # i == 1
        return 0.0 <= x <= 1 ? 1.0 : 0.0
    end
end

ϕ(i::Int, j::Int) = x -> ϕ(i,j,x)

function Dϕ(i,j,x)
    i == 1              && return 0.0
    i > 2               && return (mi = m(i); Dϕ(x*mi - j) * mi)
    i == 2 && j == 0    && return 0 <= x <= 0.5 ?  - 2.0  : 0.0
    i == 2 && j == 2    && return 0.5 <= x <= 1 ?    2.0  : 0.0
    throw(ArgumentError("i=$i, j=$j are not valid indices"))
end


function leftchild(i::Int,j::Int)
    i > 2               && return (i+1, 2j - 1)
    i == 2 && j == 0    && return (i+1, -1)
    i == 2 && j == 2    && return (i+1, 3)
    i == 1              && return (i+1, 0)
    throw(ArgumentError("i=$i, j=$j are not valid indices"))
end

function rightchild(i::Int,j::Int)
    i > 2               && return (i+1, 2j + 1)
    i == 2  && j == 0   && return (i+1, 1)
    i == 2  && j == 2   && return (i+1, -1)
    i == 1              && return (i+1, 2)
    throw(ArgumentError("i=$i, j=$j are not valid indices"))
end

function parent(i::Int,j::Int)
    i >  3              && return (i-1, div(j, 4) * 2 + 1)
    i == 3  && j == 1   && return (2, 0)
    i == 3  && j == 3   && return (2, 2)
    i == 2  && j == 0   && return (1,1)
    i == 2  && j == 2   && return (1,1)
    i == 1              && return (0,0)
    throw(ArgumentError("i=$i, j=$j are not valid indices"))
end


################################################################################
#################### Nodes #####################################################
################################################################################
KTuple{N,T} = Union{NTuple{N,T},
                    NamedTuple{S,V} where {S, V <: NTuple{N,T}}} where {N,T}
KTuple{N}   = KTuple{N,T} where T

mutable struct Node{D,K,T<:KTuple{K}}
    # --- Hot fields (accessed on every eval/integrate visit) ---
    α::T
    x::SVector{D, Float64}
    l::NTuple{D, Int}
    # Pre-linked children along each dimension. left[d] / right[d] hold
    # the left / right child in dimension d, or `nothing` if no such child
    # exists in the grid. Populated by link_to_parents! as nodes are
    # inserted; consumed by evaluate_recursive / integrate_recursive! so
    # traversal avoids Dict lookups entirely.
    left::NTuple{D, Union{Nothing, Node{D,K,T}}}
    right::NTuple{D, Union{Nothing, Node{D,K,T}}}
    # --- Cold fields (fit-only) ---
    fx::T
    i::NTuple{D, Int}
    depth::Int

    function Node{D,K,T}(α::T, x::SVector{D,Float64}, fx::T,
                         l::NTuple{D,Int}, i::NTuple{D,Int},
                         depth::Int) where {D,K,T<:KTuple{K}}
        CT  = NTuple{D, Union{Nothing, Node{D,K,T}}}
        nils = CT(nothing for _ in 1:D)
        return new{D,K,T}(α, x, l, nils, nils, fx, i, depth)
    end
end

# Outer constructor so existing call sites don't need to know the
# type parameters explicitly.
Node(α::T, x::SVector{D,Float64}, fx::T, l::NTuple{D,Int}, i::NTuple{D,Int},
     depth::Int) where {D,K,T<:KTuple{K}} = Node{D,K,T}(α, x, fx, l, i, depth)

function Node(T::Type{S}, l, i) where {S <: KTuple}
    lt = Tuple(l)
    it = Tuple(i)
    N  = length(lt)

    return Node(getzero(T),
                SVector{N}(Y.(lt,it)),
                getzero(T),
                lt, it, sum(l) - N + 1)
end

function Node(α, l::NTuple{N,Int}, i::NTuple{N,Int}, depth) where N
    x  = SVector{N,Float64}(Y.(l, i))
    fx = getzero(α)
    n  = Node(typeof(α), l,i)
    n.α      = α
    n.fx     = fx
    return n
end

Node(T::KTuple, l, i) = Node(typeof(T), l, i)
Node(l, i)            = Node(Tuple{Float64}, l, i)

# Dense, bits-typed snapshot of a Node for the evaluation hot path.
# Built from Node by sync_eval! and stored in a contiguous Vector on the
# grid. Child links are 1-based Int32 indices into that Vector (0 means
# no child along that dimension).
#
# Cold metadata (fx, depth) stays on the mutable Node used during fit!;
# EvalNode holds only what evaluate_recursive / integrate_recursive!
# actually read.
struct EvalNode{D, K, T<:KTuple{K}}
    α::T
    x::SVector{D, Float64}
    l::NTuple{D, Int}
    i::NTuple{D, Int}
    left::NTuple{D, Int32}
    right::NTuple{D, Int32}
end

getx(n::Node) = n.x
getT(::Node{N,K,T}) where {N,K,T} = T
dims(fun::Node{N,K,T}) where {N,K,T} = (N, K)

getzero(t::T) where {T <: KTuple}                   = T(zero(tv) for tv in t)
getzero(TT::Type{K}) where {N,T, K <: KTuple{N,T}}  = TT(zero(T) for i in 1:N)
getzero(t::Vector)                                  = Tuple(zeros(size(t)))
getzero(t::Number)                                  = (zero(t),)

function leftchild(p::Node, d)
    # Compute child along the dth dimension
    lc, ic = leftchild(p.l[d], p.i[d])

    # When l[d] == 2, there's the possibility that there is no left child
    ic == -1 && return nothing

    return Node(getT(p),
                (p.l[1:d-1]..., lc, p.l[d+1:end]...),
                (p.i[1:d-1]..., ic, p.i[d+1:end]...))
end

function rightchild(idx::Int, p::Node{D,K}, d) where {D, K}
    # Compute child along the dth dimension
    lc, ic = rightchild(p.l[d], p.i[d])

    # When l[d] == 2, there's the possibility that there is no right child
    ic == -1 && return nothing

    return Node(getT(p),
                (p.l[1:d-1]..., lc, p.l[d+1:end]...),
                (p.i[1:d-1]..., ic, p.i[d+1:end]...))
end

@inline ϕ(p::Node,     x, d) = @inbounds ϕ(p.l[d], p.i[d], x[d])
@inline ϕ(p::EvalNode, x, d) = @inbounds ϕ(p.l[d], p.i[d], x[d])

function ϕ(p::Node, x)
    D, K = dims(p)
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

struct Index{N}
    l::NTuple{N,Int}
    i::NTuple{N,Int}
    function Index(l::NTuple{N,Int}, i::NTuple{N,Int}, check=true) where N
        if check
            for (lk, ik) in zip(l,i)
                ik > m(lk) && throw(ArgumentError("($l,$i) is not a valid Index"))
            end
        end
        return new{N}(l, i)
    end
end
Base.getindex(idx::Index, k) = k == 1 ? idx.l : idx.i
Base.length(idx::Index)      = 2
Base.iterate(idx::Index)     = (idx[1], 2)
Base.iterate(idx::Index, s)  = s == 2 ? (idx[s], 3) : nothing


function Base.hash(idx::Index, h::UInt)
    u = zero(UInt)
    s = one(UInt)
    for (l, i) in zip(idx.l, idx.i)
        u += i * s
        s *= m(l)
    end
    return hash(u, h)
end

# Recursively apply
for f in [:leftchild, :rightchild, :parent]
    """
    ```
    $f(idx::$Index, d)
    ```
    Apply $f to the dth dimension of the index idx.
    """
    @eval function $f(idx::Index, d)
        l       = idx[1]
        i       = idx[2]

        if d == 1
            lc, ic = $f(l[1], i[1])
            return Index((lc, Base.tail(l)...), (ic, Base.tail(i)...), false)
        elseif d > 1
            lcc, icc = $f(Index(Base.tail(l), Base.tail(i), false), d-1)
            return Index((l[1], lcc...), (i[1], icc...), false)
        else
            throw(ArgumentError("Dimension $d must be positive"))
        end
    end
end

mutable struct AdaptiveSparseGrid{N, K, L, T}
    nodes::Dict{Index{N}, Node{N, K, T}}
    bounds::SMatrix{N, 2, Float64, L}
    depth::Int
    max_depth::Int
    # Flat eval view. Slot 1 is always the root. Kept in sync with `nodes`
    # by `sync_eval!`, which runs after every `drive_to_college!`.
    _eval::Vector{EvalNode{N, K, T}}
    _id::Dict{Index{N}, Int32}
end

function AdaptiveSparseGrid(nodes::Dict{Index{N}, Node{N,K,T}}, bounds::SMatrix{N,2,Float64,L},
                             depth::Int, max_depth::Int) where {N,K,L,T}
    fun = AdaptiveSparseGrid{N,K,L,T}(nodes, bounds, depth, max_depth,
                                       EvalNode{N,K,T}[], Dict{Index{N}, Int32}())
    sync_eval!(fun)
    return fun
end

getT(::AdaptiveSparseGrid{N,K,L,T}) where {N,K,L,T} = T
max_depth(f::AdaptiveSparseGrid) = f.max_depth

dims(::AdaptiveSparseGrid{N,K,L,T}) where {N,K,L,T} = (N, K)
dims(::Type{AdaptiveSparseGrid{N,K,L,T}}) where {N,K,L,T} = (N, K)
dims(fun, i) = dims(fun)[i]

function AdaptiveSparseGrid(f, lb, ub; tol = 1e-3, max_depth = 10, train = true)
    N  = length(lb)
    @assert N == length(ub)

    # Evaluate the function once to get the output dimensions/types
    fx = f((lb .+ ub)./2)


    # Make the initial node
    l    = Tuple(1 for i in 1:N)
    i    = Tuple(1 for i in 1:N)
    head = Node(getzero(fx), l, i)
    nodes = Dict(Index(l,i) => head)

    # Bounds
    bounds = SMatrix{N, 2}(hcat(SVector{N}(lb), SVector{N}(ub)))

    # Construct the approximation, and then fit it
    fun = AdaptiveSparseGrid(nodes, bounds, 1, max_depth)

    if train
        fit!(f, fun, tol = tol)
    end

    return fun
end

function Base.show(io::IO, fun::AdaptiveSparseGrid)
    N, K = dims(fun)
    println(io, "Sparse Adaptive Function Representation: R^$N → R^$K")
    println(io, "    nodes: $(fun.nodes |> length)")
    println(io, "    depth: $(max_depth(fun))")
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
    return bounds[:, 1] .+ SVector{N}(x) .* (bounds[:,2] - bounds[:,1])
end

function scale(fun::AdaptiveSparseGrid, x)
    N, K            = dims(fun)
    @unpack bounds  = fun

    # Check the bounds
    for d in 1:N
        bounds[d,1] <= x[d] <= bounds[d,2] || throw(ArgumentError("$x is out of bounds"))
    end

    return (SVector{N}(x) .- bounds[:,1]) ./ (bounds[:,2] .- bounds[:,1])
end

function base(fun::AdaptiveSparseGrid)
    if @generated
        N = dims(fun, 1)
        ex = Expr(:tuple, (:1 for i in 1:N)...) 
        return :(Index($ex, $ex))
    else
        N = dims(fun, 1)
        return Index(NTuple{N}(1 for i in 1:N), NTuple{N}(1 for i in 1:N))
    end
end

function evaluate(fun::AdaptiveSparseGrid, x)
    K = dims(fun, 2)
    y = @SVector zeros(K)
    evaluate!(y, fun, x)
end

evaluate(fun::AdaptiveSparseGrid, x, k)         = evaluate_recursive(makework(fun,x),    fun._eval, root(fun), 1, x, k)
evaluate!(y, fun::AdaptiveSparseGrid, x)        = evaluate_recursive(y, makework(fun,x), fun._eval, root(fun), 1, x)
evaluate!(y, wrk, fun::AdaptiveSparseGrid, x)   = evaluate_recursive(y, wrk,             fun._eval, root(fun), 1, x)

# Root EvalNode lives in slot 1 of _eval (invariant maintained by sync_eval!).
@inline root(fun::AdaptiveSparseGrid) = @inbounds fun._eval[1]

"""
    evaluate!(ys, fun, xs)

Evaluate `fun` at each point in `xs` in parallel, writing `fun(xs[i])` to
`ys[i]`. Traversal only reads from the grid, so it's safe to call across
threads once `fun` has been trained.
"""
function evaluate!(ys::AbstractVector, fun::AdaptiveSparseGrid,
                   xs::AbstractVector{<:Union{AbstractVector, Tuple}})
    length(ys) == length(xs) || throw(DimensionMismatch(
        "length(ys) = $(length(ys)) ≠ length(xs) = $(length(xs))"))
    _bulk_evaluate!(ys, fun, xs)
    return ys
end

# For scalar-codomain grids (K = 1) we run a single batch-coherent tree walk
# that shares every node load across the whole point cloud. For multi-codomain
# we fall back to a point-parallel walk (still thread-parallel, but one
# traversal per point).
_bulk_evaluate!(ys, fun::AdaptiveSparseGrid{N,1}, xs) where {N} =
    _bulk_evaluate_batched!(ys, fun, xs)

function _bulk_evaluate!(ys, fun::AdaptiveSparseGrid, xs)
    Threads.@threads for i in eachindex(xs)
        @inbounds ys[i] = fun(xs[i])
    end
    return ys
end

function _bulk_evaluate_batched!(ys, fun::AdaptiveSparseGrid{N,1,L,T},
                                  xs) where {N,L,T}
    M = length(xs)
    M == 0 && return ys
    fill!(ys, zero(eltype(ys)))

    nth = max(1, Threads.nthreads())
    chunk = cld(M, nth)

    Threads.@threads for t in 1:nth
        lo = (t-1)*chunk + 1
        hi = min(M, t*chunk)
        lo > hi && continue
        L_chunk = hi - lo + 1
        xs_scaled = [scale(fun, xs[i]) for i in lo:hi]
        wrk       = ones(Float64, N, L_chunk)
        subset    = collect(Int32(1):Int32(L_chunk))
        saved     = Float64[]
        left_buf  = Int32[]
        right_buf = Int32[]
        vys       = view(ys, lo:hi)
        @inbounds _batch_recurse!(vys, wrk, fun._eval, fun._eval[1],
                                   1, xs_scaled, subset,
                                   saved, left_buf, right_buf)
    end
    return ys
end

# In-place batch recursion. `subset` holds 1-based Int32 indices into `xs`
# (and `wrk`'s columns). `dimshift` is the dimension whose basis value is new
# at this node relative to its parent; `wrk`'s other rows are valid for every
# active point already. `saved`, `left_buf`, `right_buf` are per-thread
# reusable scratch buffers — resized but never freshly allocated inside the
# recursion. `saved` is used as a stack: each frame pushes its snapshot at the
# tail on entry and pops it on return, so child frames can't clobber the
# parent's restore data. Partition subsets are copied once per recursion to
# isolate the shared `left_buf`/`right_buf` state across sibling recursions.
function _batch_recurse!(ys, wrk::Matrix{Float64},
                         arr::Vector{EvalNode{D,K,T}},
                         node::EvalNode{D,K,T}, dimshift::Int,
                         xs, subset::AbstractVector{Int32},
                         saved::Vector{Float64},
                         left_buf::Vector{Int32},
                         right_buf::Vector{Int32}) where {D,K,T}
    isempty(subset) && return ys

    l_ds = node.l[dimshift]; i_ds = node.i[dimshift]
    α1 = node.α[1]
    nsub = length(subset)

    # Push this frame's snapshot of wrk[dimshift, i] at the tail of `saved`
    # (used as a stack), then write this node's basis value.
    save_off = length(saved)
    resize!(saved, save_off + nsub)
    @inbounds for (k, i) in pairs(subset)
        saved[save_off + k] = wrk[dimshift, i]
        wrk[dimshift, i] = ϕ(l_ds, i_ds, xs[i][dimshift])
    end

    # u[i] = prod_d wrk[d, i]; ys[i] += u * α
    @inbounds for i in subset
        u = 1.0
        for d in 1:D
            u *= wrk[d, i]
        end
        if u > 0
            ys[i] += u * α1
        end
    end

    @inbounds for d in 1:D
        lc, rc = node.left[d], node.right[d]
        if lc != Int32(0) || rc != Int32(0)
            empty!(left_buf); empty!(right_buf)
            nxd = node.x[d]
            for i in subset
                xd = xs[i][d]
                if xd < nxd
                    push!(left_buf, i)
                elseif xd > nxd
                    push!(right_buf, i)
                end
            end
            # Snapshot BOTH child subsets before any recursion. left_buf and
            # right_buf are shared scratch — the left child's recursion
            # reuses them for its own partitioning, clobbering right_buf and
            # the r_copy we would otherwise take after the left recursion.
            have_left  = lc != Int32(0) && !isempty(left_buf)
            have_right = rc != Int32(0) && !isempty(right_buf)
            if have_left && have_right
                l_copy = copy(left_buf)
                r_copy = copy(right_buf)
                _batch_recurse!(ys, wrk, arr, arr[lc], d, xs, l_copy,
                                 saved, left_buf, right_buf)
                _batch_recurse!(ys, wrk, arr, arr[rc], d, xs, r_copy,
                                 saved, left_buf, right_buf)
            elseif have_left
                l_copy = copy(left_buf)
                _batch_recurse!(ys, wrk, arr, arr[lc], d, xs, l_copy,
                                 saved, left_buf, right_buf)
            elseif have_right
                r_copy = copy(right_buf)
                _batch_recurse!(ys, wrk, arr, arr[rc], d, xs, r_copy,
                                 saved, left_buf, right_buf)
            end
        end
        node.l[d] > 1 && break
    end

    # Restore parent's wrk[dimshift, i] from this frame's stack slice and pop.
    @inbounds for (k, i) in pairs(subset)
        wrk[dimshift, i] = saved[save_off + k]
    end
    resize!(saved, save_off)
    return ys
end

function makework(fun, x::AbstractVector)
    N = dims(fun,1)
    T = promote_type(Float64, eltype(x))
    return @SVector ones(T, N)
end

function makework(fun, x)
    N = dims(fun, 1)
    T = promote_type(Float64, mapreduce(typeof, promote_type, x))
    return @SVector ones(T, N)
end

function evaluate_recursive(y, wrk, arr::Vector{EvalNode{D,K,T}},
                             node::EvalNode{D,K,T}, dimshift, x) where {D,K,T}
    # We have stored the basis function evaluations for every dimension except
    # dimshift
    wrk = @inbounds setindex(wrk, ϕ(node, x, dimshift), dimshift)

    # Compute the product across all the dimensions
    u = 1.0
    @inbounds for d in 1:D
        u *= wrk[d]
    end

    # Add in the the contribution of this node to the running sum
    @inbounds @simd for k in 1:K
        y = setindex(y, y[k] + u * node.α[k], k)
    end

    # Descend into children via Int32 indices into the flat array.
    # A child id of 0 means no child along that side/dimension.
    if u > 0
        @inbounds for d in 1:D
            kd = childsplit(node, x, d)
            if kd > 0
                cid = kd == 1 ? node.left[d] : node.right[d]
                if cid != Int32(0)
                    y = evaluate_recursive(y, wrk, arr, arr[cid], d, x)
                end
            end

            # Descend through the nodes lexicographically
            if node.l[d] > 1
                break
            end
        end
    end

    return y
end

@inline function childsplit(n::Node, x, d)
    @inbounds nxd = n.x[d]
    @inbounds xd  = x[d]
    nxd > xd && return 1
    nxd < xd && return 2
    return 0
end

@inline function childsplit(n::EvalNode, x, d)
    @inbounds nxd = n.x[d]
    @inbounds xd  = x[d]
    nxd > xd && return 1
    nxd < xd && return 2
    return 0
end

get(x::KTuple, i::Int)      = x[i]
get(x::KTuple, s::Symbol)   = getproperty(x, s)

function evaluate_recursive(wrk, arr::Vector{EvalNode{D,K,T}},
                             node::EvalNode{D,K,T}, dimshift, x, k) where {D,K,T}
    wrk = @inbounds setindex(wrk, ϕ(node, x, dimshift), dimshift)

    u = 1.0
    @inbounds for d in 1:D
        u *= wrk[d]
    end

    y = u * @inbounds(get(node.α, k))

    if u > 0
        @inbounds for d in 1:D
            kd = childsplit(node, x, d)
            if kd > 0
                cid = kd == 1 ? node.left[d] : node.right[d]
                if cid != Int32(0)
                    y += evaluate_recursive(wrk, arr, arr[cid], d, x, k)
                end
            end

            if node.l[d] > 1
                break
            end
        end
    end

    return y
end

################################################################################
#################### Fitting the Interpolant ###################################
################################################################################

function fit!(f, fun::AdaptiveSparseGrid; kwargs...)
    # We need to evaluate f on the base node
    train!(f, fun, collect(values(fun.nodes)))
    # The root node was already in fun.nodes before fit!, so drive_to_college!
    # (which is the normal sync point) never fired for it. Capture its
    # freshly-trained α in the flat _eval view before the first refine step.
    sync_eval!(fun)

    while true
        n = refinegrid!(f, fun; kwargs...)
        n == 0 && break
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

    raise_children!(f, fun, children)

    # Increment the depth counter
    fun.depth += 1
    return length(children)
end

function raise_children!(f, fun::AdaptiveSparseGrid, children)

    # Delete duplicates (each node should only be the child of a single parent)
    sort!(children, by = id)
    unique!(id, children)

    # Make sure all of the parents for each child are present (in the function)
    #   Note: this is trivial in the 1D case, but can be violated in higher
    #   dimensions unless we explicitly check for it
    child_support!(f, fun, children)

    # Compute the gain for each child
    train!(f, fun, children)

    # Insert the new children into the main function
    drive_to_college!(fun, children)

end

id(n::Node)    = (n.l..., n.i...)
index(n::Node) = Index(n.l, n.i)


depth(idx::Index{N}) where N = sum(idx[1]) + 1 - N
function procreate!(fun; tol = 1e-3)
    # Dimensions of function (Domain -> Codomain)
    N, K = dims(fun)

    # Get the list of possible child nodes
    TN       = eltype(values(nodes(fun)))
    children = Vector{TN}(undef, 0)
    for node in values(fun.nodes)
        if node.depth == fun.depth
            # Check the error at this node -- if it's low enough, we can stop
            # refining in this area.
            #
            # Note: We insist on refining up to at least the 3rd layer to make
            # sure that we don't stop prematurely
            node.depth > 6 && err(node) < tol   && continue


            # Add in the children -- this should be a separate function
            for d in 1:N
                # We also have a width criteria -- we won't continue adding nodes
                # past a certain level of refinement in each dimension
                node.l[d] >= max_depth(fun)     && continue
                addchildren!(children, node, d)
            end
        end
    end
    return children
end

function child_support!(f, fun, children)

    # Find the parents who aren't in the function representation
    TN       = eltype(values(nodes(fun)))
    parents  = Vector{TN}(undef, 0)
    for child in children
        find_parents!(fun, parents, child)
    end

    # If we have any orphan children, then raise their parents (recursively
    # searching for grandparents, etc...)
    if length(parents) > 0
        raise_children!(f, fun, parents)
    end
end


"""
Find the parents of the node `child` and insert them into the list of parents
"""
function find_parents!(fun::AdaptiveSparseGrid, parents, child)
    N, K = dims(fun)
    for d in 1:N
        child.l[d] == 1 && continue
        p = parent(index(child), d)
        if ! haskey(nodes(fun), p)
            push!(parents, Node(getT(fun), p...))
        end
    end
end


function hasparents(fun::AdaptiveSparseGrid, idx::Index)
    # Base case
    all(isequal(1), idx[1]) && return true

    # Recursively check for parents in each dimension
    N, K = dims(fun)
    for d in 1:N
        p = parent(idx, d)
        any(isequal(0), p[1]) && continue
        if (! haskey(nodes(fun), p)) || (! hasparents(fun, p))
            return false
        end
    end
    return true
end


function train!(f, fun::AdaptiveSparseGrid{N,K,L,T}, children) where {N,K,L,T}
    # Evaluate the function and compute the gain for each of the children
    # Note: This should be done in parallel, since this is where all of the hard
    # work (computing function evaluations) happens
    Threads.@threads for child in children
        x           = getx(child)
        child.fx    = T(f(rescale(fun, x)))
        ux          = evaluate(fun, x)
        child.α     = T(Tuple(child.fx) .- ux)
    end
    return
end

"""
This function inserts the children into the list of nodes for the function approximation
"""
function drive_to_college!(fun, children)
    for child in children
        fun.nodes[Index(child.l, child.i)] = child
    end
    link_to_parents!(fun, children)
    sync_eval!(fun)
end

"""
Rebuild the flat `_eval` vector from the current mutable `nodes` Dict.
Slot 1 always holds the root (the all-ones Index). This is invoked after
every `drive_to_college!` so subsequent `evaluate` calls (used inside
`train!` to compute the current approximation at new points) see an
up-to-date eval view.
"""
function sync_eval!(fun::AdaptiveSparseGrid{N,K,L,T}) where {N,K,L,T}
    isempty(fun.nodes) && (empty!(fun._eval); empty!(fun._id); return fun)

    # Assign stable positions: root → 1, then the rest in Dict order.
    empty!(fun._id); sizehint!(fun._id, length(fun.nodes))
    root_idx = base(fun)
    fun._id[root_idx] = Int32(1)
    next::Int32 = 2
    for idx in keys(fun.nodes)
        idx == root_idx && continue
        fun._id[idx] = next
        next += Int32(1)
    end

    resize!(fun._eval, length(fun.nodes))
    for (idx, n) in fun.nodes
        id = fun._id[idx]
        fun._eval[id] = EvalNode{N,K,T}(
            n.α, n.x, n.l, n.i,
            ntuple(d -> _eval_child_id(fun._id, n.left[d]),  Val(N)),
            ntuple(d -> _eval_child_id(fun._id, n.right[d]), Val(N)),
        )
    end
    return fun
end

_eval_child_id(::Dict, ::Nothing) = Int32(0)
_eval_child_id(id::Dict, n::Node) = Base.get(id, Index(n.l, n.i), Int32(0))

"""
Update each new child's parent node(s) so that the parent's `children`
field points to this child in the appropriate slot. This maintains the
invariant `evaluate_recursive` relies on: if a child exists in `fun.nodes`
then its parent's corresponding `children` slot holds a reference to it
(and the slot is `nothing` otherwise).
"""
function link_to_parents!(fun, new_children)
    for c in new_children
        D = length(c.l)
        for d in 1:D
            c.l[d] == 1 && continue
            p_idx = parent(Index(c.l, c.i), d)
            p = Base.get(fun.nodes, p_idx, nothing)
            p === nothing && continue
            p_l_d = p.l[d]; p_i_d = p.i[d]
            lch_ld, lch_id = leftchild(p_l_d, p_i_d)
            if (c.l[d], c.i[d]) == (lch_ld, lch_id)
                _set_left!(p, d, c)
            else
                rch_ld, rch_id = rightchild(p_l_d, p_i_d)
                if (c.l[d], c.i[d]) == (rch_ld, rch_id)
                    _set_right!(p, d, c)
                end
            end
        end
    end
end

function _set_left!(p::Node{D,K,T}, d::Int, c::Node{D,K,T}) where {D,K,T}
    old = p.left
    p.left = ntuple(s -> s == d ? c : old[s], Val(D))
    return
end

function _set_right!(p::Node{D,K,T}, d::Int, c::Node{D,K,T}) where {D,K,T}
    old = p.right
    p.right = ntuple(s -> s == d ? c : old[s], Val(D))
    return
end

function addchildren!(children, node, d)
    # Add the left child
    child = makeleftchild(node, d)
    if !isnothing(child)
        push!(children, child)
    end

    # Add the right child
    child = makerightchild(node, d)
    if !isnothing(child)
        push!(children, child)
    end
end

for f in [:rightchild, :leftchild]
    ff = Symbol(:make, f)
    @eval function $ff(node::Node, d)
        idx = $f(index(node), d)
        if any(isequal(-1), idx[2])
            return nothing
        else
            return Node(getT(node), idx...)
        end
    end
end



function whichchild(parent, child)
    # Which dimension is different
    Δl = child.l .- parent.l
    sum(Δl) == 1 || throw(error("Something has gone horribly wrong"))
    d  = findfirst(isequal(1), Δl)

    # Did we split up or down?
    if child.i[d] == leftchild(index(parent))[2]
        return (d, 1)
    elseif child.i[d] == rightchild(index(child))[2]
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


function AdaptiveIntegral(f, lb, ub, dd; kwargs...)
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
        return int(@SVector Float64[])
    else
        throw(ArgumentError("You must specify a point to evaluate the integral at"))
    end
end

function intx(int::AdaptiveIntegral, x)
    N  = dims(int.fun, 1)
    T  = promote_type(eltype(x), Float64)
    xx = @SVector zeros(T, N)

    i = 0
    for d in 1:N
        in(d, int.dims) && continue
        i += 1
        xx = setindex(xx, x[i], d)
    end
    return xx
end

function integrate(int::AdaptiveIntegral, x)
    T    = promote_type(eltype(x), Float64)
    y    = @SVector zeros(T, dims(int.fun, 2))
    wrk  = makework(int.fun, x)
    return integrate_recursive!(y, wrk, int, int.fun._eval, root(int.fun), 1, x)
end

function scale(int::AdaptiveIntegral, x)
    N, K            = dims(int.fun)
    @unpack bounds  = int.fun

    for d in int.idims
        bounds[d,1] <= x[d] <= bounds[d,2] || throw(ArgumentError("$x is out of bounds"))
    end

    return (x .- bounds[:,1]) ./ (bounds[:,2] .- bounds[:,1])
end


function integrate_recursive!(y, wrk, int::AdaptiveIntegral,
                               arr::Vector{EvalNode{D,K,T}},
                               node::EvalNode{D,K,T},
                               dimshift, x) where {D,K,T}
    # Integration dim? Use basis integral; else evaluate basis at x.
    newval = in(dimshift, int.dims) ?
                I(node, dimshift)  :
                ϕ(node, x, dimshift)
    wrk = @inbounds setindex(wrk, newval, dimshift)

    u = 1.0
    @inbounds for d in 1:D
        u *= wrk[d]
    end

    @inbounds @simd for k in 1:K
        y = setindex(y, y[k] + u * node.α[k], k)
    end

    if u > 0
        @inbounds for d in 1:D
            dd = in(d, int.dims)
            kd = dd ? 0 : childsplit(node, x, d)

            for split in 1:2
                !dd && kd != split && continue
                cid = split == 1 ? node.left[d] : node.right[d]
                if cid != Int32(0)
                    y = integrate_recursive!(y, wrk, int, arr, arr[cid], d, x)
                end
            end

            if node.l[d] > 1
                break
            end
        end
    end

    return y
end

@inline function I(l)
    # Hot path from integrate_recursive!. Only called with l >= 1 on
    # the grid; no throw branch so LLVM can drop the GC frame.
    l > 2  && return 1/(2 << (l-2))
    l == 2 && return 1/4
    return 1.0                     # l == 1
end

@inline I(n::Node,     d) = @inbounds I(n.l[d])
@inline I(n::EvalNode, d) = @inbounds I(n.l[d])
################################################################################
##################### Helper Utilities #########################################
################################################################################

function norm(f1::T, f2::T, p=Inf; dim = :) where {T <: AdaptiveSparseGrid}
    # Get the set of evaluation points (union of both functions)
    Xs = vcat(rescale.(f1, getx.(values(f1.nodes))),
              rescale.(f2, getx.(values(f2.nodes)))) |> sort! |> unique!

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
    return rescale.(fun, getx.(values(fun.nodes)))
end

getα(fun::AdaptiveSparseGrid)  = [n.α  for n in fun.nodes]
getf(fun::AdaptiveSparseGrid)  = [n.fx for n in fun.nodes]
nodes(fun::AdaptiveSparseGrid) = fun.nodes

Base.length(fun::AdaptiveSparseGrid) = length(fun.nodes)

export getx, getα, getf, nodes

end
