#=
Small, reproducible benchmark used to track per-commit performance changes.
Designed to be fast (~30s total) and deterministic so output can be pasted
directly into commit messages.

Reports: single-point eval time, bulk-eval throughput, nodes visited per call.

Run:  julia --project=bench -t 1 bench/compare.jl
(single thread on purpose — traversal is serial today, so threads just
 add noise. `construction` uses threads internally via Threads.@threads.)
=#

using AdaptiveSparseGrids
using BenchmarkTools
using Printf
using Random
using StaticArrays

const SEED = 0xB0BACAFE

gauss(x) = exp(-sum(abs2, x) / length(x))
runge(x) = 1.0 / (1.0 + sum(abs2, x))

# Fixed workloads. Kept small enough for fast iteration.
const WORKLOADS = [
    (:gauss_3d, gauss, 3, 1e-3, 10),
    (:gauss_4d, gauss, 4, 1e-2,  9),
    (:runge_4d, runge, 4, 1e-2,  9),
    (:gauss_5d, gauss, 5, 1e-2,  8),
]

function build(f, N, tol, depth)
    lb = fill(-1.0, N); ub = fill(1.0, N)
    return AdaptiveSparseGrid(f, lb, ub; tol = tol, max_depth = depth)
end

function random_points(N, M; seed = SEED)
    rng = MersenneTwister(seed)
    return [SVector{N,Float64}(2 .* rand(rng, N) .- 1) for _ in 1:M]
end

function fmt_ns(ns)
    ns < 1e3   && return @sprintf("%6.1f ns", ns)
    ns < 1e6   && return @sprintf("%6.2f µs", ns / 1e3)
    return @sprintf("%6.2f ms", ns / 1e6)
end

function main()
    println("AdaptiveSparseGrids.jl — compare")
    println("threads=", Threads.nthreads(), "  julia=", VERSION, "  bench=", basename(@__FILE__))
    println()
    nthreads = Threads.nthreads()
    header = nthreads > 1 ?
        @sprintf("%-12s | %6s | %8s | %11s | %10s | %12s | %8s",
                 "workload", "N", "nodes", "eval (1 pt)", "bulk ser.",
                 "bulk par.", "visits") :
        @sprintf("%-12s | %6s | %8s | %11s | %10s | %8s",
                 "workload", "N", "nodes", "eval (1 pt)", "bulk ser.",
                 "visits")
    println(header)
    println("-" ^ length(header))

    for (name, f, N, tol, depth) in WORKLOADS
        fun = build(f, N, tol, depth)
        pts = random_points(N, 5_000)

        # Warmup
        for x in pts[1:100]; fun(x); end

        # Single-point eval (1 of the 5_000 points, median over many reps)
        b = @benchmark $fun($(pts[1])) samples=200 evals=5 seconds=2

        # Bulk serial throughput (Mcalls/s)
        ts = Float64[]
        for _ in 1:3
            t = @elapsed for x in pts; fun(x); end
            push!(ts, t)
        end
        sort!(ts); tser = ts[2]
        mser = length(pts) / tser / 1e6

        visits = count_visits(fun, pts[1])

        if nthreads > 1
            # Bulk parallel throughput via evaluate!(ys, fun, xs)
            ys = Vector{Float64}(undef, length(pts))
            AdaptiveSparseGrids.evaluate!(ys, fun, pts)  # warmup
            tp = Float64[]
            for _ in 1:3
                t = @elapsed AdaptiveSparseGrids.evaluate!(ys, fun, pts)
                push!(tp, t)
            end
            sort!(tp); tpar = tp[2]
            mpar = length(pts) / tpar / 1e6

            println(@sprintf("%-12s | %6d | %8d | %11s | %10.2f | %12.2f | %8d",
                              string(name), N, length(fun),
                              fmt_ns(minimum(b.times)), mser, mpar, visits))
        else
            println(@sprintf("%-12s | %6d | %8d | %11s | %10.2f | %8d",
                              string(name), N, length(fun),
                              fmt_ns(minimum(b.times)), mser, visits))
        end
    end
end

# Count node visits for one call, using the same traversal logic as
# evaluate_recursive. Only used for the "visits/call" column — not the hot path.
using AdaptiveSparseGrids: childsplit, scale, ϕ

function count_visits(fun, x)
    xs = scale(fun, x)
    c = Ref(0)
    _visit!(c, fun._eval, fun._eval[1], 1, xs)
    return c[]
end

function _visit!(c, arr, node, dimshift, x)
    c[] += 1
    D = length(node.l)
    u = 1.0
    for d in 1:D; u *= ϕ(node, x, d); end
    u > 0 || return
    for d in 1:D
        kd = childsplit(node, x, d)
        if kd > 0
            cid = kd == 1 ? node.left[d] : node.right[d]
            if cid != Int32(0)
                _visit!(c, arr, arr[cid], d, x)
            end
        end
        node.l[d] > 1 && break
    end
    return
end

main()
