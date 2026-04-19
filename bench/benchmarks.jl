#=
Performance benchmarks for AdaptiveSparseGrids.jl.

Measures:
  1. Grid construction (`AdaptiveSparseGrid(f, lb, ub; tol, max_depth)`)
     across varying dimensions and tolerances.
  2. Single-point evaluation of a trained interpolant (latency + allocations).
  3. Bulk evaluation on a cloud of points (throughput).
  4. Integration (`AdaptiveIntegral`).

Run with:

    julia --project=bench -t auto bench/benchmarks.jl
=#

using AdaptiveSparseGrids
using BenchmarkTools
using Printf
using Random
using StaticArrays
using Statistics

const SEED = 0xB0BACAFE

# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

# Smooth-ish target functions with tunable dimensionality. They stay cheap
# so we time the grid machinery, not the integrand.
gauss(x)  = exp(-sum(abs2, x) / length(x))
runge(x)  = 1.0 / (1.0 + sum(abs2, x))
oscil(x)  = sum(sin(2π * xi) for xi in x) / length(x)

const FUNCS = ((:gauss, gauss), (:runge, runge), (:oscil, oscil))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

row(cols...) = join(cols, " | ")

function header(title)
    println()
    println("=" ^ 78)
    println(title)
    println("=" ^ 78)
end

function build(f, N; tol = 1e-3, max_depth = 10)
    lb = fill(-1.0, N)
    ub = fill( 1.0,  N)
    return AdaptiveSparseGrid(f, lb, ub; tol = tol, max_depth = max_depth)
end

function random_points(N, M; seed = SEED)
    rng = MersenneTwister(seed)
    return [SVector{N,Float64}(2 .* rand(rng, N) .- 1) for _ in 1:M]
end

# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

function bench_construction()
    header("Construction: time / allocations / #nodes")
    println(row(lpad("function", 8), lpad("N", 3), lpad("tol", 8),
                lpad("depth", 5), lpad("nodes", 8), lpad("time (ms)", 12),
                lpad("alloc (MiB)", 12)))
    println("-" ^ 78)

    # (N, tol, max_depth) configurations. Kept modest so the full suite
    # completes in a couple of minutes.
    configs = [
        (2,  1e-4, 10),
        (3,  1e-3, 10),
        (4,  1e-2,  9),
        (5,  1e-2,  8),
    ]

    results = Dict{Tuple{Symbol,Int},Any}()
    for (fname, f) in FUNCS
        for (N, tol, depth) in configs
            b = @benchmark build($f, $N; tol = $tol, max_depth = $depth) samples=2 evals=1 seconds=20
            fun = build(f, N; tol = tol, max_depth = depth)
            results[(fname, N)] = fun
            println(row(
                lpad(string(fname), 8),
                lpad(N, 3),
                lpad(@sprintf("%.0e", tol), 8),
                lpad(depth, 5),
                lpad(length(fun), 8),
                lpad(@sprintf("%.2f", minimum(b.times) / 1e6), 12),
                lpad(@sprintf("%.2f", b.memory / 2^20), 12),
            ))
        end
    end
    return results
end

# ---------------------------------------------------------------------------
# 2. Single-point evaluation (latency)
# ---------------------------------------------------------------------------

function bench_single_eval(funs)
    header("Single-point eval: time per call / allocations")
    println(row(lpad("function", 8), lpad("N", 3), lpad("nodes", 8),
                lpad("ns/call", 10), lpad("allocs", 8)))
    println("-" ^ 78)

    for ((fname, N), fun) in sort(collect(funs); by = first)
        pts = random_points(N, 64)
        b = @benchmark for x in $pts; $fun(x); end samples=50 evals=1 seconds=10
        per_call_ns = minimum(b.times) / length(pts)
        println(row(
            lpad(string(fname), 8),
            lpad(N, 3),
            lpad(length(fun), 8),
            lpad(@sprintf("%.1f", per_call_ns), 10),
            lpad(b.allocs, 8),
        ))
    end
end

# ---------------------------------------------------------------------------
# 3. Bulk evaluation (throughput, serial)
# ---------------------------------------------------------------------------

function bench_bulk_eval(funs; npts = 50_000)
    header("Bulk serial eval: throughput on $npts random points")
    println(row(lpad("function", 8), lpad("N", 3), lpad("nodes", 8),
                lpad("total (ms)", 12), lpad("Mcalls/s", 10)))
    println("-" ^ 78)

    for ((fname, N), fun) in sort(collect(funs); by = first)
        pts = random_points(N, npts)
        # Single warm-up + timed pass. Use @elapsed for fewer repetitions
        # since the work dominates.
        # Warm up
        let s = 0.0
            for x in pts; s += fun(x)[1]; end
        end
        t = @elapsed for x in pts; fun(x); end
        println(row(
            lpad(string(fname), 8),
            lpad(N, 3),
            lpad(length(fun), 8),
            lpad(@sprintf("%.2f", t * 1e3), 12),
            lpad(@sprintf("%.2f", npts / t / 1e6), 10),
        ))
    end
end

# ---------------------------------------------------------------------------
# 4. Integration
# ---------------------------------------------------------------------------

function bench_integration()
    header("AdaptiveIntegral: construction + call")
    println(row(lpad("N", 3), lpad("tol", 8), lpad("depth", 5),
                lpad("build (ms)", 12), lpad("call (µs)", 10)))
    println("-" ^ 78)

    for N in 2:3
        lb = fill(-1.0, N); ub = fill(1.0, N)
        b_build = @benchmark AdaptiveIntegral(gauss, $lb, $ub, 1;
                                              tol = 1e-2, max_depth = 9) samples=2 evals=1 seconds=15
        fun = AdaptiveIntegral(gauss, lb, ub, 1; tol = 1e-2, max_depth = 9)
        xs  = random_points(N - 1, 128)
        b_call = @benchmark for x in $xs; $fun(x); end samples=10 evals=1 seconds=5
        println(row(
            lpad(N, 3),
            lpad(@sprintf("%.0e", 1e-3), 8),
            lpad(10, 5),
            lpad(@sprintf("%.2f", minimum(b_build.times) / 1e6), 12),
            lpad(@sprintf("%.2f", (minimum(b_call.times) / length(xs)) / 1e3), 10),
        ))
    end
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

function main()
    println("AdaptiveSparseGrids.jl benchmark")
    println("Threads: ", Threads.nthreads())
    println("Julia  : ", VERSION)

    funs = bench_construction()
    bench_single_eval(funs)
    bench_bulk_eval(funs)
    bench_integration()
    println()
end

main()
