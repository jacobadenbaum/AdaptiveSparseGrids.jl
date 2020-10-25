using Test
using AdaptiveSparseGrids
using ForwardDiff
using StaticArrays
using QuadGK

@testset "Adaptive Sparse Grid Tests" begin

@testset "Basis Functions" begin
    import AdaptiveSparseGrids: m, Y, h, ϕ, Dϕ, leftchild, rightchild

    @testset "m(i) : Dimension of basis at level i" begin
        @test m(1) == 1
        @test m(2) == 2
        @test m(3) == 4
        @test m(4) == 8
        @test_throws ArgumentError  m(0)
        @test_throws MethodError    m(1.0)
        @test_throws MethodError    m(:test)
    end

    @testset "Y(i,j) : Get the coordinate of the jth node in the ith layer" begin
        @test Y(1, 1) == 0.5
        @test Y(2, 0) == 0.0
        @test Y(2, 1) == 0.5
        @test Y(2, 2) == 1.0

        @testset "Check raw numbers" begin
            for i = 3:10, j = 0:m(i)
                @test Y(i, j) == j/2^(i-1)
            end
        end

        @testset "Check Nesting" begin
            for i = 3:10, j = 0:m(i)
                !iseven(j) && continue
                @test Y(i,j) == Y(i-1, Int(j/2))
            end
        end

        @test_throws ArgumentError Y(0, 3)
        @test_throws ArgumentError Y(1, 2)
        @test_throws ArgumentError Y(1, -1)
        @test_throws MethodError   Y(1.0, 1)
        @test_throws MethodError   Y(1, 1.0)
    end

    @testset "h(i): Width of basis elements in ith level" begin
        @test h(1) == 1.0
        @test h(2) == 0.5
        @test h(3) == 0.25
        for i in 4:10
            @test h(i) == 2.0^(-i + 1)
        end

        @test_throws MethodError h(1.0)
        @test_throws ArgumentError h(-1)
    end

    @testset "ϕ(x): Hat function" begin
        @test ϕ(-2.0) == 0.0
        @test ϕ(-1.0) == 0.0
        @test ϕ(0.0)  == 1.0
        @test ϕ(1.0)  == 0.0
        @test ϕ(2.0)  == 0.0

        @test ϕ(0.5)  == 0.5
        @test ϕ(-0.5) == 0.5
        @test ϕ(0.25) == 0.75
        @test ϕ(0.75) == 0.25

        for x in LinRange(0, 1, 5)
            # Test Symmetry
            @test ϕ(x) == ϕ(-x)

            # Test Values
            if x <= 1
                @test ϕ(x) == 1 - x
            else
                @test ϕ(x) == 0.0
            end
        end

        # Technically it should work with integers
        @test typeof(ϕ(1.0)) == Float64
        @test typeof(ϕ(1))   == Int
        @test ϕ(-1)== ϕ(-1.0)
        @test ϕ(0) == ϕ(0.0)
        @test ϕ(1) == ϕ(1.0)
        @test ϕ(0) == ϕ(0.0)
        @test ϕ(2) == ϕ(2.0)
    end

    @testset "Dϕ: Derivatives" begin
        for x in LinRange(-1, 1, 5)
            @test Dϕ(x) == ForwardDiff.derivative(ϕ, x)
        end
    end

    @testset "ϕ(i,j,x): Hierarchical Hat Functions" begin
        @test ϕ(1, 1, 0.0) == 1.0
        @test ϕ(1, 1, 0.5) == 1.0
        @test ϕ(1, 1, 1.0) == 1.0

        @test ϕ(2, 0, Y(2, 0)) == 1.0
        @test ϕ(2, 0, Y(2, 1)) == 0.0
        @test ϕ(2, 2, Y(2, 1)) == 0.0
        @test ϕ(2, 2, Y(2, 2)) == 1.0

        for i = 3:10, j = 1:m(i) - 1
            if isodd(j)
                # Boundaries
                @test ϕ(i, j, Y(i, j-1))    == 0.0
                @test ϕ(i, j, Y(i, j+1))    == 0.0

                # Interior
                @test ϕ(i, j, Y(i, j))      == 1.0
                @test ϕ(i, j, Y(i+1, 2j-1)) == 0.5
                @test ϕ(i, j, Y(i+1, 2j+1)) == 0.5
                @test ϕ(i, j, Y(i+2, 4j-3)) == 0.25
                @test ϕ(i, j, Y(i+2, 4j-1)) == 0.75
                @test ϕ(i, j, Y(i+2, 4j+1)) == 0.75
                @test ϕ(i, j, Y(i+2, 4j+3)) == 0.25
            end
        end
    end

    @testset "lefchild and rightchild" begin

        @testset "Base Values" begin
            @test leftchild(1, 1)  == (2, 0)
            @test rightchild(1, 1) == (2, 2)
            @test rightchild(2, 0) == (3, 1)
            @test leftchild(2, 2)  == (3, 3)
            @test leftchild(2, 0)  == (3, -1)
            @test rightchild(2, 2) == (3, -1)
        end

        @testset "Higher Level Hierarchy" begin
            for i in 3:10, j in 1:2:m(i)
                @test ϕ(i,j, Y(leftchild(i,j)...))  == 0.5
                @test ϕ(i,j, Y(rightchild(i,j)...)) == 0.5

                @test ϕ(leftchild(i,j)..., Y(i,j))  == 0.0
                @test ϕ(rightchild(i,j)..., Y(i,j)) == 0.0

                # They do overlap in their support with their children
                for c in (leftchild(i,j), rightchild(i,j))
                    @test ϕ(i,j,  (Y(i,j) + Y(c...))/2) == 0.75
                    @test ϕ(c..., (Y(i,j) + Y(c...))/2) == 0.5
                end

            end


        end

        @testset "Base Level Errors" begin
            @test_throws ArgumentError leftchild(2, 1)
            @test_throws ArgumentError rightchild(2, 1)
            @test_throws ArgumentError leftchild(0, 1)
            @test_throws ArgumentError rightchild(0, 1)
        end

        @testset "No Floats" begin
            @test_throws MethodError leftchild(1.0, 1)
            @test_throws MethodError leftchild(1,  1.0)
            @test_throws MethodError rightchild(1.0, 1)
            @test_throws MethodError rightchild(1,  1.0)
        end


    end
end

@testset "Nodes" begin

    @testset "KTuples" begin
        import AdaptiveSparseGrids: KTuple
        @test (1,2) isa KTuple
        @test (1,2) isa KTuple{2}
        @test (1,2) isa KTuple{2, Int}
        @test (1.0,2.0) isa KTuple
        @test (1.0,2.0) isa KTuple{2}
        @test (1.0,2.0) isa KTuple{2, Float64}

        @test (1,2,3) isa KTuple
        @test (1,2,3) isa KTuple{3}
        @test (1,2,3) isa KTuple{3, Int}
        @test (1.0,2.0,3.0) isa KTuple
        @test (1.0,2.0,3.0) isa KTuple{3, Float64}

        @test ! ((1, 2.0) isa KTuple)

        @test (a = 1.0, b = 2.0) isa KTuple
        @test (a = 1.0, b = 2.0) isa KTuple{2}
        @test (a = 1.0, b = 2.0) isa KTuple{2, Float64}

        @test !( (a = 1.0, b = 2) isa KTuple )
        @test !( (a = 1.0, b = 2) isa KTuple{2} )
        @test !( (a = 1.0, b = 2) isa KTuple{2, Float64} )
    end

    @testset "getzero" begin
        import AdaptiveSparseGrids: getzero
        @test getzero((a=1, b=2))               == (a = 0,   b = 0)
        @test getzero((a=1.0, b=2.0))           == (a = 0.0, b = 0.0)
        @test getzero(typeof((a=1, b=2)))       == (a = 0,   b = 0)
        @test getzero(typeof((a=1.0, b=2.0)))   == (a = 0.0, b = 0.0)
        @test getzero((1, 2, 3))                == (0,0,0)
        @test getzero((1.0, 2.0, 3.0))          == (0.0,0.0,0.0)
        @test getzero([1, 2, 3])                == (0,0,0)
        @test getzero([1, 2.0, 3])              == (0.0,0.0,0.0)
        @test getzero(1.0)                      == (0.0,)
        @test getzero(0)                        == (0,)
    end

    @testset "Node 1D" begin
        import AdaptiveSparseGrids: Node, getx, ϕ, makeleftchild, makerightchild
        n = Node(1,1)
        @test ϕ(n, (0.0,), 1) == 1.0
        @test ϕ(n, (0.5,), 1) == 1.0
        @test ϕ(n, (1.0,), 1) == 1.0

        # Check the children
        lc = makeleftchild(n, 1)
        @test getx(lc) == @SVector [0.0]
        @test ϕ(lc, 0.0, 1) == 1.0
        @test ϕ(lc, 0, 1)   == 1.0
        @test ϕ(lc, 0.5, 1) == 0.0
        @test ϕ(lc, 0.25, 1) == 0.5

        rc = makerightchild(n, 1)
        @test getx(rc) == @SVector [1.0]
        @test ϕ(rc, 1.0, 1) == 1.0
        @test ϕ(rc, 1, 1)   == 1.0
        @test ϕ(rc, 0.5, 1) == 0.0
        @test ϕ(rc, 0.75, 1) == 0.5

        for l in 3:10, j in 1:2:m(l)
            n = Node(l, j)
            @test getx(n) == @SVector [Y(l, j)]
            @test ϕ(n, getx(n), 1) == 1.0
            @test ϕ(n, Y(leftchild(l,j)...), 1)  == 0.5
            @test ϕ(n, Y(rightchild(l,j)...), 1) == 0.5
            j > 1       && @test ϕ(n, Y(l, j-1), 1) == 0.0
            j < m(l)- 1 && @test ϕ(n, Y(l, j+1), 1) == 0.0
            j > 2       && @test ϕ(n, Y(l, j-2), 1) == 0.0
            j < m(l)-1  && @test ϕ(n, Y(l, j+2), 1) == 0.0
        end
    end

    @testset "Node 2D" begin
        import AdaptiveSparseGrids: Node, getx, ϕ, m, Y
        for l1 in 3:5, l2 in 3:5, i1 in 1:2:m(l1), i2 in 1:2:m(l2)
            n = Node((l1,l2), (i1, i2))

            # Check that we're linearly interpolating to the corners
            for d1 in -1:1, d2 in -1:1
                c = @SVector [Y(l1, i1 + d1), Y(l2, i2+d2)]
                for θ in LinRange(0,1, 5)
                    x = θ * c + (1-θ) * getx(n)
                    @test ϕ(n, x) == (1-θ)^(abs(d1) + abs(d2))
                end
            end
        end
    end
end

@testset "Scaling" begin
    import AdaptiveSparseGrids: scale, rescale

    f = AdaptiveSparseGrid([0.0,1.0], [2.0, 3.0]) do (x,)
        x^2
    end

    @test scale(f, [0.0, 1.0]) == [0.0, 0.0]
    @test scale(f, (0.0, 1.0)) == [0.0, 0.0]
    @test scale(f, @SVector [0.0, 1.0]) == [0.0, 0.0]

    @test scale(f, [1.0, 1.0]) == [0.5, 0.0]
    @test scale(f, [1.0, 2.0]) == [0.5, 0.5]
    @test scale(f, [1.0, 3.0]) == [0.5, 1.0]

    @test scale(f, [2.0, 1.0]) == [1.0, 0.0]
    @test scale(f, [2.0, 2.0]) == [1.0, 0.5]
    @test scale(f, [2.0, 3.0]) == [1.0, 1.0]


    @test rescale(f, [0.5, 0.0]) == [1.0, 1.0]
    @test rescale(f, [0.5, 0.5]) == [1.0, 2.0]
    @test rescale(f, [0.5, 1.0]) == [1.0, 3.0]

    @test rescale(f, [1.0, 0.0]) == [2.0, 1.0]
    @test rescale(f, [1.0, 0.5]) == [2.0, 2.0]
    @test rescale(f, [1.0, 1.0]) == [2.0, 3.0]
end

@testset "Interpolation Tests (1D)" begin

    for g in [x -> x^2, sin, cos, exp]
        f = AdaptiveSparseGrid([0.0], [10.0], tol=1e-8, max_depth = 20) do x
            g.(x)
        end

        for x in LinRange(0, 10, 1_000)
            @test abs(f(x) - g(x))/max(g(x),1) < 1e-8
        end
    end

    # Test the named tuple features
    f = AdaptiveSparseGrid([0.0], [10.0], tol=1e-8, max_depth = 20) do (x,)
        (sin = sin(x), cos = cos(x), x2 = x^2)
    end

    @test f(pi/2, :sin) ≈ 1.0
    @test f(pi/2, :cos) ≈ 0.0 atol=1e-8
    @test f(pi  , :sin) ≈ 0.0 atol=1e-8
    @test f(pi  , :cos) ≈ -1.0

    @test f(pi/2, :sin) == f(pi/2, 1)
    @test f(pi/2, :cos) == f(pi/2, 2)
end

@testset "Integration Tests (1D)" begin

    for g in [x -> x^2, sin, cos, exp]

        f = AdaptiveIntegral([0.0], [10.0], 1, tol=1e-8, max_depth = 20) do x
            g.(x)
        end

        v = quadgk(g, 0, 10)[1]
        @test abs(f()[1] - v)/max(v,1) < 1e-8
    end
end


@testset "Interpolation Tests (2D)" begin

    h((x,y)) = sin(x) * cos(x)
    h(x...)  = h(x)

    f = AdaptiveSparseGrid(h, [0.0, 0.0], [2 * pi, 2 * pi],
                           tol=1e-6, max_depth = 20, train=false)

    for x in LinRange(0,2*pi,100), y in LinRange(0,2*pi,100)
        @test f((x,y)) ≈ h(x,y) atol = 1e-6
        @test f((x,y)) == f(x,y)
    end

    # Check that using integers works fine
    for x in LinRange(0, 2*pi, 100)
        @test f((x,1)) == f(x, 1.0)
        @test f((1,x)) == f(1.0, x)
        @test f((x,2)) == f(x, 2.0)
        @test f((2,x)) == f(2.0, x)
    end

    # Check that the max depth parameter works
    @test mapreduce(x -> x.depth, max, values(nodes(f))) <= f.max_depth

    # Now a version from R^2 →  R^2
    h((x,y)) = (a = sin(x) * cos(y), b = x^2 + y^2)

    f = AdaptiveSparseGrid(h, [0.0, 0.0], [2 * pi, 2 * pi],
                           tol=1e-6, max_depth = 30)

    for x in LinRange(0,2*pi,100), y in LinRange(0,2*pi,100)
        @test f((x,y),:a) ≈ h(x,y).a atol = 1e-5
    end

    # Check that using integers works fine
    for x in LinRange(0, 2*pi, 3)
        @test f((x,1)) == f((x, 1.0))
    end

    for x in LinRange(0, 2*pi, 3)
        @test f((1,x)) == f((1.0, x))
    end


end




end
