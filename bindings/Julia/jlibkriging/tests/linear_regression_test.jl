using Test
using Random
using LinearAlgebra
using jlibkriging

@testset "LinearRegression" begin
    @testset "Exact fit (n=$n, m=$m)" for n in [40, 100], m in [3, 6]
        rng = MersenneTwister(123)
        sol = randn(rng, m)

        X = randn(rng, n, m)
        X[:, 1] .= 1.0
        y = X * sol

        lr = LinearRegression()
        fit!(lr, y, X)
        result = predict(lr, X)
        y_pred = result.mean

        @test norm(y - y_pred, Inf) <= 1e-5
    end

    @testset "Noisy fit (n=$n, m=$m)" for n in [40, 100], m in [3, 6]
        rng = MersenneTwister(123)
        sol = randn(rng, m)

        X = randn(rng, n, m)
        X[:, 1] .= 1.0
        y = X * sol

        # Add small multiplicative noise
        rng2 = MersenneTwister(456)
        y_noisy = y .* (1.0 .+ 1e-8 .* randn(rng2, n))

        lr = LinearRegression()
        fit!(lr, y_noisy, X)
        result = predict(lr, X)
        y_pred = result.mean

        @test norm(y_noisy - y_pred, Inf) <= 1e-5
    end

    @testset "Predict returns stdev" begin
        rng = MersenneTwister(123)
        n, m = 40, 3
        X = randn(rng, n, m)
        X[:, 1] .= 1.0
        y = X * randn(rng, m)

        lr = LinearRegression()
        fit!(lr, y, X)
        result = predict(lr, X)

        @test length(result.mean) == n
        @test length(result.stdev) == n
        @test all(result.stdev .>= 0.0)
    end
end
