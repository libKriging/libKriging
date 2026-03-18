using Test
using Random
using jlibkriging

# Multi-dimensional test function: product of sines
function f_parametric(X::Matrix{Float64})
    n = size(X, 1)
    y = ones(n)
    for j in axes(X, 2)
        y .*= sin.((X[:, j] .- 0.5) .^ 2)
    end
    return y
end

@testset "Kriging Parametric" begin
    @testset "Kriging fit and predict (n=$n, d=$d)" for n in [40, 100], d in [3, 6]
        rng = MersenneTwister(123)
        X = rand(rng, n, d)
        y = f_parametric(X)

        k = Kriging(y, X, "matern5_2")

        @test kernel(k) == "matern5_2"
        @test length(get_y(k)) == n
        @test size(get_X(k)) == (n, d)
        @test length(get_theta(k)) == d

        result = predict(k, X; return_stdev=true, return_cov=true)
        @test length(result.mean) == n
        @test length(result.stdev) == n
        @test size(result.cov) == (n, n)
        @test all(isfinite.(result.mean))

        # Predictions at training points should be close to training values
        @test all(abs.(result.mean .- y) .< 0.5)
    end

    @testset "Kriging with gauss kernel (n=$n, d=$d)" for n in [40, 100], d in [3, 6]
        rng = MersenneTwister(456)
        X = rand(rng, n, d)
        y = f_parametric(X)

        k = Kriging(y, X, "gauss")
        @test kernel(k) == "gauss"

        result = predict(k, X)
        @test length(result.mean) == n
        @test all(isfinite.(result.mean))
    end

    @testset "Kriging predict on new data (n=$n, d=$d)" for n in [40, 100], d in [3, 6]
        rng = MersenneTwister(123)
        X_train = rand(rng, n, d)
        y_train = f_parametric(X_train)

        k = Kriging(y_train, X_train, "matern5_2")

        m_test = 15
        rng2 = MersenneTwister(789)
        X_test = rand(rng2, m_test, d)

        result = predict(k, X_test)
        @test length(result.mean) == m_test
        @test length(result.stdev) == m_test
        @test all(result.stdev .>= 0.0)
        @test all(isfinite.(result.mean))
    end

    @testset "Kriging with different regression models (n=$n)" for n in [40, 100]
        rng = MersenneTwister(123)
        d = 3
        X = rand(rng, n, d)
        y = f_parametric(X)

        for rm in ["constant", "linear"]
            k = Kriging(y, X, "matern5_2"; regmodel=rm)
            @test regmodel(k) == rm
            result = predict(k, X)
            @test length(result.mean) == n
        end
    end
end
