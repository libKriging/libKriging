using Test
using Random
using Statistics
using jlibkriging

# f2d matches the C++/R/Python predictCG test suites' test function.
f2d(x1, x2) = sin(3.0 * x1) + cos(5.0 * x2) + x1 * x2

function make_data(n::Int; seed::Int=123)
    Random.seed!(seed)
    X = rand(n, 2)
    y = [f2d(X[i, 1], X[i, 2]) for i in 1:n]
    return X, y
end

# Fixed, moderate theta (optim="none"): on this noise-free deterministic
# test function, a free BFGS fit is known to drift theta toward a
# near-singular correlation matrix (unrelated to predictCG itself -- see
# docs/math/PredictCG.md / KrigingNystromTest.cpp for the same issue),
# which would make predict/predictCG's agreement noisy rather than a clean
# correctness signal.
make_fixed_theta_model(y, X; theta_val::Float64=0.3) =
    Kriging(y, X, "matern5_2"; optim="none", theta=[theta_val, theta_val], sigma2=1.0)

@testset "Kriging predictCG" begin
    @testset "mean/stdev match exact predict at a moderate theta" begin
        X, y = make_data(60)
        k = make_fixed_theta_model(y, X)
        Xt = rand(20, 2)

        p_exact = predict(k, Xt; return_stdev=true)
        p_cg = predictCG(k, Xt; return_stdev=true)

        sdy = std(y)
        @test maximum(abs.(p_exact.mean .- p_cg.mean)) < 0.05 * sdy
        @test maximum(abs.(p_exact.stdev .- p_cg.stdev)) < 0.05 * sdy
    end

    @testset "defaults to mean only (stdev is nothing)" begin
        X, y = make_data(40)
        k = make_fixed_theta_model(y, X)
        Xt = rand(5, 2)

        p = predictCG(k, Xt)
        @test length(p.mean) == 5
        @test p.stdev === nothing
    end

    @testset "interpolates the training data" begin
        X, y = make_data(30)
        k = make_fixed_theta_model(y, X)

        p = predictCG(k, X; return_stdev=true)
        sdy = std(y)
        @test maximum(abs.(p.mean .- y)) < 0.05 * sdy
        @test maximum(p.stdev) < 0.05 * sdy
    end

    @testset "rejects a negative max_iter" begin
        X, y = make_data(20)
        k = make_fixed_theta_model(y, X)
        Xt = rand(5, 2)

        @test_throws ArgumentError predictCG(k, Xt; max_iter=-1)
    end
end
