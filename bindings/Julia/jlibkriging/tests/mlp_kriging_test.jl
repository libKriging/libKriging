using Test
using jlibkriging

f_test(x) = 1.0 - 0.5 * (sin(12.0 * x) / (1.0 + x) + 2.0 * cos(7.0 * x) * x^5 + 0.7)

function branin(x1, x2)
    a, b, cc = 1.0, 5.1 / (4 * pi^2), 5.0 / pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * pi)
    return a * (x2 - b * x1^2 + cc * x1 - r)^2 + s * (1 - t) * cos(x1) + s
end

@testset "MLPKriging" begin

    @testset "1D gauss" begin
        X = reshape(collect(range(0.01, 0.99; length=10)), :, 1)
        y = [f_test(x) for x in X[:, 1]]

        k = MLPKriging(y, X, [16, 8], 2;
                       activation="selu", kernel="gauss",
                       normalize=true,
                       parameters=Dict("max_iter_adam" => "300"))

        s = jlibkriging.summary(k)
        @test isa(s, String) && length(s) > 0

        result = predict(k, X)
        @test length(result.mean) == 10
        @test all(isfinite.(result.mean))

        x_pred = reshape(collect(range(0.01, 0.99; length=50)), :, 1)
        result2 = predict(k, x_pred)
        y_true = [f_test(x) for x in x_pred[:, 1]]
        rmse = sqrt(sum((result2.mean .- y_true).^2) / length(y_true))
        @test isfinite(rmse)

        @test kernel(k) == "gauss"
        @test activation(k) == "selu"
        @test get_hidden_dims(k) == [16, 8]
        @test feature_dim(k) == 2
        @test get_sigma2(k) > 0
        @test isfinite(log_likelihood(k))
        @test is_fitted(k)
    end

    @testset "Branin 2D matern" begin
        n = 20
        rng_vals = [0.37454, 0.95071, 0.73199, 0.59866, 0.15601,
                    0.15599, 0.05808, 0.86617, 0.70807, 0.02058,
                    0.96991, 0.83244, 0.21233, 0.18183, 0.18340,
                    0.30424, 0.52475, 0.43194, 0.29122, 0.61185,
                    0.13949, 0.29214, 0.36636, 0.45607, 0.78518,
                    0.19968, 0.51423, 0.59242, 0.04646, 0.60754,
                    0.17052, 0.06505, 0.94889, 0.96563, 0.80840,
                    0.30462, 0.09767, 0.68423, 0.44015, 0.12203]
        X = reshape(rng_vals, n, 2)
        y = [branin(X[i, 1] * 15 - 5, X[i, 2] * 15) for i in 1:n]

        k = MLPKriging(y, X, [32, 16], 3;
                       activation="selu", kernel="matern5_2",
                       normalize=true,
                       parameters=Dict("max_iter_adam" => "300"))

        X_test = reshape([0.5, 0.3, 0.8, 0.1, 0.6,
                          0.4, 0.7, 0.2, 0.9, 0.5,
                          0.2, 0.8, 0.4, 0.9, 0.1,
                          0.6, 0.3, 0.7, 0.5, 0.8], 10, 2)

        result = predict(k, X_test)
        @test length(result.mean) == 10
        @test all(isfinite.(result.mean))
        @test all(isfinite.(result.stdev))

        sims = simulate(k, 20, 42, X_test)
        @test size(sims) == (10, 20)
        @test all(isfinite.(sims))
    end

    @testset "Update" begin
        X = reshape(collect(range(0.01, 0.99; length=10)), :, 1)
        y = [f_test(x) for x in X[:, 1]]

        k = MLPKriging(y, X, [16, 8], 2;
                       activation="selu", kernel="gauss",
                       parameters=Dict("max_iter_adam" => "100"))

        X_test = reshape(collect(range(0.05, 0.95; length=5)), :, 1)
        p_before = predict(k, X_test)

        X_new = reshape([0.15, 0.35, 0.55], :, 1)
        y_new = [f_test(x) for x in X_new[:, 1]]
        update!(k, y_new, X_new)

        p_after = predict(k, X_test)
        @test p_before.mean != p_after.mean
    end

    @testset "Predict with derivatives" begin
        n = 20
        X = reshape(collect(range(0.01, 0.99; length=n)), :, 1)
        y = [sin(3.0 * x) + 0.5 * x for x in X[:, 1]]

        k = MLPKriging(y, X, [16, 8], 2;
                       activation="selu", kernel="gauss",
                       parameters=Dict("max_iter_adam" => "100"))

        X_new = reshape([0.1, 0.3, 0.5, 0.7, 0.9], :, 1)
        result = predict(k, X_new; return_deriv=true)
        @test size(result.mean_deriv) == (5, 1)
        @test size(result.stdev_deriv) == (5, 1)
        @test all(isfinite.(result.mean_deriv))
        @test all(isfinite.(result.stdev_deriv))
    end

    @testset "Predict with covariance" begin
        n = 20
        X = reshape(collect(range(0.01, 0.99; length=n)), :, 1)
        y = [sin(3.0 * x) + 0.5 * x for x in X[:, 1]]

        k = MLPKriging(y, X, [16, 8], 2;
                       activation="selu", kernel="gauss",
                       parameters=Dict("max_iter_adam" => "100"))

        X_new = reshape([0.1, 0.3, 0.5, 0.7, 0.9], :, 1)
        result = predict(k, X_new; return_stdev=true, return_cov=true)
        @test result.cov !== nothing
        @test size(result.cov) == (5, 5)
        @test all(isfinite.(result.cov))
        @test maximum(abs.(result.cov - result.cov')) < 1e-10
    end

    @testset "Getters" begin
        X = reshape(collect(range(0.01, 0.99; length=8)), :, 1)
        y = [f_test(x) for x in X[:, 1]]

        k = MLPKriging(y, X, [16, 8], 2;
                       activation="selu", kernel="gauss",
                       parameters=Dict("max_iter_adam" => "100"))

        @test size(get_X(k)) == (8, 1)
        @test length(get_y(k)) == 8
        @test length(get_theta(k)) > 0
        @test get_sigma2(k) > 0.0
        @test kernel(k) == "gauss"
        @test activation(k) == "selu"
        @test get_hidden_dims(k) == [16, 8]
        @test feature_dim(k) == 2
        @test is_fitted(k)
    end

    @testset "Log-likelihood" begin
        X = reshape(collect(range(0.01, 0.99; length=8)), :, 1)
        y = [f_test(x) for x in X[:, 1]]

        k = MLPKriging(y, X, [16, 8], 2;
                       activation="selu", kernel="gauss",
                       parameters=Dict("max_iter_adam" => "100"))

        ll = log_likelihood(k)
        @test isfinite(ll)

        theta = get_theta(k)
        ll_res = log_likelihood_fun(k, theta)
        @test isfinite(ll_res.ll)
        @test ll_res.grad === nothing

        ll_res_grad = log_likelihood_fun(k, theta; return_grad=true)
        @test isfinite(ll_res_grad.ll)
        @test length(ll_res_grad.grad) == length(theta)
    end

    @testset "Empty constructor then fit" begin
        k = MLPKriging([16, 8], 2; activation="selu", kernel="gauss")
        @test !is_fitted(k)

        X = reshape(collect(range(0.01, 0.99; length=10)), :, 1)
        y = [f_test(x) for x in X[:, 1]]
        fit!(k, y, X; parameters=Dict("max_iter_adam" => "100"))
        @test is_fitted(k)

        result = predict(k, X)
        @test length(result.mean) == 10
        @test all(isfinite.(result.mean))
    end
end
