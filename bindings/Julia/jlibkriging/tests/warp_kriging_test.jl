using Test
using jlibkriging

f_test(x) = 1.0 - 0.5 * (sin(12.0 * x) / (1.0 + x) + 2.0 * cos(7.0 * x) * x^5 + 0.7)

function branin(x1, x2)
    a, b, cc = 1.0, 5.1 / (4 * pi^2), 5.0 / pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * pi)
    return a * (x2 - b * x1^2 + cc * x1 - r)^2 + s * (1 - t) * cos(x1) + s
end

@testset "WarpKriging" begin

    @testset "Kumaraswamy 1D" begin
        X = reshape(collect(range(0.01, 0.99; length=8)), :, 1)
        y = [f_test(x) for x in X[:, 1]]

        k = WarpKriging(y, X, ["kumaraswamy"], "gauss";
                        parameters=Dict("max_iter_adam" => "200"))
        s = jlibkriging.summary(k)
        @test isa(s, String) && length(s) > 0

        result = predict(k, X)
        @test length(result.mean) == 8
        @test all(isfinite.(result.mean))

        x_pred = reshape(collect(range(0.01, 0.99; length=50)), :, 1)
        result2 = predict(k, x_pred)
        y_true = [f_test(x) for x in x_pred[:, 1]]
        rmse = sqrt(sum((result2.mean .- y_true).^2) / length(y_true))
        @test isfinite(rmse)
    end

    @testset "Categorical embedding" begin
        mu = [1.0, 5.0, 3.0]
        n = 15
        levels = repeat([0, 1, 2], ceil(Int, n / 3))[1:n]
        X = reshape(Float64.(levels), :, 1)
        y = [mu[levels[i]+1] + 0.1 * sin(Float64(i)) for i in 1:n]

        k = WarpKriging(y, X, ["categorical(3,2)"], "gauss";
                        parameters=Dict("max_iter_adam" => "200"))

        X_test = reshape([0.0, 1.0, 2.0], :, 1)
        result = predict(k, X_test)
        @test length(result.mean) == 3
        @test all(isfinite.(result.mean))
    end

    @testset "Mixed continuous + categorical" begin
        offset = [1.0, 2.0, 0.5]
        n = 30
        x1 = collect(range(0.05, 0.95; length=n))
        x2 = repeat([0, 1, 2], ceil(Int, n / 3))[1:n]
        X = hcat(x1, Float64.(x2))
        y = [sin(2 * pi * x1[i]) * offset[x2[i]+1] for i in 1:n]

        k = WarpKriging(y, X, ["kumaraswamy", "categorical(3,2)"], "matern5_2";
                        parameters=Dict("max_iter_adam" => "300"))

        xc = collect(range(0.01, 0.99; length=20))
        for cat_idx in 0:2
            X_test = hcat(xc, fill(Float64(cat_idx), 20))
            result = predict(k, X_test)
            @test length(result.mean) == 20
            @test all(isfinite.(result.mean))
        end
    end

    @testset "Ordinal" begin
        L = 5
        n = 20
        levels = repeat(collect(0:L-1), ceil(Int, n / L))[1:n]
        X = reshape(Float64.(levels), :, 1)
        y = Float64.(levels .^ 2) .+ [0.1 * sin(Float64(i)) for i in 1:n]

        k = WarpKriging(y, X, ["ordinal(5)"], "gauss";
                        parameters=Dict("max_iter_adam" => "200"))

        X_test = reshape(Float64.(collect(0:L-1)), :, 1)
        result = predict(k, X_test)
        @test length(result.mean) == L
        @test all(isfinite.(result.mean))
    end

    @testset "Neural mono" begin
        X = reshape(collect(range(0.01, 0.99; length=10)), :, 1)
        y = [f_test(x) for x in X[:, 1]]

        k = WarpKriging(y, X, ["neural_mono(8)"], "gauss";
                        parameters=Dict("max_iter_adam" => "200"))

        result = predict(k, X)
        @test all(isfinite.(result.mean))
    end

    @testset "MLP warping 1D" begin
        X = reshape(collect(range(0.01, 0.99; length=10)), :, 1)
        y = [f_test(x) for x in X[:, 1]]

        k = WarpKriging(y, X, ["mlp(16:8,3,selu)"], "gauss";
                        parameters=Dict("max_iter_adam" => "300"))

        x_pred = reshape(collect(range(0.01, 0.99; length=50)), :, 1)
        result = predict(k, x_pred)
        y_true = [f_test(x) for x in x_pred[:, 1]]
        rmse = sqrt(sum((result.mean .- y_true).^2) / length(y_true))
        @test isfinite(rmse)
    end

    @testset "MLP + categorical mixed" begin
        offset = [1.0, 2.0, 0.5]
        n = 30
        x1 = collect(range(0.05, 0.95; length=n))
        x2 = repeat([0, 1, 2], ceil(Int, n / 3))[1:n]
        X = hcat(x1, Float64.(x2))
        y = [sin(2 * pi * x1[i]) * offset[x2[i]+1] for i in 1:n]

        k = WarpKriging(y, X, ["mlp(16:8,2,tanh)", "categorical(3,2)"], "matern5_2";
                        parameters=Dict("max_iter_adam" => "300"))

        xc = collect(range(0.01, 0.99; length=20))
        for cat_idx in 0:2
            X_test = hcat(xc, fill(Float64(cat_idx), 20))
            result = predict(k, X_test)
            @test length(result.mean) == 20
            @test all(isfinite.(result.mean))
        end
    end

    @testset "Branin 2D per-variable MLP" begin
        n = 25
        x1 = collect(range(0.05, 0.95; length=n))
        x2 = collect(range(0.05, 0.95; length=n))
        X = hcat(x1, x2)
        y = [branin(X[i, 1] * 15 - 5, X[i, 2] * 15) for i in 1:n]

        k = WarpKriging(y, X,
                        ["mlp(16:8,2,selu)", "mlp(16:8,2,selu)"], "matern5_2";
                        normalize=true,
                        parameters=Dict("max_iter_adam" => "300"))

        X_test = hcat(collect(range(0.1, 0.9; length=15)),
                      collect(range(0.1, 0.9; length=15)))
        result = predict(k, X_test)
        @test length(result.mean) == 15
        @test all(isfinite.(result.stdev))

        sims = simulate(k, 20, 42, X_test)
        @test size(sims) == (15, 20)
    end

    @testset "None vs kumaraswamy vs MLP comparison" begin
        X = reshape(collect(range(0.01, 0.99; length=12)), :, 1)
        y = [f_test(x) for x in X[:, 1]]
        xp = reshape(collect(range(0.01, 0.99; length=50)), :, 1)
        ytrue = [f_test(x) for x in xp[:, 1]]

        k_none = WarpKriging(y, X, ["none"], "gauss";
                             parameters=Dict("max_iter_adam" => "200"))
        k_kuma = WarpKriging(y, X, ["kumaraswamy"], "gauss";
                             parameters=Dict("max_iter_adam" => "200"))
        k_mlp = WarpKriging(y, X, ["mlp(16:8,2,selu)"], "gauss";
                            parameters=Dict("max_iter_adam" => "300"))

        for k in (k_none, k_kuma, k_mlp)
            result = predict(k, xp)
            rmse = sqrt(sum((result.mean .- ytrue).^2) / length(ytrue))
            @test isfinite(rmse)
            @test isfinite(log_likelihood(k))
        end
    end

    @testset "Simulate" begin
        n = 20
        x1 = collect(range(0.05, 0.95; length=n))
        x2 = repeat([0, 1], ceil(Int, n / 2))[1:n]
        X = hcat(x1, Float64.(x2))
        y = [sin(2 * pi * x1[i]) * [1.0, 3.0][x2[i]+1] for i in 1:n]

        k = WarpKriging(y, X, ["affine", "categorical(2,2)"], "gauss";
                        parameters=Dict("max_iter_adam" => "200"))

        X_sim = hcat(collect(range(0.1, 0.9; length=10)), zeros(10))
        sims = simulate(k, 30, 123, X_sim)
        @test size(sims) == (10, 30)
        @test all(isfinite.(sims))
    end

    @testset "Update" begin
        X0 = [0.1 0.0; 0.5 1.0; 0.9 0.0]
        y0 = [1.0, 3.0, 0.5]

        k = WarpKriging(y0, X0, ["none", "categorical(2,1)"], "gauss";
                        parameters=Dict("max_iter_adam" => "100"))

        X_new = [0.3 1.0; 0.7 0.0]
        update!(k, [2.0, 1.5], X_new)

        result = predict(k, X0)
        @test all(isfinite.(result.mean))
    end

    @testset "Predict with derivatives" begin
        X = reshape(collect(range(0.01, 0.99; length=10)), :, 1)
        y = [f_test(x) for x in X[:, 1]]

        k = WarpKriging(y, X, ["affine"], "gauss";
                        parameters=Dict("max_iter_adam" => "100"))

        X_new = reshape([0.1, 0.3, 0.5, 0.7, 0.9], :, 1)
        result = predict(k, X_new; return_deriv=true)
        @test size(result.mean_deriv) == (5, 1)
        @test size(result.stdev_deriv) == (5, 1)
        @test all(isfinite.(result.mean_deriv))
        @test all(isfinite.(result.stdev_deriv))
    end

    @testset "Predict with covariance" begin
        X = reshape(collect(range(0.01, 0.99; length=8)), :, 1)
        y = [f_test(x) for x in X[:, 1]]

        k = WarpKriging(y, X, ["affine"], "gauss";
                        parameters=Dict("max_iter_adam" => "100"))

        X_new = reshape([0.1, 0.3, 0.5, 0.7, 0.9], :, 1)
        result = predict(k, X_new; return_stdev=true, return_cov=true)
        @test result.cov !== nothing
        @test size(result.cov) == (5, 5)
        @test all(isfinite.(result.cov))
        @test maximum(abs.(result.cov - result.cov')) < 1e-10
    end

    @testset "Accessors and summary" begin
        X = reshape(collect(range(0.01, 0.99; length=6)), :, 1)
        y = [f_test(x) for x in X[:, 1]]

        k = WarpKriging(y, X, ["boxcox"], "matern3_2";
                        parameters=Dict("max_iter_adam" => "100"))

        s = jlibkriging.summary(k)
        @test isa(s, String) && length(s) > 0

        th = get_theta(k)
        @test length(th) > 0
        @test all(th .> 0)

        @test get_sigma2(k) > 0
        @test kernel(k) == "matern3_2"
        @test isfinite(log_likelihood(k))
        @test get_warping(k) == ["boxcox"]
        @test is_fitted(k)
    end

    @testset "Log-likelihood gradient" begin
        X = reshape(collect(range(0.01, 0.99; length=8)), :, 1)
        y = [f_test(x) for x in X[:, 1]]

        k = WarpKriging(y, X, ["affine"], "gauss"; optim="none")

        # Evaluate at a fixed benign theta (not the optimum) so FD is well-conditioned.
        th = fill(0.3, length(get_theta(k)))
        ll_res = log_likelihood_fun(k, th; return_grad=true)
        @test isfinite(ll_res.ll)
        @test all(isfinite.(ll_res.grad))

        # Numerical gradient check
        h = 1e-5
        grad_num = zeros(length(th))
        for i in 1:length(th)
            tp = copy(th)
            tm = copy(th)
            tp[i] += h
            tm[i] -= h
            ll_p = log_likelihood_fun(k, tp)
            ll_m = log_likelihood_fun(k, tm)
            grad_num[i] = (ll_p.ll - ll_m.ll) / (2 * h)
        end

        rel = sqrt(sum((ll_res.grad .- grad_num).^2)) / (sqrt(sum(grad_num.^2)) + 1e-12)
        @test rel < 1e-4
    end

    @testset "Empty constructor then fit" begin
        wk = WarpKriging(["affine"], "gauss")
        @test !is_fitted(wk)

        X = reshape(collect(range(0.01, 0.99; length=8)), :, 1)
        y = [f_test(x) for x in X[:, 1]]
        fit!(wk, y, X)
        @test is_fitted(wk)

        result = predict(wk, X)
        @test length(result.mean) == 8
        @test all(isfinite.(result.mean))
    end

    @testset "Copy" begin
        X = reshape(collect(range(0.01, 0.99; length=8)), :, 1)
        y = [f_test(x) for x in X[:, 1]]

        k = WarpKriging(y, X, ["kumaraswamy"], "gauss";
                        parameters=Dict("max_iter_adam" => "100"))

        k2 = Base.copy(k)
        @test is_fitted(k2)
        @test kernel(k2) == kernel(k)
        @test get_warping(k2) == get_warping(k)

        result1 = predict(k, X)
        result2 = predict(k2, X)
        @test all(isfinite.(result2.mean))
    end

    @testset "Getters" begin
        X = reshape(collect(range(0.01, 0.99; length=8)), :, 1)
        y = [f_test(x) for x in X[:, 1]]

        k = WarpKriging(y, X, ["kumaraswamy"], "gauss";
                        parameters=Dict("max_iter_adam" => "100"))

        @test size(get_X(k)) == (8, 1)
        @test length(get_y(k)) == 8
        @test length(get_theta(k)) > 0
        @test get_sigma2(k) > 0.0
        @test feature_dim(k) > 0
    end
end
