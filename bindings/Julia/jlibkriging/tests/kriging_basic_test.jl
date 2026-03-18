using Test
using jlibkriging

# Test function: f(x) = 1 - 1/2 * (sin(12x)/(1+x) + 2*cos(7x)*x^5 + 0.7)
f_test(x) = 1.0 - 0.5 * (sin(12.0 * x) / (1.0 + x) + 2.0 * cos(7.0 * x) * x^5 + 0.7)

@testset "Kriging Basic" begin
    # Training data
    X_train_vec = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    n_train = length(X_train_vec)
    X_train = reshape(Float64.(X_train_vec), :, 1)
    y_train = [f_test(x) for x in X_train_vec]

    # Test data
    X_test_vec = collect(range(0.05, 0.95; length=10))
    n_test = length(X_test_vec)
    X_test = reshape(X_test_vec, :, 1)

    @testset "Construction and fit" begin
        k = Kriging("gauss")
        @test kernel(k) == "gauss"

        fit!(k, y_train, X_train)
        @test kernel(k) == "gauss"
    end

    @testset "Direct construction with data" begin
        k = Kriging(y_train, X_train, "gauss")
        @test kernel(k) == "gauss"
        @test optim(k) == "BFGS"
        @test objective(k) == "LL"
        @test regmodel(k) == "constant"
    end

    @testset "Predict" begin
        k = Kriging(y_train, X_train, "gauss")
        result = predict(k, X_test)

        @test length(result.mean) == n_test
        @test length(result.stdev) == n_test
        @test all(result.stdev .>= 0.0)
        @test result.cov === nothing
        @test result.mean_deriv === nothing
        @test result.stdev_deriv === nothing
    end

    @testset "Predict with covariance" begin
        k = Kriging(y_train, X_train, "gauss")
        result = predict(k, X_test; return_cov=true)

        @test size(result.cov) == (n_test, n_test)
    end

    @testset "Predict with derivatives" begin
        k = Kriging(y_train, X_train, "gauss")
        result = predict(k, X_test; return_deriv=true)

        @test size(result.mean_deriv) == (n_test, 1)
        @test size(result.stdev_deriv) == (n_test, 1)
    end

    @testset "Predict at training points" begin
        k = Kriging(y_train, X_train, "gauss")
        result = predict(k, X_train)

        for i in 1:n_train
            @test abs(result.mean[i] - y_train[i]) < 1e-3
        end
        for i in 1:n_train
            @test result.stdev[i] < 0.1
        end
    end

    @testset "Simulate" begin
        k = Kriging(y_train, X_train, "gauss")
        nsim = 5
        sim = simulate(k, nsim, 123, X_test)

        @test size(sim) == (n_test, nsim)
        @test all(isfinite.(sim))
    end

    @testset "Update" begin
        k = Kriging(y_train, X_train, "gauss")
        p_before = predict(k, X_test)

        X_new = reshape([0.3], :, 1)
        y_new = [f_test(0.3)]
        update!(k, y_new, X_new)

        p_after = predict(k, X_test)
        @test p_before.mean != p_after.mean
    end

    @testset "Summary" begin
        k = Kriging(y_train, X_train, "gauss")
        s = jlibkriging.summary(k)
        @test isa(s, String)
        @test length(s) > 0
    end

    @testset "Getters" begin
        k = Kriging(y_train, X_train, "gauss")

        @test get_X(k) == X_train
        @test get_y(k) ≈ y_train
        @test length(get_theta(k)) > 0
        @test get_sigma2(k) > 0.0
        @test length(get_beta(k)) > 0
        @test isa(get_centerY(k), Float64)
        @test isa(get_scaleY(k), Float64)
        @test length(get_centerX(k)) == 1
        @test length(get_scaleX(k)) == 1
        @test size(get_F(k), 1) == n_train
        @test size(get_T(k), 1) == n_train
        @test length(get_z(k)) == n_train

        @test isa(is_beta_estim(k), Bool)
        @test isa(is_theta_estim(k), Bool)
        @test isa(is_sigma2_estim(k), Bool)
        @test isa(is_normalize(k), Bool)
    end

    @testset "Log-likelihood functions" begin
        k = Kriging(y_train, X_train, "gauss")
        theta = get_theta(k)

        ll = log_likelihood(k)
        @test isfinite(ll)

        ll_res = log_likelihood_fun(k, theta)
        @test isfinite(ll_res.ll)
        @test ll_res.grad === nothing

        ll_res_grad = log_likelihood_fun(k, theta; return_grad=true)
        @test isfinite(ll_res_grad.ll)
        @test length(ll_res_grad.grad) == length(theta)

        loo_val = leave_one_out(k)
        @test isfinite(loo_val)

        loo_res = leave_one_out_fun(k, theta)
        @test isfinite(loo_res.loo)

        lmp = log_marg_post(k)
        @test isfinite(lmp)

        lmp_res = log_marg_post_fun(k, theta)
        @test isfinite(lmp_res.lmp)
    end

    @testset "Covariance matrix" begin
        k = Kriging(y_train, X_train, "gauss")
        C = cov_mat(k, X_test, X_test)
        @test size(C) == (n_test, n_test)
        @test all(isfinite.(C))
        # Covariance matrix should be symmetric
        @test C ≈ C' atol=1e-10
    end

    @testset "Leave-one-out vector" begin
        k = Kriging(y_train, X_train, "gauss")
        theta = get_theta(k)
        loo_vec = leave_one_out_vec(k, theta)
        @test length(loo_vec.yhat) == n_train
        @test length(loo_vec.stderr) == n_train
        @test all(isfinite.(loo_vec.yhat))
        @test all(loo_vec.stderr .>= 0.0)
    end
end
