using Test
using Random
using jlibkriging

f_test(x) = 1.0 - 0.5 * (sin(12.0 * x) / (1.0 + x) + 2.0 * cos(7.0 * x) * x^5 + 0.7)

@testset "NuggetKriging" begin
    # Training data with slight noise to give the nugget something to estimate
    X_train_vec = collect(range(0.0, 1.0; length=20))
    n_train = length(X_train_vec)
    X_train = reshape(X_train_vec, :, 1)
    rng = MersenneTwister(42)
    y_train = [f_test(x) for x in X_train_vec] .+ 0.01 .* randn(rng, n_train)

    X_test_vec = collect(range(0.05, 0.95; length=10))
    n_test = length(X_test_vec)
    X_test = reshape(X_test_vec, :, 1)

    @testset "Construction with kernel" begin
        nk = NuggetKriging("matern3_2")
        @test kernel(nk) == "matern3_2"
    end

    @testset "Construction with data" begin
        nk = NuggetKriging(y_train, X_train, "matern5_2")
        @test kernel(nk) == "matern5_2"
        @test optim(nk) == "BFGS"
        @test objective(nk) == "LL"
        @test regmodel(nk) == "constant"
    end

    @testset "Fit and predict" begin
        nk = NuggetKriging("matern5_2")
        fit!(nk, y_train, X_train)

        result = predict(nk, X_test)
        @test length(result.mean) == n_test
        @test length(result.stdev) == n_test
        @test all(result.stdev .>= 0.0)
        @test all(isfinite.(result.mean))
    end

    @testset "Predict with covariance and derivatives" begin
        nk = NuggetKriging(y_train, X_train, "matern5_2")

        result = predict(nk, X_test; return_cov=true, return_deriv=true)
        @test size(result.cov) == (n_test, n_test)
        @test size(result.mean_deriv) == (n_test, 1)
        @test size(result.stdev_deriv) == (n_test, 1)
    end

    @testset "Simulate with nugget" begin
        nk = NuggetKriging(y_train, X_train, "matern5_2")
        nsim = 5

        sim_with = simulate(nk, nsim, 123, X_test; with_nugget=true)
        @test size(sim_with) == (n_test, nsim)
        @test all(isfinite.(sim_with))
    end

    @testset "Simulate without nugget" begin
        nk = NuggetKriging(y_train, X_train, "matern5_2")
        nsim = 5

        sim_without = simulate(nk, nsim, 123, X_test; with_nugget=false)
        @test size(sim_without) == (n_test, nsim)
        @test all(isfinite.(sim_without))
    end

    @testset "Simulate with vs without nugget differ" begin
        nk = NuggetKriging(y_train, X_train, "matern5_2")
        nsim = 5

        sim_with = simulate(nk, nsim, 123, X_test; with_nugget=true)
        sim_without = simulate(nk, nsim, 123, X_test; with_nugget=false)
        # Both should produce valid results
        @test all(isfinite.(sim_with))
        @test all(isfinite.(sim_without))
        # Check that nugget was estimated
        nug = get_nugget(nk)
        if nug > 1e-10
            @test sim_with != sim_without
        else
            @test_skip "nugget too small to produce different simulations"
        end
    end

    @testset "Update" begin
        nk = NuggetKriging(y_train, X_train, "matern5_2")
        p_before = predict(nk, X_test)

        X_new = reshape([0.35], :, 1)
        y_new = [f_test(0.35)]
        update!(nk, y_new, X_new)

        p_after = predict(nk, X_test)
        @test p_before.mean != p_after.mean
    end

    @testset "Summary" begin
        nk = NuggetKriging(y_train, X_train, "matern5_2")
        s = jlibkriging.summary(nk)
        @test isa(s, String)
        @test length(s) > 0
    end

    @testset "Getters" begin
        nk = NuggetKriging(y_train, X_train, "matern5_2")

        @test size(get_X(nk)) == (n_train, 1)
        @test length(get_y(nk)) == n_train
        @test length(get_theta(nk)) > 0
        @test get_sigma2(nk) > 0.0
        @test get_nugget(nk) >= 0.0
        @test length(get_beta(nk)) > 0
        @test isa(get_centerY(nk), Float64)
        @test isa(get_scaleY(nk), Float64)
        @test length(get_centerX(nk)) == 1
        @test length(get_scaleX(nk)) == 1
        @test size(get_F(nk), 1) == n_train
        @test size(get_T(nk), 1) == n_train
        @test length(get_z(nk)) == n_train

        @test isa(is_beta_estim(nk), Bool)
        @test isa(is_theta_estim(nk), Bool)
        @test isa(is_sigma2_estim(nk), Bool)
        @test isa(is_nugget_estim(nk), Bool)
        @test isa(is_normalize(nk), Bool)
    end

    @testset "Log-likelihood" begin
        nk = NuggetKriging(y_train, X_train, "matern5_2")
        theta = get_theta(nk)

        ll = log_likelihood(nk)
        @test isfinite(ll)

        ll_res = log_likelihood_fun(nk, theta)
        @test isfinite(ll_res.ll)
        @test ll_res.grad === nothing

        ll_res_grad = log_likelihood_fun(nk, theta; return_grad=true)
        @test isfinite(ll_res_grad.ll)
        @test length(ll_res_grad.grad) == length(theta)

        lmp = log_marg_post(nk)
        @test isfinite(lmp)
    end

    @testset "Covariance matrix" begin
        nk = NuggetKriging(y_train, X_train, "matern5_2")
        C = cov_mat(nk, X_test, X_test)
        @test size(C) == (n_test, n_test)
        @test all(isfinite.(C))
        @test C ≈ C' atol=1e-10
    end

    @testset "Copy" begin
        nk1 = NuggetKriging(y_train, X_train, "matern5_2")
        nk2 = copy(nk1)
        @test nk1.ptr != nk2.ptr

        p1 = predict(nk1, X_test)
        p2 = predict(nk2, X_test)
        @test p1.mean == p2.mean
        @test p1.stdev == p2.stdev
    end
end
