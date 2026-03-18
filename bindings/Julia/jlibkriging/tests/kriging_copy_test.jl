using Test
using jlibkriging

f_test(x) = 1.0 - 0.5 * (sin(12.0 * x) / (1.0 + x) + 2.0 * cos(7.0 * x) * x^5 + 0.7)

@testset "Kriging Copy" begin
    # Training data
    X_train_vec = [0.0, 0.2, 0.5, 0.8, 1.0]
    X_train = reshape(Float64.(X_train_vec), :, 1)
    y_train = [f_test(x) for x in X_train_vec]

    # Prediction grid
    X_pred_vec = collect(range(0.0, stop=1.0, length=99))
    X_pred = reshape(X_pred_vec, :, 1)

    @testset "Copied Kriging returns same result" begin
        k1 = Kriging(y_train, X_train, "gauss"; sigma2=1.0, is_theta_estim=true)
        k2 = copy(k1)

        # Different objects
        @test k1.ptr != k2.ptr

        p1 = predict(k1, X_pred; return_stdev=true, return_cov=true, return_deriv=true)
        p2 = predict(k2, X_pred; return_stdev=true, return_cov=true, return_deriv=true)

        @test p1.mean == p2.mean
        @test p1.stdev == p2.stdev
        @test p1.cov == p2.cov
        @test p1.mean_deriv == p2.mean_deriv

        # Filter non-finite derivatives (may not be derivable at design points)
        sd1 = p1.stdev_deriv
        sd2 = p2.stdev_deriv
        finite_mask = isfinite.(sd1) .& isfinite.(sd2)
        if any(finite_mask)
            @test sd1[finite_mask] == sd2[finite_mask]
        end
    end

    @testset "Copied and updated Kriging returns different result" begin
        k1 = Kriging(y_train, X_train, "gauss"; sigma2=1.0, is_theta_estim=false)
        k2 = copy(k1)

        # Different objects
        @test k1.ptr != k2.ptr

        # Predict on original
        p1 = predict(k1, X_pred; return_stdev=true)

        # Update the copy with new data
        X_new = reshape([0.6], :, 1)
        y_new = [f_test(0.6)]
        update!(k2, y_new, X_new)

        # Predict on updated copy
        p2 = predict(k2, X_pred; return_stdev=true)

        # Predictions should differ after update
        @test p1.mean != p2.mean
        @test p1.stdev != p2.stdev
    end

    @testset "Original unchanged after copy is updated" begin
        k1 = Kriging(y_train, X_train, "gauss")
        p1_before = predict(k1, X_pred)

        k2 = copy(k1)
        X_new = reshape([0.3, 0.7], :, 1)
        y_new = [f_test(0.3), f_test(0.7)]
        update!(k2, y_new, X_new)

        p1_after = predict(k1, X_pred)

        # Original should be unchanged
        @test p1_before.mean == p1_after.mean
        @test p1_before.stdev == p1_after.stdev
    end

    @testset "Copy of NuggetKriging" begin
        X_train_nk = reshape(collect(range(0.0, 1.0; length=15)), :, 1)
        y_train_nk = [f_test(x) for x in X_train_nk[:, 1]] .+ 0.01 .* randn(15)

        nk1 = NuggetKriging(y_train_nk, X_train_nk, "matern5_2")
        nk2 = copy(nk1)

        @test nk1.ptr != nk2.ptr

        p1 = predict(nk1, X_pred)
        p2 = predict(nk2, X_pred)
        @test p1.mean == p2.mean
        @test p1.stdev == p2.stdev
    end

    @testset "Copy of NoiseKriging" begin
        X_train_nk = reshape(collect(range(0.0, 1.0; length=15)), :, 1)
        y_train_nk = [f_test(x) for x in X_train_nk[:, 1]] .+ 0.01 .* randn(15)
        noise_vec = fill(0.01^2, 15)

        nk1 = NoiseKriging(y_train_nk, noise_vec, X_train_nk, "matern5_2")
        nk2 = copy(nk1)

        @test nk1.ptr != nk2.ptr

        p1 = predict(nk1, X_pred)
        p2 = predict(nk2, X_pred)
        @test p1.mean == p2.mean
        @test p1.stdev == p2.stdev
    end
end
