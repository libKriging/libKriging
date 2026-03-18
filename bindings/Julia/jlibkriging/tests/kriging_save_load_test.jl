using Test
using jlibkriging

f_test(x) = 1.0 - 0.5 * (sin(12.0 * x) / (1.0 + x) + 2.0 * cos(7.0 * x) * x^5 + 0.7)

@testset "Kriging Save/Load" begin
    # Training data
    X_train_vec = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]
    X_train = reshape(Float64.(X_train_vec), :, 1)
    y_train = [f_test(x) for x in X_train_vec]

    # Prediction grid
    X_pred_vec = collect(range(0.05, 0.95; length=20))
    X_pred = reshape(X_pred_vec, :, 1)

    @testset "Save and load Kriging" begin
        k1 = Kriging(y_train, X_train, "gauss")
        p1 = predict(k1, X_pred)

        tmpfile = tempname() * ".json"
        try
            save(k1, tmpfile)
            @test isfile(tmpfile)

            k2 = load_kriging(tmpfile)
            p2 = predict(k2, X_pred)

            @test p1.mean ≈ p2.mean atol=1e-12
            @test p1.stdev ≈ p2.stdev atol=1e-12

            # Verify getters match
            @test kernel(k1) == kernel(k2)
            @test get_sigma2(k1) ≈ get_sigma2(k2) atol=1e-12
            @test get_theta(k1) ≈ get_theta(k2) atol=1e-12
            @test get_beta(k1) ≈ get_beta(k2) atol=1e-12
        finally
            isfile(tmpfile) && rm(tmpfile)
        end
    end

    @testset "Save and load Kriging with different kernels" begin
        for kern in ["matern3_2", "matern5_2", "gauss"]
            k1 = Kriging(y_train, X_train, kern)
            p1 = predict(k1, X_pred)

            tmpfile = tempname() * ".json"
            try
                save(k1, tmpfile)
                k2 = load_kriging(tmpfile)
                p2 = predict(k2, X_pred)

                @test kernel(k2) == kern
                @test p1.mean ≈ p2.mean atol=1e-12
            finally
                isfile(tmpfile) && rm(tmpfile)
            end
        end
    end

    @testset "Save and load NuggetKriging" begin
        X_train_nk = reshape(collect(range(0.0, 1.0; length=20)), :, 1)
        y_train_nk = [f_test(x) for x in X_train_nk[:, 1]] .+ 0.01 .* randn(20)

        nk1 = NuggetKriging(y_train_nk, X_train_nk, "matern5_2")
        p1 = predict(nk1, X_pred)

        tmpfile = tempname() * ".json"
        try
            save(nk1, tmpfile)
            @test isfile(tmpfile)

            nk2 = load_nugget_kriging(tmpfile)
            p2 = predict(nk2, X_pred)

            @test p1.mean ≈ p2.mean atol=1e-12
            @test p1.stdev ≈ p2.stdev atol=1e-12

            @test kernel(nk1) == kernel(nk2)
            @test get_sigma2(nk1) ≈ get_sigma2(nk2) atol=1e-12
            @test get_nugget(nk1) ≈ get_nugget(nk2) atol=1e-12
            @test get_theta(nk1) ≈ get_theta(nk2) atol=1e-12
        finally
            isfile(tmpfile) && rm(tmpfile)
        end
    end

    @testset "Save and load NoiseKriging" begin
        X_train_nk = reshape(collect(range(0.0, 1.0; length=20)), :, 1)
        noise_vec = fill(0.01^2, 20)
        y_train_nk = [f_test(x) for x in X_train_nk[:, 1]] .+ 0.01 .* randn(20)

        nk1 = NoiseKriging(y_train_nk, noise_vec, X_train_nk, "matern5_2")
        p1 = predict(nk1, X_pred)

        tmpfile = tempname() * ".json"
        try
            save(nk1, tmpfile)
            @test isfile(tmpfile)

            nk2 = load_noise_kriging(tmpfile)
            p2 = predict(nk2, X_pred)

            @test p1.mean ≈ p2.mean atol=1e-12
            @test p1.stdev ≈ p2.stdev atol=1e-12

            @test kernel(nk1) == kernel(nk2)
            @test get_sigma2(nk1) ≈ get_sigma2(nk2) atol=1e-12
            @test get_theta(nk1) ≈ get_theta(nk2) atol=1e-12
            @test get_noise(nk1) ≈ get_noise(nk2) atol=1e-12
        finally
            isfile(tmpfile) && rm(tmpfile)
        end
    end

    @testset "Loaded model can be updated" begin
        k1 = Kriging(y_train, X_train, "gauss")

        tmpfile = tempname() * ".json"
        try
            save(k1, tmpfile)
            k2 = load_kriging(tmpfile)

            X_new = reshape([0.42], :, 1)
            y_new = [f_test(0.42)]
            update!(k2, y_new, X_new)

            p = predict(k2, X_pred)
            @test length(p.mean) == length(X_pred_vec)
            @test all(isfinite.(p.mean))
        finally
            isfile(tmpfile) && rm(tmpfile)
        end
    end
end
