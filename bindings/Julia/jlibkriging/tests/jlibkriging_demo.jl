using Test
using jlibkriging

# Test function: f(x) = 1 - 1/2 * (sin(12x)/(1+x) + 2*cos(7x)*x^5 + 0.7)
f_demo(x) = 1.0 - 0.5 * (sin(12.0 * x) / (1.0 + x) + 2.0 * cos(7.0 * x) * x^5 + 0.7)

@testset "jlibkriging Demo" begin
    # Design points
    X_design_vec = [0.0, 0.2, 0.5, 0.8, 1.0]
    n_design = length(X_design_vec)
    X_design = reshape(Float64.(X_design_vec), :, 1)
    y_design = [f_demo(x) for x in X_design_vec]

    # Prediction grid
    X_pred_vec = collect(range(0.0, stop=1.0, length=99))
    n_pred = length(X_pred_vec)
    X_pred = reshape(X_pred_vec, :, 1)
    y_true = [f_demo(x) for x in X_pred_vec]

    # --- Phase 1: Fit Kriging model ---
    @testset "Fit model" begin
        k = Kriging(y_design, X_design, "gauss"; sigma2=1.0, is_theta_estim=false)
        @test kernel(k) == "gauss"
        s = jlibkriging.summary(k)
        @test length(s) > 0
        println("Model summary after fit:")
        println(s)
    end

    k = Kriging(y_design, X_design, "gauss"; sigma2=1.0, is_theta_estim=false)

    # --- Phase 2: Predict ---
    @testset "Predict" begin
        result = predict(k, X_pred; return_stdev=true, return_cov=true)

        @test length(result.mean) == n_pred
        @test length(result.stdev) == n_pred
        @test size(result.cov) == (n_pred, n_pred)
        @test all(isfinite.(result.mean))
        @test all(result.stdev .>= 0.0)

        # Check that predictions are in a reasonable range
        @test all(-2.0 .< result.mean .< 3.0)

        println("Prediction range: [$(minimum(result.mean)), $(maximum(result.mean))]")
        println("Max stdev: $(maximum(result.stdev))")
    end

    # --- Phase 3: Simulate ---
    @testset "Simulate" begin
        nsim = 10
        sim = simulate(k, nsim, 123, X_pred)

        @test size(sim) == (n_pred, nsim)
        @test all(isfinite.(sim))

        println("Simulation shape: $(size(sim))")
        println("Simulation range: [$(minimum(sim)), $(maximum(sim))]")
    end

    # --- Phase 4: Update with new points ---
    @testset "Update" begin
        s_before = jlibkriging.summary(k)
        println("\nSummary before update:")
        println(s_before)

        X_new = reshape([0.3, 0.4], :, 1)
        y_new = [f_demo(0.3), f_demo(0.4)]
        update!(k, y_new, X_new)

        s_after = jlibkriging.summary(k)
        println("\nSummary after update:")
        println(s_after)

        # Model should now have more training points
        @test length(get_y(k)) == n_design + 2
        @test size(get_X(k), 1) == n_design + 2
    end

    # --- Phase 5: Predict again after update ---
    @testset "Predict after update" begin
        result = predict(k, X_pred; return_stdev=true)

        @test length(result.mean) == n_pred
        @test all(isfinite.(result.mean))
        @test all(result.stdev .>= 0.0)

        # With more data, max stdev should be reasonable
        @test maximum(result.stdev) < 5.0

        println("Post-update prediction range: [$(minimum(result.mean)), $(maximum(result.mean))]")
        println("Post-update max stdev: $(maximum(result.stdev))")
    end

    # --- Phase 6: Full workflow with getters ---
    @testset "Model properties" begin
        @test kernel(k) == "gauss"
        @test get_sigma2(k) > 0.0
        @test length(get_theta(k)) == 1
        @test length(get_beta(k)) > 0
        @test isa(log_likelihood(k), Float64)
        @test isfinite(log_likelihood(k))

        println("\nFinal model properties:")
        println("  kernel: $(kernel(k))")
        println("  sigma2: $(get_sigma2(k))")
        println("  theta: $(get_theta(k))")
        println("  beta: $(get_beta(k))")
        println("  log_likelihood: $(log_likelihood(k))")
    end
end
