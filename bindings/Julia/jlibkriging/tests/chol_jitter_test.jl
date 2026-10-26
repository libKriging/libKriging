using Test
using jlibkriging

f_test(x) = 1.0 - 0.5 * (sin(12.0 * x) / (1.0 + x) + 2.0 * cos(7.0 * x) * x^5 + 0.7)

@testset "LinearAlgebra settings" begin
    saved_jitter = get_chol_jitter()
    saved_nugget = get_num_nugget()

    @testset "chol_jitter setter / getter" begin
        set_chol_jitter(1e-9)
        @test get_chol_jitter() == 1e-9
        set_chol_jitter(0)
        @test get_chol_jitter() == 0.0
        set_chol_jitter(saved_jitter)
        @test get_chol_jitter() == saved_jitter
    end

    @testset "num_nugget setter / getter" begin
        set_num_nugget(1e-12)
        @test get_num_nugget() == 1e-12
        set_num_nugget(saved_nugget)
        @test get_num_nugget() == saved_nugget
    end

    @testset "linearalgebra_check (#429)" begin
        c = linearalgebra_check(2)
        @test !isempty(c.backend)
        @test startswith(c.report, "backend: ")
        @test occursin("single thread:", c.report)
        @test occursin("2 concurrent threads", c.report)
        c.ok || println(c.report)
        @test c.ok
    end

    @testset "gauss fit with and without the jitter" begin
        X = reshape(collect(range(0.0, 1.0; length=10)), :, 1)
        y = f_test.(vec(X))
        for jitter in (saved_jitter, 0.0)
            set_chol_jitter(jitter)
            k = Kriging(y, X, "gauss")
            m = predict(k, X)[1]
            @test all(isfinite, m)
            # the committed model (jitter <= num_nugget) interpolates
            @test maximum(abs.(m .- y)) < 1e-4
        end
        set_chol_jitter(saved_jitter)
    end
end

@testset "jitters of a fitted model, kept by save / load" begin
    X = reshape(collect(range(0.0, 1.0; length=10)), :, 1)
    y = sin.(6 .* vec(X))
    models = [Kriging(y, X, "gauss"),
              WarpKriging(y, X, ["kumaraswamy"], "gauss"),
              MLPKriging(y, X, [4], 1; kernel="gauss", parameters=Dict("max_iter_adam" => "20"))]
    for k in models
        @test chol_jitter(k) > 0  # gauss, noise-free
        @test 0 <= model_chol_jitter(k) <= chol_jitter(k)
        f = tempname() * ".json"
        save(k, f)
        k2 = load(f)
        @test chol_jitter(k2) == chol_jitter(k)
        @test model_chol_jitter(k2) == model_chol_jitter(k)
        rm(f)
    end
    @test model_chol_jitter(models[1]) == 0  # Kriging: exact committed model
end

