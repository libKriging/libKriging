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

    @testset "check_blas (#429)" begin
        c = check_blas(2)
        @test !isempty(c.library)
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
