using Test
using Random
using jlibkriging

f2d(x) = sin(3.0 * x[1]) + cos(5.0 * x[2])
df2d(x) = [3.0 * cos(3.0 * x[1]), -5.0 * sin(5.0 * x[2])]

function make_design(n::Int, seed::Int)
    Random.seed!(seed)
    X = rand(n, 2)
    y = [f2d(X[i, :]) for i in 1:n]
    dY = permutedims(hcat([df2d(X[i, :]) for i in 1:n]...))
    return X, y, dY
end

@testset "Gradient-enhanced kriging (dydX)" begin
    @testset "dydX interpolates values and gradients" begin
        X, y, dY = make_design(20, 1)
        k = Kriging(y, X, "gauss"; dydX=dY)

        @test size(dy(k)) == (20, 2)

        p = predict(k, X; return_stdev=true, return_deriv=true)
        @test maximum(abs.(p.mean .- y)) < 1e-4
        @test maximum(abs.(p.mean_deriv .- dY)) < 1e-3
    end

    @testset "dydX=nothing is a value-only fit" begin
        X, y, _dY = make_design(20, 1)
        k = Kriging(y, X, "gauss")
        @test isempty(dy(k))
    end

    @testset "dydX beats a value-only fit out of sample" begin
        X, y, dY = make_design(15, 72)
        Xt, yt, _ = make_design(200, 720)

        k_plain = Kriging(y, X, "gauss")
        k_grad = Kriging(y, X, "gauss"; dydX=dY)

        mean_plain = predict(k_plain, Xt; return_stdev=false).mean
        mean_grad = predict(k_grad, Xt; return_stdev=false).mean

        rmse_plain = sqrt(sum((mean_plain .- yt) .^ 2) / length(yt))
        rmse_grad = sqrt(sum((mean_grad .- yt) .^ 2) / length(yt))
        @test rmse_grad < rmse_plain
    end

    @testset "fit! without dydX clears previous gradient observations" begin
        X, y, dY = make_design(20, 1)
        k = Kriging(y, X, "gauss"; dydX=dY)
        @test !isempty(dy(k))

        fit!(k, y, X)
        @test isempty(dy(k))
    end

    @testset "dydX rejects a non-differentiable kernel" begin
        X, y, dY = make_design(10, 1)
        @test_throws ErrorException Kriging(y, X, "exp"; dydX=dY)
    end

    @testset "dydX rejects a wrongly shaped matrix" begin
        X, y, dY = make_design(10, 1)
        @test_throws AssertionError Kriging(y, X, "gauss"; dydX=dY[:, 1:1])
    end
end
