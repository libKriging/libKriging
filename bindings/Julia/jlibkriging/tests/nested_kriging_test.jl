using Test
using jlibkriging

f_test(x) = sin(3.0 * x[1]) + cos(5.0 * x[2]) + x[1] * x[2]

@testset "NestedKriging" begin
    import Random
    rng = Random.MersenneTwister(123)
    n, d = 200, 2
    X = rand(rng, n, d)
    y = [f_test(X[i, :]) for i in 1:n]
    Xt = rand(rng, 100, d)
    yt = [f_test(Xt[i, :]) for i in 1:100]

    @testset "Construction and accessors" begin
        k = NestedKriging(y, X, "gauss", 4)
        @test kernel(k) == "gauss"
        @test aggregation(k) == "NK"
        @test nb_groups(k) == 4
        @test length(theta(k)) == d
        @test sigma2(k) > 0
        @test occursin("groups", jlibkriging.summary(k))
    end

    @testset "All aggregations predict" begin
        for agg in ["PoE", "gPoE", "BCM", "rBCM", "NK"]
            k = NestedKriging(y, X, "matern5_2", 4; aggregation=agg)
            p = predict(k, Xt)
            @test length(p.mean) == 100
            @test length(p.stdev) == 100
            @test all(isfinite, p.mean)
            @test all(p.stdev .>= 0)
            rmse = sqrt(sum((p.mean .- yt) .^ 2) / length(yt))
            @test rmse < 0.5 * sqrt(sum((y .- sum(y) / n) .^ 2) / n)
        end
    end

    @testset "NK interpolates the design" begin
        k = NestedKriging(y, X, "matern5_2", 4; aggregation="NK")
        p = predict(k, X)
        @test maximum(abs.(p.mean .- y)) < 1e-3
        @test maximum(p.stdev) < 1e-2
    end

    @testset "Reproducibility" begin
        k1 = NestedKriging(y, X, "gauss", 5; partition="random", seed=42)
        k2 = NestedKriging(y, X, "gauss", 5; partition="random", seed=42)
        p1 = predict(k1, Xt)
        p2 = predict(k2, Xt)
        @test p1.mean == p2.mean
        @test p1.stdev == p2.stdev
    end

    @testset "Input validation" begin
        @test_throws ErrorException NestedKriging(y, X, "gauss", 4; aggregation="median")
        @test_throws ErrorException NestedKriging(y, X, "gauss", 1000)
        @test_throws ErrorException NestedKriging(y, X, "gauss", 4; aggregation="NK", regmodel="linear")
    end
end

@testset "NestedKriging with warping" begin
    import Random
    rng = Random.MersenneTwister(123)
    X = rand(rng, 120, 2)
    y = [sin(3.0 * X[i, 1]) + cos(5.0 * X[i, 2]) for i in 1:120]
    k = NestedKriging(y, X, "gauss", 3; warping=["kumaraswamy", "kumaraswamy"])
    p = predict(k, X)
    @test maximum(abs.(p.mean .- y)) < 1e-3  # NK interpolates under warping
    @test all(p.stdev .>= 0)
end

@testset "NestedKriging parameters= dict" begin
    import Random
    rng = Random.MersenneTwister(123)
    n, d = 100, 2
    X = rand(rng, n, d)
    y = [f_test(X[i, :]) for i in 1:n]
    k = NestedKriging(y, X, "matern5_2", 4; parameters=Dict("theta" => fill(0.3, d)))
    p = predict(k, X)
    @test length(p.mean) == n
    @test all(isfinite, p.mean)
end

@testset "NestedKriging save/load" begin
    import Random
    rng = Random.MersenneTwister(123)
    n, d = 120, 2
    X = rand(rng, n, d)
    y = [f_test(X[i, :]) for i in 1:n]
    Xt = rand(rng, 30, d)
    for agg in ["NK", "gPoE"]
        k = NestedKriging(y, X, "matern5_2", 3; aggregation=agg)
        file = tempname() * ".json"
        try
            save(k, file)
            for k2 in (load_nested_kriging(file), jlibkriging.load(file))
                @test k2 isa NestedKriging
                @test aggregation(k2) == agg
                @test nb_groups(k2) == nb_groups(k)
                @test theta(k2) == theta(k)
                p1 = predict(k, Xt)
                p2 = predict(k2, Xt)
                @test maximum(abs.(p2.mean .- p1.mean)) < 1e-10
                @test maximum(abs.(p2.stdev .- p1.stdev)) < 1e-10
            end
        finally
            isfile(file) && rm(file)
        end
    end
end

@testset "NestedKriging accessors and setters" begin
    import Random
    rng = Random.MersenneTwister(7)
    n, d = 60, 2
    X = rand(rng, n, d)
    y = [f_test(X[i, :]) for i in 1:n]
    k = NestedKriging(y, X, "gauss", 3)
    @test jlibkriging.X(k) == X
    @test jlibkriging.y(k) == y
    g = groups(k)
    @test length(g) == 3
    @test sort(vcat(g...)) == collect(1:n)  # 1-based partition of the rows
    @test isempty(warping(k))
    Xt = rand(rng, 10, d)
    set_predict_chunk!(k, 3)
    p1 = predict(k, Xt)
    set_predict_chunk!(k, 128)
    p2 = predict(k, Xt)
    @test maximum(abs.(p1.mean .- p2.mean)) < 1e-8  # up to BLAS blocking / summation order
    @test set_warp_subsample!(k, 500) === k
    @test_throws ErrorException set_predict_chunk!(k, 0)
    kw = NestedKriging(y, X, "gauss", 2; warping=["kumaraswamy", "kumaraswamy"])
    @test warping(kw) == ["kumaraswamy", "kumaraswamy"]
end
