using Test
using jlibkriging

f_test(x) = 1.0 - 0.5 * (sin(12.0 * x) / (1.0 + x) + 2.0 * cos(7.0 * x) * x^5 + 0.7)

@testset "Loading" begin
    @testset "Kriging constructor with kernel" begin
        k = Kriging("matern3_2")
        @test kernel(k) == "matern3_2"
    end

    @testset "Kriging constructor with different kernels" begin
        for kern in ["matern3_2", "matern5_2", "gauss"]
            k = Kriging(kern)
            @test kernel(k) == kern
        end
    end

    @testset "Generic load dispatch" begin
        X = reshape(collect(range(0.01, 0.99; length=8)), :, 1)
        y = [f_test(x) for x in X[:, 1]]
        files = ["loading_test_k.json", "loading_test_wk.json", "loading_test_mlp.json"]

        try
            k = Kriging(y, X, "gauss")
            save(k, files[1])
            @test jlibkriging.load(files[1]) isa Kriging

            wk = WarpKriging(y, X, ["kumaraswamy"], "gauss")
            save(wk, files[2])
            @test jlibkriging.load(files[2]) isa WarpKriging

            mk = MLPKriging(y, X, [8, 4], 2; activation="selu", kernel="gauss")
            save(mk, files[3])
            @test jlibkriging.load(files[3]) isa MLPKriging
        finally
            for file in files
                isfile(file) && rm(file)
            end
        end
    end
end
