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
        files = ["loading_test_k.json", "loading_test_wk.json", "loading_test_mlp.json",
                 "loading_test_nk.json"]

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

            X2 = [0.1 0.2; 0.3 0.9; 0.5 0.4; 0.7 0.1; 0.9 0.6; 0.2 0.7; 0.6 0.8; 0.8 0.3;
                  0.4 0.5; 0.15 0.35; 0.45 0.95; 0.65 0.25; 0.85 0.75; 0.25 0.05; 0.55 0.6; 0.95 0.15]
            y2 = [sin(3.0 * X2[i, 1]) + X2[i, 2] for i in 1:size(X2, 1)]
            nk = NestedKriging(y2, X2, "gauss", 2)
            save(nk, files[4])
            @test jlibkriging.load(files[4]) isa NestedKriging
        finally
            for file in files
                isfile(file) && rm(file)
            end
        end
    end
end
