using Test
using jlibkriging

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

    @testset "NuggetKriging constructor" begin
        nk = NuggetKriging("matern3_2")
        @test kernel(nk) == "matern3_2"
    end

    @testset "NoiseKriging constructor" begin
        nk = NoiseKriging("matern3_2")
        @test kernel(nk) == "matern3_2"
    end

    @testset "LinearRegression constructor" begin
        lr = LinearRegression()
        @test lr.ptr != C_NULL
    end
end
