using Test
using jlibkriging

@testset "GPU" begin
    @test gpu_compiled_backends() isa String
    @test gpu_available() isa Bool
    @test gpu_backend() in ("none", "cuda", "hip", "sycl", "metal")
    @test gpu_enabled() == (gpu_backend() != "none")

    initial = gpu_enabled()
    try
        @test set_gpu_enabled(false) == false
        @test gpu_backend() == "none"
        # never turns on a backend without a device
        @test set_gpu_enabled(true) == gpu_available()
    finally
        set_gpu_enabled(initial)
    end
end
