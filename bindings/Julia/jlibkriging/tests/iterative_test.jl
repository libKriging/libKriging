using Test

std_y(v) = sqrt(sum((v .- sum(v) / length(v)) .^ 2) / (length(v) - 1))
using jlibkriging

# Method "iterative(...)": matrix-free fit / predict (CG, SLQ log-determinant),
# objective "LL" and noise model None only; Gpu and LinearAlgebra settings.

f2d(x1, x2) = sin(3 * x1) + cos(5 * x2) + x1 * x2
const n = 40
X = hcat(mod.(0.3819660 .* (1:n), 1.0), mod.(0.7548777 .* (1:n), 1.0))
y = f2d.(X[:, 1], X[:, 2])
Xn = [0.15 0.3; 0.45 0.6; 0.75 0.2]

@testset "predict method iterative on an exact fit" begin
    k = Kriging(y, X, "matern5_2")
    pe = predict(k, Xn; return_stdev=true)
    pc = predict(k, Xn; return_stdev=true, method="iterative(30,0,20,10,1e-12)")
    @test maximum(abs.(pe.mean .- pc.mean)) < 1e-5 * std_y(y)
    @test maximum(abs.(pe.stdev .- pc.stdev)) < 1e-4 * std_y(y)
end

@testset "iterative fit" begin
    k = Kriging(y, X, "matern5_2"; optim="BFGS", method="iterative(8)")
    @test method(k) == "iterative(8)"
    p = predict(k, Xn; method="iterative")
    @test all(isfinite, p.mean)
    @test isfinite(log_likelihood(k; method="iterative"))
    @test_throws Exception predict(k, Xn)  # no exact factorization
    @test_throws Exception Kriging(y, X, "matern5_2"; objective="LOO", method="iterative")
end

@testset "WarpKriging predict method iterative" begin
    w = WarpKriging(y, X, ["kumaraswamy", "kumaraswamy"], "matern5_2")
    pe = predict(w, Xn; return_stdev=true)
    pc = predict(w, Xn; return_stdev=true, method="iterative(30,0,20,10,1e-12)")
    @test maximum(abs.(pe.mean .- pc.mean)) < 1e-4 * std_y(y)
end

@testset "Gpu and LinearAlgebra settings" begin
    G = jlibkriging.Gpu
    @test G.compiled_backends() isa String
    @test G.active_backend() in ("none", "cuda", "hip", "sycl", "metal")
    e = G.get_enabled()
    G.set_enabled(false)
    @test !G.get_enabled()
    @test G.active_backend() == "none"
    G.set_enabled(e)
    mb = G.get_dense_max_mb()
    G.set_dense_max_mb(0)
    @test G.get_dense_max_mb() == 0
    @test_throws Exception G.set_dense_max_mb(-1)
    G.set_dense_max_mb(mb)
    hmb = get_iterative_dense_max_mb()
    set_iterative_dense_max_mb(0)
    @test get_iterative_dense_max_mb() == 0
    set_iterative_dense_max_mb(hmb)
    set_cg_warning(true)
    set_cg_auto_precond(true)
end
