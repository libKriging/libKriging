using Test
using jlibkriging
import Random
# no Statistics dependency
mean(a) = sum(a) / length(a)
mean(a, d) = sum(a; dims=d) ./ size(a, d)
std(a) = sqrt(sum((a .- mean(a)) .^ 2) / (length(a) - 1))

# Damped oscillation sampled at q = 30 time steps, two inputs
const t_out = collect(range(0.5, 10, length=30))
code(x) = exp.(-(0.1 + 0.5 * x[2]) .* t_out) .* cos.(2pi * (0.5 + 1.5 * x[1]) .* t_out ./ 5)
outputs(X) = Matrix{Float64}(reduce(vcat, [code(X[i, :])' for i in 1:size(X, 1)]))

@testset "MultiOutputKriging" begin
    rng = Random.MersenneTwister(1)
    X = rand(rng, 40, 2)
    Y = outputs(X)
    Xt = rand(rng, 10, 2)
    Yt = outputs(Xt)

    @testset "pca" begin
        k = MultiOutputKriging(Y, X, "matern5_2"; output_model="pca(0.999)")
        @test nb_outputs(k) == 30
        K = nb_components(k)
        @test size(pca_basis(k)) == (30, K)
        @test output_model(k) == "pca(0.999)"
        p = predict(k, Xt; return_cov=true, return_deriv=true)
        @test size(p.mean) == (10, 30)
        @test size(p.stdev) == (10, 30)
        @test size(p.cov) == (300, 300)
        @test size(p.mean_deriv) == (10, 2, 30)
        @test maximum(abs.(sqrt.([p.cov[i, i] for i in 1:300]) .- vec(p.stdev))) < 1e-8
        @test sqrt(mean((p.mean .- Yt) .^ 2)) < 0.25 * std(Y)
        km = component(k, 1)
        @test km isa Kriging
        @test size(jlibkriging.X(km), 1) == 40
    end

    @testset "shared with one output is Kriging" begin
        y = Y[:, 5]
        km = Kriging(y, X, "matern5_2")
        k = MultiOutputKriging(y, X, "matern5_2"; output_model="shared")
        @test isapprox(theta(k), theta(km); rtol=1e-6)
        @test isapprox(log_likelihood(k), log_likelihood(km); rtol=1e-8)
        pk = predict(km, Xt)
        pm = predict(k, Xt)
        @test isapprox(vec(pm.mean), pk.mean; rtol=1e-6)
        @test isapprox(vec(pm.stdev), pk.stdev; rtol=1e-6)
    end

    @testset "shared gradient" begin
        k = MultiOutputKriging(Y, X, "matern5_2"; output_model="shared")
        th = theta(k) .* 1.3
        r = log_likelihood_fun(k, th; return_grad=true)
        h = 1e-4  # smaller steps are dominated by the rounding noise of the LL
        for i in 1:2
            e = zeros(2); e[i] = h
            fd = (log_likelihood_fun(k, th .+ e).ll - log_likelihood_fun(k, th .- e).ll) / (2h)
            @test isapprox(r.grad[i], fd; rtol=1e-4, atol=1e-6)
        end
    end

    @testset "separable" begin
        Y3 = Y[:, [3, 10, 20]]
        k = MultiOutputKriging(Y3, X, "matern5_2"; output_model="separable")
        S = output_cov(k)
        @test size(S) == (3, 3)
        f = predict_cov_factors(k, Xt)
        p = predict(k, Xt; return_cov=true)
        @test maximum(abs.(kron(f.Sigma, f.Cx) .- p.cov)) < 1e-8 * maximum(abs.(p.cov))
        sims = simulate(k, 2000, 3, Xt)
        @test size(sims) == (10, 3, 2000)
        emp = dropdims(mean(sims, 3); dims=3)
        @test maximum(abs.(emp .- p.mean) ./ max.(p.stdev, 1e-12)) < 15 / sqrt(2000)
    end

    @testset "update and update_simulate" begin
        for om in ["pca(0.999)", "shared"]
            k = MultiOutputKriging(Y, X, "matern5_2"; output_model=om)
            Xu, Yu, xs = Xt[1:3, :], Yt[1:3, :], Xt[4:10, :]
            @test_throws ErrorException update_simulate(k, Yu, Xu)
            s0 = simulate(k, 1000, 5, xs; will_update=true)
            s1 = update_simulate(k, Yu, Xu)
            @test size(s1) == size(s0)
            update!(k, Yu, Xu; refit=false)
            @test size(jlibkriging.X(k), 1) == 43
            p = predict(k, xs)
            emp = dropdims(mean(s1, 3); dims=3)
            @test maximum(abs.(emp .- p.mean)) < 5 * maximum(p.stdev) / sqrt(1000) + 1e-8
        end
    end

    @testset "leave-one-out and fixed theta" begin
        k = MultiOutputKriging(Y, X, "matern5_2"; output_model="shared")
        l = leave_one_out_mat(k)
        @test size(l.mean) == size(Y)
        @test isapprox(leave_one_out(k), mean((Y .- l.mean) .^ 2); rtol=1e-10)
        k2 = MultiOutputKriging("matern5_2"; output_model="shared")
        fit!(k2, Y, X; optim="none", theta=[0.3, 0.4])
        @test theta(k2) ≈ [0.3, 0.4]
        @test_throws ErrorException MultiOutputKriging(Y, X, "matern5_2"; output_model="foo")
    end
end
