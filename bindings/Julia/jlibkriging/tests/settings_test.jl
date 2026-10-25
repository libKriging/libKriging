using Test
using jlibkriging

# Optim (jlibkriging.Optim) and LinearAlgebra settings, Kriging model():
# the API of the other bindings (tools/checks/binding_api)

@testset "Optim settings" begin
    O = jlibkriging.Optim

    orig = O.get_max_iteration()
    O.set_max_iteration(42)
    @test O.get_max_iteration() == 42
    O.set_max_iteration(orig)

    orig = O.get_gradient_tolerance()
    O.set_gradient_tolerance(1e-4)
    @test O.get_gradient_tolerance() == 1e-4
    O.set_gradient_tolerance(orig)

    orig = O.get_objective_rel_tolerance()
    O.set_objective_rel_tolerance(1e-5)
    @test O.get_objective_rel_tolerance() == 1e-5
    O.set_objective_rel_tolerance(orig)

    orig = O.get_log_level()
    O.set_log_level(2)
    @test O.get_log_level() == 2
    O.set_log_level(orig)

    orig = O.is_reparametrized()
    O.use_reparametrize(!orig)
    @test O.is_reparametrized() == !orig
    O.use_reparametrize(orig)

    orig = O.variogram_bounds_heuristic_used()
    O.use_variogram_bounds_heuristic(!orig)
    @test O.variogram_bounds_heuristic_used() == !orig
    O.use_variogram_bounds_heuristic(orig)

    lo, up = O.get_theta_lower_factor(), O.get_theta_upper_factor()
    O.set_theta_lower_factor(0.05)
    O.set_theta_upper_factor(15.0)
    @test O.get_theta_lower_factor() == 0.05
    @test O.get_theta_upper_factor() == 15.0
    O.set_theta_lower_factor(lo)
    O.set_theta_upper_factor(up)

    orig = O.get_thread_start_delay_ms()
    O.set_thread_start_delay_ms(5)
    @test O.get_thread_start_delay_ms() == 5
    O.set_thread_start_delay_ms(orig)

    orig = O.get_thread_pool_size()
    O.set_thread_pool_size(2)
    @test O.get_thread_pool_size() == 2
    O.set_thread_pool_size(orig)

    orig = O.get_num_threads_percent()
    O.set_num_threads_percent(100)
    full = O.get_num_threads()
    @test full >= 1
    O.set_num_threads_percent(50)
    @test O.get_num_threads_percent() == 50
    @test 1 <= O.get_num_threads() <= full
    @test_throws ErrorException O.set_num_threads_percent(-1)
    O.set_num_threads_percent(orig)
end

@testset "LinearAlgebra settings" begin
    orig = chol_rcond_checked()
    check_chol_rcond(false)
    @test !chol_rcond_checked()
    check_chol_rcond(true)
    @test chol_rcond_checked()
    check_chol_rcond(orig)
    set_chol_warning(false)
end

@testset "Kriging model()" begin
    X = reshape(collect(range(0.0, 1.0; length=10)), :, 1)
    y = sin.(6 .* vec(X))
    m = model(Kriging(y, X, "matern3_2"))
    @test m["kernel"] == "matern3_2"
    @test m["noise_model"] == "none"
    @test m["X"] == X
    @test length(m["theta"]) == 1
    @test !haskey(m, "nugget")

    m = model(Kriging(y, X, "matern3_2"; noise="nugget"))
    @test m["noise_model"] == "nugget"
    @test haskey(m, "nugget") && haskey(m, "is_nugget_estim")
end

# arguments aligned with the other bindings (tools/checks/binding_api)
@testset "Arguments as in Python / R" begin
    X = reshape(collect(range(0.0, 1.0; length=12)), :, 1)
    y = sin.(6 .* vec(X))

    # fit!(...; parameters) of Kriging and NestedKriging
    k = Kriging("gauss")
    fit!(k, y, X; parameters=Dict("theta" => [0.3], "is_theta_estim" => false))
    @test vec(theta(k)) ≈ [0.3]
    @test !is_theta_estim(k)
    @test_throws ArgumentError fit!(k, y, X; parameters=Dict("thetas" => [0.3]))
    X2 = hcat(X, reverse(X; dims=1))
    nk = NestedKriging(y, X2, "gauss", 2)
    fit!(nk, y, X2, 2; parameters=Dict("theta" => [0.4, 0.5], "is_theta_estim" => false))
    @test vec(theta(nk)) ≈ [0.4, 0.5]

    # simulate(model, X_n; nsim=1, seed=123): nsim / seed defaults
    k = Kriging(y, X, "gauss")
    Xn = reshape([0.25, 0.75], :, 1)
    @test simulate(k, Xn) == simulate(k, 1, 123, Xn)
    @test simulate(k, Xn; nsim=3, seed=7) == simulate(k, 3, 7, Xn)
    wk = WarpKriging(y, X, ["none"], "gauss")
    @test simulate(wk, Xn; nsim=2) == simulate(wk, 2, 123, Xn)
    mk = MLPKriging(y, X, [2], 1; kernel="gauss", parameters=Dict("max_iter_adam" => "20"))
    @test simulate(mk, Xn; seed=5) == simulate(mk, 1, 5, Xn)

    # Kriging nugget model: with_noise=b is with_nugget=b
    nk = Kriging(y, X, "gauss"; noise="nugget")
    @test simulate(nk, 2, 1, Xn; with_noise=false) == simulate(nk, 2, 1, Xn; with_nugget=false)
    @test simulate(nk, 2, 1, Xn; with_noise=true) == simulate(nk, 2, 1, Xn; with_nugget=true)
    @test_throws ArgumentError simulate(nk, 2, 1, Xn; with_noise=true, with_nugget=true)

    # bench of the Kriging objective functions (prints timings, same values)
    t = vec(theta(k))
    @test log_likelihood_fun(k, t; bench=true).ll ≈ log_likelihood_fun(k, t).ll
    @test leave_one_out_fun(k, t; bench=true).loo ≈ leave_one_out_fun(k, t).loo
    @test log_marg_post_fun(k, t; bench=true).lmp ≈ log_marg_post_fun(k, t).lmp

end
