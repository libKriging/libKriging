module jlibkriging

using Libdl

# Library loading
const _lib = Ref{Ptr{Nothing}}(C_NULL)

function _get_lib()
    if _lib[] == C_NULL
        path = get(ENV, "JLIBKRIGING_LIB_PATH", "")
        if isempty(path)
            # @__DIR__ is bindings/Julia/jlibkriging/src/
            # repo root is four levels up
            repo_root = joinpath(@__DIR__, "..", "..", "..", "..")
            build_subdir = joinpath("bindings", "Julia", "jlibkriging")
            candidates = String[]
            for build_dir in ["build", "build-Release", "build-Debug"]
                for libname in ["libkriging_c.so", "libkriging_c.dylib", "libkriging_c.dll"]
                    push!(candidates, joinpath(repo_root, build_dir, build_subdir, libname))
                end
            end
            push!(candidates, "libkriging_c")
            for candidate in candidates
                h = dlopen(candidate; throw_error=false)
                if h !== nothing
                    _lib[] = h
                    return _lib[]
                end
            end
            error("Cannot find libkriging_c shared library. " *
                  "Set JLIBKRIGING_LIB_PATH environment variable or build with " *
                  "cmake -DENABLE_JULIA_BINDING=ON.")
        else
            _lib[] = dlopen(path)
        end
    end
    return _lib[]
end

function _lk()
    lib = _get_lib()
    return lib
end

function _check_error(ret::Cint)
    if ret != 0
        msg = unsafe_string(ccall(dlsym(_lk(), :lk_get_last_error), Cstring, ()))
        error("libKriging error: $msg")
    end
end

function _check_ptr(ptr::Ptr{Nothing})
    if ptr == C_NULL
        msg = unsafe_string(ccall(dlsym(_lk(), :lk_get_last_error), Cstring, ()))
        error("libKriging error: $msg")
    end
    return ptr
end

# ─── LinearRegression ──────────────────────────────────────────────

mutable struct LinearRegression
    ptr::Ptr{Nothing}

    function LinearRegression()
        ptr = ccall(dlsym(_lk(), :lk_linear_regression_new), Ptr{Nothing}, ())
        obj = new(_check_ptr(ptr))
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall(dlsym(_lk(), :lk_linear_regression_delete), Nothing, (Ptr{Nothing},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

function fit!(lr::LinearRegression, y::Vector{Float64}, X::Matrix{Float64})
    n, d = size(X)
    @assert length(y) == n
    ret = ccall(dlsym(_lk(), :lk_linear_regression_fit), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint),
                lr.ptr, y, n, X, n, d)
    _check_error(ret)
    return lr
end

function predict(lr::LinearRegression, X::Matrix{Float64})
    m, d = size(X)
    mean_out = Vector{Float64}(undef, m)
    stdev_out = Vector{Float64}(undef, m)
    ret = ccall(dlsym(_lk(), :lk_linear_regression_predict), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Float64}),
                lr.ptr, X, m, d, mean_out, stdev_out)
    _check_error(ret)
    return (mean=mean_out, stdev=stdev_out)
end

# ─── Kriging ──────────────────────────────────────────────────────

mutable struct Kriging
    ptr::Ptr{Nothing}

    function Kriging(ptr::Ptr{Nothing})
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall(dlsym(_lk(), :lk_kriging_delete), Nothing, (Ptr{Nothing},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

function Kriging(kernel::String)
    ptr = ccall(dlsym(_lk(), :lk_kriging_new), Ptr{Nothing}, (Cstring,), kernel)
    return Kriging(_check_ptr(ptr))
end

function Kriging(y::Vector{Float64}, X::Matrix{Float64}, kernel::String;
                 regmodel::String="constant",
                 normalize::Bool=false,
                 optim::String="BFGS",
                 objective::String="LL",
                 sigma2::Union{Nothing,Float64}=nothing,
                 is_sigma2_estim::Bool=true,
                 theta::Union{Nothing,Vector{Float64}}=nothing,
                 is_theta_estim::Bool=true,
                 beta::Union{Nothing,Vector{Float64}}=nothing,
                 is_beta_estim::Bool=true)
    n, d = size(X)
    @assert length(y) == n
    sigma2_ptr = sigma2 === nothing ? C_NULL : Ref(sigma2)
    theta_ptr = theta === nothing ? C_NULL : pointer(theta)
    theta_n = theta === nothing ? 0 : length(theta)
    beta_ptr = beta === nothing ? C_NULL : pointer(beta)
    beta_n = beta === nothing ? 0 : length(beta)
    ptr = ccall(dlsym(_lk(), :lk_kriging_new_fit), Ptr{Nothing},
                (Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint, Cint,
                 Cstring, Cstring, Cint, Cstring, Cstring,
                 Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint, Cint,
                 Ptr{Float64}, Cint, Cint),
                y, n, X, n, d,
                kernel, regmodel, normalize ? 1 : 0, optim, objective,
                sigma2_ptr, is_sigma2_estim ? 1 : 0,
                theta_ptr, theta_n, is_theta_estim ? 1 : 0,
                beta_ptr, beta_n, is_beta_estim ? 1 : 0)
    return Kriging(_check_ptr(ptr))
end

function Base.copy(k::Kriging)
    ptr = ccall(dlsym(_lk(), :lk_kriging_copy), Ptr{Nothing}, (Ptr{Nothing},), k.ptr)
    return Kriging(_check_ptr(ptr))
end

function fit!(k::Kriging, y::Vector{Float64}, X::Matrix{Float64};
              regmodel::String="constant",
              normalize::Bool=false,
              optim::String="BFGS",
              objective::String="LL")
    n, d = size(X)
    @assert length(y) == n
    ret = ccall(dlsym(_lk(), :lk_kriging_fit), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint,
                 Cstring, Cint, Cstring, Cstring),
                k.ptr, y, n, X, n, d, regmodel, normalize ? 1 : 0, optim, objective)
    _check_error(ret)
    return k
end

function predict(k::Kriging, X_n::Matrix{Float64};
                 return_stdev::Bool=true,
                 return_cov::Bool=false,
                 return_deriv::Bool=false)
    m, d = size(X_n)
    mean_out = Vector{Float64}(undef, m)
    stdev_out = return_stdev ? Vector{Float64}(undef, m) : Float64[]
    cov_out = return_cov ? Matrix{Float64}(undef, m, m) : Matrix{Float64}(undef, 0, 0)
    mean_deriv_out = return_deriv ? Matrix{Float64}(undef, m, d) : Matrix{Float64}(undef, 0, 0)
    stdev_deriv_out = return_deriv ? Matrix{Float64}(undef, m, d) : Matrix{Float64}(undef, 0, 0)

    ret = ccall(dlsym(_lk(), :lk_kriging_predict), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Cint,
                 Cint, Cint, Cint,
                 Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
                k.ptr, X_n, m, d,
                return_stdev ? 1 : 0, return_cov ? 1 : 0, return_deriv ? 1 : 0,
                mean_out,
                return_stdev ? stdev_out : C_NULL,
                return_cov ? cov_out : C_NULL,
                return_deriv ? mean_deriv_out : C_NULL,
                return_deriv ? stdev_deriv_out : C_NULL)
    _check_error(ret)
    return (mean=mean_out,
            stdev=return_stdev ? stdev_out : nothing,
            cov=return_cov ? cov_out : nothing,
            mean_deriv=return_deriv ? mean_deriv_out : nothing,
            stdev_deriv=return_deriv ? stdev_deriv_out : nothing)
end

function simulate(k::Kriging, nsim::Int, seed::Int, X_n::Matrix{Float64};
                  will_update::Bool=false)
    m, d = size(X_n)
    sim_out = Matrix{Float64}(undef, m, nsim)
    ret = ccall(dlsym(_lk(), :lk_kriging_simulate), Cint,
                (Ptr{Nothing}, Cint, Cint, Ptr{Float64}, Cint, Cint, Cint, Ptr{Float64}),
                k.ptr, nsim, seed, X_n, m, d, will_update ? 1 : 0, sim_out)
    _check_error(ret)
    return sim_out
end

function update!(k::Kriging, y_u::Vector{Float64}, X_u::Matrix{Float64};
                 refit::Bool=true)
    n, d = size(X_u)
    @assert length(y_u) == n
    ret = ccall(dlsym(_lk(), :lk_kriging_update), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint, Cint),
                k.ptr, y_u, n, X_u, n, d, refit ? 1 : 0)
    _check_error(ret)
    return k
end

function update_simulate(k::Kriging, y_u::Vector{Float64}, X_u::Matrix{Float64})
    n, d = size(X_u)
    @assert length(y_u) == n
    # First call to get dimensions
    nsim_out = Ref{Cint}(0)
    m_out = Ref{Cint}(0)
    ret = ccall(dlsym(_lk(), :lk_kriging_update_simulate), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint,
                 Ptr{Float64}, Ptr{Cint}, Ptr{Cint}),
                k.ptr, y_u, n, X_u, n, d, C_NULL, nsim_out, m_out)
    _check_error(ret)
    sim_out = Matrix{Float64}(undef, m_out[], nsim_out[])
    ret = ccall(dlsym(_lk(), :lk_kriging_update_simulate), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint,
                 Ptr{Float64}, Ptr{Cint}, Ptr{Cint}),
                k.ptr, y_u, n, X_u, n, d, sim_out, nsim_out, m_out)
    _check_error(ret)
    return sim_out
end

function save(k::Kriging, filename::String)
    ret = ccall(dlsym(_lk(), :lk_kriging_save), Cint, (Ptr{Nothing}, Cstring), k.ptr, filename)
    _check_error(ret)
end

function load_kriging(filename::String)
    ptr = ccall(dlsym(_lk(), :lk_kriging_load), Ptr{Nothing}, (Cstring,), filename)
    return Kriging(_check_ptr(ptr))
end

function summary(k::Kriging)
    s = ccall(dlsym(_lk(), :lk_kriging_summary), Cstring, (Ptr{Nothing},), k.ptr)
    return unsafe_string(s)
end

function log_likelihood_fun(k::Kriging, theta::Vector{Float64};
                             return_grad::Bool=false,
                             return_hess::Bool=false)
    n = length(theta)
    ll = Ref{Float64}(0.0)
    grad = return_grad ? Vector{Float64}(undef, n) : Float64[]
    hess = return_hess ? Matrix{Float64}(undef, n, n) : Matrix{Float64}(undef, 0, 0)
    ret = ccall(dlsym(_lk(), :lk_kriging_log_likelihood_fun), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Cint, Cint,
                 Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
                k.ptr, theta, n, return_grad ? 1 : 0, return_hess ? 1 : 0,
                ll, return_grad ? grad : C_NULL, return_hess ? hess : C_NULL)
    _check_error(ret)
    return (ll=ll[],
            grad=return_grad ? grad : nothing,
            hess=return_hess ? hess : nothing)
end

function leave_one_out_fun(k::Kriging, theta::Vector{Float64};
                            return_grad::Bool=false)
    n = length(theta)
    loo = Ref{Float64}(0.0)
    grad = return_grad ? Vector{Float64}(undef, n) : Float64[]
    ret = ccall(dlsym(_lk(), :lk_kriging_leave_one_out_fun), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Float64}),
                k.ptr, theta, n, return_grad ? 1 : 0, loo, return_grad ? grad : C_NULL)
    _check_error(ret)
    return (loo=loo[], grad=return_grad ? grad : nothing)
end

function log_marg_post_fun(k::Kriging, theta::Vector{Float64};
                            return_grad::Bool=false)
    n = length(theta)
    lmp = Ref{Float64}(0.0)
    grad = return_grad ? Vector{Float64}(undef, n) : Float64[]
    ret = ccall(dlsym(_lk(), :lk_kriging_log_marg_post_fun), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Float64}),
                k.ptr, theta, n, return_grad ? 1 : 0, lmp, return_grad ? grad : C_NULL)
    _check_error(ret)
    return (lmp=lmp[], grad=return_grad ? grad : nothing)
end

function log_likelihood(k::Kriging)
    return ccall(dlsym(_lk(), :lk_kriging_log_likelihood), Float64, (Ptr{Nothing},), k.ptr)
end

function leave_one_out(k::Kriging)
    return ccall(dlsym(_lk(), :lk_kriging_leave_one_out), Float64, (Ptr{Nothing},), k.ptr)
end

function log_marg_post(k::Kriging)
    return ccall(dlsym(_lk(), :lk_kriging_log_marg_post), Float64, (Ptr{Nothing},), k.ptr)
end

function leave_one_out_vec(k::Kriging, theta::Vector{Float64})
    n_theta = length(theta)
    n_ref = Ref{Cint}(0)
    # query size
    ccall(dlsym(_lk(), :lk_kriging_get_y), Cint,
          (Ptr{Nothing}, Ptr{Float64}, Ptr{Cint}), k.ptr, C_NULL, n_ref)
    n_obs = n_ref[]
    yhat = Vector{Float64}(undef, n_obs)
    stderr_out = Vector{Float64}(undef, n_obs)
    ret = ccall(dlsym(_lk(), :lk_kriging_leave_one_out_vec), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64}),
                k.ptr, theta, n_theta, yhat, stderr_out)
    _check_error(ret)
    return (yhat=yhat, stderr=stderr_out)
end

function cov_mat(k::Kriging, X1::Matrix{Float64}, X2::Matrix{Float64})
    n1, d1 = size(X1)
    n2, d2 = size(X2)
    @assert d1 == d2
    out = Matrix{Float64}(undef, n1, n2)
    ret = ccall(dlsym(_lk(), :lk_kriging_cov_mat), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}),
                k.ptr, X1, n1, d1, X2, n2, d2, out)
    _check_error(ret)
    return out
end

# Kriging getters
function kernel(k::Kriging)
    return unsafe_string(ccall(dlsym(_lk(), :lk_kriging_kernel), Cstring, (Ptr{Nothing},), k.ptr))
end

function optim(k::Kriging)
    return unsafe_string(ccall(dlsym(_lk(), :lk_kriging_optim), Cstring, (Ptr{Nothing},), k.ptr))
end

function objective(k::Kriging)
    return unsafe_string(ccall(dlsym(_lk(), :lk_kriging_objective), Cstring, (Ptr{Nothing},), k.ptr))
end

function is_normalize(k::Kriging)
    return ccall(dlsym(_lk(), :lk_kriging_is_normalize), Cint, (Ptr{Nothing},), k.ptr) != 0
end

function regmodel(k::Kriging)
    return unsafe_string(ccall(dlsym(_lk(), :lk_kriging_regmodel), Cstring, (Ptr{Nothing},), k.ptr))
end

function _get_vec(sym::Symbol, ptr::Ptr{Nothing})
    n = Ref{Cint}(0)
    ccall(dlsym(_lk(), sym), Cint, (Ptr{Nothing}, Ptr{Float64}, Ptr{Cint}), ptr, C_NULL, n)
    out = Vector{Float64}(undef, n[])
    ccall(dlsym(_lk(), sym), Cint, (Ptr{Nothing}, Ptr{Float64}, Ptr{Cint}), ptr, out, n)
    return out
end

function _get_mat(sym::Symbol, ptr::Ptr{Nothing})
    n = Ref{Cint}(0)
    d = Ref{Cint}(0)
    ccall(dlsym(_lk(), sym), Cint, (Ptr{Nothing}, Ptr{Float64}, Ptr{Cint}, Ptr{Cint}), ptr, C_NULL, n, d)
    out = Matrix{Float64}(undef, n[], d[])
    ccall(dlsym(_lk(), sym), Cint, (Ptr{Nothing}, Ptr{Float64}, Ptr{Cint}, Ptr{Cint}), ptr, out, n, d)
    return out
end

function _get_rowvec(sym::Symbol, ptr::Ptr{Nothing})
    d = Ref{Cint}(0)
    ccall(dlsym(_lk(), sym), Cint, (Ptr{Nothing}, Ptr{Float64}, Ptr{Cint}), ptr, C_NULL, d)
    out = Vector{Float64}(undef, d[])
    ccall(dlsym(_lk(), sym), Cint, (Ptr{Nothing}, Ptr{Float64}, Ptr{Cint}), ptr, out, d)
    return out
end

get_X(k::Kriging) = _get_mat(:lk_kriging_get_X, k.ptr)
get_centerX(k::Kriging) = _get_rowvec(:lk_kriging_get_centerX, k.ptr)
get_scaleX(k::Kriging) = _get_rowvec(:lk_kriging_get_scaleX, k.ptr)
get_y(k::Kriging) = _get_vec(:lk_kriging_get_y, k.ptr)
get_centerY(k::Kriging) = ccall(dlsym(_lk(), :lk_kriging_get_centerY), Float64, (Ptr{Nothing},), k.ptr)
get_scaleY(k::Kriging) = ccall(dlsym(_lk(), :lk_kriging_get_scaleY), Float64, (Ptr{Nothing},), k.ptr)
get_F(k::Kriging) = _get_mat(:lk_kriging_get_F, k.ptr)
get_T(k::Kriging) = _get_mat(:lk_kriging_get_T, k.ptr)
get_M(k::Kriging) = _get_mat(:lk_kriging_get_M, k.ptr)
get_z(k::Kriging) = _get_vec(:lk_kriging_get_z, k.ptr)
get_beta(k::Kriging) = _get_vec(:lk_kriging_get_beta, k.ptr)
get_theta(k::Kriging) = _get_vec(:lk_kriging_get_theta, k.ptr)
get_sigma2(k::Kriging) = ccall(dlsym(_lk(), :lk_kriging_get_sigma2), Float64, (Ptr{Nothing},), k.ptr)
is_beta_estim(k::Kriging) = ccall(dlsym(_lk(), :lk_kriging_is_beta_estim), Cint, (Ptr{Nothing},), k.ptr) != 0
is_theta_estim(k::Kriging) = ccall(dlsym(_lk(), :lk_kriging_is_theta_estim), Cint, (Ptr{Nothing},), k.ptr) != 0
is_sigma2_estim(k::Kriging) = ccall(dlsym(_lk(), :lk_kriging_is_sigma2_estim), Cint, (Ptr{Nothing},), k.ptr) != 0

# ─── NuggetKriging ─────────────────────────────────────────────────

mutable struct NuggetKriging
    ptr::Ptr{Nothing}

    function NuggetKriging(ptr::Ptr{Nothing})
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall(dlsym(_lk(), :lk_nugget_kriging_delete), Nothing, (Ptr{Nothing},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

function NuggetKriging(kernel::String)
    ptr = ccall(dlsym(_lk(), :lk_nugget_kriging_new), Ptr{Nothing}, (Cstring,), kernel)
    return NuggetKriging(_check_ptr(ptr))
end

function NuggetKriging(y::Vector{Float64}, X::Matrix{Float64}, kernel::String;
                        regmodel::String="constant",
                        normalize::Bool=false,
                        optim::String="BFGS",
                        objective::String="LL",
                        sigma2::Union{Nothing,Vector{Float64}}=nothing,
                        is_sigma2_estim::Bool=true,
                        theta::Union{Nothing,Vector{Float64}}=nothing,
                        is_theta_estim::Bool=true,
                        beta::Union{Nothing,Vector{Float64}}=nothing,
                        is_beta_estim::Bool=true,
                        nugget::Union{Nothing,Vector{Float64}}=nothing,
                        is_nugget_estim::Bool=true)
    n, d = size(X)
    @assert length(y) == n
    sigma2_ptr = sigma2 === nothing ? C_NULL : pointer(sigma2)
    sigma2_n = sigma2 === nothing ? 0 : length(sigma2)
    theta_ptr = theta === nothing ? C_NULL : pointer(theta)
    theta_n = theta === nothing ? 0 : length(theta)
    beta_ptr = beta === nothing ? C_NULL : pointer(beta)
    beta_n = beta === nothing ? 0 : length(beta)
    nugget_ptr = nugget === nothing ? C_NULL : pointer(nugget)
    nugget_n = nugget === nothing ? 0 : length(nugget)
    ptr = ccall(dlsym(_lk(), :lk_nugget_kriging_new_fit), Ptr{Nothing},
                (Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint, Cint,
                 Cstring, Cstring, Cint, Cstring, Cstring,
                 Ptr{Float64}, Cint, Cint,
                 Ptr{Float64}, Cint, Cint,
                 Ptr{Float64}, Cint, Cint,
                 Ptr{Float64}, Cint, Cint),
                y, n, X, n, d,
                kernel, regmodel, normalize ? 1 : 0, optim, objective,
                sigma2_ptr, sigma2_n, is_sigma2_estim ? 1 : 0,
                theta_ptr, theta_n, is_theta_estim ? 1 : 0,
                beta_ptr, beta_n, is_beta_estim ? 1 : 0,
                nugget_ptr, nugget_n, is_nugget_estim ? 1 : 0)
    return NuggetKriging(_check_ptr(ptr))
end

function Base.copy(k::NuggetKriging)
    ptr = ccall(dlsym(_lk(), :lk_nugget_kriging_copy), Ptr{Nothing}, (Ptr{Nothing},), k.ptr)
    return NuggetKriging(_check_ptr(ptr))
end

function fit!(k::NuggetKriging, y::Vector{Float64}, X::Matrix{Float64};
              regmodel::String="constant", normalize::Bool=false,
              optim::String="BFGS", objective::String="LL")
    n, d = size(X)
    ret = ccall(dlsym(_lk(), :lk_nugget_kriging_fit), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint,
                 Cstring, Cint, Cstring, Cstring),
                k.ptr, y, n, X, n, d, regmodel, normalize ? 1 : 0, optim, objective)
    _check_error(ret)
    return k
end

function predict(k::NuggetKriging, X_n::Matrix{Float64};
                 return_stdev::Bool=true, return_cov::Bool=false, return_deriv::Bool=false)
    m, d = size(X_n)
    mean_out = Vector{Float64}(undef, m)
    stdev_out = return_stdev ? Vector{Float64}(undef, m) : Float64[]
    cov_out = return_cov ? Matrix{Float64}(undef, m, m) : Matrix{Float64}(undef, 0, 0)
    mean_deriv_out = return_deriv ? Matrix{Float64}(undef, m, d) : Matrix{Float64}(undef, 0, 0)
    stdev_deriv_out = return_deriv ? Matrix{Float64}(undef, m, d) : Matrix{Float64}(undef, 0, 0)
    ret = ccall(dlsym(_lk(), :lk_nugget_kriging_predict), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Cint, Cint, Cint, Cint,
                 Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
                k.ptr, X_n, m, d,
                return_stdev ? 1 : 0, return_cov ? 1 : 0, return_deriv ? 1 : 0,
                mean_out,
                return_stdev ? stdev_out : C_NULL,
                return_cov ? cov_out : C_NULL,
                return_deriv ? mean_deriv_out : C_NULL,
                return_deriv ? stdev_deriv_out : C_NULL)
    _check_error(ret)
    return (mean=mean_out, stdev=return_stdev ? stdev_out : nothing,
            cov=return_cov ? cov_out : nothing,
            mean_deriv=return_deriv ? mean_deriv_out : nothing,
            stdev_deriv=return_deriv ? stdev_deriv_out : nothing)
end

function simulate(k::NuggetKriging, nsim::Int, seed::Int, X_n::Matrix{Float64};
                  with_nugget::Bool=true, will_update::Bool=false)
    m, d = size(X_n)
    sim_out = Matrix{Float64}(undef, m, nsim)
    ret = ccall(dlsym(_lk(), :lk_nugget_kriging_simulate), Cint,
                (Ptr{Nothing}, Cint, Cint, Ptr{Float64}, Cint, Cint, Cint, Cint, Ptr{Float64}),
                k.ptr, nsim, seed, X_n, m, d, with_nugget ? 1 : 0, will_update ? 1 : 0, sim_out)
    _check_error(ret)
    return sim_out
end

function update!(k::NuggetKriging, y_u::Vector{Float64}, X_u::Matrix{Float64}; refit::Bool=true)
    n, d = size(X_u)
    ret = ccall(dlsym(_lk(), :lk_nugget_kriging_update), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint, Cint),
                k.ptr, y_u, n, X_u, n, d, refit ? 1 : 0)
    _check_error(ret)
    return k
end

function save(k::NuggetKriging, filename::String)
    ret = ccall(dlsym(_lk(), :lk_nugget_kriging_save), Cint, (Ptr{Nothing}, Cstring), k.ptr, filename)
    _check_error(ret)
end

function load_nugget_kriging(filename::String)
    ptr = ccall(dlsym(_lk(), :lk_nugget_kriging_load), Ptr{Nothing}, (Cstring,), filename)
    return NuggetKriging(_check_ptr(ptr))
end

function summary(k::NuggetKriging)
    return unsafe_string(ccall(dlsym(_lk(), :lk_nugget_kriging_summary), Cstring, (Ptr{Nothing},), k.ptr))
end

function log_likelihood_fun(k::NuggetKriging, theta::Vector{Float64};
                             return_grad::Bool=false, return_hess::Bool=false)
    n = length(theta)
    ll = Ref{Float64}(0.0)
    grad = return_grad ? Vector{Float64}(undef, n) : Float64[]
    hess = return_hess ? Matrix{Float64}(undef, n, n) : Matrix{Float64}(undef, 0, 0)
    ret = ccall(dlsym(_lk(), :lk_nugget_kriging_log_likelihood_fun), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Cint, Cint,
                 Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
                k.ptr, theta, n, return_grad ? 1 : 0, return_hess ? 1 : 0,
                ll, return_grad ? grad : C_NULL, return_hess ? hess : C_NULL)
    _check_error(ret)
    return (ll=ll[], grad=return_grad ? grad : nothing, hess=return_hess ? hess : nothing)
end

function log_likelihood(k::NuggetKriging)
    return ccall(dlsym(_lk(), :lk_nugget_kriging_log_likelihood), Float64, (Ptr{Nothing},), k.ptr)
end

function log_marg_post(k::NuggetKriging)
    return ccall(dlsym(_lk(), :lk_nugget_kriging_log_marg_post), Float64, (Ptr{Nothing},), k.ptr)
end

function cov_mat(k::NuggetKriging, X1::Matrix{Float64}, X2::Matrix{Float64})
    n1, d1 = size(X1)
    n2, d2 = size(X2)
    out = Matrix{Float64}(undef, n1, n2)
    ret = ccall(dlsym(_lk(), :lk_nugget_kriging_cov_mat), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}),
                k.ptr, X1, n1, d1, X2, n2, d2, out)
    _check_error(ret)
    return out
end

kernel(k::NuggetKriging) = unsafe_string(ccall(dlsym(_lk(), :lk_nugget_kriging_kernel), Cstring, (Ptr{Nothing},), k.ptr))
optim(k::NuggetKriging) = unsafe_string(ccall(dlsym(_lk(), :lk_nugget_kriging_optim), Cstring, (Ptr{Nothing},), k.ptr))
objective(k::NuggetKriging) = unsafe_string(ccall(dlsym(_lk(), :lk_nugget_kriging_objective), Cstring, (Ptr{Nothing},), k.ptr))
is_normalize(k::NuggetKriging) = ccall(dlsym(_lk(), :lk_nugget_kriging_is_normalize), Cint, (Ptr{Nothing},), k.ptr) != 0
regmodel(k::NuggetKriging) = unsafe_string(ccall(dlsym(_lk(), :lk_nugget_kriging_regmodel), Cstring, (Ptr{Nothing},), k.ptr))
get_X(k::NuggetKriging) = _get_mat(:lk_nugget_kriging_get_X, k.ptr)
get_centerX(k::NuggetKriging) = _get_rowvec(:lk_nugget_kriging_get_centerX, k.ptr)
get_scaleX(k::NuggetKriging) = _get_rowvec(:lk_nugget_kriging_get_scaleX, k.ptr)
get_y(k::NuggetKriging) = _get_vec(:lk_nugget_kriging_get_y, k.ptr)
get_centerY(k::NuggetKriging) = ccall(dlsym(_lk(), :lk_nugget_kriging_get_centerY), Float64, (Ptr{Nothing},), k.ptr)
get_scaleY(k::NuggetKriging) = ccall(dlsym(_lk(), :lk_nugget_kriging_get_scaleY), Float64, (Ptr{Nothing},), k.ptr)
get_F(k::NuggetKriging) = _get_mat(:lk_nugget_kriging_get_F, k.ptr)
get_T(k::NuggetKriging) = _get_mat(:lk_nugget_kriging_get_T, k.ptr)
get_M(k::NuggetKriging) = _get_mat(:lk_nugget_kriging_get_M, k.ptr)
get_z(k::NuggetKriging) = _get_vec(:lk_nugget_kriging_get_z, k.ptr)
get_beta(k::NuggetKriging) = _get_vec(:lk_nugget_kriging_get_beta, k.ptr)
get_theta(k::NuggetKriging) = _get_vec(:lk_nugget_kriging_get_theta, k.ptr)
get_sigma2(k::NuggetKriging) = ccall(dlsym(_lk(), :lk_nugget_kriging_get_sigma2), Float64, (Ptr{Nothing},), k.ptr)
get_nugget(k::NuggetKriging) = ccall(dlsym(_lk(), :lk_nugget_kriging_get_nugget), Float64, (Ptr{Nothing},), k.ptr)
is_beta_estim(k::NuggetKriging) = ccall(dlsym(_lk(), :lk_nugget_kriging_is_beta_estim), Cint, (Ptr{Nothing},), k.ptr) != 0
is_theta_estim(k::NuggetKriging) = ccall(dlsym(_lk(), :lk_nugget_kriging_is_theta_estim), Cint, (Ptr{Nothing},), k.ptr) != 0
is_sigma2_estim(k::NuggetKriging) = ccall(dlsym(_lk(), :lk_nugget_kriging_is_sigma2_estim), Cint, (Ptr{Nothing},), k.ptr) != 0
is_nugget_estim(k::NuggetKriging) = ccall(dlsym(_lk(), :lk_nugget_kriging_is_nugget_estim), Cint, (Ptr{Nothing},), k.ptr) != 0

# ─── NoiseKriging ──────────────────────────────────────────────────

mutable struct NoiseKriging
    ptr::Ptr{Nothing}

    function NoiseKriging(ptr::Ptr{Nothing})
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall(dlsym(_lk(), :lk_noise_kriging_delete), Nothing, (Ptr{Nothing},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

function NoiseKriging(kernel::String)
    ptr = ccall(dlsym(_lk(), :lk_noise_kriging_new), Ptr{Nothing}, (Cstring,), kernel)
    return NoiseKriging(_check_ptr(ptr))
end

function NoiseKriging(y::Vector{Float64}, noise::Vector{Float64}, X::Matrix{Float64}, kernel::String;
                       regmodel::String="constant",
                       normalize::Bool=false,
                       optim::String="BFGS",
                       objective::String="LL",
                       sigma2::Union{Nothing,Vector{Float64}}=nothing,
                       is_sigma2_estim::Bool=true,
                       theta::Union{Nothing,Vector{Float64}}=nothing,
                       is_theta_estim::Bool=true,
                       beta::Union{Nothing,Vector{Float64}}=nothing,
                       is_beta_estim::Bool=true)
    n, d = size(X)
    @assert length(y) == n
    @assert length(noise) == n
    sigma2_ptr = sigma2 === nothing ? C_NULL : pointer(sigma2)
    sigma2_n = sigma2 === nothing ? 0 : length(sigma2)
    theta_ptr = theta === nothing ? C_NULL : pointer(theta)
    theta_n = theta === nothing ? 0 : length(theta)
    beta_ptr = beta === nothing ? C_NULL : pointer(beta)
    beta_n = beta === nothing ? 0 : length(beta)
    ptr = ccall(dlsym(_lk(), :lk_noise_kriging_new_fit), Ptr{Nothing},
                (Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint, Cint,
                 Cstring, Cstring, Cint, Cstring, Cstring,
                 Ptr{Float64}, Cint, Cint,
                 Ptr{Float64}, Cint, Cint,
                 Ptr{Float64}, Cint, Cint),
                y, n, noise, n, X, n, d,
                kernel, regmodel, normalize ? 1 : 0, optim, objective,
                sigma2_ptr, sigma2_n, is_sigma2_estim ? 1 : 0,
                theta_ptr, theta_n, is_theta_estim ? 1 : 0,
                beta_ptr, beta_n, is_beta_estim ? 1 : 0)
    return NoiseKriging(_check_ptr(ptr))
end

function Base.copy(k::NoiseKriging)
    ptr = ccall(dlsym(_lk(), :lk_noise_kriging_copy), Ptr{Nothing}, (Ptr{Nothing},), k.ptr)
    return NoiseKriging(_check_ptr(ptr))
end

function fit!(k::NoiseKriging, y::Vector{Float64}, noise::Vector{Float64}, X::Matrix{Float64};
              regmodel::String="constant", normalize::Bool=false,
              optim::String="BFGS", objective::String="LL")
    n, d = size(X)
    ret = ccall(dlsym(_lk(), :lk_noise_kriging_fit), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint, Cint, Cstring, Cint, Cstring, Cstring),
                k.ptr, y, n, noise, n, X, n, d, regmodel, normalize ? 1 : 0, optim, objective)
    _check_error(ret)
    return k
end

function predict(k::NoiseKriging, X_n::Matrix{Float64};
                 return_stdev::Bool=true, return_cov::Bool=false, return_deriv::Bool=false)
    m, d = size(X_n)
    mean_out = Vector{Float64}(undef, m)
    stdev_out = return_stdev ? Vector{Float64}(undef, m) : Float64[]
    cov_out = return_cov ? Matrix{Float64}(undef, m, m) : Matrix{Float64}(undef, 0, 0)
    mean_deriv_out = return_deriv ? Matrix{Float64}(undef, m, d) : Matrix{Float64}(undef, 0, 0)
    stdev_deriv_out = return_deriv ? Matrix{Float64}(undef, m, d) : Matrix{Float64}(undef, 0, 0)
    ret = ccall(dlsym(_lk(), :lk_noise_kriging_predict), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Cint, Cint, Cint, Cint,
                 Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
                k.ptr, X_n, m, d,
                return_stdev ? 1 : 0, return_cov ? 1 : 0, return_deriv ? 1 : 0,
                mean_out,
                return_stdev ? stdev_out : C_NULL,
                return_cov ? cov_out : C_NULL,
                return_deriv ? mean_deriv_out : C_NULL,
                return_deriv ? stdev_deriv_out : C_NULL)
    _check_error(ret)
    return (mean=mean_out, stdev=return_stdev ? stdev_out : nothing,
            cov=return_cov ? cov_out : nothing,
            mean_deriv=return_deriv ? mean_deriv_out : nothing,
            stdev_deriv=return_deriv ? stdev_deriv_out : nothing)
end

function simulate(k::NoiseKriging, nsim::Int, seed::Int, X_n::Matrix{Float64},
                  with_noise::Vector{Float64}; will_update::Bool=false)
    m, d = size(X_n)
    sim_out = Matrix{Float64}(undef, m, nsim)
    ret = ccall(dlsym(_lk(), :lk_noise_kriging_simulate), Cint,
                (Ptr{Nothing}, Cint, Cint, Ptr{Float64}, Cint, Cint,
                 Ptr{Float64}, Cint, Cint, Ptr{Float64}),
                k.ptr, nsim, seed, X_n, m, d,
                with_noise, length(with_noise), will_update ? 1 : 0, sim_out)
    _check_error(ret)
    return sim_out
end

function update!(k::NoiseKriging, y_u::Vector{Float64}, noise_u::Vector{Float64},
                 X_u::Matrix{Float64}; refit::Bool=true)
    n, d = size(X_u)
    ret = ccall(dlsym(_lk(), :lk_noise_kriging_update), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint, Cint),
                k.ptr, y_u, n, noise_u, n, X_u, n, d, refit ? 1 : 0)
    _check_error(ret)
    return k
end

function save(k::NoiseKriging, filename::String)
    ret = ccall(dlsym(_lk(), :lk_noise_kriging_save), Cint, (Ptr{Nothing}, Cstring), k.ptr, filename)
    _check_error(ret)
end

function load_noise_kriging(filename::String)
    ptr = ccall(dlsym(_lk(), :lk_noise_kriging_load), Ptr{Nothing}, (Cstring,), filename)
    return NoiseKriging(_check_ptr(ptr))
end

function summary(k::NoiseKriging)
    return unsafe_string(ccall(dlsym(_lk(), :lk_noise_kriging_summary), Cstring, (Ptr{Nothing},), k.ptr))
end

function log_likelihood_fun(k::NoiseKriging, theta::Vector{Float64};
                             return_grad::Bool=false, return_hess::Bool=false)
    n = length(theta)
    ll = Ref{Float64}(0.0)
    grad = return_grad ? Vector{Float64}(undef, n) : Float64[]
    hess = return_hess ? Matrix{Float64}(undef, n, n) : Matrix{Float64}(undef, 0, 0)
    ret = ccall(dlsym(_lk(), :lk_noise_kriging_log_likelihood_fun), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Cint, Cint,
                 Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
                k.ptr, theta, n, return_grad ? 1 : 0, return_hess ? 1 : 0,
                ll, return_grad ? grad : C_NULL, return_hess ? hess : C_NULL)
    _check_error(ret)
    return (ll=ll[], grad=return_grad ? grad : nothing, hess=return_hess ? hess : nothing)
end

function log_likelihood(k::NoiseKriging)
    return ccall(dlsym(_lk(), :lk_noise_kriging_log_likelihood), Float64, (Ptr{Nothing},), k.ptr)
end

function cov_mat(k::NoiseKriging, X1::Matrix{Float64}, X2::Matrix{Float64})
    n1, d1 = size(X1)
    n2, d2 = size(X2)
    out = Matrix{Float64}(undef, n1, n2)
    ret = ccall(dlsym(_lk(), :lk_noise_kriging_cov_mat), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}),
                k.ptr, X1, n1, d1, X2, n2, d2, out)
    _check_error(ret)
    return out
end

kernel(k::NoiseKriging) = unsafe_string(ccall(dlsym(_lk(), :lk_noise_kriging_kernel), Cstring, (Ptr{Nothing},), k.ptr))
optim(k::NoiseKriging) = unsafe_string(ccall(dlsym(_lk(), :lk_noise_kriging_optim), Cstring, (Ptr{Nothing},), k.ptr))
objective(k::NoiseKriging) = unsafe_string(ccall(dlsym(_lk(), :lk_noise_kriging_objective), Cstring, (Ptr{Nothing},), k.ptr))
is_normalize(k::NoiseKriging) = ccall(dlsym(_lk(), :lk_noise_kriging_is_normalize), Cint, (Ptr{Nothing},), k.ptr) != 0
regmodel(k::NoiseKriging) = unsafe_string(ccall(dlsym(_lk(), :lk_noise_kriging_regmodel), Cstring, (Ptr{Nothing},), k.ptr))
get_X(k::NoiseKriging) = _get_mat(:lk_noise_kriging_get_X, k.ptr)
get_centerX(k::NoiseKriging) = _get_rowvec(:lk_noise_kriging_get_centerX, k.ptr)
get_scaleX(k::NoiseKriging) = _get_rowvec(:lk_noise_kriging_get_scaleX, k.ptr)
get_y(k::NoiseKriging) = _get_vec(:lk_noise_kriging_get_y, k.ptr)
get_centerY(k::NoiseKriging) = ccall(dlsym(_lk(), :lk_noise_kriging_get_centerY), Float64, (Ptr{Nothing},), k.ptr)
get_scaleY(k::NoiseKriging) = ccall(dlsym(_lk(), :lk_noise_kriging_get_scaleY), Float64, (Ptr{Nothing},), k.ptr)
get_noise(k::NoiseKriging) = _get_vec(:lk_noise_kriging_get_noise, k.ptr)
get_F(k::NoiseKriging) = _get_mat(:lk_noise_kriging_get_F, k.ptr)
get_T(k::NoiseKriging) = _get_mat(:lk_noise_kriging_get_T, k.ptr)
get_M(k::NoiseKriging) = _get_mat(:lk_noise_kriging_get_M, k.ptr)
get_z(k::NoiseKriging) = _get_vec(:lk_noise_kriging_get_z, k.ptr)
get_beta(k::NoiseKriging) = _get_vec(:lk_noise_kriging_get_beta, k.ptr)
get_theta(k::NoiseKriging) = _get_vec(:lk_noise_kriging_get_theta, k.ptr)
get_sigma2(k::NoiseKriging) = ccall(dlsym(_lk(), :lk_noise_kriging_get_sigma2), Float64, (Ptr{Nothing},), k.ptr)
is_beta_estim(k::NoiseKriging) = ccall(dlsym(_lk(), :lk_noise_kriging_is_beta_estim), Cint, (Ptr{Nothing},), k.ptr) != 0
is_theta_estim(k::NoiseKriging) = ccall(dlsym(_lk(), :lk_noise_kriging_is_theta_estim), Cint, (Ptr{Nothing},), k.ptr) != 0
is_sigma2_estim(k::NoiseKriging) = ccall(dlsym(_lk(), :lk_noise_kriging_is_sigma2_estim), Cint, (Ptr{Nothing},), k.ptr) != 0

# ─── Exports ──────────────────────────────────────────────────────

export LinearRegression, Kriging, NuggetKriging, NoiseKriging
export fit!, predict, simulate, update!, update_simulate, save, summary
export load_kriging, load_nugget_kriging, load_noise_kriging
export log_likelihood_fun, leave_one_out_fun, log_marg_post_fun
export log_likelihood, leave_one_out, log_marg_post
export leave_one_out_vec, cov_mat
export kernel, optim, objective, is_normalize, regmodel
export get_X, get_centerX, get_scaleX, get_y, get_centerY, get_scaleY
export get_F, get_T, get_M, get_z, get_beta, get_theta, get_sigma2
export is_beta_estim, is_theta_estim, is_sigma2_estim
export get_nugget, is_nugget_estim, get_noise

end # module
