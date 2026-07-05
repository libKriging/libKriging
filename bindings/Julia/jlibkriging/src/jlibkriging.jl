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
                for config_subdir in ["", "Release", "Debug"]
                    for libname in ["libkriging_c.so", "libkriging_c.dylib", "libkriging_c.dll"]
                        push!(candidates, joinpath(repo_root, build_dir, build_subdir, config_subdir, libname))
                    end
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

function Kriging(kernel::String; noise::Union{Nothing,String}=nothing)
    nm = noise === nothing ? "none" : noise
    ptr = ccall(dlsym(_lk(), :lk_kriging_new), Ptr{Nothing}, (Cstring, Cstring), kernel, nm)
    return Kriging(_check_ptr(ptr))
end

function Kriging(y::Vector{Float64}, X::Matrix{Float64}, kernel::String;
                 noise::Union{Nothing,String,Float64,Vector{Float64}}=nothing,
                 regmodel::String="constant",
                 normalize::Bool=false,
                 optim::String="BFGS",
                 objective::String="LL",
                 sigma2::Union{Nothing,Float64}=nothing,
                 is_sigma2_estim::Bool=true,
                 theta::Union{Nothing,Vector{Float64}}=nothing,
                 is_theta_estim::Bool=true,
                 beta::Union{Nothing,Vector{Float64}}=nothing,
                 is_beta_estim::Bool=true,
                 nugget::Union{Nothing,Float64}=nothing,
                 is_nugget_estim::Bool=true)
    n, d = size(X)
    @assert length(y) == n

    # Determine noise_model string and noise vector
    if noise === nothing
        noise_model_str = "none"
        noise_ptr = C_NULL
        noise_n = 0
    elseif noise isa String
        noise_model_str = noise  # "nugget" or "none"
        noise_ptr = C_NULL
        noise_n = 0
    elseif noise isa Float64
        noise_model_str = "heterogeneous"
        noise_vec = fill(noise, n)
        noise_ptr = pointer(noise_vec)
        noise_n = n
    elseif noise isa Vector{Float64}
        noise_model_str = "heterogeneous"
        noise_ptr = pointer(noise)
        noise_n = length(noise)
    else
        error("noise must be nothing, a String, Float64, or Vector{Float64}")
    end

    sigma2_ptr = sigma2 === nothing ? C_NULL : Ref(sigma2)
    theta_ptr = theta === nothing ? C_NULL : pointer(theta)
    theta_n = theta === nothing ? 0 : length(theta)
    beta_ptr = beta === nothing ? C_NULL : pointer(beta)
    beta_n = beta === nothing ? 0 : length(beta)
    nugget_ptr = nugget === nothing ? C_NULL : Ref(nugget)

    if noise isa Float64
        # Keep noise_vec alive during ccall
        GC.@preserve noise_vec begin
            ptr = ccall(dlsym(_lk(), :lk_kriging_new_fit), Ptr{Nothing},
                        (Ptr{Float64}, Cint,
                         Ptr{Float64}, Cint,
                         Ptr{Float64}, Cint, Cint,
                         Cstring, Cstring, Cstring, Cint, Cstring, Cstring,
                         Ptr{Float64}, Cint,
                         Ptr{Float64}, Cint, Cint,
                         Ptr{Float64}, Cint, Cint,
                         Ptr{Float64}, Cint),
                        y, n,
                        noise_ptr, noise_n,
                        X, n, d,
                        kernel, noise_model_str, regmodel, normalize ? 1 : 0, optim, objective,
                        sigma2_ptr, is_sigma2_estim ? 1 : 0,
                        theta_ptr, theta_n, is_theta_estim ? 1 : 0,
                        beta_ptr, beta_n, is_beta_estim ? 1 : 0,
                        nugget_ptr, is_nugget_estim ? 1 : 0)
        end
    else
        ptr = ccall(dlsym(_lk(), :lk_kriging_new_fit), Ptr{Nothing},
                    (Ptr{Float64}, Cint,
                     Ptr{Float64}, Cint,
                     Ptr{Float64}, Cint, Cint,
                     Cstring, Cstring, Cstring, Cint, Cstring, Cstring,
                     Ptr{Float64}, Cint,
                     Ptr{Float64}, Cint, Cint,
                     Ptr{Float64}, Cint, Cint,
                     Ptr{Float64}, Cint),
                    y, n,
                    noise_ptr, noise_n,
                    X, n, d,
                    kernel, noise_model_str, regmodel, normalize ? 1 : 0, optim, objective,
                    sigma2_ptr, is_sigma2_estim ? 1 : 0,
                    theta_ptr, theta_n, is_theta_estim ? 1 : 0,
                    beta_ptr, beta_n, is_beta_estim ? 1 : 0,
                    nugget_ptr, is_nugget_estim ? 1 : 0)
    end
    return Kriging(_check_ptr(ptr))
end

function Base.copy(k::Kriging)
    ptr = ccall(dlsym(_lk(), :lk_kriging_copy), Ptr{Nothing}, (Ptr{Nothing},), k.ptr)
    return Kriging(_check_ptr(ptr))
end

function fit!(k::Kriging, y::Vector{Float64}, X::Matrix{Float64};
              noise::Union{Nothing,Vector{Float64}}=nothing,
              regmodel::String="constant",
              normalize::Bool=false,
              optim::String="BFGS",
              objective::String="LL")
    n, d = size(X)
    @assert length(y) == n
    noise_ptr = noise === nothing ? C_NULL : pointer(noise)
    noise_n = noise === nothing ? 0 : length(noise)
    ret = ccall(dlsym(_lk(), :lk_kriging_fit), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint, Cint,
                 Cstring, Cint, Cstring, Cstring),
                k.ptr, y, n,
                noise_ptr, noise_n,
                X, n, d, regmodel, normalize ? 1 : 0, optim, objective)
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
                  with_nugget::Bool=true,
                  with_noise::Union{Nothing,Vector{Float64}}=nothing,
                  will_update::Bool=false)
    m, d = size(X_n)
    sim_out = Matrix{Float64}(undef, m, nsim)
    noise_ptr = with_noise === nothing ? C_NULL : pointer(with_noise)
    noise_n = with_noise === nothing ? 0 : length(with_noise)
    ret = ccall(dlsym(_lk(), :lk_kriging_simulate), Cint,
                (Ptr{Nothing}, Cint, Cint, Ptr{Float64}, Cint, Cint,
                 Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}),
                k.ptr, nsim, seed, X_n, m, d,
                with_nugget ? 1 : 0, noise_ptr, noise_n,
                will_update ? 1 : 0, sim_out)
    _check_error(ret)
    return sim_out
end

function update!(k::Kriging, y_u::Vector{Float64}, X_u::Matrix{Float64};
                 noise_u::Union{Nothing,Vector{Float64}}=nothing,
                 refit::Bool=true)
    n, d = size(X_u)
    @assert length(y_u) == n
    nu_ptr = noise_u === nothing ? C_NULL : pointer(noise_u)
    nu_n = noise_u === nothing ? 0 : length(noise_u)
    ret = ccall(dlsym(_lk(), :lk_kriging_update), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint, Cint, Cint),
                k.ptr, y_u, n,
                nu_ptr, nu_n,
                X_u, n, d, refit ? 1 : 0)
    _check_error(ret)
    return k
end

function update_simulate(k::Kriging, y_u::Vector{Float64}, X_u::Matrix{Float64};
                         noise_u::Union{Nothing,Vector{Float64}}=nothing)
    n, d = size(X_u)
    @assert length(y_u) == n
    nu_ptr = noise_u === nothing ? C_NULL : pointer(noise_u)
    nu_n = noise_u === nothing ? 0 : length(noise_u)
    # First call to get dimensions
    nsim_out = Ref{Cint}(0)
    m_out = Ref{Cint}(0)
    ret = ccall(dlsym(_lk(), :lk_kriging_update_simulate), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint, Cint,
                 Ptr{Float64}, Ptr{Cint}, Ptr{Cint}),
                k.ptr, y_u, n,
                nu_ptr, nu_n,
                X_u, n, d, C_NULL, nsim_out, m_out)
    _check_error(ret)
    sim_out = Matrix{Float64}(undef, m_out[], nsim_out[])
    ret = ccall(dlsym(_lk(), :lk_kriging_update_simulate), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint, Cint,
                 Ptr{Float64}, Ptr{Cint}, Ptr{Cint}),
                k.ptr, y_u, n,
                nu_ptr, nu_n,
                X_u, n, d, sim_out, nsim_out, m_out)
    _check_error(ret)
    return sim_out
end

function save(k::Kriging, filename::String)
    ret = ccall(dlsym(_lk(), :lk_kriging_save), Cint, (Ptr{Nothing}, Cstring), k.ptr, filename)
    _check_error(ret)
end

function _saved_content(filename::String)
    m = match(r"\"content\"\s*:\s*\"([^\"]+)\"", read(filename, String))
    return m === nothing ? "" : m.captures[1]
end

function load(filename::String)
    content = _saved_content(filename)
    if content == "Kriging" || content == "NoiseKriging" || content == "NuggetKriging"
        return load_kriging(filename)
    elseif content == "WarpKriging"
        return load_warp_kriging(filename)
    elseif content == "MLPKriging"
        return load_mlp_kriging(filename)
    else
        error("Unknown Kriging type in file: $filename")
    end
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

function normalize(k::Kriging)
    return ccall(dlsym(_lk(), :lk_kriging_is_normalize), Cint, (Ptr{Nothing},), k.ptr) != 0
end

# Deprecated alias
function is_normalize(k::Kriging)
    Base.depwarn("`is_normalize` is deprecated, use `normalize` instead", :is_normalize)
    return normalize(k)
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

X(k::Kriging) = _get_mat(:lk_kriging_get_X, k.ptr)
centerX(k::Kriging) = _get_rowvec(:lk_kriging_get_centerX, k.ptr)
scaleX(k::Kriging) = _get_rowvec(:lk_kriging_get_scaleX, k.ptr)
y(k::Kriging) = _get_vec(:lk_kriging_get_y, k.ptr)
centerY(k::Kriging) = ccall(dlsym(_lk(), :lk_kriging_get_centerY), Float64, (Ptr{Nothing},), k.ptr)
scaleY(k::Kriging) = ccall(dlsym(_lk(), :lk_kriging_get_scaleY), Float64, (Ptr{Nothing},), k.ptr)
F(k::Kriging) = _get_mat(:lk_kriging_get_F, k.ptr)
T(k::Kriging) = _get_mat(:lk_kriging_get_T, k.ptr)
M(k::Kriging) = _get_mat(:lk_kriging_get_M, k.ptr)
z(k::Kriging) = _get_vec(:lk_kriging_get_z, k.ptr)
beta(k::Kriging) = _get_vec(:lk_kriging_get_beta, k.ptr)
theta(k::Kriging) = _get_vec(:lk_kriging_get_theta, k.ptr)
sigma2(k::Kriging) = ccall(dlsym(_lk(), :lk_kriging_get_sigma2), Float64, (Ptr{Nothing},), k.ptr)
is_beta_estim(k::Kriging) = ccall(dlsym(_lk(), :lk_kriging_is_beta_estim), Cint, (Ptr{Nothing},), k.ptr) != 0
is_theta_estim(k::Kriging) = ccall(dlsym(_lk(), :lk_kriging_is_theta_estim), Cint, (Ptr{Nothing},), k.ptr) != 0
is_sigma2_estim(k::Kriging) = ccall(dlsym(_lk(), :lk_kriging_is_sigma2_estim), Cint, (Ptr{Nothing},), k.ptr) != 0
noise_model(k::Kriging) = unsafe_string(ccall(dlsym(_lk(), :lk_kriging_noise_model), Cstring, (Ptr{Nothing},), k.ptr))
nugget(k::Kriging) = ccall(dlsym(_lk(), :lk_kriging_get_nugget), Float64, (Ptr{Nothing},), k.ptr)
is_nugget_estim(k::Kriging) = ccall(dlsym(_lk(), :lk_kriging_is_nugget_estim), Cint, (Ptr{Nothing},), k.ptr) != 0
noise(k::Kriging) = _get_vec(:lk_kriging_get_noise, k.ptr)

# Deprecated get_*/is_normalize aliases for Kriging
for (_old, _new) in [(:get_X, :X), (:get_y, :y), (:get_theta, :theta), (:get_sigma2, :sigma2),
                     (:get_beta, :beta), (:get_nugget, :nugget), (:get_noise, :noise),
                     (:get_centerX, :centerX), (:get_scaleX, :scaleX),
                     (:get_centerY, :centerY), (:get_scaleY, :scaleY),
                     (:get_F, :F), (:get_T, :T), (:get_M, :M), (:get_z, :z)]
    @eval function $(_old)(k::Kriging)
        Base.depwarn("`$($_old)` is deprecated, use `$($_new)` instead", $(_old))
        return $(_new)(k)
    end
end

# ─── WarpKriging ──────────────────────────────────────────────────

mutable struct WarpKriging
    ptr::Ptr{Nothing}

    function WarpKriging(ptr::Ptr{Nothing})
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall(dlsym(_lk(), :lk_warp_kriging_delete), Nothing, (Ptr{Nothing},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

function WarpKriging(warping::Vector{String}, kernel::String="gauss")
    ptrs = [Base.unsafe_convert(Cstring, s) for s in warping]
    GC.@preserve warping begin
        ptr = ccall(dlsym(_lk(), :lk_warp_kriging_new), Ptr{Nothing},
                    (Ptr{Cstring}, Cint, Cstring),
                    ptrs, length(warping), kernel)
    end
    return WarpKriging(_check_ptr(ptr))
end

function WarpKriging(y::Vector{Float64}, X::Matrix{Float64},
                     warping::Vector{String}, kernel::String="gauss";
                     regmodel::String="constant",
                     normalize::Bool=false,
                     optim::String="BFGS+Adam",
                     objective::String="LL",
                     parameters::Union{Nothing,Dict{String,String}}=nothing,
                     noise::Union{Nothing,Vector{Float64}}=nothing)
    n, d = size(X)
    @assert length(y) == n
    @assert length(warping) == d || length(warping) == 1
    ptrs = [Base.unsafe_convert(Cstring, s) for s in warping]
    n_params = parameters === nothing ? 0 : length(parameters)
    if noise !== nothing
        if n_params > 0
            keys_arr = collect(keys(parameters))
            vals_arr = [parameters[k] for k in keys_arr]
            keys_c = [Base.unsafe_convert(Cstring, k) for k in keys_arr]
            vals_c = [Base.unsafe_convert(Cstring, v) for v in vals_arr]
            GC.@preserve warping keys_arr vals_arr begin
                ptr = ccall(dlsym(_lk(), :lk_warp_kriging_new_fit_noise), Ptr{Nothing},
                            (Ptr{Float64}, Cint,
                             Ptr{Float64}, Cint,
                             Ptr{Float64}, Cint, Cint,
                             Ptr{Cstring}, Cint,
                             Cstring, Cstring, Cint, Cstring, Cstring,
                             Ptr{Cstring}, Ptr{Cstring}, Cint),
                            y, n, noise, length(noise), X, n, d,
                            ptrs, length(warping),
                            kernel, regmodel, normalize ? 1 : 0, optim, objective,
                            keys_c, vals_c, n_params)
            end
        else
            GC.@preserve warping begin
                ptr = ccall(dlsym(_lk(), :lk_warp_kriging_new_fit_noise), Ptr{Nothing},
                            (Ptr{Float64}, Cint,
                             Ptr{Float64}, Cint,
                             Ptr{Float64}, Cint, Cint,
                             Ptr{Cstring}, Cint,
                             Cstring, Cstring, Cint, Cstring, Cstring,
                             Ptr{Cstring}, Ptr{Cstring}, Cint),
                            y, n, noise, length(noise), X, n, d,
                            ptrs, length(warping),
                            kernel, regmodel, normalize ? 1 : 0, optim, objective,
                            C_NULL, C_NULL, 0)
            end
        end
    elseif n_params > 0
        keys_arr = collect(keys(parameters))
        vals_arr = [parameters[k] for k in keys_arr]
        keys_c = [Base.unsafe_convert(Cstring, k) for k in keys_arr]
        vals_c = [Base.unsafe_convert(Cstring, v) for v in vals_arr]
        GC.@preserve warping keys_arr vals_arr begin
            ptr = ccall(dlsym(_lk(), :lk_warp_kriging_new_fit), Ptr{Nothing},
                        (Ptr{Float64}, Cint,
                         Ptr{Float64}, Cint, Cint,
                         Ptr{Cstring}, Cint,
                         Cstring, Cstring, Cint, Cstring, Cstring,
                         Ptr{Cstring}, Ptr{Cstring}, Cint),
                        y, n, X, n, d,
                        ptrs, length(warping),
                        kernel, regmodel, normalize ? 1 : 0, optim, objective,
                        keys_c, vals_c, n_params)
        end
    else
        GC.@preserve warping begin
            ptr = ccall(dlsym(_lk(), :lk_warp_kriging_new_fit), Ptr{Nothing},
                        (Ptr{Float64}, Cint,
                         Ptr{Float64}, Cint, Cint,
                         Ptr{Cstring}, Cint,
                         Cstring, Cstring, Cint, Cstring, Cstring,
                         Ptr{Cstring}, Ptr{Cstring}, Cint),
                        y, n, X, n, d,
                        ptrs, length(warping),
                        kernel, regmodel, normalize ? 1 : 0, optim, objective,
                        C_NULL, C_NULL, 0)
        end
    end
    return WarpKriging(_check_ptr(ptr))
end

function Base.copy(wk::WarpKriging)
    ptr = ccall(dlsym(_lk(), :lk_warp_kriging_copy), Ptr{Nothing}, (Ptr{Nothing},), wk.ptr)
    return WarpKriging(_check_ptr(ptr))
end

function save(wk::WarpKriging, filename::String)
    ret = ccall(dlsym(_lk(), :lk_warp_kriging_save), Cint, (Ptr{Nothing}, Cstring), wk.ptr, filename)
    _check_error(ret)
end

function load_warp_kriging(filename::String)
    ptr = ccall(dlsym(_lk(), :lk_warp_kriging_load), Ptr{Nothing}, (Cstring,), filename)
    return WarpKriging(_check_ptr(ptr))
end

function fit!(wk::WarpKriging, y::Vector{Float64}, X::Matrix{Float64};
              regmodel::String="constant",
              normalize::Bool=false,
              optim::String="BFGS+Adam",
              objective::String="LL",
              parameters::Union{Nothing,Dict{String,String}}=nothing,
              noise::Union{Nothing,Vector{Float64}}=nothing)
    n, d = size(X)
    @assert length(y) == n
    n_params = parameters === nothing ? 0 : length(parameters)
    if noise !== nothing
        if n_params > 0
            keys_arr = collect(keys(parameters))
            vals_arr = [parameters[k] for k in keys_arr]
            keys_c = [Base.unsafe_convert(Cstring, k) for k in keys_arr]
            vals_c = [Base.unsafe_convert(Cstring, v) for v in vals_arr]
            GC.@preserve keys_arr vals_arr begin
                ret = ccall(dlsym(_lk(), :lk_warp_kriging_fit_noise), Cint,
                            (Ptr{Nothing}, Ptr{Float64}, Cint,
                             Ptr{Float64}, Cint,
                             Ptr{Float64}, Cint, Cint,
                             Cstring, Cint, Cstring, Cstring,
                             Ptr{Cstring}, Ptr{Cstring}, Cint),
                            wk.ptr, y, n, noise, length(noise), X, n, d,
                            regmodel, normalize ? 1 : 0, optim, objective,
                            keys_c, vals_c, n_params)
            end
        else
            ret = ccall(dlsym(_lk(), :lk_warp_kriging_fit_noise), Cint,
                        (Ptr{Nothing}, Ptr{Float64}, Cint,
                         Ptr{Float64}, Cint,
                         Ptr{Float64}, Cint, Cint,
                         Cstring, Cint, Cstring, Cstring,
                         Ptr{Cstring}, Ptr{Cstring}, Cint),
                        wk.ptr, y, n, noise, length(noise), X, n, d,
                        regmodel, normalize ? 1 : 0, optim, objective,
                        C_NULL, C_NULL, 0)
        end
    elseif n_params > 0
        keys_arr = collect(keys(parameters))
        vals_arr = [parameters[k] for k in keys_arr]
        keys_c = [Base.unsafe_convert(Cstring, k) for k in keys_arr]
        vals_c = [Base.unsafe_convert(Cstring, v) for v in vals_arr]
        GC.@preserve keys_arr vals_arr begin
            ret = ccall(dlsym(_lk(), :lk_warp_kriging_fit), Cint,
                        (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint,
                         Cstring, Cint, Cstring, Cstring,
                         Ptr{Cstring}, Ptr{Cstring}, Cint),
                        wk.ptr, y, n, X, n, d,
                        regmodel, normalize ? 1 : 0, optim, objective,
                        keys_c, vals_c, n_params)
        end
    else
        ret = ccall(dlsym(_lk(), :lk_warp_kriging_fit), Cint,
                    (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint,
                     Cstring, Cint, Cstring, Cstring,
                     Ptr{Cstring}, Ptr{Cstring}, Cint),
                    wk.ptr, y, n, X, n, d,
                    regmodel, normalize ? 1 : 0, optim, objective,
                    C_NULL, C_NULL, 0)
    end
    _check_error(ret)
    return wk
end

function predict(wk::WarpKriging, X_n::Matrix{Float64};
                 return_stdev::Bool=true,
                 return_cov::Bool=false,
                 return_deriv::Bool=false)
    m, d = size(X_n)
    mean_out = Vector{Float64}(undef, m)
    stdev_out = return_stdev ? Vector{Float64}(undef, m) : Float64[]
    cov_out = return_cov ? Matrix{Float64}(undef, m, m) : Matrix{Float64}(undef, 0, 0)
    mean_deriv_out = return_deriv ? Matrix{Float64}(undef, m, d) : Matrix{Float64}(undef, 0, 0)
    stdev_deriv_out = return_deriv ? Matrix{Float64}(undef, m, d) : Matrix{Float64}(undef, 0, 0)

    ret = ccall(dlsym(_lk(), :lk_warp_kriging_predict), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Cint,
                 Cint, Cint, Cint,
                 Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
                wk.ptr, X_n, m, d,
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

function simulate(wk::WarpKriging, nsim::Int, seed::Int, X_n::Matrix{Float64}; will_update::Bool=false)
    m, d = size(X_n)
    sim_out = Matrix{Float64}(undef, m, nsim)
    ret = ccall(dlsym(_lk(), :lk_warp_kriging_simulate), Cint,
                (Ptr{Nothing}, Cint, Cint, Ptr{Float64}, Cint, Cint, Cint, Ptr{Float64}),
                wk.ptr, nsim, seed, X_n, m, d, will_update ? 1 : 0, sim_out)
    _check_error(ret)
    return sim_out
end

function update_simulate(wk::WarpKriging, y_u::Vector{Float64}, X_u::Matrix{Float64})
    n, d = size(X_u)
    @assert length(y_u) == n
    nsim_out = Ref{Cint}(0)
    m_out = Ref{Cint}(0)
    ret = ccall(dlsym(_lk(), :lk_warp_kriging_update_simulate), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Cint}, Ptr{Cint}),
                wk.ptr, y_u, n, X_u, n, d, C_NULL, nsim_out, m_out)
    _check_error(ret)
    sim = Matrix{Float64}(undef, m_out[], nsim_out[])
    ret = ccall(dlsym(_lk(), :lk_warp_kriging_update_simulate), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Cint}, Ptr{Cint}),
                wk.ptr, y_u, n, X_u, n, d, sim, nsim_out, m_out)
    _check_error(ret)
    return sim
end

function update!(wk::WarpKriging, y_u::Vector{Float64}, X_u::Matrix{Float64}; refit::Bool=true)
    n, d = size(X_u)
    @assert length(y_u) == n
    ret = ccall(dlsym(_lk(), :lk_warp_kriging_update), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint, Cint),
                wk.ptr, y_u, n, X_u, n, d, refit ? 1 : 0)
    _check_error(ret)
    return wk
end

function summary(wk::WarpKriging)
    s = ccall(dlsym(_lk(), :lk_warp_kriging_summary), Cstring, (Ptr{Nothing},), wk.ptr)
    return unsafe_string(s)
end

function log_likelihood(wk::WarpKriging)
    return ccall(dlsym(_lk(), :lk_warp_kriging_log_likelihood), Float64, (Ptr{Nothing},), wk.ptr)
end

function log_likelihood_fun(wk::WarpKriging, theta::Vector{Float64};
                             return_grad::Bool=false,
                             return_hess::Bool=false)
    n = length(theta)
    ll = Ref{Float64}(0.0)
    grad = return_grad ? Vector{Float64}(undef, n) : Float64[]
    hess = return_hess ? Matrix{Float64}(undef, n, n) : Matrix{Float64}(undef, 0, 0)
    ret = ccall(dlsym(_lk(), :lk_warp_kriging_log_likelihood_fun), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Cint, Cint,
                 Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
                wk.ptr, theta, n, return_grad ? 1 : 0, return_hess ? 1 : 0,
                ll, return_grad ? grad : C_NULL, return_hess ? hess : C_NULL)
    _check_error(ret)
    return (ll=ll[],
            grad=return_grad ? grad : nothing,
            hess=return_hess ? hess : nothing)
end

kernel(wk::WarpKriging) = unsafe_string(ccall(dlsym(_lk(), :lk_warp_kriging_kernel), Cstring, (Ptr{Nothing},), wk.ptr))
normalize(wk::WarpKriging) = ccall(dlsym(_lk(), :lk_warp_kriging_get_normalize), Cint, (Ptr{Nothing},), wk.ptr) != 0
regmodel(wk::WarpKriging) = unsafe_string(ccall(dlsym(_lk(), :lk_warp_kriging_get_regmodel), Cstring, (Ptr{Nothing},), wk.ptr))
is_fitted(wk::WarpKriging) = ccall(dlsym(_lk(), :lk_warp_kriging_is_fitted), Cint, (Ptr{Nothing},), wk.ptr) != 0
feature_dim(wk::WarpKriging) = Int(ccall(dlsym(_lk(), :lk_warp_kriging_feature_dim), Cint, (Ptr{Nothing},), wk.ptr))
X(wk::WarpKriging) = _get_mat(:lk_warp_kriging_get_X, wk.ptr)
centerX(wk::WarpKriging) = _get_rowvec(:lk_warp_kriging_get_centerX, wk.ptr)
scaleX(wk::WarpKriging) = _get_rowvec(:lk_warp_kriging_get_scaleX, wk.ptr)
y(wk::WarpKriging) = _get_vec(:lk_warp_kriging_get_y, wk.ptr)
centerY(wk::WarpKriging) = ccall(dlsym(_lk(), :lk_warp_kriging_get_centerY), Float64, (Ptr{Nothing},), wk.ptr)
scaleY(wk::WarpKriging) = ccall(dlsym(_lk(), :lk_warp_kriging_get_scaleY), Float64, (Ptr{Nothing},), wk.ptr)
F(wk::WarpKriging) = _get_mat(:lk_warp_kriging_get_F, wk.ptr)
T(wk::WarpKriging) = _get_mat(:lk_warp_kriging_get_T, wk.ptr)
M(wk::WarpKriging) = _get_mat(:lk_warp_kriging_get_M, wk.ptr)
z(wk::WarpKriging) = _get_vec(:lk_warp_kriging_get_z, wk.ptr)
beta(wk::WarpKriging) = _get_vec(:lk_warp_kriging_get_beta, wk.ptr)
theta(wk::WarpKriging) = _get_vec(:lk_warp_kriging_get_theta, wk.ptr)
sigma2(wk::WarpKriging) = ccall(dlsym(_lk(), :lk_warp_kriging_get_sigma2), Float64, (Ptr{Nothing},), wk.ptr)

# Deprecated aliases for WarpKriging
for (_old, _new) in [(:get_X, :X), (:get_y, :y), (:get_theta, :theta), (:get_sigma2, :sigma2), (:get_warping, :warping)]
    @eval function $(_old)(wk::WarpKriging)
        Base.depwarn("`$($_old)` is deprecated, use `$($_new)` instead", $(_old))
        return $(_new)(wk)
    end
end

function warping(wk::WarpKriging)
    n_ref = Ref{Cint}(0)
    ret = ccall(dlsym(_lk(), :lk_warp_kriging_get_warping), Cint,
                (Ptr{Nothing}, Ptr{Ptr{Cchar}}, Ptr{Cint}),
                wk.ptr, C_NULL, n_ref)
    _check_error(ret)
    n = n_ref[]
    ptrs = Vector{Ptr{Cchar}}(undef, n)
    ret = ccall(dlsym(_lk(), :lk_warp_kriging_get_warping), Cint,
                (Ptr{Nothing}, Ptr{Ptr{Cchar}}, Ptr{Cint}),
                wk.ptr, ptrs, n_ref)
    _check_error(ret)
    return [unsafe_string(p) for p in ptrs]
end

# ─── WarpKriging: String column encoding helpers ─────────────────

"""
    encode_string_columns(X::Matrix, warping::Vector{String})

Detect string columns in a mixed-type matrix, encode them as integers
0..L-1, and rewrite warping specs to include level names.
Returns `(X_num::Matrix{Float64}, warping_out::Vector{String})`.
"""
function encode_string_columns(X::Matrix, warping::Vector{String})
    n, d = size(X)
    X_num = Matrix{Float64}(undef, n, d)
    warping_out = copy(warping)

    for j in 1:d
        col = X[:, j]
        if eltype(col) <: AbstractString || any(x -> x isa AbstractString, col)
            str_vals = String.(col)
            labels = sort(unique(str_vals))
            label_map = Dict(lab => i - 1 for (i, lab) in enumerate(labels))
            X_num[:, j] = Float64[label_map[v] for v in str_vals]

            spec = strip(warping_out[j])
            spec_lower = lowercase(spec)
            names_str = "[" * join(["\"$lab\"" for lab in labels], ",") * "]"

            if startswith(spec_lower, "categorical")
                embed_dim = 2
                m = match(r"\(([^)]*)\)", spec)
                if m !== nothing && !isempty(m.captures[1])
                    parts = strip.(split(m.captures[1], ","))
                    if length(parts) >= 2
                        embed_dim = parse(Int, parts[end])
                    end
                end
                warping_out[j] = "categorical($(names_str),$(embed_dim))"
            elseif startswith(spec_lower, "ordinal")
                warping_out[j] = "ordinal($(names_str))"
            else
                error("Column $j contains strings but warping spec '$spec' is not 'categorical' or 'ordinal'")
            end
        else
            X_num[:, j] = Float64.(col)
        end
    end

    return X_num, warping_out
end

function _has_string_columns(X::Matrix)
    for j in 1:size(X, 2)
        if any(x -> x isa AbstractString, X[:, j])
            return true
        end
    end
    return false
end

# Constructor accepting mixed-type Matrix{Any}
function WarpKriging(y::Vector{Float64}, X::Matrix{Any},
                     warping::Vector{String}, kernel::String="gauss";
                     kwargs...)
    X_num, warping_enc = encode_string_columns(X, warping)
    return WarpKriging(y, X_num, warping_enc, kernel; kwargs...)
end

# fit! accepting mixed-type Matrix{Any}
function fit!(wk::WarpKriging, y::Vector{Float64}, X::Matrix{Any}; kwargs...)
    X_num, _ = encode_string_columns(X, get_warping(wk))
    return fit!(wk, y, X_num; kwargs...)
end

# predict accepting mixed-type Matrix{Any}
function predict(wk::WarpKriging, X_n::Matrix{Any}; kwargs...)
    X_num, _ = encode_string_columns(X_n, get_warping(wk))
    return predict(wk, X_num; kwargs...)
end

# simulate accepting mixed-type Matrix{Any}
function simulate(wk::WarpKriging, nsim::Int, seed::Int, X_n::Matrix{Any})
    X_num, _ = encode_string_columns(X_n, get_warping(wk))
    return simulate(wk, nsim, seed, X_num)
end

# update! accepting mixed-type Matrix{Any}
function update!(wk::WarpKriging, y_u::Vector{Float64}, X_u::Matrix{Any})
    X_num, _ = encode_string_columns(X_u, get_warping(wk))
    return update!(wk, y_u, X_num)
end

# ─── MLPKriging ──────────────────────────────────────────────────

mutable struct MLPKriging
    ptr::Ptr{Nothing}

    function MLPKriging(ptr::Ptr{Nothing})
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall(dlsym(_lk(), :lk_mlp_kriging_delete), Nothing, (Ptr{Nothing},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

function MLPKriging(hidden_dims::Vector{Int}, d_out::Int=2;
                    activation::String="selu", kernel::String="gauss")
    c_dims = Cint.(hidden_dims)
    ptr = ccall(dlsym(_lk(), :lk_mlp_kriging_new), Ptr{Nothing},
                (Ptr{Cint}, Cint, Cint, Cstring, Cstring),
                c_dims, length(c_dims), d_out, activation, kernel)
    return MLPKriging(_check_ptr(ptr))
end

function MLPKriging(y::Vector{Float64}, X::Matrix{Float64},
                    hidden_dims::Vector{Int}, d_out::Int=2;
                    activation::String="selu", kernel::String="gauss",
                    regmodel::String="constant",
                    normalize::Bool=false,
                    optim::String="BFGS+Adam",
                    objective::String="LL",
                    parameters::Union{Nothing,Dict{String,String}}=nothing)
    n, d = size(X)
    @assert length(y) == n
    c_dims = Cint.(hidden_dims)
    n_params = parameters === nothing ? 0 : length(parameters)
    if n_params > 0
        keys_arr = collect(keys(parameters))
        vals_arr = [parameters[k] for k in keys_arr]
        keys_c = [Base.unsafe_convert(Cstring, k) for k in keys_arr]
        vals_c = [Base.unsafe_convert(Cstring, v) for v in vals_arr]
        GC.@preserve keys_arr vals_arr begin
            ptr = ccall(dlsym(_lk(), :lk_mlp_kriging_new_fit), Ptr{Nothing},
                        (Ptr{Float64}, Cint,
                         Ptr{Float64}, Cint, Cint,
                         Ptr{Cint}, Cint, Cint,
                         Cstring, Cstring,
                         Cstring, Cint, Cstring, Cstring,
                         Ptr{Cstring}, Ptr{Cstring}, Cint),
                        y, n, X, n, d,
                        c_dims, length(c_dims), d_out,
                        activation, kernel,
                        regmodel, normalize ? 1 : 0, optim, objective,
                        keys_c, vals_c, n_params)
        end
    else
        ptr = ccall(dlsym(_lk(), :lk_mlp_kriging_new_fit), Ptr{Nothing},
                    (Ptr{Float64}, Cint,
                     Ptr{Float64}, Cint, Cint,
                     Ptr{Cint}, Cint, Cint,
                     Cstring, Cstring,
                     Cstring, Cint, Cstring, Cstring,
                     Ptr{Cstring}, Ptr{Cstring}, Cint),
                    y, n, X, n, d,
                    c_dims, length(c_dims), d_out,
                    activation, kernel,
                    regmodel, normalize ? 1 : 0, optim, objective,
                    C_NULL, C_NULL, 0)
    end
    return MLPKriging(_check_ptr(ptr))
end

function fit!(mk::MLPKriging, y::Vector{Float64}, X::Matrix{Float64};
              regmodel::String="constant",
              normalize::Bool=false,
              optim::String="BFGS+Adam",
              objective::String="LL",
              parameters::Union{Nothing,Dict{String,String}}=nothing)
    n, d = size(X)
    @assert length(y) == n
    n_params = parameters === nothing ? 0 : length(parameters)
    if n_params > 0
        keys_arr = collect(keys(parameters))
        vals_arr = [parameters[k] for k in keys_arr]
        keys_c = [Base.unsafe_convert(Cstring, k) for k in keys_arr]
        vals_c = [Base.unsafe_convert(Cstring, v) for v in vals_arr]
        GC.@preserve keys_arr vals_arr begin
            ret = ccall(dlsym(_lk(), :lk_mlp_kriging_fit), Cint,
                        (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint,
                         Cstring, Cint, Cstring, Cstring,
                         Ptr{Cstring}, Ptr{Cstring}, Cint),
                        mk.ptr, y, n, X, n, d,
                        regmodel, normalize ? 1 : 0, optim, objective,
                        keys_c, vals_c, n_params)
        end
    else
        ret = ccall(dlsym(_lk(), :lk_mlp_kriging_fit), Cint,
                    (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint,
                     Cstring, Cint, Cstring, Cstring,
                     Ptr{Cstring}, Ptr{Cstring}, Cint),
                    mk.ptr, y, n, X, n, d,
                    regmodel, normalize ? 1 : 0, optim, objective,
                    C_NULL, C_NULL, 0)
    end
    _check_error(ret)
    return mk
end

function predict(mk::MLPKriging, X_n::Matrix{Float64};
                 return_stdev::Bool=true,
                 return_cov::Bool=false,
                 return_deriv::Bool=false)
    m, d = size(X_n)
    mean_out = Vector{Float64}(undef, m)
    stdev_out = return_stdev ? Vector{Float64}(undef, m) : Float64[]
    cov_out = return_cov ? Matrix{Float64}(undef, m, m) : Matrix{Float64}(undef, 0, 0)
    mean_deriv_out = return_deriv ? Matrix{Float64}(undef, m, d) : Matrix{Float64}(undef, 0, 0)
    stdev_deriv_out = return_deriv ? Matrix{Float64}(undef, m, d) : Matrix{Float64}(undef, 0, 0)

    ret = ccall(dlsym(_lk(), :lk_mlp_kriging_predict), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Cint,
                 Cint, Cint, Cint,
                 Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
                mk.ptr, X_n, m, d,
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

function simulate(mk::MLPKriging, nsim::Int, seed::Int, X_n::Matrix{Float64}; will_update::Bool=false)
    m, d = size(X_n)
    sim_out = Matrix{Float64}(undef, m, nsim)
    ret = ccall(dlsym(_lk(), :lk_mlp_kriging_simulate), Cint,
                (Ptr{Nothing}, Cint, Cint, Ptr{Float64}, Cint, Cint, Cint, Ptr{Float64}),
                mk.ptr, nsim, seed, X_n, m, d, will_update ? 1 : 0, sim_out)
    _check_error(ret)
    return sim_out
end

function update_simulate(mk::MLPKriging, y_u::Vector{Float64}, X_u::Matrix{Float64})
    n, d = size(X_u)
    @assert length(y_u) == n
    nsim_out = Ref{Cint}(0)
    m_out = Ref{Cint}(0)
    ret = ccall(dlsym(_lk(), :lk_mlp_kriging_update_simulate), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Cint}, Ptr{Cint}),
                mk.ptr, y_u, n, X_u, n, d, C_NULL, nsim_out, m_out)
    _check_error(ret)
    sim = Matrix{Float64}(undef, m_out[], nsim_out[])
    ret = ccall(dlsym(_lk(), :lk_mlp_kriging_update_simulate), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Cint}, Ptr{Cint}),
                mk.ptr, y_u, n, X_u, n, d, sim, nsim_out, m_out)
    _check_error(ret)
    return sim
end

function update!(mk::MLPKriging, y_u::Vector{Float64}, X_u::Matrix{Float64}; refit::Bool=true)
    n, d = size(X_u)
    @assert length(y_u) == n
    ret = ccall(dlsym(_lk(), :lk_mlp_kriging_update), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Cint, Cint),
                mk.ptr, y_u, n, X_u, n, d, refit ? 1 : 0)
    _check_error(ret)
    return mk
end

function summary(mk::MLPKriging)
    s = ccall(dlsym(_lk(), :lk_mlp_kriging_summary), Cstring, (Ptr{Nothing},), mk.ptr)
    return unsafe_string(s)
end

function log_likelihood(mk::MLPKriging)
    return ccall(dlsym(_lk(), :lk_mlp_kriging_log_likelihood), Float64, (Ptr{Nothing},), mk.ptr)
end

function log_likelihood_fun(mk::MLPKriging, theta::Vector{Float64};
                             return_grad::Bool=false,
                             return_hess::Bool=false)
    n = length(theta)
    ll = Ref{Float64}(0.0)
    grad = return_grad ? Vector{Float64}(undef, n) : Float64[]
    hess = return_hess ? Matrix{Float64}(undef, n, n) : Matrix{Float64}(undef, 0, 0)
    ret = ccall(dlsym(_lk(), :lk_mlp_kriging_log_likelihood_fun), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Cint, Cint,
                 Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
                mk.ptr, theta, n, return_grad ? 1 : 0, return_hess ? 1 : 0,
                ll, return_grad ? grad : C_NULL, return_hess ? hess : C_NULL)
    _check_error(ret)
    return (ll=ll[],
            grad=return_grad ? grad : nothing,
            hess=return_hess ? hess : nothing)
end

kernel(mk::MLPKriging) = unsafe_string(ccall(dlsym(_lk(), :lk_mlp_kriging_kernel), Cstring, (Ptr{Nothing},), mk.ptr))
activation(mk::MLPKriging) = unsafe_string(ccall(dlsym(_lk(), :lk_mlp_kriging_activation), Cstring, (Ptr{Nothing},), mk.ptr))
normalize(mk::MLPKriging) = ccall(dlsym(_lk(), :lk_mlp_kriging_get_normalize), Cint, (Ptr{Nothing},), mk.ptr) != 0
regmodel(mk::MLPKriging) = unsafe_string(ccall(dlsym(_lk(), :lk_mlp_kriging_get_regmodel), Cstring, (Ptr{Nothing},), mk.ptr))
is_fitted(mk::MLPKriging) = ccall(dlsym(_lk(), :lk_mlp_kriging_is_fitted), Cint, (Ptr{Nothing},), mk.ptr) != 0
feature_dim(mk::MLPKriging) = Int(ccall(dlsym(_lk(), :lk_mlp_kriging_feature_dim), Cint, (Ptr{Nothing},), mk.ptr))
X(mk::MLPKriging) = _get_mat(:lk_mlp_kriging_get_X, mk.ptr)
centerX(mk::MLPKriging) = _get_rowvec(:lk_mlp_kriging_get_centerX, mk.ptr)
scaleX(mk::MLPKriging) = _get_rowvec(:lk_mlp_kriging_get_scaleX, mk.ptr)
y(mk::MLPKriging) = _get_vec(:lk_mlp_kriging_get_y, mk.ptr)
centerY(mk::MLPKriging) = ccall(dlsym(_lk(), :lk_mlp_kriging_get_centerY), Float64, (Ptr{Nothing},), mk.ptr)
scaleY(mk::MLPKriging) = ccall(dlsym(_lk(), :lk_mlp_kriging_get_scaleY), Float64, (Ptr{Nothing},), mk.ptr)
F(mk::MLPKriging) = _get_mat(:lk_mlp_kriging_get_F, mk.ptr)
T(mk::MLPKriging) = _get_mat(:lk_mlp_kriging_get_T, mk.ptr)
M(mk::MLPKriging) = _get_mat(:lk_mlp_kriging_get_M, mk.ptr)
z(mk::MLPKriging) = _get_vec(:lk_mlp_kriging_get_z, mk.ptr)
beta(mk::MLPKriging) = _get_vec(:lk_mlp_kriging_get_beta, mk.ptr)
theta(mk::MLPKriging) = _get_vec(:lk_mlp_kriging_get_theta, mk.ptr)
sigma2(mk::MLPKriging) = ccall(dlsym(_lk(), :lk_mlp_kriging_get_sigma2), Float64, (Ptr{Nothing},), mk.ptr)

# Deprecated aliases for MLPKriging
for (_old, _new) in [(:get_X, :X), (:get_y, :y), (:get_theta, :theta), (:get_sigma2, :sigma2), (:get_hidden_dims, :hidden_dims)]
    @eval function $(_old)(mk::MLPKriging)
        Base.depwarn("`$($_old)` is deprecated, use `$($_new)` instead", $(_old))
        return $(_new)(mk)
    end
end

function hidden_dims(mk::MLPKriging)
    n_ref = Ref{Cint}(0)
    ret = ccall(dlsym(_lk(), :lk_mlp_kriging_get_hidden_dims), Cint,
                (Ptr{Nothing}, Ptr{Cint}, Ptr{Cint}),
                mk.ptr, C_NULL, n_ref)
    _check_error(ret)
    n = n_ref[]
    out = Vector{Cint}(undef, n)
    ret = ccall(dlsym(_lk(), :lk_mlp_kriging_get_hidden_dims), Cint,
                (Ptr{Nothing}, Ptr{Cint}, Ptr{Cint}),
                mk.ptr, out, n_ref)
    _check_error(ret)
    return Int.(out)
end

function Base.copy(mk::MLPKriging)
    ptr = ccall(dlsym(_lk(), :lk_mlp_kriging_copy), Ptr{Nothing}, (Ptr{Nothing},), mk.ptr)
    return MLPKriging(_check_ptr(ptr))
end

function save(mk::MLPKriging, filename::String)
    ret = ccall(dlsym(_lk(), :lk_mlp_kriging_save), Cint, (Ptr{Nothing}, Cstring), mk.ptr, filename)
    _check_error(ret)
end

function load_mlp_kriging(filename::String)
    ptr = ccall(dlsym(_lk(), :lk_mlp_kriging_load), Ptr{Nothing}, (Cstring,), filename)
    return MLPKriging(_check_ptr(ptr))
end

# ─── Exports ──────────────────────────────────────────────────────

# ─── NestedKriging ────────────────────────────────────────────────

mutable struct NestedKriging
    ptr::Ptr{Nothing}

    function NestedKriging(ptr::Ptr{Nothing})
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL
                ccall(dlsym(_lk(), :lk_nested_kriging_delete), Nothing, (Ptr{Nothing},), o.ptr)
                o.ptr = C_NULL
            end
        end
        return obj
    end
end

"""
    NestedKriging(y, X, kernel, nb_groups; aggregation="NK", partition="kmeans", ...)

Divide-and-conquer Kriging for large designs: the data are partitioned in
`nb_groups` groups, one `Kriging` submodel is fitted per group (all sharing a
common prior after hyperparameter unification), and predictions are aggregated
with `aggregation` in `"NK"` (optimal nested-kriging aggregation, default),
`"PoE"`, `"gPoE"`, `"BCM"` or `"rBCM"`.
"""
function NestedKriging(y::Vector{Float64}, X::Matrix{Float64}, kernel::String, nb_groups::Int;
                       aggregation::String="NK",
                       partition::String="kmeans",
                       seed::Int=123,
                       regmodel::String="constant",
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
    theta_ptr = theta === nothing ? C_NULL : theta
    theta_n = theta === nothing ? 0 : length(theta)
    beta_ptr = beta === nothing ? C_NULL : beta
    beta_n = beta === nothing ? 0 : length(beta)
    ptr = ccall(dlsym(_lk(), :lk_nested_kriging_new_fit), Ptr{Nothing},
                (Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint, Cint,
                 Cstring, Cint, Cstring, Cstring, Cint,
                 Cstring, Cstring, Cstring,
                 Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint, Cint,
                 Ptr{Float64}, Cint, Cint),
                y, n,
                X, n, d,
                kernel, nb_groups, aggregation, partition, seed,
                regmodel, optim, objective,
                sigma2_ptr, is_sigma2_estim ? 1 : 0,
                theta_ptr, theta_n, is_theta_estim ? 1 : 0,
                beta_ptr, beta_n, is_beta_estim ? 1 : 0)
    return NestedKriging(_check_ptr(ptr))
end

function fit!(k::NestedKriging, y::Vector{Float64}, X::Matrix{Float64}, nb_groups::Int;
              regmodel::String="constant",
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
    theta_ptr = theta === nothing ? C_NULL : theta
    theta_n = theta === nothing ? 0 : length(theta)
    beta_ptr = beta === nothing ? C_NULL : beta
    beta_n = beta === nothing ? 0 : length(beta)
    ret = ccall(dlsym(_lk(), :lk_nested_kriging_fit), Cint,
                (Ptr{Nothing},
                 Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint, Cint,
                 Cint, Cstring, Cstring, Cstring,
                 Ptr{Float64}, Cint,
                 Ptr{Float64}, Cint, Cint,
                 Ptr{Float64}, Cint, Cint),
                k.ptr,
                y, n,
                X, n, d,
                nb_groups, regmodel, optim, objective,
                sigma2_ptr, is_sigma2_estim ? 1 : 0,
                theta_ptr, theta_n, is_theta_estim ? 1 : 0,
                beta_ptr, beta_n, is_beta_estim ? 1 : 0)
    _check_error(ret)
    return k
end

function predict(k::NestedKriging, X_n::Matrix{Float64}; return_stdev::Bool=true)
    m, d = size(X_n)
    mean_out = Vector{Float64}(undef, m)
    stdev_out = return_stdev ? Vector{Float64}(undef, m) : Float64[]
    ret = ccall(dlsym(_lk(), :lk_nested_kriging_predict), Cint,
                (Ptr{Nothing}, Ptr{Float64}, Cint, Cint, Cint, Ptr{Float64}, Ptr{Float64}),
                k.ptr, X_n, m, d, return_stdev ? 1 : 0,
                mean_out, return_stdev ? stdev_out : C_NULL)
    _check_error(ret)
    return (mean=mean_out, stdev=return_stdev ? stdev_out : nothing)
end

function summary(k::NestedKriging)
    s = ccall(dlsym(_lk(), :lk_nested_kriging_summary), Cstring, (Ptr{Nothing},), k.ptr)
    s == C_NULL && _check_error(Cint(1))
    return unsafe_string(s)
end

kernel(k::NestedKriging) = unsafe_string(ccall(dlsym(_lk(), :lk_nested_kriging_kernel), Cstring, (Ptr{Nothing},), k.ptr))
aggregation(k::NestedKriging) = unsafe_string(ccall(dlsym(_lk(), :lk_nested_kriging_aggregation), Cstring, (Ptr{Nothing},), k.ptr))
nb_groups(k::NestedKriging) = Int(ccall(dlsym(_lk(), :lk_nested_kriging_nb_groups), Cint, (Ptr{Nothing},), k.ptr))
theta(k::NestedKriging) = _get_vec(:lk_nested_kriging_get_theta, k.ptr)
sigma2(k::NestedKriging) = ccall(dlsym(_lk(), :lk_nested_kriging_get_sigma2), Float64, (Ptr{Nothing},), k.ptr)
beta0(k::NestedKriging) = ccall(dlsym(_lk(), :lk_nested_kriging_get_beta0), Float64, (Ptr{Nothing},), k.ptr)

Base.show(io::IO, k::NestedKriging) = print(io, summary(k))


export Kriging, WarpKriging, MLPKriging, NestedKriging
export nb_groups, aggregation, beta0
export fit!, predict, simulate, update!, update_simulate, save, summary
export load, load_kriging, load_warp_kriging, load_mlp_kriging
export log_likelihood_fun, leave_one_out_fun, log_marg_post_fun
export log_likelihood, leave_one_out, log_marg_post
export leave_one_out_vec, cov_mat
export kernel, optim, objective, normalize, regmodel, noise_model
export X, centerX, scaleX, y, centerY, scaleY
export F, T, M, z, beta, theta, sigma2
export is_beta_estim, is_theta_estim, is_sigma2_estim
export nugget, is_nugget_estim, noise
export is_fitted, feature_dim, warping
export activation, hidden_dims
# Deprecated: get_X, get_y, get_theta, get_sigma2, get_beta, get_nugget, get_noise,
#             get_centerX, get_scaleX, get_centerY, get_scaleY, get_F, get_T, get_M, get_z,
#             get_warping, get_hidden_dims, is_normalize
export get_X, get_centerX, get_scaleX, get_y, get_centerY, get_scaleY
export get_F, get_T, get_M, get_z, get_beta, get_theta, get_sigma2
export get_nugget, get_noise, get_warping, get_hidden_dims, is_normalize

end # module
