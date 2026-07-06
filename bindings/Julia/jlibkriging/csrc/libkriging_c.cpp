#include "libkriging_c.h"

#include <libKriging/Kriging.hpp>
#include <libKriging/MLPKriging.hpp>
#include <libKriging/NestedKriging.hpp>

#include <libKriging/Trend.hpp>
#include <libKriging/WarpKriging.hpp>
#include <libKriging/utils/ExplicitCopySpecifier.hpp>

#include <cstring>
#include <limits>
#include <map>
#include <string>

static thread_local std::string g_last_error;

#define CATCH_RETURN                \
  catch (const std::exception& e) { \
    g_last_error = e.what();        \
    return -1;                      \
  }

#define CATCH_RETURN_NULL           \
  catch (const std::exception& e) { \
    g_last_error = e.what();        \
    return nullptr;                 \
  }

#define CATCH_RETURN_NAN                             \
  catch (const std::exception& e) {                  \
    g_last_error = e.what();                         \
    return std::numeric_limits<double>::quiet_NaN(); \
  }

/* ========================================================================== */
/*  Error handling                                                            */
/* ========================================================================== */

const char* lk_get_last_error(void) {
  return g_last_error.c_str();
}

/* ========================================================================== */
/*  Kriging                                                                   */
/* ========================================================================== */

static Kriging::NoiseModel parse_noise_model(const char* s) {
  if (!s || std::string(s) == "none")
    return Kriging::NoiseModel::None;
  if (std::string(s) == "nugget")
    return Kriging::NoiseModel::Nugget;
  if (std::string(s) == "heterogeneous")
    return Kriging::NoiseModel::Heterogeneous;
  throw std::runtime_error(std::string("Unknown noise_model: ") + s);
}

static const char* noise_model_to_string(Kriging::NoiseModel nm) {
  switch (nm) {
    case Kriging::NoiseModel::None:
      return "none";
    case Kriging::NoiseModel::Nugget:
      return "nugget";
    case Kriging::NoiseModel::Heterogeneous:
      return "heterogeneous";
  }
  return "none";
}

void* lk_kriging_new(const char* kernel, const char* noise_model) {
  try {
    return new Kriging(kernel ? kernel : "matern3_2", parse_noise_model(noise_model));
  }
  CATCH_RETURN_NULL
}

void* lk_kriging_new_fit(const double* y,
                         int n,
                         const double* noise,
                         int noise_n,
                         const double* X,
                         int nX,
                         int d,
                         const char* kernel,
                         const char* noise_model_str,
                         const char* regmodel,
                         int normalize,
                         const char* optim,
                         const char* objective,
                         const double* sigma2,
                         int is_sigma2_estim,
                         const double* theta,
                         int theta_n,
                         int is_theta_estim,
                         const double* beta,
                         int beta_n,
                         int is_beta_estim,
                         const double* nugget,
                         int is_nugget_estim) {
  try {
    auto nm = parse_noise_model(noise_model_str);
    arma::vec y_v(const_cast<double*>(y), n, false, true);
    arma::mat X_m(const_cast<double*>(X), nX, d, false, true);

    Kriging::Parameters params;
    if (sigma2)
      params.sigma2 = *sigma2;
    params.is_sigma2_estim = is_sigma2_estim != 0;
    if (theta && theta_n > 0)
      params.theta = arma::mat(const_cast<double*>(theta), 1, theta_n, false, true);
    params.is_theta_estim = is_theta_estim != 0;
    if (beta && beta_n > 0)
      params.beta = arma::vec(const_cast<double*>(beta), beta_n, false, true);
    params.is_beta_estim = is_beta_estim != 0;
    if (nugget)
      params.nugget = *nugget;
    params.is_nugget_estim = is_nugget_estim != 0;

    auto* kr = new Kriging(kernel ? kernel : "matern3_2", nm);
    if (nm == Kriging::NoiseModel::Heterogeneous && noise && noise_n > 0) {
      arma::vec noise_v(const_cast<double*>(noise), noise_n, false, true);
      kr->fit(y_v,
              noise_v,
              X_m,
              Trend::fromString(regmodel ? regmodel : "constant"),
              normalize != 0,
              optim ? optim : "BFGS",
              objective ? objective : "LL",
              params);
    } else {
      kr->fit(y_v,
              X_m,
              Trend::fromString(regmodel ? regmodel : "constant"),
              normalize != 0,
              optim ? optim : "BFGS",
              objective ? objective : "LL",
              params);
    }
    return kr;
  }
  CATCH_RETURN_NULL
}

void lk_kriging_delete(void* ptr) {
  delete static_cast<Kriging*>(ptr);
}

void* lk_kriging_copy(void* ptr) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    return new Kriging(*k, ExplicitCopySpecifier{});
  }
  CATCH_RETURN_NULL
}

int lk_kriging_fit(void* ptr,
                   const double* y,
                   int n,
                   const double* noise,
                   int noise_n,
                   const double* X,
                   int nX,
                   int d,
                   const char* regmodel,
                   int normalize,
                   const char* optim,
                   const char* objective) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    arma::vec y_v(const_cast<double*>(y), n, false, true);
    arma::mat X_m(const_cast<double*>(X), nX, d, false, true);
    if (noise && noise_n > 0) {
      arma::vec noise_v(const_cast<double*>(noise), noise_n, false, true);
      k->fit(y_v,
             noise_v,
             X_m,
             Trend::fromString(regmodel ? regmodel : "constant"),
             normalize != 0,
             optim ? optim : "BFGS",
             objective ? objective : "LL");
    } else {
      k->fit(y_v,
             X_m,
             Trend::fromString(regmodel ? regmodel : "constant"),
             normalize != 0,
             optim ? optim : "BFGS",
             objective ? objective : "LL");
    }
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_predict(void* ptr,
                       const double* X_n,
                       int m,
                       int d,
                       int return_stdev,
                       int return_cov,
                       int return_deriv,
                       double* mean_out,
                       double* stdev_out,
                       double* cov_out,
                       double* mean_deriv_out,
                       double* stdev_deriv_out) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    arma::mat X_m(const_cast<double*>(X_n), m, d, false, true);
    auto [mean_v, stdev_v, cov_m, mean_deriv_m, stdev_deriv_m]
        = k->predict(X_m, return_stdev != 0, return_cov != 0, return_deriv != 0);
    if (mean_out)
      std::memcpy(mean_out, mean_v.memptr(), mean_v.n_elem * sizeof(double));
    if (stdev_out && return_stdev)
      std::memcpy(stdev_out, stdev_v.memptr(), stdev_v.n_elem * sizeof(double));
    if (cov_out && return_cov)
      std::memcpy(cov_out, cov_m.memptr(), cov_m.n_elem * sizeof(double));
    if (mean_deriv_out && return_deriv)
      std::memcpy(mean_deriv_out, mean_deriv_m.memptr(), mean_deriv_m.n_elem * sizeof(double));
    if (stdev_deriv_out && return_deriv)
      std::memcpy(stdev_deriv_out, stdev_deriv_m.memptr(), stdev_deriv_m.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_simulate(void* ptr,
                        int nsim,
                        int seed,
                        const double* X_n,
                        int m,
                        int d,
                        int with_nugget,
                        const double* with_noise,
                        int noise_n,
                        int will_update,
                        double* sim_out) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    arma::mat X_m(const_cast<double*>(X_n), m, d, false, true);
    arma::mat sim;
    if (with_noise && noise_n > 0) {
      arma::vec noise_v(const_cast<double*>(with_noise), noise_n, false, true);
      sim = k->simulate(nsim, seed, X_m, noise_v, will_update != 0);
    } else if (k->noise_model() == Kriging::NoiseModel::Nugget) {
      sim = k->simulate(nsim, seed, X_m, with_nugget != 0, will_update != 0);
    } else {
      sim = k->simulate(nsim, seed, X_m, will_update != 0);
    }
    if (sim_out)
      std::memcpy(sim_out, sim.memptr(), sim.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_update(void* ptr,
                      const double* y_u,
                      int n,
                      const double* noise_u,
                      int noise_n,
                      const double* X_u,
                      int nX,
                      int d,
                      int refit) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    arma::vec y_v(const_cast<double*>(y_u), n, false, true);
    arma::mat X_m(const_cast<double*>(X_u), nX, d, false, true);
    if (noise_u && noise_n > 0) {
      arma::vec noise_v(const_cast<double*>(noise_u), noise_n, false, true);
      k->update(y_v, noise_v, X_m, refit != 0);
    } else {
      k->update(y_v, X_m, refit != 0);
    }
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_update_simulate(void* ptr,
                               const double* y_u,
                               int n,
                               const double* noise_u,
                               int noise_n,
                               const double* X_u,
                               int nX,
                               int d,
                               double* sim_out,
                               int* nsim_out,
                               int* m_out) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    arma::vec y_v(const_cast<double*>(y_u), n, false, true);
    arma::mat X_m(const_cast<double*>(X_u), nX, d, false, true);
    arma::mat sim;
    if (noise_u && noise_n > 0) {
      arma::vec noise_v(const_cast<double*>(noise_u), noise_n, false, true);
      sim = k->update_simulate(y_v, noise_v, X_m);
    } else {
      sim = k->update_simulate(y_v, X_m);
    }
    if (nsim_out)
      *nsim_out = static_cast<int>(sim.n_cols);
    if (m_out)
      *m_out = static_cast<int>(sim.n_rows);
    if (sim_out)
      std::memcpy(sim_out, sim.memptr(), sim.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_save(void* ptr, const char* filename) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    k->save(filename);
    return 0;
  }
  CATCH_RETURN
}

void* lk_kriging_load(const char* filename) {
  try {
    return new Kriging(Kriging::load(filename));
  }
  CATCH_RETURN_NULL
}

const char* lk_kriging_summary(void* ptr) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    static thread_local std::string buf;
    buf = k->summary();
    return buf.c_str();
  }
  CATCH_RETURN_NULL
}

int lk_kriging_log_likelihood_fun(void* ptr,
                                  const double* theta,
                                  int theta_n,
                                  int return_grad,
                                  int return_hess,
                                  double* ll_out,
                                  double* grad_out,
                                  double* hess_out) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    arma::vec theta_v(const_cast<double*>(theta), theta_n, false, true);
    auto [ll, grad] = k->logLikelihoodFun(theta_v, return_grad != 0, false);
    if (ll_out)
      *ll_out = ll;
    if (grad_out && return_grad)
      std::memcpy(grad_out, grad.memptr(), grad.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_leave_one_out_fun(void* ptr,
                                 const double* theta,
                                 int theta_n,
                                 int return_grad,
                                 double* loo_out,
                                 double* grad_out) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    arma::vec theta_v(const_cast<double*>(theta), theta_n, false, true);
    auto [loo, grad] = k->leaveOneOutFun(theta_v, return_grad != 0, false);
    if (loo_out)
      *loo_out = loo;
    if (grad_out && return_grad)
      std::memcpy(grad_out, grad.memptr(), grad.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_log_marg_post_fun(void* ptr,
                                 const double* theta,
                                 int theta_n,
                                 int return_grad,
                                 double* lmp_out,
                                 double* grad_out) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    arma::vec theta_v(const_cast<double*>(theta), theta_n, false, true);
    auto [lmp, grad] = k->logMargPostFun(theta_v, return_grad != 0, false);
    if (lmp_out)
      *lmp_out = lmp;
    if (grad_out && return_grad)
      std::memcpy(grad_out, grad.memptr(), grad.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

double lk_kriging_log_likelihood(void* ptr) {
  try {
    return static_cast<Kriging*>(ptr)->logLikelihood();
  }
  CATCH_RETURN_NAN
}

double lk_kriging_leave_one_out(void* ptr) {
  try {
    return static_cast<Kriging*>(ptr)->leaveOneOut();
  }
  CATCH_RETURN_NAN
}

double lk_kriging_log_marg_post(void* ptr) {
  try {
    return static_cast<Kriging*>(ptr)->logMargPost();
  }
  CATCH_RETURN_NAN
}

int lk_kriging_leave_one_out_vec(void* ptr, const double* theta, int theta_n, double* yhat_out, double* stderr_out) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    arma::vec theta_v(const_cast<double*>(theta), theta_n, false, true);
    auto [yhat, stderr_v] = k->leaveOneOutVec(theta_v);
    if (yhat_out)
      std::memcpy(yhat_out, yhat.memptr(), yhat.n_elem * sizeof(double));
    if (stderr_out)
      std::memcpy(stderr_out, stderr_v.memptr(), stderr_v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_cov_mat(void* ptr, const double* X1, int n1, int d1, const double* X2, int n2, int d2, double* cov_out) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    arma::mat X1_m(const_cast<double*>(X1), n1, d1, false, true);
    arma::mat X2_m(const_cast<double*>(X2), n2, d2, false, true);
    arma::mat cov = k->covMat(X1_m, X2_m);
    if (cov_out)
      std::memcpy(cov_out, cov.memptr(), cov.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

/* Kriging string getters */

const char* lk_kriging_kernel(void* ptr) {
  try {
    return static_cast<Kriging*>(ptr)->kernel().c_str();
  }
  CATCH_RETURN_NULL
}

const char* lk_kriging_optim(void* ptr) {
  try {
    return static_cast<Kriging*>(ptr)->optim().c_str();
  }
  CATCH_RETURN_NULL
}

const char* lk_kriging_objective(void* ptr) {
  try {
    return static_cast<Kriging*>(ptr)->objective().c_str();
  }
  CATCH_RETURN_NULL
}

int lk_kriging_is_normalize(void* ptr) {
  try {
    return static_cast<Kriging*>(ptr)->normalize() ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

const char* lk_kriging_regmodel(void* ptr) {
  try {
    static thread_local std::string buf;
    buf = Trend::toString(static_cast<Kriging*>(ptr)->regmodel());
    return buf.c_str();
  }
  CATCH_RETURN_NULL
}

/* Kriging array getters */

int lk_kriging_get_X(void* ptr, double* out, int* n, int* d) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    const arma::mat& v = k->X();
    if (n)
      *n = static_cast<int>(v.n_rows);
    if (d)
      *d = static_cast<int>(v.n_cols);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_get_centerX(void* ptr, double* out, int* d) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    const arma::rowvec& v = k->centerX();
    if (d)
      *d = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_get_scaleX(void* ptr, double* out, int* d) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    const arma::rowvec& v = k->scaleX();
    if (d)
      *d = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_get_y(void* ptr, double* out, int* n) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    const arma::vec& v = k->y();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

double lk_kriging_get_centerY(void* ptr) {
  try {
    return static_cast<Kriging*>(ptr)->centerY();
  }
  CATCH_RETURN_NAN
}

double lk_kriging_get_scaleY(void* ptr) {
  try {
    return static_cast<Kriging*>(ptr)->scaleY();
  }
  CATCH_RETURN_NAN
}

int lk_kriging_get_F(void* ptr, double* out, int* n, int* d) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    const arma::mat& v = k->F();
    if (n)
      *n = static_cast<int>(v.n_rows);
    if (d)
      *d = static_cast<int>(v.n_cols);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_get_T(void* ptr, double* out, int* n, int* d) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    const arma::mat& v = k->T();
    if (n)
      *n = static_cast<int>(v.n_rows);
    if (d)
      *d = static_cast<int>(v.n_cols);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_get_M(void* ptr, double* out, int* n, int* d) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    const arma::mat& v = k->M();
    if (n)
      *n = static_cast<int>(v.n_rows);
    if (d)
      *d = static_cast<int>(v.n_cols);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_get_z(void* ptr, double* out, int* n) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    const arma::vec& v = k->z();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_get_beta(void* ptr, double* out, int* n) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    const arma::vec& v = k->beta();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_is_beta_estim(void* ptr) {
  try {
    return static_cast<Kriging*>(ptr)->is_beta_estim() ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

int lk_kriging_get_theta(void* ptr, double* out, int* n) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    const arma::vec& v = k->theta();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_is_theta_estim(void* ptr) {
  try {
    return static_cast<Kriging*>(ptr)->is_theta_estim() ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

double lk_kriging_get_sigma2(void* ptr) {
  try {
    return static_cast<Kriging*>(ptr)->sigma2();
  }
  CATCH_RETURN_NAN
}

int lk_kriging_is_sigma2_estim(void* ptr) {
  try {
    return static_cast<Kriging*>(ptr)->is_sigma2_estim() ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

const char* lk_kriging_noise_model(void* ptr) {
  try {
    return noise_model_to_string(static_cast<Kriging*>(ptr)->noise_model());
  }
  CATCH_RETURN_NULL
}

double lk_kriging_get_nugget(void* ptr) {
  try {
    return static_cast<Kriging*>(ptr)->nugget();
  }
  CATCH_RETURN_NAN
}

int lk_kriging_is_nugget_estim(void* ptr) {
  try {
    return static_cast<Kriging*>(ptr)->is_nugget_estim() ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

int lk_kriging_get_noise(void* ptr, double* out, int* n) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    const arma::vec& v = k->noise();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

/* ========================================================================== */
/*  WarpKriging                                                               */
/* ========================================================================== */

using libKriging::WarpKriging;

static std::vector<std::string> to_string_vec(const char** arr, int n) {
  std::vector<std::string> result(n);
  for (int i = 0; i < n; ++i) {
    result[i] = arr[i];
  }
  return result;
}

static std::map<std::string, std::string> to_param_map(const char** keys, const char** vals, int n) {
  std::map<std::string, std::string> m;
  if (keys && vals) {
    for (int i = 0; i < n; ++i)
      m[keys[i]] = vals[i];
  }
  return m;
}

// Per-object storage for strings returned to the caller
static thread_local std::string g_wk_summary;
static thread_local std::string g_wk_kernel;
static thread_local std::vector<std::string> g_wk_warping;
static thread_local std::vector<char*> g_wk_warping_ptrs;

void* lk_warp_kriging_new(const char** warping, int n_warping, const char* kernel) {
  try {
    return new WarpKriging(to_string_vec(warping, n_warping), kernel);
  }
  CATCH_RETURN_NULL
}

void* lk_warp_kriging_new_fit(const double* y,
                              int n,
                              const double* X,
                              int nX,
                              int d,
                              const char** warping,
                              int n_warping,
                              const char* kernel,
                              const char* regmodel,
                              int normalize,
                              const char* optim,
                              const char* objective,
                              const char** param_keys,
                              const char** param_vals,
                              int n_params) {
  try {
    arma::vec y_vec(const_cast<double*>(y), n, false, true);
    arma::mat X_mat(const_cast<double*>(X), nX, d, false, true);
    return new WarpKriging(y_vec,
                           X_mat,
                           to_string_vec(warping, n_warping),
                           kernel,
                           Trend::fromString(regmodel ? regmodel : "constant"),
                           normalize != 0,
                           optim,
                           objective,
                           to_param_map(param_keys, param_vals, n_params));
  }
  CATCH_RETURN_NULL
}

void* lk_warp_kriging_new_fit_noise(const double* y,
                                    int n,
                                    const double* noise,
                                    int n_noise,
                                    const double* X,
                                    int nX,
                                    int d,
                                    const char** warping,
                                    int n_warping,
                                    const char* kernel,
                                    const char* regmodel,
                                    int normalize,
                                    const char* optim,
                                    const char* objective,
                                    const char** param_keys,
                                    const char** param_vals,
                                    int n_params) {
  try {
    arma::vec y_vec(const_cast<double*>(y), n, false, true);
    arma::mat X_mat(const_cast<double*>(X), nX, d, false, true);
    auto* wk = new WarpKriging(to_string_vec(warping, n_warping), kernel);
    WarpKriging::Parameters wparams;
    wparams.noise = arma::vec(const_cast<double*>(noise), n_noise, false, true);
    wk->fit(y_vec, X_mat, Trend::fromString(regmodel ? regmodel : "constant"), normalize != 0, optim, objective, wparams);
    return wk;
  }
  CATCH_RETURN_NULL
}

int lk_warp_kriging_fit_noise(void* ptr,
                               const double* y,
                               int n,
                               const double* noise,
                               int n_noise,
                               const double* X,
                               int nX,
                               int d,
                               const char* regmodel,
                               int normalize,
                               const char* optim,
                               const char* objective,
                               const char** param_keys,
                               const char** param_vals,
                               int n_params) {
  try {
    arma::vec y_vec(const_cast<double*>(y), n, false, true);
    arma::mat X_mat(const_cast<double*>(X), nX, d, false, true);
    WarpKriging::Parameters wparams;
    wparams.noise = arma::vec(const_cast<double*>(noise), n_noise, false, true);
    static_cast<WarpKriging*>(ptr)->fit(y_vec, X_mat, Trend::fromString(regmodel ? regmodel : "constant"), normalize != 0, optim, objective, wparams);
    return 0;
  }
  CATCH_RETURN
}

void lk_warp_kriging_delete(void* ptr) {
  delete static_cast<WarpKriging*>(ptr);
}

void* lk_warp_kriging_copy(void* ptr) {
  try {
    auto* wk = static_cast<WarpKriging*>(ptr);
    auto* clone = new WarpKriging(wk->warping_strings(), wk->kernel());
    if (wk->is_fitted()) {
      clone->fit(wk->y(), wk->X());
    }
    return clone;
  }
  CATCH_RETURN_NULL
}

int lk_warp_kriging_fit(void* ptr,
                        const double* y,
                        int n,
                        const double* X,
                        int nX,
                        int d,
                        const char* regmodel,
                        int normalize,
                        const char* optim,
                        const char* objective,
                        const char** param_keys,
                        const char** param_vals,
                        int n_params) {
  try {
    arma::vec y_vec(const_cast<double*>(y), n, false, true);
    arma::mat X_mat(const_cast<double*>(X), nX, d, false, true);
    static_cast<WarpKriging*>(ptr)->fit(y_vec,
                                        X_mat,
                                        Trend::fromString(regmodel ? regmodel : "constant"),
                                        normalize != 0,
                                        optim,
                                        objective,
                                        to_param_map(param_keys, param_vals, n_params));
    return 0;
  }
  CATCH_RETURN
}

int lk_warp_kriging_predict(void* ptr,
                            const double* X_n,
                            int m,
                            int d,
                            int return_stdev,
                            int return_cov,
                            int return_deriv,
                            double* mean_out,
                            double* stdev_out,
                            double* cov_out,
                            double* mean_deriv_out,
                            double* stdev_deriv_out) {
  try {
    arma::mat X_mat(const_cast<double*>(X_n), m, d, false, true);
    auto [mean, stdev, cov, mean_deriv, stdev_deriv]
        = static_cast<WarpKriging*>(ptr)->predict(X_mat, return_stdev != 0, return_cov != 0, return_deriv != 0);
    if (mean_out)
      std::memcpy(mean_out, mean.memptr(), mean.n_elem * sizeof(double));
    if (return_stdev && stdev_out)
      std::memcpy(stdev_out, stdev.memptr(), stdev.n_elem * sizeof(double));
    if (return_cov && cov_out)
      std::memcpy(cov_out, cov.memptr(), cov.n_elem * sizeof(double));
    if (return_deriv && mean_deriv_out)
      std::memcpy(mean_deriv_out, mean_deriv.memptr(), mean_deriv.n_elem * sizeof(double));
    if (return_deriv && stdev_deriv_out)
      std::memcpy(stdev_deriv_out, stdev_deriv.memptr(), stdev_deriv.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_warp_kriging_simulate(void* ptr, int nsim, int seed, const double* X_n, int m, int d, int will_update, double* sim_out) {
  try {
    arma::mat X_mat(const_cast<double*>(X_n), m, d, false, true);
    auto result = static_cast<WarpKriging*>(ptr)->simulate(nsim, seed, X_mat, will_update != 0);
    if (sim_out)
      std::memcpy(sim_out, result.memptr(), result.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_warp_kriging_update(void* ptr, const double* y_u, int n, const double* X_u, int nX, int d, int refit) {
  try {
    arma::vec y_vec(const_cast<double*>(y_u), n, false, true);
    arma::mat X_mat(const_cast<double*>(X_u), nX, d, false, true);
    static_cast<WarpKriging*>(ptr)->update(y_vec, X_mat, refit != 0);
    return 0;
  }
  CATCH_RETURN
}

int lk_warp_kriging_update_simulate(void* ptr,
                                    const double* y_u,
                                    int n,
                                    const double* X_u,
                                    int nX,
                                    int d,
                                    double* sim_out,
                                    int* nsim_out,
                                    int* m_out) {
  try {
    auto* wk = static_cast<WarpKriging*>(ptr);
    arma::vec y_vec(const_cast<double*>(y_u), n, false, true);
    arma::mat X_mat(const_cast<double*>(X_u), nX, d, false, true);
    arma::mat sim = wk->update_simulate(y_vec, X_mat);
    if (nsim_out)
      *nsim_out = static_cast<int>(sim.n_cols);
    if (m_out)
      *m_out = static_cast<int>(sim.n_rows);
    if (sim_out)
      std::memcpy(sim_out, sim.memptr(), sim.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

const char* lk_warp_kriging_summary(void* ptr) {
  try {
    g_wk_summary = static_cast<WarpKriging*>(ptr)->summary();
    return g_wk_summary.c_str();
  } catch (...) {
    return "";
  }
}

double lk_warp_kriging_log_likelihood(void* ptr) {
  try {
    return static_cast<WarpKriging*>(ptr)->logLikelihood();
  }
  CATCH_RETURN_NAN
}

int lk_warp_kriging_log_likelihood_fun(void* ptr,
                                       const double* theta,
                                       int theta_n,
                                       int return_grad,
                                       int return_hess,
                                       double* ll_out,
                                       double* grad_out,
                                       double* hess_out) {
  try {
    arma::vec theta_vec(const_cast<double*>(theta), theta_n, false, true);
    auto [ll, grad, hess]
        = static_cast<WarpKriging*>(ptr)->logLikelihoodFun(theta_vec, return_grad != 0, return_hess != 0);
    if (ll_out)
      *ll_out = ll;
    if (return_grad && grad_out)
      std::memcpy(grad_out, grad.memptr(), grad.n_elem * sizeof(double));
    if (return_hess && hess_out)
      std::memcpy(hess_out, hess.memptr(), hess.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

const char* lk_warp_kriging_kernel(void* ptr) {
  try {
    g_wk_kernel = static_cast<WarpKriging*>(ptr)->kernel();
    return g_wk_kernel.c_str();
  } catch (...) {
    return "";
  }
}

int lk_warp_kriging_get_normalize(void* ptr) {
  try {
    return static_cast<WarpKriging*>(ptr)->normalize() ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

const char* lk_warp_kriging_get_regmodel(void* ptr) {
  try {
    static thread_local std::string buf;
    buf = Trend::toString(static_cast<WarpKriging*>(ptr)->regmodel());
    return buf.c_str();
  }
  CATCH_RETURN_NULL
}

int lk_warp_kriging_is_fitted(void* ptr) {
  try {
    return static_cast<WarpKriging*>(ptr)->is_fitted() ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

int lk_warp_kriging_feature_dim(void* ptr) {
  try {
    return static_cast<int>(static_cast<WarpKriging*>(ptr)->feature_dim());
  } catch (...) {
    return -1;
  }
}

int lk_warp_kriging_get_X(void* ptr, double* out, int* n, int* d) {
  try {
    const arma::mat& X = static_cast<WarpKriging*>(ptr)->X();
    if (n)
      *n = static_cast<int>(X.n_rows);
    if (d)
      *d = static_cast<int>(X.n_cols);
    if (out)
      std::memcpy(out, X.memptr(), X.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_warp_kriging_get_centerX(void* ptr, double* out, int* n) {
  try {
    const arma::rowvec& v = static_cast<WarpKriging*>(ptr)->centerX();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_warp_kriging_get_scaleX(void* ptr, double* out, int* n) {
  try {
    const arma::rowvec& v = static_cast<WarpKriging*>(ptr)->scaleX();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_warp_kriging_get_y(void* ptr, double* out, int* n) {
  try {
    const arma::vec& v = static_cast<WarpKriging*>(ptr)->y();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

double lk_warp_kriging_get_centerY(void* ptr) {
  try {
    return static_cast<WarpKriging*>(ptr)->centerY();
  }
  CATCH_RETURN_NAN
}

double lk_warp_kriging_get_scaleY(void* ptr) {
  try {
    return static_cast<WarpKriging*>(ptr)->scaleY();
  }
  CATCH_RETURN_NAN
}

int lk_warp_kriging_get_F(void* ptr, double* out, int* n, int* d) {
  try {
    const arma::mat& v = static_cast<WarpKriging*>(ptr)->F();
    if (n)
      *n = static_cast<int>(v.n_rows);
    if (d)
      *d = static_cast<int>(v.n_cols);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_warp_kriging_get_T(void* ptr, double* out, int* n, int* d) {
  try {
    const arma::mat& v = static_cast<WarpKriging*>(ptr)->T();
    if (n)
      *n = static_cast<int>(v.n_rows);
    if (d)
      *d = static_cast<int>(v.n_cols);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_warp_kriging_get_M(void* ptr, double* out, int* n, int* d) {
  try {
    const arma::mat& v = static_cast<WarpKriging*>(ptr)->M();
    if (n)
      *n = static_cast<int>(v.n_rows);
    if (d)
      *d = static_cast<int>(v.n_cols);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_warp_kriging_get_z(void* ptr, double* out, int* n) {
  try {
    const arma::vec& v = static_cast<WarpKriging*>(ptr)->z();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_warp_kriging_get_beta(void* ptr, double* out, int* n) {
  try {
    const arma::vec& v = static_cast<WarpKriging*>(ptr)->beta();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_warp_kriging_get_theta(void* ptr, double* out, int* n) {
  try {
    arma::vec v = static_cast<WarpKriging*>(ptr)->theta();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

double lk_warp_kriging_get_sigma2(void* ptr) {
  try {
    return static_cast<WarpKriging*>(ptr)->sigma2();
  }
  CATCH_RETURN_NAN
}

int lk_warp_kriging_get_warping(void* ptr, char** out, int* n_warping) {
  try {
    g_wk_warping = static_cast<WarpKriging*>(ptr)->warping_strings();
    if (n_warping)
      *n_warping = static_cast<int>(g_wk_warping.size());
    if (out) {
      g_wk_warping_ptrs.resize(g_wk_warping.size());
      for (size_t i = 0; i < g_wk_warping.size(); ++i) {
        g_wk_warping_ptrs[i] = const_cast<char*>(g_wk_warping[i].c_str());
      }
      std::memcpy(out, g_wk_warping_ptrs.data(), g_wk_warping.size() * sizeof(char*));
    }
    return 0;
  }
  CATCH_RETURN
}

/* ========================================================================== */
/*  MLPKriging                                                                */
/* ========================================================================== */

using MLPKriging = libKriging::MLPKriging;

static thread_local std::string g_mk_summary;
static thread_local std::string g_mk_kernel;
static thread_local std::string g_mk_activation;

static std::vector<arma::uword> to_uword_vec(const int* arr, int n) {
  std::vector<arma::uword> v(n);
  for (int i = 0; i < n; ++i)
    v[i] = static_cast<arma::uword>(arr[i]);
  return v;
}

void* lk_mlp_kriging_new(const int* hidden_dims, int n_hidden, int d_out, const char* activation, const char* kernel) {
  try {
    return new MLPKriging(to_uword_vec(hidden_dims, n_hidden), static_cast<arma::uword>(d_out), activation, kernel);
  }
  CATCH_RETURN_NULL
}

void* lk_mlp_kriging_new_fit(const double* y,
                             int n,
                             const double* X,
                             int nX,
                             int d,
                             const int* hidden_dims,
                             int n_hidden,
                             int d_out,
                             const char* activation,
                             const char* kernel,
                             const char* regmodel,
                             int normalize,
                             const char* optim,
                             const char* objective,
                             const char** param_keys,
                             const char** param_vals,
                             int n_params) {
  try {
    arma::vec y_vec(const_cast<double*>(y), n, false, true);
    arma::mat X_mat(const_cast<double*>(X), nX, d, false, true);
    return new MLPKriging(y_vec,
                          X_mat,
                          to_uword_vec(hidden_dims, n_hidden),
                          static_cast<arma::uword>(d_out),
                          activation,
                          kernel,
                          Trend::fromString(regmodel ? regmodel : "constant"),
                          normalize != 0,
                          optim,
                          objective,
                          to_param_map(param_keys, param_vals, n_params));
  }
  CATCH_RETURN_NULL
}

void lk_mlp_kriging_delete(void* ptr) {
  delete static_cast<MLPKriging*>(ptr);
}

void* lk_mlp_kriging_copy(void* ptr) {
  try {
    auto* mk = static_cast<MLPKriging*>(ptr);
    auto* clone = new MLPKriging(mk->hidden_dims(), mk->d_out(), mk->activation(), mk->kernel());
    if (mk->is_fitted()) {
      clone->fit(mk->y(), mk->X());
    }
    return clone;
  }
  CATCH_RETURN_NULL
}

int lk_mlp_kriging_fit(void* ptr,
                       const double* y,
                       int n,
                       const double* X,
                       int nX,
                       int d,
                       const char* regmodel,
                       int normalize,
                       const char* optim,
                       const char* objective,
                       const char** param_keys,
                       const char** param_vals,
                       int n_params) {
  try {
    arma::vec y_vec(const_cast<double*>(y), n, false, true);
    arma::mat X_mat(const_cast<double*>(X), nX, d, false, true);
    static_cast<MLPKriging*>(ptr)->fit(y_vec,
                                       X_mat,
                                       Trend::fromString(regmodel ? regmodel : "constant"),
                                       normalize != 0,
                                       optim,
                                       objective,
                                       to_param_map(param_keys, param_vals, n_params));
    return 0;
  }
  CATCH_RETURN
}

int lk_mlp_kriging_predict(void* ptr,
                           const double* X_n,
                           int m,
                           int d,
                           int return_stdev,
                           int return_cov,
                           int return_deriv,
                           double* mean_out,
                           double* stdev_out,
                           double* cov_out,
                           double* mean_deriv_out,
                           double* stdev_deriv_out) {
  try {
    arma::mat X_mat(const_cast<double*>(X_n), m, d, false, true);
    auto [mean, stdev, cov, mean_deriv, stdev_deriv]
        = static_cast<MLPKriging*>(ptr)->predict(X_mat, return_stdev != 0, return_cov != 0, return_deriv != 0);
    if (mean_out)
      std::memcpy(mean_out, mean.memptr(), mean.n_elem * sizeof(double));
    if (return_stdev && stdev_out)
      std::memcpy(stdev_out, stdev.memptr(), stdev.n_elem * sizeof(double));
    if (return_cov && cov_out)
      std::memcpy(cov_out, cov.memptr(), cov.n_elem * sizeof(double));
    if (return_deriv && mean_deriv_out)
      std::memcpy(mean_deriv_out, mean_deriv.memptr(), mean_deriv.n_elem * sizeof(double));
    if (return_deriv && stdev_deriv_out)
      std::memcpy(stdev_deriv_out, stdev_deriv.memptr(), stdev_deriv.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_mlp_kriging_simulate(void* ptr, int nsim, int seed, const double* X_n, int m, int d, int will_update, double* sim_out) {
  try {
    arma::mat X_mat(const_cast<double*>(X_n), m, d, false, true);
    auto result = static_cast<MLPKriging*>(ptr)->simulate(nsim, seed, X_mat, will_update != 0);
    if (sim_out)
      std::memcpy(sim_out, result.memptr(), result.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_mlp_kriging_update(void* ptr, const double* y_u, int n, const double* X_u, int nX, int d, int refit) {
  try {
    arma::vec y_vec(const_cast<double*>(y_u), n, false, true);
    arma::mat X_mat(const_cast<double*>(X_u), nX, d, false, true);
    static_cast<MLPKriging*>(ptr)->update(y_vec, X_mat, refit != 0);
    return 0;
  }
  CATCH_RETURN
}

int lk_mlp_kriging_update_simulate(void* ptr,
                                   const double* y_u,
                                   int n,
                                   const double* X_u,
                                   int nX,
                                   int d,
                                   double* sim_out,
                                   int* nsim_out,
                                   int* m_out) {
  try {
    auto* mk = static_cast<MLPKriging*>(ptr);
    arma::vec y_vec(const_cast<double*>(y_u), n, false, true);
    arma::mat X_mat(const_cast<double*>(X_u), nX, d, false, true);
    arma::mat sim = mk->update_simulate(y_vec, X_mat);
    if (nsim_out)
      *nsim_out = static_cast<int>(sim.n_cols);
    if (m_out)
      *m_out = static_cast<int>(sim.n_rows);
    if (sim_out)
      std::memcpy(sim_out, sim.memptr(), sim.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

const char* lk_mlp_kriging_summary(void* ptr) {
  try {
    g_mk_summary = static_cast<MLPKriging*>(ptr)->summary();
    return g_mk_summary.c_str();
  } catch (...) {
    return "";
  }
}

double lk_mlp_kriging_log_likelihood(void* ptr) {
  try {
    return static_cast<MLPKriging*>(ptr)->logLikelihood();
  }
  CATCH_RETURN_NAN
}

int lk_mlp_kriging_log_likelihood_fun(void* ptr,
                                      const double* theta,
                                      int theta_n,
                                      int return_grad,
                                      int return_hess,
                                      double* ll_out,
                                      double* grad_out,
                                      double* hess_out) {
  try {
    arma::vec theta_vec(const_cast<double*>(theta), theta_n, false, true);
    auto [ll, grad, hess]
        = static_cast<MLPKriging*>(ptr)->logLikelihoodFun(theta_vec, return_grad != 0, return_hess != 0);
    if (ll_out)
      *ll_out = ll;
    if (return_grad && grad_out)
      std::memcpy(grad_out, grad.memptr(), grad.n_elem * sizeof(double));
    if (return_hess && hess_out)
      std::memcpy(hess_out, hess.memptr(), hess.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

const char* lk_mlp_kriging_kernel(void* ptr) {
  try {
    g_mk_kernel = static_cast<MLPKriging*>(ptr)->kernel();
    return g_mk_kernel.c_str();
  } catch (...) {
    return "";
  }
}

const char* lk_mlp_kriging_activation(void* ptr) {
  try {
    g_mk_activation = static_cast<MLPKriging*>(ptr)->activation();
    return g_mk_activation.c_str();
  } catch (...) {
    return "";
  }
}

int lk_mlp_kriging_get_normalize(void* ptr) {
  try {
    return static_cast<MLPKriging*>(ptr)->normalize() ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

const char* lk_mlp_kriging_get_regmodel(void* ptr) {
  try {
    static thread_local std::string buf;
    buf = Trend::toString(static_cast<MLPKriging*>(ptr)->regmodel());
    return buf.c_str();
  }
  CATCH_RETURN_NULL
}

int lk_mlp_kriging_is_fitted(void* ptr) {
  try {
    return static_cast<MLPKriging*>(ptr)->is_fitted() ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

int lk_mlp_kriging_feature_dim(void* ptr) {
  try {
    return static_cast<int>(static_cast<MLPKriging*>(ptr)->feature_dim());
  } catch (...) {
    return -1;
  }
}

int lk_mlp_kriging_get_X(void* ptr, double* out, int* n, int* d) {
  try {
    const arma::mat& X = static_cast<MLPKriging*>(ptr)->X();
    if (n)
      *n = static_cast<int>(X.n_rows);
    if (d)
      *d = static_cast<int>(X.n_cols);
    if (out)
      std::memcpy(out, X.memptr(), X.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_mlp_kriging_get_centerX(void* ptr, double* out, int* n) {
  try {
    const arma::rowvec& v = static_cast<MLPKriging*>(ptr)->centerX();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_mlp_kriging_get_scaleX(void* ptr, double* out, int* n) {
  try {
    const arma::rowvec& v = static_cast<MLPKriging*>(ptr)->scaleX();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_mlp_kriging_get_y(void* ptr, double* out, int* n) {
  try {
    const arma::vec& v = static_cast<MLPKriging*>(ptr)->y();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

double lk_mlp_kriging_get_centerY(void* ptr) {
  try {
    return static_cast<MLPKriging*>(ptr)->centerY();
  }
  CATCH_RETURN_NAN
}

double lk_mlp_kriging_get_scaleY(void* ptr) {
  try {
    return static_cast<MLPKriging*>(ptr)->scaleY();
  }
  CATCH_RETURN_NAN
}

int lk_mlp_kriging_get_F(void* ptr, double* out, int* n, int* d) {
  try {
    const arma::mat& v = static_cast<MLPKriging*>(ptr)->F();
    if (n)
      *n = static_cast<int>(v.n_rows);
    if (d)
      *d = static_cast<int>(v.n_cols);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_mlp_kriging_get_T(void* ptr, double* out, int* n, int* d) {
  try {
    const arma::mat& v = static_cast<MLPKriging*>(ptr)->T();
    if (n)
      *n = static_cast<int>(v.n_rows);
    if (d)
      *d = static_cast<int>(v.n_cols);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_mlp_kriging_get_M(void* ptr, double* out, int* n, int* d) {
  try {
    const arma::mat& v = static_cast<MLPKriging*>(ptr)->M();
    if (n)
      *n = static_cast<int>(v.n_rows);
    if (d)
      *d = static_cast<int>(v.n_cols);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_mlp_kriging_get_z(void* ptr, double* out, int* n) {
  try {
    const arma::vec& v = static_cast<MLPKriging*>(ptr)->z();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_mlp_kriging_get_beta(void* ptr, double* out, int* n) {
  try {
    const arma::vec& v = static_cast<MLPKriging*>(ptr)->beta();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_mlp_kriging_get_theta(void* ptr, double* out, int* n) {
  try {
    arma::vec v = static_cast<MLPKriging*>(ptr)->theta();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

double lk_mlp_kriging_get_sigma2(void* ptr) {
  try {
    return static_cast<MLPKriging*>(ptr)->sigma2();
  }
  CATCH_RETURN_NAN
}

int lk_mlp_kriging_get_hidden_dims(void* ptr, int* out, int* n) {
  try {
    const auto& dims = static_cast<MLPKriging*>(ptr)->hidden_dims();
    if (n)
      *n = static_cast<int>(dims.size());
    if (out) {
      for (size_t i = 0; i < dims.size(); ++i)
        out[i] = static_cast<int>(dims[i]);
    }
    return 0;
  }
  CATCH_RETURN
}

int lk_warp_kriging_save(void* ptr, const char* filename) {
  try {
    auto* wk = static_cast<WarpKriging*>(ptr);
    wk->save(filename);
    return 0;
  }
  CATCH_RETURN
}

void* lk_warp_kriging_load(const char* filename) {
  try {
    return new WarpKriging(WarpKriging::load(filename));
  }
  CATCH_RETURN_NULL
}

int lk_mlp_kriging_save(void* ptr, const char* filename) {
  try {
    auto* mk = static_cast<MLPKriging*>(ptr);
    mk->save(filename);
    return 0;
  }
  CATCH_RETURN
}

void* lk_mlp_kriging_load(const char* filename) {
  try {
    return new MLPKriging(MLPKriging::load(filename));
  }
  CATCH_RETURN_NULL
}

/* ═══ NestedKriging ═══════════════════════════════════════════════ */

static Kriging::Parameters lk_nested_make_params(const double* sigma2,
                                                 int is_sigma2_estim,
                                                 const double* theta,
                                                 int theta_n,
                                                 int is_theta_estim,
                                                 const double* beta,
                                                 int beta_n,
                                                 int is_beta_estim) {
  Kriging::Parameters params;
  if (sigma2)
    params.sigma2 = *sigma2;
  params.is_sigma2_estim = is_sigma2_estim != 0;
  if (theta && theta_n > 0)
    params.theta = arma::mat(const_cast<double*>(theta), 1, theta_n, false, true);
  params.is_theta_estim = is_theta_estim != 0;
  if (beta && beta_n > 0)
    params.beta = arma::colvec(const_cast<double*>(beta), beta_n, false, true);
  params.is_beta_estim = is_beta_estim != 0;
  return params;
}

static std::vector<std::string> lk_nested_make_warping(const char** warping, int n_warping) {
  std::vector<std::string> out;
  for (int i = 0; i < n_warping; ++i)
    out.push_back(warping[i] ? warping[i] : "none");
  return out;
}

static NestedKriging::Partition lk_nested_parse_partition(const char* s) {
  const std::string p = s ? s : "kmeans";
  if (p == "random")
    return NestedKriging::Partition::Random;
  if (p == "kmeans" || p.empty())
    return NestedKriging::Partition::KMeans;
  throw std::invalid_argument("Unknown partition: '" + p + "'. Expected 'kmeans' or 'random'.");
}

void* lk_nested_kriging_new_fit(const double* y,
                                int n,
                                const double* X,
                                int nX,
                                int d,
                                const char* kernel,
                                int nb_groups,
                                const char* aggregation,
                                const char* partition,
                                int seed,
                                const char** warping,
                                int n_warping,
                                const char* regmodel,
                                const char* optim,
                                const char* objective,
                                const double* sigma2,
                                int is_sigma2_estim,
                                const double* theta,
                                int theta_n,
                                int is_theta_estim,
                                const double* beta,
                                int beta_n,
                                int is_beta_estim) {
  try {
    arma::vec y_v(const_cast<double*>(y), n, false, true);
    arma::mat X_m(const_cast<double*>(X), nX, d, false, true);
    return new NestedKriging(
        y_v,
        X_m,
        kernel ? kernel : "matern3_2",
        static_cast<arma::uword>(nb_groups),
        NestedKriging::aggregationFromString(aggregation ? aggregation : "NK"),
        lk_nested_parse_partition(partition),
        seed,
        Trend::fromString(regmodel ? regmodel : "constant"),
        optim ? optim : "BFGS",
        objective ? objective : "LL",
        lk_nested_make_params(sigma2, is_sigma2_estim, theta, theta_n, is_theta_estim, beta, beta_n, is_beta_estim),
        lk_nested_make_warping(warping, n_warping));
  }
  CATCH_RETURN_NULL
}

void lk_nested_kriging_delete(void* ptr) {
  delete static_cast<NestedKriging*>(ptr);
}

int lk_nested_kriging_fit(void* ptr,
                          const double* y,
                          int n,
                          const double* X,
                          int nX,
                          int d,
                          int nb_groups,
                          const char** warping,
                          int n_warping,
                          const char* regmodel,
                          const char* optim,
                          const char* objective,
                          const double* sigma2,
                          int is_sigma2_estim,
                          const double* theta,
                          int theta_n,
                          int is_theta_estim,
                          const double* beta,
                          int beta_n,
                          int is_beta_estim) {
  try {
    auto* k = static_cast<NestedKriging*>(ptr);
    arma::vec y_v(const_cast<double*>(y), n, false, true);
    arma::mat X_m(const_cast<double*>(X), nX, d, false, true);
    k->fit(y_v,
           X_m,
           static_cast<arma::uword>(nb_groups),
           Trend::fromString(regmodel ? regmodel : "constant"),
           optim ? optim : "BFGS",
           objective ? objective : "LL",
           lk_nested_make_params(sigma2, is_sigma2_estim, theta, theta_n, is_theta_estim, beta, beta_n, is_beta_estim),
           lk_nested_make_warping(warping, n_warping));
    return 0;
  }
  CATCH_RETURN
}

int lk_nested_kriging_predict(void* ptr,
                              const double* X_n,
                              int m,
                              int d,
                              int return_stdev,
                              double* mean_out,
                              double* stdev_out) {
  try {
    auto* k = static_cast<NestedKriging*>(ptr);
    arma::mat X_m(const_cast<double*>(X_n), m, d, false, true);
    auto [mean_v, stdev_v] = k->predict(X_m, return_stdev != 0);
    if (mean_out)
      std::memcpy(mean_out, mean_v.memptr(), mean_v.n_elem * sizeof(double));
    if (stdev_out && return_stdev)
      std::memcpy(stdev_out, stdev_v.memptr(), stdev_v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

const char* lk_nested_kriging_summary(void* ptr) {
  try {
    auto* k = static_cast<NestedKriging*>(ptr);
    static thread_local std::string buf;
    buf = k->summary();
    return buf.c_str();
  }
  CATCH_RETURN_NULL
}

const char* lk_nested_kriging_kernel(void* ptr) {
  try {
    auto* k = static_cast<NestedKriging*>(ptr);
    static thread_local std::string buf;
    buf = k->kernel();
    return buf.c_str();
  }
  CATCH_RETURN_NULL
}

const char* lk_nested_kriging_aggregation(void* ptr) {
  try {
    auto* k = static_cast<NestedKriging*>(ptr);
    static thread_local std::string buf;
    buf = NestedKriging::aggregationToString(k->aggregation());
    return buf.c_str();
  }
  CATCH_RETURN_NULL
}

int lk_nested_kriging_nb_groups(void* ptr) {
  try {
    return static_cast<int>(static_cast<NestedKriging*>(ptr)->nb_groups());
  }
  CATCH_RETURN
}

int lk_nested_kriging_get_theta(void* ptr, double* out, int* n) {
  try {
    auto* k = static_cast<NestedKriging*>(ptr);
    const arma::vec& v = k->theta();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

double lk_nested_kriging_get_sigma2(void* ptr) {
  return static_cast<NestedKriging*>(ptr)->sigma2();
}

double lk_nested_kriging_get_beta0(void* ptr) {
  return static_cast<NestedKriging*>(ptr)->beta0();
}
