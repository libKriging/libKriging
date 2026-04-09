#include "libkriging_c.h"

#include <libKriging/Kriging.hpp>
#include <libKriging/LinearRegression.hpp>
#include <libKriging/NoiseKriging.hpp>
#include <libKriging/NuggetKriging.hpp>
#include <libKriging/Trend.hpp>
#include <libKriging/WarpKriging.hpp>
#include <libKriging/utils/ExplicitCopySpecifier.hpp>

#include <cstring>
#include <limits>
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
/*  LinearRegression                                                          */
/* ========================================================================== */

void* lk_linear_regression_new(void) {
  try {
    return new LinearRegression();
  }
  CATCH_RETURN_NULL
}

void lk_linear_regression_delete(void* ptr) {
  delete static_cast<LinearRegression*>(ptr);
}

int lk_linear_regression_fit(void* ptr, const double* y, int n, const double* X, int nX, int d) {
  try {
    auto* lr = static_cast<LinearRegression*>(ptr);
    arma::vec y_v(const_cast<double*>(y), n, false, true);
    arma::mat X_m(const_cast<double*>(X), nX, d, false, true);
    lr->fit(y_v, X_m);
    return 0;
  }
  CATCH_RETURN
}

int lk_linear_regression_predict(void* ptr, const double* X, int m, int d, double* mean_out, double* stdev_out) {
  try {
    auto* lr = static_cast<LinearRegression*>(ptr);
    arma::mat X_m(const_cast<double*>(X), m, d, false, true);
    auto [mean_v, stdev_v] = lr->predict(X_m);
    if (mean_out)
      std::memcpy(mean_out, mean_v.memptr(), mean_v.n_elem * sizeof(double));
    if (stdev_out)
      std::memcpy(stdev_out, stdev_v.memptr(), stdev_v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

/* ========================================================================== */
/*  Kriging                                                                   */
/* ========================================================================== */

void* lk_kriging_new(const char* kernel) {
  try {
    return new Kriging(kernel);
  }
  CATCH_RETURN_NULL
}

void* lk_kriging_new_fit(const double* y,
                         int n,
                         const double* X,
                         int nX,
                         int d,
                         const char* kernel,
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
                         int is_beta_estim) {
  try {
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

    return new Kriging(y_v,
                       X_m,
                       kernel ? kernel : "matern3_2",
                       Trend::fromString(regmodel ? regmodel : "constant"),
                       normalize != 0,
                       optim ? optim : "BFGS",
                       objective ? objective : "LL",
                       params);
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
    k->fit(y_v,
           X_m,
           Trend::fromString(regmodel ? regmodel : "constant"),
           normalize != 0,
           optim ? optim : "BFGS",
           objective ? objective : "LL");
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
                        int will_update,
                        double* sim_out) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    arma::mat X_m(const_cast<double*>(X_n), m, d, false, true);
    arma::mat sim = k->simulate(nsim, seed, X_m, will_update != 0);
    if (sim_out)
      std::memcpy(sim_out, sim.memptr(), sim.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_update(void* ptr, const double* y_u, int n, const double* X_u, int nX, int d, int refit) {
  try {
    auto* k = static_cast<Kriging*>(ptr);
    arma::vec y_v(const_cast<double*>(y_u), n, false, true);
    arma::mat X_m(const_cast<double*>(X_u), nX, d, false, true);
    k->update(y_v, X_m, refit != 0);
    return 0;
  }
  CATCH_RETURN
}

int lk_kriging_update_simulate(void* ptr,
                               const double* y_u,
                               int n,
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
    arma::mat sim = k->update_simulate(y_v, X_m);
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
    auto [ll, grad, hess] = k->logLikelihoodFun(theta_v, return_grad != 0, return_hess != 0, false);
    if (ll_out)
      *ll_out = ll;
    if (grad_out && return_grad)
      std::memcpy(grad_out, grad.memptr(), grad.n_elem * sizeof(double));
    if (hess_out && return_hess)
      std::memcpy(hess_out, hess.memptr(), hess.n_elem * sizeof(double));
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

/* ========================================================================== */
/*  NuggetKriging                                                             */
/* ========================================================================== */

void* lk_nugget_kriging_new(const char* kernel) {
  try {
    return new NuggetKriging(kernel);
  }
  CATCH_RETURN_NULL
}

void* lk_nugget_kriging_new_fit(const double* y,
                                int n,
                                const double* X,
                                int nX,
                                int d,
                                const char* kernel,
                                const char* regmodel,
                                int normalize,
                                const char* optim,
                                const char* objective,
                                const double* sigma2,
                                int sigma2_n,
                                int is_sigma2_estim,
                                const double* theta,
                                int theta_n,
                                int is_theta_estim,
                                const double* beta,
                                int beta_n,
                                int is_beta_estim,
                                const double* nugget,
                                int nugget_n,
                                int is_nugget_estim) {
  try {
    arma::vec y_v(const_cast<double*>(y), n, false, true);
    arma::mat X_m(const_cast<double*>(X), nX, d, false, true);

    NuggetKriging::Parameters params;
    if (sigma2 && sigma2_n > 0)
      params.sigma2 = arma::vec(const_cast<double*>(sigma2), sigma2_n, false, true);
    params.is_sigma2_estim = is_sigma2_estim != 0;
    if (theta && theta_n > 0)
      params.theta = arma::mat(const_cast<double*>(theta), 1, theta_n, false, true);
    params.is_theta_estim = is_theta_estim != 0;
    if (beta && beta_n > 0)
      params.beta = arma::vec(const_cast<double*>(beta), beta_n, false, true);
    params.is_beta_estim = is_beta_estim != 0;
    if (nugget && nugget_n > 0)
      params.nugget = arma::vec(const_cast<double*>(nugget), nugget_n, false, true);
    params.is_nugget_estim = is_nugget_estim != 0;

    return new NuggetKriging(y_v,
                             X_m,
                             kernel ? kernel : "matern3_2",
                             Trend::fromString(regmodel ? regmodel : "constant"),
                             normalize != 0,
                             optim ? optim : "BFGS",
                             objective ? objective : "LL",
                             params);
  }
  CATCH_RETURN_NULL
}

void lk_nugget_kriging_delete(void* ptr) {
  delete static_cast<NuggetKriging*>(ptr);
}

void* lk_nugget_kriging_copy(void* ptr) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
    return new NuggetKriging(*k, ExplicitCopySpecifier{});
  }
  CATCH_RETURN_NULL
}

int lk_nugget_kriging_fit(void* ptr,
                          const double* y,
                          int n,
                          const double* X,
                          int nX,
                          int d,
                          const char* regmodel,
                          int normalize,
                          const char* optim,
                          const char* objective) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
    arma::vec y_v(const_cast<double*>(y), n, false, true);
    arma::mat X_m(const_cast<double*>(X), nX, d, false, true);
    k->fit(y_v,
           X_m,
           Trend::fromString(regmodel ? regmodel : "constant"),
           normalize != 0,
           optim ? optim : "BFGS",
           objective ? objective : "LL");
    return 0;
  }
  CATCH_RETURN
}

int lk_nugget_kriging_predict(void* ptr,
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
    auto* k = static_cast<NuggetKriging*>(ptr);
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

int lk_nugget_kriging_simulate(void* ptr,
                               int nsim,
                               int seed,
                               const double* X_n,
                               int m,
                               int d,
                               int with_nugget,
                               int will_update,
                               double* sim_out) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
    arma::mat X_m(const_cast<double*>(X_n), m, d, false, true);
    arma::mat sim = k->simulate(nsim, seed, X_m, with_nugget != 0, will_update != 0);
    if (sim_out)
      std::memcpy(sim_out, sim.memptr(), sim.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_nugget_kriging_update(void* ptr, const double* y_u, int n, const double* X_u, int nX, int d, int refit) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
    arma::vec y_v(const_cast<double*>(y_u), n, false, true);
    arma::mat X_m(const_cast<double*>(X_u), nX, d, false, true);
    k->update(y_v, X_m, refit != 0);
    return 0;
  }
  CATCH_RETURN
}

int lk_nugget_kriging_update_simulate(void* ptr,
                                      const double* y_u,
                                      int n,
                                      const double* X_u,
                                      int nX,
                                      int d,
                                      double* sim_out,
                                      int* nsim_out,
                                      int* m_out) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
    arma::vec y_v(const_cast<double*>(y_u), n, false, true);
    arma::mat X_m(const_cast<double*>(X_u), nX, d, false, true);
    arma::mat sim = k->update_simulate(y_v, X_m);
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

int lk_nugget_kriging_save(void* ptr, const char* filename) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
    k->save(filename);
    return 0;
  }
  CATCH_RETURN
}

void* lk_nugget_kriging_load(const char* filename) {
  try {
    return new NuggetKriging(NuggetKriging::load(filename));
  }
  CATCH_RETURN_NULL
}

const char* lk_nugget_kriging_summary(void* ptr) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
    static thread_local std::string buf;
    buf = k->summary();
    return buf.c_str();
  }
  CATCH_RETURN_NULL
}

int lk_nugget_kriging_log_likelihood_fun(void* ptr,
                                         const double* theta,
                                         int theta_n,
                                         int return_grad,
                                         int return_hess,
                                         double* ll_out,
                                         double* grad_out,
                                         double* hess_out) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
    arma::vec theta_v(const_cast<double*>(theta), theta_n, false, true);
    // NuggetKriging::logLikelihoodFun does not support hessian output
    auto [ll, grad] = k->logLikelihoodFun(theta_v, return_grad != 0, false);
    if (ll_out)
      *ll_out = ll;
    if (grad_out && return_grad)
      std::memcpy(grad_out, grad.memptr(), grad.n_elem * sizeof(double));
    (void)return_hess;
    (void)hess_out;
    return 0;
  }
  CATCH_RETURN
}

int lk_nugget_kriging_log_marg_post_fun(void* ptr,
                                        const double* theta,
                                        int theta_n,
                                        int return_grad,
                                        double* lmp_out,
                                        double* grad_out) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
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

double lk_nugget_kriging_log_likelihood(void* ptr) {
  try {
    return static_cast<NuggetKriging*>(ptr)->logLikelihood();
  }
  CATCH_RETURN_NAN
}

double lk_nugget_kriging_log_marg_post(void* ptr) {
  try {
    return static_cast<NuggetKriging*>(ptr)->logMargPost();
  }
  CATCH_RETURN_NAN
}

int lk_nugget_kriging_cov_mat(void* ptr,
                              const double* X1,
                              int n1,
                              int d1,
                              const double* X2,
                              int n2,
                              int d2,
                              double* cov_out) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
    arma::mat X1_m(const_cast<double*>(X1), n1, d1, false, true);
    arma::mat X2_m(const_cast<double*>(X2), n2, d2, false, true);
    arma::mat cov = k->covMat(X1_m, X2_m);
    if (cov_out)
      std::memcpy(cov_out, cov.memptr(), cov.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

/* NuggetKriging string getters */

const char* lk_nugget_kriging_kernel(void* ptr) {
  try {
    return static_cast<NuggetKriging*>(ptr)->kernel().c_str();
  }
  CATCH_RETURN_NULL
}

const char* lk_nugget_kriging_optim(void* ptr) {
  try {
    return static_cast<NuggetKriging*>(ptr)->optim().c_str();
  }
  CATCH_RETURN_NULL
}

const char* lk_nugget_kriging_objective(void* ptr) {
  try {
    return static_cast<NuggetKriging*>(ptr)->objective().c_str();
  }
  CATCH_RETURN_NULL
}

int lk_nugget_kriging_is_normalize(void* ptr) {
  try {
    return static_cast<NuggetKriging*>(ptr)->normalize() ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

const char* lk_nugget_kriging_regmodel(void* ptr) {
  try {
    static thread_local std::string buf;
    buf = Trend::toString(static_cast<NuggetKriging*>(ptr)->regmodel());
    return buf.c_str();
  }
  CATCH_RETURN_NULL
}

/* NuggetKriging array getters */

int lk_nugget_kriging_get_X(void* ptr, double* out, int* n, int* d) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
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

int lk_nugget_kriging_get_centerX(void* ptr, double* out, int* d) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
    const arma::rowvec& v = k->centerX();
    if (d)
      *d = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_nugget_kriging_get_scaleX(void* ptr, double* out, int* d) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
    const arma::rowvec& v = k->scaleX();
    if (d)
      *d = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_nugget_kriging_get_y(void* ptr, double* out, int* n) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
    const arma::vec& v = k->y();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

double lk_nugget_kriging_get_centerY(void* ptr) {
  try {
    return static_cast<NuggetKriging*>(ptr)->centerY();
  }
  CATCH_RETURN_NAN
}

double lk_nugget_kriging_get_scaleY(void* ptr) {
  try {
    return static_cast<NuggetKriging*>(ptr)->scaleY();
  }
  CATCH_RETURN_NAN
}

int lk_nugget_kriging_get_F(void* ptr, double* out, int* n, int* d) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
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

int lk_nugget_kriging_get_T(void* ptr, double* out, int* n, int* d) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
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

int lk_nugget_kriging_get_M(void* ptr, double* out, int* n, int* d) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
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

int lk_nugget_kriging_get_z(void* ptr, double* out, int* n) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
    const arma::vec& v = k->z();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_nugget_kriging_get_beta(void* ptr, double* out, int* n) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
    const arma::vec& v = k->beta();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_nugget_kriging_is_beta_estim(void* ptr) {
  try {
    return static_cast<NuggetKriging*>(ptr)->is_beta_estim() ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

int lk_nugget_kriging_get_theta(void* ptr, double* out, int* n) {
  try {
    auto* k = static_cast<NuggetKriging*>(ptr);
    const arma::vec& v = k->theta();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_nugget_kriging_is_theta_estim(void* ptr) {
  try {
    return static_cast<NuggetKriging*>(ptr)->is_theta_estim() ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

double lk_nugget_kriging_get_sigma2(void* ptr) {
  try {
    return static_cast<NuggetKriging*>(ptr)->sigma2();
  }
  CATCH_RETURN_NAN
}

int lk_nugget_kriging_is_sigma2_estim(void* ptr) {
  try {
    return static_cast<NuggetKriging*>(ptr)->is_sigma2_estim() ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

double lk_nugget_kriging_get_nugget(void* ptr) {
  try {
    return static_cast<NuggetKriging*>(ptr)->nugget();
  }
  CATCH_RETURN_NAN
}

int lk_nugget_kriging_is_nugget_estim(void* ptr) {
  try {
    return static_cast<NuggetKriging*>(ptr)->is_nugget_estim() ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

/* ========================================================================== */
/*  NoiseKriging                                                              */
/* ========================================================================== */

void* lk_noise_kriging_new(const char* kernel) {
  try {
    return new NoiseKriging(kernel);
  }
  CATCH_RETURN_NULL
}

void* lk_noise_kriging_new_fit(const double* y,
                               int n,
                               const double* noise,
                               int noise_n,
                               const double* X,
                               int nX,
                               int d,
                               const char* kernel,
                               const char* regmodel,
                               int normalize,
                               const char* optim,
                               const char* objective,
                               const double* sigma2,
                               int sigma2_n,
                               int is_sigma2_estim,
                               const double* theta,
                               int theta_n,
                               int is_theta_estim,
                               const double* beta,
                               int beta_n,
                               int is_beta_estim) {
  try {
    arma::vec y_v(const_cast<double*>(y), n, false, true);
    arma::vec noise_v(const_cast<double*>(noise), noise_n, false, true);
    arma::mat X_m(const_cast<double*>(X), nX, d, false, true);

    NoiseKriging::Parameters params;
    if (sigma2 && sigma2_n > 0)
      params.sigma2 = arma::vec(const_cast<double*>(sigma2), sigma2_n, false, true);
    params.is_sigma2_estim = is_sigma2_estim != 0;
    if (theta && theta_n > 0)
      params.theta = arma::mat(const_cast<double*>(theta), 1, theta_n, false, true);
    params.is_theta_estim = is_theta_estim != 0;
    if (beta && beta_n > 0)
      params.beta = arma::vec(const_cast<double*>(beta), beta_n, false, true);
    params.is_beta_estim = is_beta_estim != 0;

    return new NoiseKriging(y_v,
                            noise_v,
                            X_m,
                            kernel ? kernel : "matern3_2",
                            Trend::fromString(regmodel ? regmodel : "constant"),
                            normalize != 0,
                            optim ? optim : "BFGS",
                            objective ? objective : "LL",
                            params);
  }
  CATCH_RETURN_NULL
}

void lk_noise_kriging_delete(void* ptr) {
  delete static_cast<NoiseKriging*>(ptr);
}

void* lk_noise_kriging_copy(void* ptr) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
    return new NoiseKriging(*k, ExplicitCopySpecifier{});
  }
  CATCH_RETURN_NULL
}

int lk_noise_kriging_fit(void* ptr,
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
    auto* k = static_cast<NoiseKriging*>(ptr);
    arma::vec y_v(const_cast<double*>(y), n, false, true);
    arma::vec noise_v(const_cast<double*>(noise), noise_n, false, true);
    arma::mat X_m(const_cast<double*>(X), nX, d, false, true);
    k->fit(y_v,
           noise_v,
           X_m,
           Trend::fromString(regmodel ? regmodel : "constant"),
           normalize != 0,
           optim ? optim : "BFGS",
           objective ? objective : "LL");
    return 0;
  }
  CATCH_RETURN
}

int lk_noise_kriging_predict(void* ptr,
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
    auto* k = static_cast<NoiseKriging*>(ptr);
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

int lk_noise_kriging_simulate(void* ptr,
                              int nsim,
                              int seed,
                              const double* X_n,
                              int m,
                              int d,
                              const double* with_noise,
                              int noise_n,
                              int will_update,
                              double* sim_out) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
    arma::mat X_m(const_cast<double*>(X_n), m, d, false, true);
    arma::vec noise_v(const_cast<double*>(with_noise), noise_n, false, true);
    arma::mat sim = k->simulate(nsim, seed, X_m, noise_v, will_update != 0);
    if (sim_out)
      std::memcpy(sim_out, sim.memptr(), sim.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_noise_kriging_update(void* ptr,
                            const double* y_u,
                            int n,
                            const double* noise_u,
                            int noise_n,
                            const double* X_u,
                            int nX,
                            int d,
                            int refit) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
    arma::vec y_v(const_cast<double*>(y_u), n, false, true);
    arma::vec noise_v(const_cast<double*>(noise_u), noise_n, false, true);
    arma::mat X_m(const_cast<double*>(X_u), nX, d, false, true);
    k->update(y_v, noise_v, X_m, refit != 0);
    return 0;
  }
  CATCH_RETURN
}

int lk_noise_kriging_update_simulate(void* ptr,
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
    auto* k = static_cast<NoiseKriging*>(ptr);
    arma::vec y_v(const_cast<double*>(y_u), n, false, true);
    arma::vec noise_v(const_cast<double*>(noise_u), noise_n, false, true);
    arma::mat X_m(const_cast<double*>(X_u), nX, d, false, true);
    arma::mat sim = k->update_simulate(y_v, noise_v, X_m);
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

int lk_noise_kriging_save(void* ptr, const char* filename) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
    k->save(filename);
    return 0;
  }
  CATCH_RETURN
}

void* lk_noise_kriging_load(const char* filename) {
  try {
    return new NoiseKriging(NoiseKriging::load(filename));
  }
  CATCH_RETURN_NULL
}

const char* lk_noise_kriging_summary(void* ptr) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
    static thread_local std::string buf;
    buf = k->summary();
    return buf.c_str();
  }
  CATCH_RETURN_NULL
}

int lk_noise_kriging_log_likelihood_fun(void* ptr,
                                        const double* theta,
                                        int theta_n,
                                        int return_grad,
                                        int return_hess,
                                        double* ll_out,
                                        double* grad_out,
                                        double* hess_out) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
    arma::vec theta_v(const_cast<double*>(theta), theta_n, false, true);
    // NoiseKriging::logLikelihoodFun does not support hessian output
    auto [ll, grad] = k->logLikelihoodFun(theta_v, return_grad != 0, false);
    if (ll_out)
      *ll_out = ll;
    if (grad_out && return_grad)
      std::memcpy(grad_out, grad.memptr(), grad.n_elem * sizeof(double));
    (void)return_hess;
    (void)hess_out;
    return 0;
  }
  CATCH_RETURN
}

double lk_noise_kriging_log_likelihood(void* ptr) {
  try {
    return static_cast<NoiseKriging*>(ptr)->logLikelihood();
  }
  CATCH_RETURN_NAN
}

int lk_noise_kriging_cov_mat(void* ptr,
                             const double* X1,
                             int n1,
                             int d1,
                             const double* X2,
                             int n2,
                             int d2,
                             double* cov_out) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
    arma::mat X1_m(const_cast<double*>(X1), n1, d1, false, true);
    arma::mat X2_m(const_cast<double*>(X2), n2, d2, false, true);
    arma::mat cov = k->covMat(X1_m, X2_m);
    if (cov_out)
      std::memcpy(cov_out, cov.memptr(), cov.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

/* NoiseKriging string getters */

const char* lk_noise_kriging_kernel(void* ptr) {
  try {
    return static_cast<NoiseKriging*>(ptr)->kernel().c_str();
  }
  CATCH_RETURN_NULL
}

const char* lk_noise_kriging_optim(void* ptr) {
  try {
    return static_cast<NoiseKriging*>(ptr)->optim().c_str();
  }
  CATCH_RETURN_NULL
}

const char* lk_noise_kriging_objective(void* ptr) {
  try {
    return static_cast<NoiseKriging*>(ptr)->objective().c_str();
  }
  CATCH_RETURN_NULL
}

int lk_noise_kriging_is_normalize(void* ptr) {
  try {
    return static_cast<NoiseKriging*>(ptr)->normalize() ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

const char* lk_noise_kriging_regmodel(void* ptr) {
  try {
    static thread_local std::string buf;
    buf = Trend::toString(static_cast<NoiseKriging*>(ptr)->regmodel());
    return buf.c_str();
  }
  CATCH_RETURN_NULL
}

/* NoiseKriging array getters */

int lk_noise_kriging_get_X(void* ptr, double* out, int* n, int* d) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
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

int lk_noise_kriging_get_centerX(void* ptr, double* out, int* d) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
    const arma::rowvec& v = k->centerX();
    if (d)
      *d = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_noise_kriging_get_scaleX(void* ptr, double* out, int* d) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
    const arma::rowvec& v = k->scaleX();
    if (d)
      *d = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_noise_kriging_get_y(void* ptr, double* out, int* n) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
    const arma::vec& v = k->y();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

double lk_noise_kriging_get_centerY(void* ptr) {
  try {
    return static_cast<NoiseKriging*>(ptr)->centerY();
  }
  CATCH_RETURN_NAN
}

double lk_noise_kriging_get_scaleY(void* ptr) {
  try {
    return static_cast<NoiseKriging*>(ptr)->scaleY();
  }
  CATCH_RETURN_NAN
}

int lk_noise_kriging_get_noise(void* ptr, double* out, int* n) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
    const arma::vec& v = k->noise();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_noise_kriging_get_F(void* ptr, double* out, int* n, int* d) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
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

int lk_noise_kriging_get_T(void* ptr, double* out, int* n, int* d) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
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

int lk_noise_kriging_get_M(void* ptr, double* out, int* n, int* d) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
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

int lk_noise_kriging_get_z(void* ptr, double* out, int* n) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
    const arma::vec& v = k->z();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_noise_kriging_get_beta(void* ptr, double* out, int* n) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
    const arma::vec& v = k->beta();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_noise_kriging_is_beta_estim(void* ptr) {
  try {
    return static_cast<NoiseKriging*>(ptr)->is_beta_estim() ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

int lk_noise_kriging_get_theta(void* ptr, double* out, int* n) {
  try {
    auto* k = static_cast<NoiseKriging*>(ptr);
    const arma::vec& v = k->theta();
    if (n)
      *n = static_cast<int>(v.n_elem);
    if (out)
      std::memcpy(out, v.memptr(), v.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_noise_kriging_is_theta_estim(void* ptr) {
  try {
    return static_cast<NoiseKriging*>(ptr)->is_theta_estim() ? 1 : 0;
  } catch (...) {
    return -1;
  }
}

double lk_noise_kriging_get_sigma2(void* ptr) {
  try {
    return static_cast<NoiseKriging*>(ptr)->sigma2();
  }
  CATCH_RETURN_NAN
}

int lk_noise_kriging_is_sigma2_estim(void* ptr) {
  try {
    return static_cast<NoiseKriging*>(ptr)->is_sigma2_estim() ? 1 : 0;
  } catch (...) {
    return -1;
  }
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
                              const char* objective) {
  try {
    arma::vec y_vec(const_cast<double*>(y), n, false, true);
    arma::mat X_mat(const_cast<double*>(X), nX, d, false, true);
    return new WarpKriging(
        y_vec, X_mat, to_string_vec(warping, n_warping), kernel, regmodel, normalize != 0, optim, objective);
  }
  CATCH_RETURN_NULL
}

void lk_warp_kriging_delete(void* ptr) {
  delete static_cast<WarpKriging*>(ptr);
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
                        const char* objective) {
  try {
    arma::vec y_vec(const_cast<double*>(y), n, false, true);
    arma::mat X_mat(const_cast<double*>(X), nX, d, false, true);
    static_cast<WarpKriging*>(ptr)->fit(y_vec, X_mat, regmodel, normalize != 0, optim, objective);
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
                            double* mean_out,
                            double* stdev_out,
                            double* cov_out) {
  try {
    arma::mat X_mat(const_cast<double*>(X_n), m, d, false, true);
    auto [mean, stdev, cov] = static_cast<WarpKriging*>(ptr)->predict(X_mat, return_stdev != 0, return_cov != 0);
    if (mean_out)
      std::memcpy(mean_out, mean.memptr(), mean.n_elem * sizeof(double));
    if (return_stdev && stdev_out)
      std::memcpy(stdev_out, stdev.memptr(), stdev.n_elem * sizeof(double));
    if (return_cov && cov_out)
      std::memcpy(cov_out, cov.memptr(), cov.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_warp_kriging_simulate(void* ptr, int nsim, int seed, const double* X_n, int m, int d, double* sim_out) {
  try {
    arma::mat X_mat(const_cast<double*>(X_n), m, d, false, true);
    auto result = static_cast<WarpKriging*>(ptr)->simulate(nsim, static_cast<uint64_t>(seed), X_mat);
    if (sim_out)
      std::memcpy(sim_out, result.memptr(), result.n_elem * sizeof(double));
    return 0;
  }
  CATCH_RETURN
}

int lk_warp_kriging_update(void* ptr, const double* y_u, int n, const double* X_u, int nX, int d) {
  try {
    arma::vec y_vec(const_cast<double*>(y_u), n, false, true);
    arma::mat X_mat(const_cast<double*>(X_u), nX, d, false, true);
    static_cast<WarpKriging*>(ptr)->update(y_vec, X_mat);
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
