#ifndef LIBKRIGING_C_H
#define LIBKRIGING_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* Error handling: all functions return 0 on success, -1 on error.
   Use lk_get_last_error() to retrieve the error message. */
const char* lk_get_last_error(void);

/* --- Kriging ---
   noise_model: "none" (default/NULL), "nugget", or "heterogeneous"
   noise:       per-observation noise vector (heterogeneous mode only, NULL otherwise)
*/

void* lk_kriging_new(const char* kernel, const char* noise_model);
void* lk_kriging_new_fit(const double* y,
                         int n,
                         const double* noise,
                         int noise_n,
                         const double* X,
                         int nX,
                         int d,
                         const char* kernel,
                         const char* noise_model,
                         const char* regmodel,
                         int normalize,
                         const char* optim,
                         const char* objective,
                         /* optional parameters (NULL to skip) */
                         const double* sigma2,
                         int is_sigma2_estim,
                         const double* theta,
                         int theta_n,
                         int is_theta_estim,
                         const double* beta,
                         int beta_n,
                         int is_beta_estim,
                         const double* nugget,
                         int is_nugget_estim,
                         /* fit method: "exact", "vecchia(m)", "nystrom(k)" (NULL = "exact") */
                         const char* method);
void lk_kriging_delete(void* ptr);
void* lk_kriging_copy(void* ptr);

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
                   const char* objective,
                   const char* method);

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
                       double* stdev_deriv_out,
                       const char* method); /* "exact" (NULL), "vecchia", "vecchia(m)" or "nystrom" */

/* Subset-of-data pre-fit reduction: select n_max rows of X (k-means
   centroids snapped to the nearest real point, or a uniform random
   subsample). idx_out must have room for min(n_max, m) ints; returns the
   actual number of indices written (== n_max unless n_max >= m, in which
   case it's a no-op returning all m indices), or -1 on error. Indices are
   0-based row-indices into X. */
int lk_kriging_subsetOfData(const double* X_n, int m, int d, int n_max, const char* method, int seed, int* idx_out);

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
                        double* sim_out,
                        const char* method); /* "exact" (NULL), "vecchia", "vecchia(m)" or "nystrom" */

int lk_kriging_update(void* ptr,
                      const double* y_u,
                      int n,
                      const double* noise_u,
                      int noise_n,
                      const double* X_u,
                      int nX,
                      int d,
                      int refit);

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
                               int* m_out);

int lk_kriging_save(void* ptr, const char* filename);
void* lk_kriging_load(const char* filename);
const char* lk_kriging_summary(void* ptr);

int lk_kriging_log_likelihood_fun(void* ptr,
                                  const double* theta,
                                  int theta_n,
                                  int return_grad,
                                  int return_hess,
                                  double* ll_out,
                                  double* grad_out,
                                  double* hess_out);

int lk_kriging_leave_one_out_fun(void* ptr,
                                 const double* theta,
                                 int theta_n,
                                 int return_grad,
                                 double* loo_out,
                                 double* grad_out);

int lk_kriging_log_marg_post_fun(void* ptr,
                                 const double* theta,
                                 int theta_n,
                                 int return_grad,
                                 double* lmp_out,
                                 double* grad_out);

double lk_kriging_log_likelihood(void* ptr);
/* log-likelihood with an explicit method ("exact", "vecchia" or "nystrom"); 0 or -1 on error */
int lk_kriging_log_likelihood_method(void* ptr, const char* method, double* ll_out);
double lk_kriging_leave_one_out(void* ptr);
double lk_kriging_log_marg_post(void* ptr);
/* LOO / log-marginal posterior with an explicit method ("exact", "vecchia" or "nystrom"); 0 or -1 on error */
int lk_kriging_leave_one_out_method(void* ptr, const char* method, double* loo_out);
int lk_kriging_log_marg_post_method(void* ptr, const char* method, double* lmp_out);

int lk_kriging_leave_one_out_vec(void* ptr, const double* theta, int theta_n, double* yhat_out, double* stderr_out);
int lk_kriging_leave_one_out_vec_method(void* ptr,
                                        const double* theta,
                                        int theta_n,
                                        const char* method,
                                        double* yhat_out,
                                        double* stderr_out);

int lk_kriging_cov_mat(void* ptr, const double* X1, int n1, int d1, const double* X2, int n2, int d2, double* cov_out);

/* Getters (return sizes via out params; pass NULL for data to query size only) */
const char* lk_kriging_kernel(void* ptr);
const char* lk_kriging_optim(void* ptr);
const char* lk_kriging_objective(void* ptr);
int lk_kriging_nystrom_rank(void* ptr);
int lk_kriging_vecchia_neighbors(void* ptr);
const char* lk_kriging_method(void* ptr);
/* Deprecation message of a deprecated fit objective ("" if none, NULL on error) */
const char* lk_deprecated_objective_message(const char* objective);
int lk_kriging_is_normalize(void* ptr);
const char* lk_kriging_regmodel(void* ptr);
int lk_kriging_get_X(void* ptr, double* out, int* n, int* d);
int lk_kriging_get_centerX(void* ptr, double* out, int* d);
int lk_kriging_get_scaleX(void* ptr, double* out, int* d);
int lk_kriging_get_y(void* ptr, double* out, int* n);
double lk_kriging_get_centerY(void* ptr);
double lk_kriging_get_scaleY(void* ptr);
int lk_kriging_get_F(void* ptr, double* out, int* n, int* d);
int lk_kriging_get_T(void* ptr, double* out, int* n, int* d);
int lk_kriging_get_M(void* ptr, double* out, int* n, int* d);
int lk_kriging_get_z(void* ptr, double* out, int* n);
int lk_kriging_get_beta(void* ptr, double* out, int* n);
int lk_kriging_is_beta_estim(void* ptr);
int lk_kriging_get_theta(void* ptr, double* out, int* n);
int lk_kriging_is_theta_estim(void* ptr);
double lk_kriging_get_sigma2(void* ptr);
int lk_kriging_is_sigma2_estim(void* ptr);
const char* lk_kriging_noise_model(void* ptr);
double lk_kriging_get_nugget(void* ptr);
int lk_kriging_is_nugget_estim(void* ptr);
int lk_kriging_get_noise(void* ptr, double* out, int* n);

/* --- WarpKriging --- */

void* lk_warp_kriging_new(const char** warping, int n_warping, const char* kernel);
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
                              int n_params);
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
                                    int n_params);
void lk_warp_kriging_delete(void* ptr);
void* lk_warp_kriging_copy(void* ptr);

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
                        int n_params);
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
                              int n_params);

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
                            double* stdev_deriv_out);

/* with_noise: observation-noise VARIANCES (raw y scale, like the fit's noise),
 * noise_n = 1 (common) or m (one per row of X_n); NULL / noise_n = 0 for no
 * noise. Negative / NaN values return an error. */
int lk_warp_kriging_simulate(void* ptr,
                             int nsim,
                             int seed,
                             const double* X_n,
                             int m,
                             int d,
                             const double* with_noise,
                             int noise_n,
                             int will_update,
                             double* sim_out);

/* noise_u: per-update-point noise variances (NULL / noise_u_n = 0 when the
 * model was fitted noise-free; required otherwise). */
int lk_warp_kriging_update(void* ptr,
                           const double* y_u,
                           int n,
                           const double* X_u,
                           int nX,
                           int d,
                           int refit,
                           const double* noise_u,
                           int noise_u_n);
int lk_warp_kriging_update_simulate(void* ptr,
                                    const double* y_u,
                                    int n,
                                    const double* X_u,
                                    int nX,
                                    int d,
                                    const double* noise_u,
                                    int noise_u_n,
                                    double* sim_out,
                                    int* nsim_out,
                                    int* m_out);

const char* lk_warp_kriging_summary(void* ptr);
double lk_warp_kriging_log_likelihood(void* ptr);

int lk_warp_kriging_log_likelihood_fun(void* ptr,
                                       const double* theta,
                                       int theta_n,
                                       int return_grad,
                                       int return_hess,
                                       double* ll_out,
                                       double* grad_out,
                                       double* hess_out);

const char* lk_warp_kriging_kernel(void* ptr);
int lk_warp_kriging_get_normalize(void* ptr);
const char* lk_warp_kriging_get_regmodel(void* ptr);
int lk_warp_kriging_is_fitted(void* ptr);
int lk_warp_kriging_feature_dim(void* ptr);
int lk_warp_kriging_get_X(void* ptr, double* out, int* n, int* d);
int lk_warp_kriging_get_centerX(void* ptr, double* out, int* n);
int lk_warp_kriging_get_scaleX(void* ptr, double* out, int* n);
int lk_warp_kriging_get_y(void* ptr, double* out, int* n);
double lk_warp_kriging_get_centerY(void* ptr);
double lk_warp_kriging_get_scaleY(void* ptr);
int lk_warp_kriging_get_F(void* ptr, double* out, int* n, int* d);
int lk_warp_kriging_get_T(void* ptr, double* out, int* n, int* d);
int lk_warp_kriging_get_M(void* ptr, double* out, int* n, int* d);
int lk_warp_kriging_get_z(void* ptr, double* out, int* n);
int lk_warp_kriging_get_beta(void* ptr, double* out, int* n);
int lk_warp_kriging_get_theta(void* ptr, double* out, int* n);
double lk_warp_kriging_get_sigma2(void* ptr);
int lk_warp_kriging_get_noise(void* ptr, double* out, int* n);
int lk_warp_kriging_get_warp_params(void* ptr, double* out, int* n);
const char* lk_warp_kriging_get_optim(void* ptr);
const char* lk_warp_kriging_get_objective(void* ptr);
int lk_warp_kriging_get_warping(void* ptr, char** out, int* n_warping);

/* Covariance matrix (n1 x n2) between two designs, using the warped kernel.
 * `out` must have room for n1 * n2 doubles (column-major). */
int lk_warp_kriging_cov_mat(void* ptr, const double* X1, int n1, int d1, const double* X2, int n2, int d2, double* out);

int lk_warp_kriging_save(void* ptr, const char* filename);
void* lk_warp_kriging_load(const char* filename);

/* --- MLPKriging --- */

void* lk_mlp_kriging_new(const int* hidden_dims, int n_hidden, int d_out, const char* activation, const char* kernel);
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
                             int n_params);
void lk_mlp_kriging_delete(void* ptr);
void* lk_mlp_kriging_copy(void* ptr);

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
                       int n_params);

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
                           double* stdev_deriv_out);

/* with_noise: observation-noise VARIANCES (raw y scale, like the fit's noise),
 * noise_n = 1 (common) or m (one per row of X_n); NULL / noise_n = 0 for no
 * noise. Negative / NaN values return an error. */
int lk_mlp_kriging_simulate(void* ptr,
                            int nsim,
                            int seed,
                            const double* X_n,
                            int m,
                            int d,
                            const double* with_noise,
                            int noise_n,
                            int will_update,
                            double* sim_out);

int lk_mlp_kriging_update(void* ptr, const double* y_u, int n, const double* X_u, int nX, int d, int refit);
int lk_mlp_kriging_update_simulate(void* ptr,
                                   const double* y_u,
                                   int n,
                                   const double* X_u,
                                   int nX,
                                   int d,
                                   double* sim_out,
                                   int* nsim_out,
                                   int* m_out);

const char* lk_mlp_kriging_summary(void* ptr);
double lk_mlp_kriging_log_likelihood(void* ptr);

int lk_mlp_kriging_log_likelihood_fun(void* ptr,
                                      const double* theta,
                                      int theta_n,
                                      int return_grad,
                                      int return_hess,
                                      double* ll_out,
                                      double* grad_out,
                                      double* hess_out);

const char* lk_mlp_kriging_kernel(void* ptr);
const char* lk_mlp_kriging_activation(void* ptr);
const char* lk_mlp_kriging_get_objective(void* ptr);
int lk_mlp_kriging_get_normalize(void* ptr);
const char* lk_mlp_kriging_get_regmodel(void* ptr);
int lk_mlp_kriging_is_fitted(void* ptr);
int lk_mlp_kriging_feature_dim(void* ptr);
int lk_mlp_kriging_get_X(void* ptr, double* out, int* n, int* d);
int lk_mlp_kriging_get_centerX(void* ptr, double* out, int* n);
int lk_mlp_kriging_get_scaleX(void* ptr, double* out, int* n);
int lk_mlp_kriging_get_y(void* ptr, double* out, int* n);
double lk_mlp_kriging_get_centerY(void* ptr);
double lk_mlp_kriging_get_scaleY(void* ptr);
int lk_mlp_kriging_get_F(void* ptr, double* out, int* n, int* d);
int lk_mlp_kriging_get_T(void* ptr, double* out, int* n, int* d);
int lk_mlp_kriging_get_M(void* ptr, double* out, int* n, int* d);
int lk_mlp_kriging_get_z(void* ptr, double* out, int* n);
int lk_mlp_kriging_get_beta(void* ptr, double* out, int* n);
int lk_mlp_kriging_get_theta(void* ptr, double* out, int* n);
double lk_mlp_kriging_get_sigma2(void* ptr);
int lk_mlp_kriging_get_hidden_dims(void* ptr, int* out, int* n);

int lk_mlp_kriging_save(void* ptr, const char* filename);
void* lk_mlp_kriging_load(const char* filename);

/* ═══ NestedKriging ═══════════════════════════════════════════════ */

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
                                int is_beta_estim,
                                const char* method);
const char* lk_nested_kriging_objective(void* ptr);
const char* lk_nested_kriging_method(void* ptr);
void lk_nested_kriging_delete(void* ptr);
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
                          int is_beta_estim,
                          const char* method);
int lk_nested_kriging_predict(void* ptr,
                              const double* X_n,
                              int m,
                              int d,
                              int return_stdev,
                              double* mean_out,
                              double* stdev_out);
const char* lk_nested_kriging_summary(void* ptr);
const char* lk_nested_kriging_kernel(void* ptr);
const char* lk_nested_kriging_aggregation(void* ptr);
int lk_nested_kriging_nb_groups(void* ptr);
int lk_nested_kriging_get_theta(void* ptr, double* out, int* n);
double lk_nested_kriging_get_sigma2(void* ptr);
double lk_nested_kriging_get_beta0(void* ptr);
int lk_nested_kriging_get_X(void* ptr, double* out, int* n, int* d);
int lk_nested_kriging_get_y(void* ptr, double* out, int* n);
int lk_nested_kriging_get_group(void* ptr, int g, int* out, int* n);
int lk_nested_kriging_get_warping(void* ptr, char** out, int* n_warping);
int lk_nested_kriging_set_predict_chunk(void* ptr, int chunk);
int lk_nested_kriging_set_warp_subsample(void* ptr, int m);
int lk_nested_kriging_save(void* ptr, const char* filename);
void* lk_nested_kriging_load(const char* filename);

/* --- LinearAlgebra settings (global) ---
   chol_jitter: fixed jitter on the correlation diagonal of noise-free fits
   (< 0: automatic per kernel, the default; 0: disabled; > 0: this value)
   num_nugget:  numerical nugget of the Cholesky fallback (default 1e-10) */
double lk_linalg_get_chol_jitter(void);
int lk_linalg_set_chol_jitter(double jitter);
double lk_linalg_get_num_nugget(void);
int lk_linalg_set_num_nugget(double nugget);
/* warning when the Cholesky fallback adds a numerical nugget; check of the
   reciprocal condition number of the factorizations */
int lk_linalg_set_chol_warning(int warn);
int lk_linalg_check_chol_rcond(int check);
int lk_linalg_chol_rcond_checked(void);

/* --- Optim settings (global), as Optim:: in C++ --- */
int lk_optim_use_reparametrize(int reparametrize);
int lk_optim_is_reparametrized(void);
int lk_optim_set_theta_lower_factor(double factor);
double lk_optim_get_theta_lower_factor(void);
int lk_optim_set_theta_upper_factor(double factor);
double lk_optim_get_theta_upper_factor(void);
int lk_optim_use_variogram_bounds_heuristic(int use);
int lk_optim_variogram_bounds_heuristic_used(void);
int lk_optim_set_log_level(int level);
int lk_optim_get_log_level(void);
int lk_optim_set_max_iteration(int max_iteration);
int lk_optim_get_max_iteration(void);
int lk_optim_set_gradient_tolerance(double tolerance);
double lk_optim_get_gradient_tolerance(void);
int lk_optim_set_objective_rel_tolerance(double tolerance);
double lk_optim_get_objective_rel_tolerance(void);
int lk_optim_set_thread_start_delay_ms(int delay_ms);
int lk_optim_get_thread_start_delay_ms(void);
int lk_optim_set_thread_pool_size(int pool_size);
int lk_optim_get_thread_pool_size(void);
int lk_optim_set_num_threads_percent(double percent);
double lk_optim_get_num_threads_percent(void);
int lk_optim_get_num_threads(void);

#ifdef __cplusplus
}
#endif

#endif /* LIBKRIGING_C_H */
