#ifndef LIBKRIGING_C_H
#define LIBKRIGING_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* Error handling: all functions return 0 on success, -1 on error.
   Use lk_get_last_error() to retrieve the error message. */
const char* lk_get_last_error(void);

/* --- LinearRegression --- */

void* lk_linear_regression_new(void);
void lk_linear_regression_delete(void* ptr);
int lk_linear_regression_fit(void* ptr, const double* y, int n, const double* X, int nX, int d);
int lk_linear_regression_predict(void* ptr, const double* X, int m, int d, double* mean_out, double* stdev_out);

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
                         int is_nugget_estim);
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
                   const char* objective);

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
                       double* stdev_deriv_out);

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
                        double* sim_out);

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
double lk_kriging_leave_one_out(void* ptr);
double lk_kriging_log_marg_post(void* ptr);

int lk_kriging_leave_one_out_vec(void* ptr, const double* theta, int theta_n, double* yhat_out, double* stderr_out);

int lk_kriging_cov_mat(void* ptr, const double* X1, int n1, int d1, const double* X2, int n2, int d2, double* cov_out);

/* Getters (return sizes via out params; pass NULL for data to query size only) */
const char* lk_kriging_kernel(void* ptr);
const char* lk_kriging_optim(void* ptr);
const char* lk_kriging_objective(void* ptr);
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

int lk_warp_kriging_simulate(void* ptr, int nsim, int seed, const double* X_n, int m, int d, double* sim_out);

int lk_warp_kriging_update(void* ptr, const double* y_u, int n, const double* X_u, int nX, int d, int refit);

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
int lk_warp_kriging_is_fitted(void* ptr);
int lk_warp_kriging_feature_dim(void* ptr);
int lk_warp_kriging_get_X(void* ptr, double* out, int* n, int* d);
int lk_warp_kriging_get_y(void* ptr, double* out, int* n);
int lk_warp_kriging_get_theta(void* ptr, double* out, int* n);
double lk_warp_kriging_get_sigma2(void* ptr);
int lk_warp_kriging_get_warping(void* ptr, char** out, int* n_warping);

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

int lk_mlp_kriging_simulate(void* ptr, int nsim, int seed, const double* X_n, int m, int d, double* sim_out);

int lk_mlp_kriging_update(void* ptr, const double* y_u, int n, const double* X_u, int nX, int d, int refit);

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
int lk_mlp_kriging_is_fitted(void* ptr);
int lk_mlp_kriging_feature_dim(void* ptr);
int lk_mlp_kriging_get_X(void* ptr, double* out, int* n, int* d);
int lk_mlp_kriging_get_y(void* ptr, double* out, int* n);
int lk_mlp_kriging_get_theta(void* ptr, double* out, int* n);
double lk_mlp_kriging_get_sigma2(void* ptr);
int lk_mlp_kriging_get_hidden_dims(void* ptr, int* out, int* n);

int lk_mlp_kriging_save(void* ptr, const char* filename);
void* lk_mlp_kriging_load(const char* filename);

#ifdef __cplusplus
}
#endif

#endif /* LIBKRIGING_C_H */
