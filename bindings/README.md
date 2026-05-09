# libKriging Bindings — Method Reference

This document lists all methods exposed by each language binding for accessing the underlying C++ library.

## Classes

| Class | Description | R | Python | Octave/Matlab | Julia |
|---|---|:---:|:---:|:---:|:---:|
| `Kriging` | Standard Kriging, supports `noise_model`: `none`, `nugget`, `heterogeneous` | ✅ | ✅ | ✅ | ✅ |
| `WarpKriging` | Kriging with input warping | ✅ | ✅ | ✅ | ✅ |
| `MLPKriging` | Kriging with MLP feature mapping | ✅ | ✅ | ✅ | ✅ |
| `LinearRegression` | Linear regression | ✅ | ✅ | ✅ | ✅ |

> **Note on noise models**: `NoiseKriging` (heterogeneous noise) and `NuggetKriging` (nugget/homoscedastic noise) are deprecated in R, Python and Julia — use `Kriging` with `noise_model="heterogeneous"` or `noise_model="nugget"`. The Octave/Matlab binding still exposes them as separate classes.

---

## Kriging

| Method | R (C++ fn) | R (object method) | Python | Octave/Matlab | Julia |
|---|---|---|---|---|---|
| Constructor | `new_Kriging(kernel)` | — | `Kriging(kernel)` / `Kriging(kernel, noise_model)` | `build(y,X,kernel,…)` | `Kriging(kernel)` / `Kriging(y,X,kernel;…)` |
| Fit | `kriging_fit(obj,y,X,…)` | `obj$fit(y,X,…)` | `obj.fit(y,X,…)` | `fit(obj,y,X,…)` | `fit!(obj,y,X;…)` |
| Copy | `kriging_copy(obj)` | `obj$copy()` | `obj.copy()` | `copy(obj)` | `copy(obj)` |
| Predict | `kriging_predict(obj,x,…)` | `obj$predict(x,…)` | `obj.predict(x,…)` | `predict(obj,x,…)` | `predict(obj,X;…)` |
| Simulate | `kriging_simulate(obj,nsim,seed,x,…)` | `obj$simulate(nsim,seed,x,…)` | `obj.simulate(nsim,seed,x,…)` | `simulate(obj,nsim,seed,x,…)` | `simulate(obj,nsim,seed,X;…)` |
| Update | `kriging_update(obj,y_u,X_u,…)` | `obj$update(y_u,X_u,…)` | `obj.update(y_u,X_u,…)` | `update(obj,y_u,X_u,…)` | `update!(obj,y_u,X_u;…)` |
| Update simulate | `kriging_update_simulate(obj,y_u,X_u)` | `obj$update_simulate(y_u,X_u)` | `obj.update_simulate(y_u,X_u)` | `update_simulate(obj,y_u,X_u)` | `update_simulate(obj,y_u,X_u)` |
| Save | `kriging_save(obj,file)` | `obj$save(file)` | `obj.save(file)` | `save(obj,file)` | `save(obj,file)` |
| Load | `kriging_load(file)` | `load.Kriging(file)` | `load_kriging(file)` | `load(file)` | `load_kriging(file)` |
| Summary | `kriging_summary(obj)` | `obj$print()` | `obj.summary()` | `summary(obj)` | `summary(obj)` |
| Covariance matrix | `kriging_covMat(obj,x1,x2)` | `obj$covMat(x1,x2)` | `obj.covMat(x1,x2)` | `covMat(obj,x1,x2)` | `cov_mat(obj,X1,X2)` |
| Log-likelihood | `kriging_logLikelihood(obj)` | `obj$logLikelihood()` | `obj.logLikelihood()` | `logLikelihood(obj)` | `log_likelihood(obj)` |
| Log-likelihood function | `kriging_logLikelihoodFun(obj,theta,…)` | `obj$logLikelihoodFun(theta,…)` | `obj.logLikelihoodFun(theta,…)` | `logLikelihoodFun(obj,theta,…)` | `log_likelihood_fun(obj,theta;…)` |
| Leave-one-out | `kriging_leaveOneOut(obj)` | `obj$leaveOneOut()` | `obj.leaveOneOut()` | — | `leave_one_out(obj)` |
| Leave-one-out function | `kriging_leaveOneOutFun(obj,theta,…)` | `obj$leaveOneOutFun(theta,…)` | `obj.leaveOneOutFun(theta,…)` | — | `leave_one_out_fun(obj,theta;…)` |
| Leave-one-out vec | `kriging_leaveOneOutVec(obj,theta)` | `obj$leaveOneOutVec(theta)` | `obj.leaveOneOutVec(theta)` | — | `leave_one_out_vec(obj,theta)` |
| Log-marginal-post | `kriging_logMargPost(obj)` | `obj$logMargPost()` | `obj.logMargPost()` | `logMargPost(obj)` | `log_marg_post(obj)` |
| Log-marginal-post function | `kriging_logMargPostFun(obj,theta,…)` | `obj$logMargPostFun(theta,…)` | `obj.logMargPostFun(theta,…)` | `logMargPostFun(obj,theta,…)` | `log_marg_post_fun(obj,theta;…)` |
| Model info | `kriging_model(obj)` | `obj$kernel()`, `obj$theta()`, … | `obj.model()` | `model(obj)` | `kernel(obj)`, `optim(obj)`, … |
| **Parameters** | | | | | |
| `kernel` | `kriging_kernel(obj)` | `obj$kernel()` | `obj.kernel()` | `kernel(obj)` | `kernel(obj)` |
| `optim` | `kriging_optim(obj)` | `obj$optim()` | `obj.optim()` | `optim(obj)` | `optim(obj)` |
| `objective` | `kriging_objective(obj)` | `obj$objective()` | `obj.objective()` | `objective(obj)` | `objective(obj)` |
| `X` | `kriging_X(obj)` | `obj$X()` | `obj.X()` | — | — |
| `centerX` / `scaleX` | `kriging_centerX(obj)` / `kriging_scaleX(obj)` | `obj$centerX()` / `obj$scaleX()` | `obj.centerX()` / `obj.scaleX()` | — | — |
| `y` | `kriging_y(obj)` | `obj$y()` | `obj.y()` | — | — |
| `centerY` / `scaleY` | `kriging_centerY(obj)` / `kriging_scaleY(obj)` | `obj$centerY()` / `obj$scaleY()` | `obj.centerY()` / `obj.scaleY()` | — | — |
| `normalize` | `kriging_normalize(obj)` | `obj$normalize()` | `obj.normalize()` | — | `normalize(obj)` |
| `regmodel` | `kriging_regmodel(obj)` | `obj$regmodel()` | `obj.regmodel()` | — | `regmodel(obj)` |
| `F`, `T`, `M`, `z` | `kriging_F(obj)`, … | `obj$F()`, … | `obj.F()`, … | — | — |
| `beta` / `is_beta_estim` | `kriging_beta(obj)` / `kriging_is_beta_estim(obj)` | `obj$beta()` / `obj$is_beta_estim()` | `obj.beta()` / `obj.is_beta_estim()` | — | — |
| `theta` / `is_theta_estim` | `kriging_theta(obj)` / `kriging_is_theta_estim(obj)` | `obj$theta()` / `obj$is_theta_estim()` | `obj.theta()` / `obj.is_theta_estim()` | — | — |
| `sigma2` / `is_sigma2_estim` | `kriging_sigma2(obj)` / `kriging_is_sigma2_estim(obj)` | `obj$sigma2()` / `obj$is_sigma2_estim()` | `obj.sigma2()` / `obj.is_sigma2_estim()` | — | — |
| `noise_model` | `kriging_noise_model(obj)` | `obj$noise_model()` | `obj.noise_model()` | — | — |
| `nugget` / `is_nugget_estim` | `kriging_nugget(obj)` / `kriging_is_nugget_estim(obj)` | `obj$nugget()` / `obj$is_nugget_estim()` | `obj.nugget()` / `obj.is_nugget_estim()` | — | — |
| `noise` | `kriging_noise(obj)` | `obj$noise()` | `obj.noise()` | `noise(obj)` | — |

---

## WarpKriging

| Method | R (C++ fn) | R (object method) | Python | Octave/Matlab | Julia |
|---|---|---|---|---|---|
| Constructor | `warpKriging_new(warping,kernel)` | — | `WarpKriging(warping,kernel)` | `build(y,X,warping,kernel,…)` | `WarpKriging(warping,kernel)` / `WarpKriging(y,X,…)` |
| Fit | `warpKriging_fit(obj,y,X,…)` | `obj$fit(y,X,…)` | `obj.fit(y,X,…)` | `fit(obj,y,X,…)` | `fit!(obj,y,X;…)` |
| Copy | `warpKriging_copy(obj)` | `obj$copy()` | `obj.copy()` | `copy(obj)` | `copy(obj)` |
| Predict | `warpKriging_predict(obj,x,…)` | `obj$predict(x,…)` | `obj.predict(x,…)` | `predict(obj,x,…)` | `predict(obj,X;…)` |
| Simulate | `warpKriging_simulate(obj,nsim,seed,x)` | `obj$simulate(nsim,seed,x)` | `obj.simulate(nsim,seed,x)` | `simulate(obj,nsim,seed,x)` | `simulate(obj,nsim,seed,X)` |
| Update | `warpKriging_update(obj,y_u,X_u,…)` | `obj$update(y_u,X_u,…)` | `obj.update(y_u,X_u,…)` | `update(obj,y_u,X_u,…)` | `update!(obj,y_u,X_u)` |
| Update simulate | `warpKriging_update_simulate(obj,y_u,X_u)` | `obj$update_simulate(y_u,X_u)` | `obj.update_simulate(y_u,X_u)` | `update_simulate(obj,y_u,X_u)` | — |
| Save | `warpKriging_save(obj,file)` | `obj$save(file)` | `obj.save(file)` | `save(obj,file)` | `save(obj,file)` |
| Load | `warpkriging_load(file)` | `load.WarpKriging(file)` | `load_warp_kriging(file)` | `load(file)` | — |
| Summary | `warpKriging_summary(obj)` | `obj$print()` | `obj.summary()` | `summary(obj)` | `summary(obj)` |
| Log-likelihood | `warpKriging_logLikelihood(obj)` | `obj$logLikelihood()` | `obj.logLikelihood()` | `logLikelihood(obj)` | `log_likelihood(obj)` |
| Log-likelihood function | `warpKriging_logLikelihoodFun(obj,theta,…)` | `obj$logLikelihoodFun(theta,…)` | `obj.logLikelihoodFun(theta,…)` | `logLikelihoodFun(obj,theta,…)` | `log_likelihood_fun(obj,theta;…)` |
| `kernel` | `warpKriging_kernel(obj)` | `obj$kernel()` | `obj.kernel()` | `kernel(obj)` | — |
| `warping` | `warpKriging_warping(obj)` | `obj$warping()` | `obj.warping()` | — | `warping(obj)` |
| `feature_dim` | `warpKriging_featureDim(obj)` | `obj$featureDim()` | `obj.feature_dim()` | — | — |
| `is_fitted` | `warpKriging_isFitted(obj)` | `obj$isFitted()` | `obj.is_fitted()` | `is_fitted(obj)` | — |
| `X`, `y`, `centerX/Y`, `scaleX/Y` | `warpKriging_X(obj)`, … | `obj$X()`, … | `obj.X()`, … | — | — |
| `normalize`, `regmodel` | `warpKriging_normalize(obj)`, … | `obj$normalize()`, … | `obj.normalize()`, … | — | — |
| `F`, `T`, `M`, `z`, `beta`, `theta`, `sigma2` | `warpKriging_F(obj)`, … | `obj$F()`, … | `obj.F()`, … | — | — |

---

## MLPKriging

| Method | R (C++ fn) | R (object method) | Python | Octave/Matlab | Julia |
|---|---|---|---|---|---|
| Constructor | `mlpKriging_new(hidden,d_out,kernel,warping)` | — | `MLPKriging(hidden,d_out,kernel,warping)` | `build(y,X,hidden,…)` | `MLPKriging(hidden,d_out;…)` / `MLPKriging(y,X,…)` |
| Fit | `mlpKriging_fit(obj,y,X,…)` | `obj$fit(y,X,…)` | `obj.fit(y,X,…)` | `fit(obj,y,X,…)` | `fit!(obj,y,X;…)` |
| Copy | `mlpKriging_copy(obj)` | `obj$copy()` | `obj.copy()` | — | `copy(obj)` |
| Predict | `mlpKriging_predict(obj,x,…)` | `obj$predict(x,…)` | `obj.predict(x,…)` | `predict(obj,x,…)` | `predict(obj,X;…)` |
| Simulate | `mlpKriging_simulate(obj,nsim,seed,x)` | `obj$simulate(nsim,seed,x)` | `obj.simulate(nsim,seed,x)` | `simulate(obj,nsim,seed,x)` | `simulate(obj,nsim,seed,X)` |
| Update | `mlpKriging_update(obj,y_u,X_u,…)` | `obj$update(y_u,X_u,…)` | `obj.update(y_u,X_u,…)` | `update(obj,y_u,X_u,…)` | `update!(obj,y_u,X_u)` |
| Update simulate | `mlpKriging_update_simulate(obj,y_u,X_u)` | `obj$update_simulate(y_u,X_u)` | `obj.update_simulate(y_u,X_u)` | `update_simulate(obj,y_u,X_u)` | — |
| Save | `mlpKriging_save(obj,file)` | `obj$save(file)` | `obj.save(file)` | `save(obj,file)` | `save(obj,file)` |
| Load | `mlpkriging_load(file)` | `load.MLPKriging(file)` | `load_mlp_kriging(file)` | `load(file)` | — |
| Summary | `mlpKriging_summary(obj)` | `obj$print()` | `obj.summary()` | `summary(obj)` | `summary(obj)` |
| Log-likelihood | `mlpKriging_logLikelihood(obj)` | `obj$logLikelihood()` | `obj.logLikelihood()` | `logLikelihood(obj)` | `log_likelihood(obj)` |
| Log-likelihood function | `mlpKriging_logLikelihoodFun(obj,theta,…)` | `obj$logLikelihoodFun(theta,…)` | `obj.logLikelihoodFun(theta,…)` | `logLikelihoodFun(obj,theta,…)` | `log_likelihood_fun(obj,theta;…)` |
| `kernel` | `mlpKriging_kernel(obj)` | `obj$kernel()` | `obj.kernel()` | `kernel(obj)` | — |
| `feature_dim` | `mlpKriging_featureDim(obj)` | `obj$featureDim()` | `obj.feature_dim()` | — | — |
| `hidden_dims` | `mlpKriging_hiddenDims(obj)` | `obj$hiddenDims()` | `obj.hidden_dims()` | — | `hidden_dims(obj)` |
| `activation` | `mlpKriging_activation(obj)` | `obj$activation()` | `obj.activation()` | — | — |
| `is_fitted` | `mlpKriging_isFitted(obj)` | `obj$isFitted()` | `obj.is_fitted()` | `is_fitted(obj)` | — |
| `X`, `y`, `centerX/Y`, `scaleX/Y` | `mlpKriging_X(obj)`, … | `obj$X()`, … | `obj.X()`, … | — | — |
| `normalize`, `regmodel` | `mlpKriging_normalize(obj)`, … | `obj$normalize()`, … | `obj.normalize()`, … | — | — |
| `F`, `T`, `M`, `z`, `beta`, `theta`, `sigma2` | `mlpKriging_F(obj)`, … | `obj$F()`, … | `obj.F()`, … | — | — |

---

## LinearRegression

| Method | R (C++ fn) | R (object method) | Python | Octave/Matlab | Julia |
|---|---|---|---|---|---|
| Constructor | — | `LinearRegression()` | `LinearRegression()` | `build(y,X)` | `LinearRegression()` |
| Fit | `linear_regression(y,X)` | — | `obj.fit(y,X)` | `fit(obj,y,X)` | `fit!(obj,y,X)` |
| Predict | `linear_regression_predict(obj,x)` | — | `obj.predict(x)` | `predict(obj,x)` | `predict(obj,X)` |
| Optimize | `linear_regression_optim(obj,…)` | — | — | — | — |

---

## Cross-language Load / Class detection

| Function | R | Python | Octave/Matlab | Julia |
|---|---|---|---|---|
| Detect saved class | `class_saved(file)` | — | — | — |
| Load Kriging | `kriging_load(file)` / `load.Kriging(file)` | `load_kriging(file)` | `load(file)` | `load_kriging(file)` |
| Load WarpKriging | `warpkriging_load(file)` / `load.WarpKriging(file)` | `load_warp_kriging(file)` | `load(file)` | — |
| Load MLPKriging | `mlpkriging_load(file)` / `load.MLPKriging(file)` | `load_mlp_kriging(file)` | `load(file)` | — |

---

## Octave/Matlab — Deprecated classes (still present)

The Octave/Matlab binding still provides separate `NoiseKriging` and `NuggetKriging` classes for backward compatibility. Use `Kriging` with `noise_model` instead.

| Method | NoiseKriging (Octave) | NuggetKriging (Octave) |
|---|---|---|
| build | `NoiseKriging_binding build` | `NuggetKriging_binding build` |
| fit / predict / simulate | ✅ | ✅ |
| update / update_simulate | ✅ | ✅ |
| logLikelihoodFun / logLikelihood | ✅ | ✅ |
| logMargPostFun / logMargPost | — | ✅ |
| covMat / model / summary | ✅ | ✅ |
| save / load / copy | ✅ | ✅ |
| `noise` accessor | ✅ | — |
| `nugget` / `is_nugget_estim` | — | ✅ |
