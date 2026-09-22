# libKriging Bindings — Method Reference

This document lists all methods exposed by each language binding for accessing the underlying C++ library.

## Classes

| Class | Description | R | Python | Octave/Matlab | Julia |
|---|---|:---:|:---:|:---:|:---:|
| `Kriging` | Unified Kriging; all bindings support `noise_model`: `none`, `nugget`, `heterogeneous` | ✅ | ✅ | ✅ | ✅ |
| `WarpKriging` | Kriging with input warping | ✅ | ✅ | ✅ | ✅ |
| `MLPKriging` | Kriging with MLP feature mapping | ✅ | ✅ | ✅ | ✅ |
| `NestedKriging` | Divide-and-conquer Kriging for large designs (see [docs/math/Nested.md](../docs/math/Nested.md)) | ✅ | ✅ | ✅ | ✅ |

> **Note on noise models**: `NoiseKriging` (heterogeneous noise) and `NuggetKriging` (nugget/homoscedastic noise) have been removed from all bindings — use `Kriging` with `noise_model="heterogeneous"` or `noise_model="nugget"`.

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
| Leave-one-out | `kriging_leaveOneOut(obj)` | `obj$leaveOneOut()` | `obj.leaveOneOut()` | `leaveOneOut(obj)` | `leave_one_out(obj)` |
| Leave-one-out function | `kriging_leaveOneOutFun(obj,theta,…)` | `obj$leaveOneOutFun(theta,…)` | `obj.leaveOneOutFun(theta,…)` | `leaveOneOutFun(obj,theta,…)` | `leave_one_out_fun(obj,theta;…)` |
| Leave-one-out vec | `kriging_leaveOneOutVec(obj,theta)` | `obj$leaveOneOutVec(theta)` | `obj.leaveOneOutVec(theta)` | `leaveOneOutVec(obj,theta)` | `leave_one_out_vec(obj,theta)` |
| Log-marginal-post | `kriging_logMargPost(obj)` | `obj$logMargPost()` | `obj.logMargPost()` | `logMargPost(obj)` | `log_marg_post(obj)` |
| Log-marginal-post function | `kriging_logMargPostFun(obj,theta,…)` | `obj$logMargPostFun(theta,…)` | `obj.logMargPostFun(theta,…)` | `logMargPostFun(obj,theta,…)` | `log_marg_post_fun(obj,theta;…)` |
| Model info | `kriging_model(obj)` | `obj$kernel()`, `obj$theta()`, … | `obj.model()` | `model(obj)` | `kernel(obj)`, `optim(obj)`, … |
| **Parameters** | | | | | |
| `kernel` | `kriging_kernel(obj)` | `obj$kernel()` | `obj.kernel()` | `kernel(obj)` | `kernel(obj)` |
| `optim` | `kriging_optim(obj)` | `obj$optim()` | `obj.optim()` | `optim(obj)` | `optim(obj)` |
| `objective` | `kriging_objective(obj)` | `obj$objective()` | `obj.objective()` | `objective(obj)` | `objective(obj)` |
| `X` | `kriging_X(obj)` | `obj$X()` | `obj.X()` | `X(obj)` | `X(obj)` |
| `centerX` / `scaleX` | `kriging_centerX(obj)` / `kriging_scaleX(obj)` | `obj$centerX()` / `obj$scaleX()` | `obj.centerX()` / `obj.scaleX()` | `centerX(obj)` / `scaleX(obj)` | `centerX(obj)` / `scaleX(obj)` |
| `y` | `kriging_y(obj)` | `obj$y()` | `obj.y()` | `y(obj)` | `y(obj)` |
| `centerY` / `scaleY` | `kriging_centerY(obj)` / `kriging_scaleY(obj)` | `obj$centerY()` / `obj$scaleY()` | `obj.centerY()` / `obj.scaleY()` | `centerY(obj)` / `scaleY(obj)` | `centerY(obj)` / `scaleY(obj)` |
| `normalize` | `kriging_normalize(obj)` | `obj$normalize()` | `obj.normalize()` | `normalize(obj)` | `normalize(obj)` |
| `regmodel` | `kriging_regmodel(obj)` | `obj$regmodel()` | `obj.regmodel()` | `regmodel(obj)` | `regmodel(obj)` |
| `F`, `T`, `M`, `z` | `kriging_F(obj)`, … | `obj$F()`, … | `obj.F()`, … | `F(obj)`, `T(obj)`, `M(obj)`, `z(obj)` | `F(obj)`, `T(obj)`, `M(obj)`, `z(obj)` |
| `beta` / `is_beta_estim` | `kriging_beta(obj)` / `kriging_is_beta_estim(obj)` | `obj$beta()` / `obj$is_beta_estim()` | `obj.beta()` / `obj.is_beta_estim()` | `beta(obj)` / `is_beta_estim(obj)` | `beta(obj)` / `is_beta_estim(obj)` |
| `theta` / `is_theta_estim` | `kriging_theta(obj)` / `kriging_is_theta_estim(obj)` | `obj$theta()` / `obj$is_theta_estim()` | `obj.theta()` / `obj.is_theta_estim()` | `theta(obj)` / `is_theta_estim(obj)` | `theta(obj)` / `is_theta_estim(obj)` |
| `sigma2` / `is_sigma2_estim` | `kriging_sigma2(obj)` / `kriging_is_sigma2_estim(obj)` | `obj$sigma2()` / `obj$is_sigma2_estim()` | `obj.sigma2()` / `obj.is_sigma2_estim()` | `sigma2(obj)` / `is_sigma2_estim(obj)` | `sigma2(obj)` / `is_sigma2_estim(obj)` |
| `noise_model` | `kriging_noise_model(obj)` | `obj$noise_model()` | `obj.noise_model()` | `noise_model(obj)` | `noise_model(obj)` |
| `nugget` / `is_nugget_estim` | `kriging_nugget(obj)` / `kriging_is_nugget_estim(obj)` | `obj$nugget()` / `obj$is_nugget_estim()` | `obj.nugget()` / `obj.is_nugget_estim()` | `nugget(obj)` / `is_nugget_estim(obj)` | `nugget(obj)` / `is_nugget_estim(obj)` |
| `noise` | `kriging_noise(obj)` | `obj$noise()` | `obj.noise()` | `noise(obj)` | `noise(obj)` |
| `nystrom_rank` | `kriging_nystrom_rank(obj)` | `obj$nystrom_rank()` | `obj.nystrom_rank()` | `nystrom_rank(obj)` | `nystrom_rank(obj)` |
| **Pre-fit data reduction** | | | | | |
| `subsetOfData` | `kriging_subsetOfData(X,n_max,method,seed)` | `subsetOfData(X,n_max,method,seed)` (plain function) | `Kriging.subsetOfData(X,n_max,method,seed)` (static) | `Kriging.subsetOfData(X,int32(n_max),method,int32(seed))` (static) | `subsetOfData(X,n_max;method,seed)` |

`subsetOfData` picks `n_max` representative rows of `X` (`method="kmeans"`, default: k-means centroids snapped to the
nearest real observation; or `"random"`) to fit on a reduced design; see [docs/math/SubsetOfData.md](../docs/math/SubsetOfData.md).
It returns row indices into `X`, **0-based in Python and Julia, 1-based in R and Octave/Matlab**; the Python result is an
`(n_max, 1)` integer array. Keep the matching `y` entries.

### Fit objectives

`Kriging` accepts `objective` = `"LL"` (default), `"LOO"`, `"LMP"`, and the two large-`n` approximations
`"LLVecchia"` / `"LLVecchia(m)"` (Vecchia, see [docs/math/Vecchia.md](../docs/math/Vecchia.md)) and `"LLNystrom"` /
`"LLNystrom(k)"` (Nystrom low-rank, see [docs/math/Nystrom.md](../docs/math/Nystrom.md)). `nystrom_rank()` returns the
rank `k` of a `LLNystrom` fit (0 otherwise). `"VLL"` / `"VLL(m)"`, the pre-1.2 spelling of the Vecchia objective, is no
longer accepted. Both approximations require the noise-free model (`noise_model` `none`). `WarpKriging` fits with `"LL"`
only.

> **C++ only**: the following `Kriging` methods are not exposed by any binding: `predictVecchia`, `predictNystrom`,
> `simulateNystrom`, `set_vecchia_exact_commit` / `vecchia_exact_commit` (the factorization-free "light" Vecchia mode),
> `is_vecchia_light`, `vecchia_neighbors`, `is_nystrom_light`, `logLikelihoodVecchiaFun` and `logLikelihoodNystromFun`.
> From a binding, `predict` is the only prediction entry point.

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
| Update simulate | `warpKriging_update_simulate(obj,y_u,X_u)` | `obj$update_simulate(y_u,X_u)` | `obj.update_simulate(y_u,X_u)` | `update_simulate(obj,y_u,X_u)` | `update_simulate(obj,y_u,X_u)` |
| Save | `warpKriging_save(obj,file)` | `obj$save(file)` | `obj.save(file)` | `save(obj,file)` | `save(obj,file)` |
| Load | `warpkriging_load(file)` | `load.WarpKriging(file)` | `load_warp_kriging(file)` | `load(file)` | `load_warp_kriging(file)` |
| Summary | `warpKriging_summary(obj)` | `obj$print()` | `obj.summary()` | `summary(obj)` | `summary(obj)` |
| Log-likelihood | `warpKriging_logLikelihood(obj)` | `obj$logLikelihood()` | `obj.logLikelihood()` | `logLikelihood(obj)` | `log_likelihood(obj)` |
| Log-likelihood function | `warpKriging_logLikelihoodFun(obj,theta,…)` | `obj$logLikelihoodFun(theta,…)` | `obj.logLikelihoodFun(theta,…)` | `logLikelihoodFun(obj,theta,…)` | `log_likelihood_fun(obj,theta;…)` |
| `kernel` | `warpKriging_kernel(obj)` | `obj$kernel()` | `obj.kernel()` | `kernel(obj)` | `kernel(obj)` |
| `warping` | `warpKriging_warping(obj)` | `obj$warping()` | `obj.warping()` | `warping(obj)` | `warping(obj)` |
| `feature_dim` | `warpKriging_featureDim(obj)` | `obj$featureDim()` | `obj.feature_dim()` | `feature_dim(obj)` | `feature_dim(obj)` |
| `is_fitted` | `warpKriging_isFitted(obj)` | `obj$isFitted()` | `obj.is_fitted()` | `is_fitted(obj)` | `is_fitted(obj)` |
| `optim` / `objective` | `warpKriging_optim(obj)` / `warpKriging_objective(obj)` | `obj$optim()` / `obj$objective()` | `obj.optim()` / `obj.objective()` | `optim(obj)` / `objective(obj)` | `optim(obj)` / `objective(obj)` |
| `noise` | `warpKriging_noise(obj)` | `obj$noise()` | `obj.noise()` | `noise(obj)` | `noise(obj)` |
| `warp_params` | `warpKriging_warpParams(obj)` | `obj$warp_params()` | `obj.warp_params()` | `warp_params(obj)` | `warp_params(obj)` |
| Covariance matrix | `warpKriging_covMat(obj,X1,X2)` | `obj$covMat(X1,X2)` | `obj.covMat(X1,X2)` | `covMat(obj,X1,X2)` | `cov_mat(obj,X1,X2)` |
| `X` / `y` | `warpKriging_X(obj)` / `warpKriging_y(obj)` | `obj$X()` / `obj$y()` | `obj.X()` / `obj.y()` | `X(obj)` / `y(obj)` | `X(obj)` / `y(obj)` |
| `centerX` / `scaleX` | `warpKriging_centerX(obj)` / `warpKriging_scaleX(obj)` | `obj$centerX()` / `obj$scaleX()` | `obj.centerX()` / `obj.scaleX()` | `centerX(obj)` / `scaleX(obj)` | `centerX(obj)` / `scaleX(obj)` |
| `centerY` / `scaleY` | `warpKriging_centerY(obj)` / `warpKriging_scaleY(obj)` | `obj$centerY()` / `obj$scaleY()` | `obj.centerY()` / `obj.scaleY()` | `centerY(obj)` / `scaleY(obj)` | `centerY(obj)` / `scaleY(obj)` |
| `normalize` | `warpKriging_normalize(obj)` | `obj$normalize()` | `obj.normalize()` | `normalize(obj)` | `normalize(obj)` |
| `regmodel` | `warpKriging_regmodel(obj)` | `obj$regmodel()` | `obj.regmodel()` | `regmodel(obj)` | `regmodel(obj)` |
| `F`, `T`, `M`, `z` | `warpKriging_F(obj)`, … | `obj$F()`, … | `obj.F()`, … | `F(obj)`, `T(obj)`, `M(obj)`, `z(obj)` | `F(obj)`, `T(obj)`, `M(obj)`, `z(obj)` |
| `beta`, `theta`, `sigma2` | `warpKriging_beta(obj)`, … | `obj$beta()`, … | `obj.beta()`, … | `beta(obj)`, `theta(obj)`, `sigma2(obj)` | `beta(obj)`, `theta(obj)`, `sigma2(obj)` |

> `WarpKriging` constructor / `fit` accept `noise=` as a per-observation variance vector (there is no `"nugget"` mode),
> and `parameters=` with numeric `theta`, `warp_params` and `noise` seeds; with `optim="none"` they freeze the
> hyper-parameters. `update` and `update_simulate` accept `noise_u=` (variances of the new points) when the model was
> fitted with noise. `WarpKriging` always fits with `objective="LL"`.

---

## MLPKriging

| Method | R (C++ fn) | R (object method) | Python | Octave/Matlab | Julia |
|---|---|---|---|---|---|
| Constructor | `mlpKriging_new(hidden,d_out,kernel,warping)` | — | `MLPKriging(hidden,d_out,kernel,warping)` | `build(y,X,hidden,…)` | `MLPKriging(hidden,d_out;…)` / `MLPKriging(y,X,…)` |
| Fit | `mlpKriging_fit(obj,y,X,…)` | `obj$fit(y,X,…)` | `obj.fit(y,X,…)` | `fit(obj,y,X,…)` | `fit!(obj,y,X;…)` |
| Copy | `mlpKriging_copy(obj)` | `obj$copy()` | `obj.copy()` | `copy(obj)` | `copy(obj)` |
| Predict | `mlpKriging_predict(obj,x,…)` | `obj$predict(x,…)` | `obj.predict(x,…)` | `predict(obj,x,…)` | `predict(obj,X;…)` |
| Simulate | `mlpKriging_simulate(obj,nsim,seed,x)` | `obj$simulate(nsim,seed,x)` | `obj.simulate(nsim,seed,x)` | `simulate(obj,nsim,seed,x)` | `simulate(obj,nsim,seed,X)` |
| Update | `mlpKriging_update(obj,y_u,X_u,…)` | `obj$update(y_u,X_u,…)` | `obj.update(y_u,X_u,…)` | `update(obj,y_u,X_u,…)` | `update!(obj,y_u,X_u)` |
| Update simulate | `mlpKriging_update_simulate(obj,y_u,X_u)` | `obj$update_simulate(y_u,X_u)` | `obj.update_simulate(y_u,X_u)` | `update_simulate(obj,y_u,X_u)` | `update_simulate(obj,y_u,X_u)` |
| Save | `mlpKriging_save(obj,file)` | `obj$save(file)` | `obj.save(file)` | `save(obj,file)` | `save(obj,file)` |
| Load | `mlpkriging_load(file)` | `load.MLPKriging(file)` | `load_mlp_kriging(file)` | `load(file)` | `load_mlp_kriging(file)` |
| Summary | `mlpKriging_summary(obj)` | `obj$print()` | `obj.summary()` | `summary(obj)` | `summary(obj)` |
| Log-likelihood | `mlpKriging_logLikelihood(obj)` | `obj$logLikelihood()` | `obj.logLikelihood()` | `logLikelihood(obj)` | `log_likelihood(obj)` |
| Log-likelihood function | `mlpKriging_logLikelihoodFun(obj,theta,…)` | `obj$logLikelihoodFun(theta,…)` | `obj.logLikelihoodFun(theta,…)` | `logLikelihoodFun(obj,theta,…)` | `log_likelihood_fun(obj,theta;…)` |
| `kernel` | `mlpKriging_kernel(obj)` | `obj$kernel()` | `obj.kernel()` | `kernel(obj)` | `kernel(obj)` |
| `feature_dim` | `mlpKriging_featureDim(obj)` | `obj$featureDim()` | `obj.feature_dim()` | `feature_dim(obj)` | `feature_dim(obj)` |
| `hidden_dims` | `mlpKriging_hiddenDims(obj)` | `obj$hiddenDims()` | `obj.hidden_dims()` | `hidden_dims(obj)` | `hidden_dims(obj)` |
| `activation` | `mlpKriging_activation(obj)` | `obj$activation()` | `obj.activation()` | `activation(obj)` | `activation(obj)` |
| `is_fitted` | `mlpKriging_isFitted(obj)` | `obj$isFitted()` | `obj.is_fitted()` | `is_fitted(obj)` | `is_fitted(obj)` |
| `X` / `y` | `mlpKriging_X(obj)` / `mlpKriging_y(obj)` | `obj$X()` / `obj$y()` | `obj.X()` / `obj.y()` | `X(obj)` / `y(obj)` | `X(obj)` / `y(obj)` |
| `centerX` / `scaleX` | `mlpKriging_centerX(obj)` / `mlpKriging_scaleX(obj)` | `obj$centerX()` / `obj$scaleX()` | `obj.centerX()` / `obj.scaleX()` | `centerX(obj)` / `scaleX(obj)` | `centerX(obj)` / `scaleX(obj)` |
| `centerY` / `scaleY` | `mlpKriging_centerY(obj)` / `mlpKriging_scaleY(obj)` | `obj$centerY()` / `obj$scaleY()` | `obj.centerY()` / `obj.scaleY()` | `centerY(obj)` / `scaleY(obj)` | `centerY(obj)` / `scaleY(obj)` |
| `normalize` | `mlpKriging_normalize(obj)` | `obj$normalize()` | `obj.normalize()` | `normalize(obj)` | `normalize(obj)` |
| `regmodel` | `mlpKriging_regmodel(obj)` | `obj$regmodel()` | `obj.regmodel()` | `regmodel(obj)` | `regmodel(obj)` |
| `F`, `T`, `M`, `z` | `mlpKriging_F(obj)`, … | `obj$F()`, … | `obj.F()`, … | `F(obj)`, `T(obj)`, `M(obj)`, `z(obj)` | `F(obj)`, `T(obj)`, `M(obj)`, `z(obj)` |
| `beta`, `theta`, `sigma2` | `mlpKriging_beta(obj)`, … | `obj$beta()`, … | `obj.beta()`, … | `beta(obj)`, `theta(obj)`, `sigma2(obj)` | `beta(obj)`, `theta(obj)`, `sigma2(obj)` |

---

## NestedKriging

| Method | R (C++ fn) | R (object method) | Python | Octave/Matlab | Julia |
|---|---|---|---|---|---|
| Constructor + fit | `new_NestedKrigingFit(y,X,kernel,nb_groups,…)` | — | `NestedKriging(y,X,kernel,nb_groups=…,…)` | `build(y,X,kernel,nb_groups,…)` | `NestedKriging(y,X,kernel,nb_groups;…)` |
| Predict | `nestedkriging_predict(obj,x,…)` | `obj$predict(x,…)` | `obj.predict(x,…)` | `predict(obj,x,…)` | `predict(obj,X;…)` |
| Summary | `nestedkriging_summary(obj)` | `obj$print()` | `obj.summary()` | `summary(obj)` | `summary(obj)` |
| `kernel` | `nestedkriging_kernel(obj)` | `obj$kernel()` | `obj.kernel()` | `kernel(obj)` | `kernel(obj)` |
| `aggregation` | `nestedkriging_aggregation(obj)` | `obj$aggregation()` | `obj.aggregation()` | `aggregation(obj)` | `aggregation(obj)` |
| `nb_groups` | `nestedkriging_nb_groups(obj)` | `obj$nb_groups()` | `obj.nb_groups()` | `nb_groups(obj)` | `nb_groups(obj)` |
| `groups` | `nestedkriging_groups(obj)` | `obj$groups()` | `obj.groups()` | — | — |
| `theta` | `nestedkriging_theta(obj)` | `obj$theta()` | `obj.theta()` | `theta(obj)` | `theta(obj)` |
| `sigma2` | `nestedkriging_sigma2(obj)` | `obj$sigma2()` | `obj.sigma2()` | `sigma2(obj)` | `sigma2(obj)` |
| `beta0` | `nestedkriging_beta0(obj)` | `obj$beta0()` | `obj.beta0()` | `beta0(obj)` | `beta0(obj)` |
| `warping` | `nestedkriging_warping(obj)` | `obj$warping()` | `obj.warping()` | — | — |
| `X` / `y` | `nestedkriging_X(obj)` / `nestedkriging_y(obj)` | `obj$X()` / `obj$y()` | `obj.X()` / `obj.y()` | — | — |
| `set_predict_chunk` | — | — | `obj.set_predict_chunk(chunk)` | — | — |
| `set_warp_subsample` | — | — | `obj.set_warp_subsample(m)` | — | — |

> No `noise=`, no `normalize=`, no `save()`/`load()` yet on `NestedKriging` — see [docs/math/Nested.md](../docs/math/Nested.md) for current limitations.

---

## Python: scikit-learn estimators

`pylibkriging.sklearn` (`pip install pylibkriging[sklearn]`) wraps each class as a scikit-learn regressor implementing
`fit` / `predict`, `get_params` / `set_params` and `clone`, so it works in `Pipeline`, `GridSearchCV` and
`cross_val_score`:

| Estimator | Wraps |
|---|---|
| `KrigingRegressor` | `Kriging` |
| `WarpKrigingRegressor` | `WarpKriging` |
| `MLPKrigingRegressor` | `MLPKriging` |
| `NestedKrigingRegressor` | `NestedKriging` |

See [bindings/Python/README.md](Python/README.md#scikit-learn-compatible-estimators) for an example.

---

## Cross-language Load / Class detection

| Function | R | Python | Octave/Matlab | Julia |
|---|---|---|---|---|
| Detect saved class | `class_saved(file)` → `"Kriging"`, `"WarpKriging"`, `"MLPKriging"` | — | `class_saved(file)` | — |
| Generic load | `load(file)` | `load(file)` | `load_kriging(file)` | `load(file)` |
| Load Kriging | `kriging_load(file)` / `load.Kriging(file)` | `load_kriging(file)` | `Kriging.load(file)` | `load_kriging(file)` |
| Load WarpKriging | `warpkriging_load(file)` / `load.WarpKriging(file)` | `load_warp_kriging(file)` | `WarpKriging.load(file)` | `load_warp_kriging(file)` |
| Load MLPKriging | `mlpkriging_load(file)` / `load.MLPKriging(file)` | `load_mlp_kriging(file)` | `MLPKriging.load(file)` | `load_mlp_kriging(file)` |

---

---

## Worked notebooks

Each notebook fits the Branin 2D function with one class or option, in each language.

| Example | Python | R | Julia | Octave |
|---|---|---|---|---|
| `Kriging` | [Python](Python/kriging_branin2d_py.ipynb) | [R](R/kriging_branin2d_r.ipynb) | [Julia](Julia/kriging_branin2d_julia.ipynb) | [Octave](Octave/kriging_branin2d_octave.ipynb) |
| `Kriging` with `objective="LLVecchia(m)"` | [Python](Python/kriging_vecchia_branin2d_py.ipynb) | [R](R/kriging_vecchia_branin2d_r.ipynb) | [Julia](Julia/kriging_vecchia_branin2d_julia.ipynb) | [Octave](Octave/kriging_vecchia_branin2d_octave.ipynb) |
| `NestedKriging` | [Python](Python/nestedkriging_branin2d_py.ipynb) | [R](R/nestedkriging_branin2d_r.ipynb) | [Julia](Julia/nestedkriging_branin2d_julia.ipynb) | [Octave](Octave/nestedkriging_branin2d_octave.ipynb) |
| `Kriging` with an estimated nugget (`noise="nugget"`) | [Python](Python/nuggetkriging_branin2d_py.ipynb) | [R](R/nuggetkriging_branin2d_r.ipynb) | [Julia](Julia/nuggetkriging_branin2d_julia.ipynb) | — |
| `Kriging` with known noise variances (`noise=` vector) | [Python](Python/noisekriging_branin2d_py.ipynb) | [R](R/noisekriging_branin2d_r.ipynb) | [Julia](Julia/noisekriging_branin2d_julia.ipynb) | — |
| `MLPKriging` | [Python](Python/mlpkriging_branin2d_py.ipynb) | [R](R/mlpkriging_branin2d_r.ipynb) | [Julia](Julia/mlpkriging_branin2d_julia.ipynb) | [Octave](Octave/mlpkriging_branin2d_octave.ipynb) |
| `WarpKriging`, `none` warping | [Python](Python/warpkriging_none_branin2d_py.ipynb) | [R](R/warpkriging_none_branin2d_r.ipynb) | [Julia](Julia/warpkriging_none_branin2d_julia.ipynb) | [Octave](Octave/warpkriging_none_branin2d_octave.ipynb) |
| `WarpKriging`, `affine` warping | [Python](Python/warpkriging_affine_branin2d_py.ipynb) | [R](R/warpkriging_affine_branin2d_r.ipynb) | [Julia](Julia/warpkriging_affine_branin2d_julia.ipynb) | [Octave](Octave/warpkriging_affine_branin2d_octave.ipynb) |
| `WarpKriging`, `boxcox` warping | [Python](Python/warpkriging_boxcox_branin2d_py.ipynb) | [R](R/warpkriging_boxcox_branin2d_r.ipynb) | [Julia](Julia/warpkriging_boxcox_branin2d_julia.ipynb) | [Octave](Octave/warpkriging_boxcox_branin2d_octave.ipynb) |
| `WarpKriging`, `kumaraswamy` warping | [Python](Python/warpkriging_kumaraswamy_branin2d_py.ipynb) | [R](R/warpkriging_kumaraswamy_branin2d_r.ipynb) | [Julia](Julia/warpkriging_kumaraswamy_branin2d_julia.ipynb) | [Octave](Octave/warpkriging_kumaraswamy_branin2d_octave.ipynb) |
| `WarpKriging`, `neural_mono` warping | [Python](Python/warpkriging_neural_mono_branin2d_py.ipynb) | [R](R/warpkriging_neural_mono_branin2d_r.ipynb) | [Julia](Julia/warpkriging_neural_mono_branin2d_julia.ipynb) | [Octave](Octave/warpkriging_neural_mono_branin2d_octave.ipynb) |
| `WarpKriging`, `knots` warping | [Python](Python/warpkriging_knots_branin2d_py.ipynb) | [R](R/warpkriging_knots_branin2d_r.ipynb) | [Julia](Julia/warpkriging_knots_branin2d_julia.ipynb) | [Octave](Octave/warpkriging_knots_branin2d_octave.ipynb) |
| `WarpKriging`, `mlp` warping | [Python](Python/warpkriging_mlp_branin2d_py.ipynb) | [R](R/warpkriging_mlp_branin2d_r.ipynb) | [Julia](Julia/warpkriging_mlp_branin2d_julia.ipynb) | [Octave](Octave/warpkriging_mlp_branin2d_octave.ipynb) |
| `WarpKriging`, `categorical` warping | [Python](Python/warpkriging_categorical_branin2d_py.ipynb) | [R](R/warpkriging_categorical_branin2d_r.ipynb) | [Julia](Julia/warpkriging_categorical_branin2d_julia.ipynb) | [Octave](Octave/warpkriging_categorical_branin2d_octave.ipynb) |
| `WarpKriging`, `ordinal` warping | [Python](Python/warpkriging_ordinal_branin2d_py.ipynb) | [R](R/warpkriging_ordinal_branin2d_r.ipynb) | [Julia](Julia/warpkriging_ordinal_branin2d_julia.ipynb) | [Octave](Octave/warpkriging_ordinal_branin2d_octave.ipynb) |

The `nuggetkriging_*` and `noisekriging_*` notebooks keep the names of the classes that were merged into `Kriging`; they
use `Kriging` with `noise=`. Other notebooks: [docs/math](../docs/math) (large-design methods against exact Cholesky) and
[docs/comparisons](../docs/comparisons) (other packages).
