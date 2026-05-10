## *************************************************************************
##  MLPKriging S3 class for rlibkriging
##
##  Deep Kernel Learning: Kriging with a joint MLP feature extractor
##  Phi : R^d -> R^{d_out}, k(x, x') = sigma^2 * k_base(Phi(x), Phi(x'); theta)
## *************************************************************************

## Register MLPKriging as a known S3 class to the S4 system so that
## packages that override generics (e.g. RobustGaSP overriding simulate)
## can still dispatch to the S3 simulate.MLPKriging method.
setOldClass("MLPKriging")

#' @title Create an MLPKriging model (Deep Kernel Learning)
#'
#' @description Kriging with a joint multi-layer perceptron applied to all
#'   inputs before the GP kernel is evaluated. The MLP weights, GP range
#'   parameters, variance and trend are jointly fitted by maximising the
#'   concentrated log-likelihood.
#'
#' @param y numeric vector of observations (n)
#' @param X numeric matrix of inputs (n x d)
#' @param hidden_dims integer vector of hidden layer sizes, e.g. \code{c(32, 16)}
#' @param d_out output feature dimensionality (default 2)
#' @param activation activation function: "relu", "selu", "tanh", "sigmoid", "elu"
#' @param kernel covariance kernel: "gauss", "matern3_2", "matern5_2", "exp"
#' @param regmodel trend: "constant", "linear", "quadratic"
#' @param normalize logical; normalise inputs?
#' @param optim optimiser (default "BFGS+Adam")
#' @param objective "LL" (log-likelihood)
#' @param parameters optional named list of tuning parameters, e.g.
#'   \code{list(max_iter_adam = "300", adam_lr = "0.001", max_iter_bfgs = "50")}
#'
#' @return An S3 object of class "MLPKriging".
#'
#' @examples
#' X <- as.matrix(seq(0.01, 0.99, length.out = 10))
#' f <- function(x) 1 - 1/2 * (sin(12*x)/(1+x) + 2*cos(7*x)*x^5 + 0.7)
#' y <- f(X)
#' k <- MLPKriging(y, X, hidden_dims = c(16, 8), d_out = 2,
#'                 activation = "selu", kernel = "gauss")
#' print(k)
#'
#' @export
MLPKriging <- function(y, X, hidden_dims,
                       d_out = 2,
                       activation = "selu",
                       kernel = "gauss",
                       regmodel = "constant",
                       normalize = FALSE,
                       optim = "BFGS+Adam",
                       objective = "LL",
                       parameters = NULL) {
  y <- as.numeric(y)
  X <- as.matrix(X)
  hidden_dims <- as.integer(hidden_dims)

  ptr <- mlpKriging_new(y, X, hidden_dims, as.integer(d_out),
                        activation, kernel, regmodel, normalize,
                        optim, objective, parameters)
  obj <- list(ptr = ptr)
  class(obj) <- "MLPKriging"
  return(obj)
}

# -----------------------------------------------------------------------
#  S3 methods
# -----------------------------------------------------------------------

#' @method print MLPKriging
#' @export
print.MLPKriging <- function(x, ...) {
  cat(mlpKriging_summary(x$ptr))
  invisible(x)
}

#' @method summary MLPKriging
#' @export
summary.MLPKriging <- function(object, ...) {
  mlpKriging_summary(object$ptr)
}

#' @title Fit an MLPKriging model to data
#'
#' @description (Re-)fit an already-constructed MLPKriging object on new
#'   data.  The MLP architecture and kernel are kept from construction.
#'
#' @param object MLPKriging object
#' @param y numeric vector of observations (n)
#' @param X numeric matrix of inputs (n x d)
#' @param regmodel trend: "constant", "linear", "quadratic"
#' @param normalize logical; normalise inputs?
#' @param optim optimiser
#' @param objective "LL" (log-likelihood)
#' @param parameters optional named list of tuning parameters
#' @param ... ignored
#'
#' @return No return value. MLPKriging object argument is modified.
#'
#' @method fit MLPKriging
#' @export
fit.MLPKriging <- function(object, y, X,
                           regmodel = "constant",
                           normalize = FALSE,
                           optim = "BFGS+Adam",
                           objective = "LL",
                           parameters = NULL, ...) {
  mlpKriging_fit(object$ptr,
                 as.numeric(y), as.matrix(X),
                 regmodel, normalize, optim, objective,
                 parameters)
  invisible(NULL)
}

#' @title Predict with an MLPKriging model
#' @param object MLPKriging object
#' @param x prediction matrix (m x d)
#' @param return_stdev return standard deviations?
#' @param return_cov return full covariance?
#' @param return_deriv return derivatives of mean and stdev wrt x?
#' @param ... ignored
#' @return list with \code{mean}, optionally \code{stdev}, \code{cov},
#'   \code{mean_deriv}, \code{stdev_deriv}
#' @method predict MLPKriging
#' @export
predict.MLPKriging <- function(object, x, return_stdev = TRUE, return_cov = FALSE,
                               return_deriv = FALSE, ...) {
  mlpKriging_predict(object$ptr, as.matrix(x), return_stdev, return_cov, return_deriv)
}

#' @title Simulate from an MLPKriging model
#' @param object MLPKriging object
#' @param nsim number of simulations
#' @param seed random seed
#' @param x simulation matrix (m x d)
#' @param ... ignored
#' @return matrix (m x nsim)
#' @method simulate MLPKriging
#' @export
simulate.MLPKriging <- function(object, nsim = 1, seed = 123, x,
                                 will_update = FALSE, ...) {
  mlpKriging_simulate(object$ptr, as.integer(nsim),
                      as.integer(seed), as.matrix(x),
                      as.logical(will_update))
}

#' @title Update simulated paths with new observations (FOXY algorithm)
#' @param object MLPKriging object (must have called simulate with will_update=TRUE)
#' @param y_u new observations
#' @param X_u new input matrix
#' @param ... ignored
#' @return matrix (m x nsim) of updated simulated paths
#' @method update_simulate MLPKriging
#' @export
update_simulate.MLPKriging <- function(object, y_u, X_u, ...) {
  mlpKriging_update_simulate(object$ptr, as.numeric(y_u), as.matrix(X_u))
}

#' @title Update an MLPKriging model with new observations
#' @param refit Logical. If \code{TRUE} the model is refitted (default is TRUE).
#' @method update MLPKriging
#' @export
update.MLPKriging <- function(object, y_u, X_u, refit = TRUE, ...) {
  mlpKriging_update(object$ptr, as.numeric(y_u), as.matrix(X_u), as.logical(refit))
  invisible(object)
}

#' @method logLikelihood MLPKriging
#' @export
logLikelihood.MLPKriging <- function(object, ...) {
  mlpKriging_logLikelihood(object$ptr)
}

#' @title Evaluate log-likelihood at given GP theta
#' @method logLikelihoodFun MLPKriging
#' @export
logLikelihoodFun.MLPKriging <- function(object, theta, return_grad = FALSE, return_hess = FALSE, ...) {
  mlpKriging_logLikelihoodFun(object$ptr, theta, return_grad, return_hess)
}

#' @method theta MLPKriging
#' @export
theta.MLPKriging <- function(object, ...) {
  mlpKriging_theta(object$ptr)
}

#' @method sigma2 MLPKriging
#' @export
sigma2.MLPKriging <- function(object, ...) {
  mlpKriging_sigma2(object$ptr)
}

#' @method kernel MLPKriging
#' @export
kernel.MLPKriging <- function(object, ...) {
  mlpKriging_kernel(object$ptr)
}

#' @title Get feature dimensionality (d_out)
#' @export
feature_dim <- function(object, ...) UseMethod("feature_dim")

#' @method feature_dim MLPKriging
#' @export
feature_dim.MLPKriging <- function(object, ...) {
  mlpKriging_featureDim(object$ptr)
}

#' @title Get hidden layer sizes
#' @export
hidden_dims <- function(object, ...) UseMethod("hidden_dims")

#' @method hidden_dims MLPKriging
#' @export
hidden_dims.MLPKriging <- function(object, ...) {
  mlpKriging_hiddenDims(object$ptr)
}

#' @title Get activation function name
#' @export
activation <- function(object, ...) UseMethod("activation")

#' @method activation MLPKriging
#' @export
activation.MLPKriging <- function(object, ...) {
  mlpKriging_activation(object$ptr)
}

#' @method is_fitted MLPKriging
#' @export
is_fitted.MLPKriging <- function(object, ...) {
  mlpKriging_isFitted(object$ptr)
}

#' @title Get training input matrix
#' @param object MLPKriging object
#' @param ... ignored
#' @return matrix of training inputs
#' @method X MLPKriging
#' @export
X.MLPKriging <- function(object, ...) {
  mlpKriging_X(object$ptr)
}

#' @title Get training output vector
#' @param object MLPKriging object
#' @param ... ignored
#' @return vector of training outputs
#' @method y MLPKriging
#' @export
y.MLPKriging <- function(object, ...) {
  mlpKriging_y(object$ptr)
}

#' @method centerX MLPKriging
#' @export
centerX.MLPKriging <- function(object, ...) {
  mlpKriging_centerX(object$ptr)
}

#' @method scaleX MLPKriging
#' @export
scaleX.MLPKriging <- function(object, ...) {
  mlpKriging_scaleX(object$ptr)
}

#' @method centerY MLPKriging
#' @export
centerY.MLPKriging <- function(object, ...) {
  mlpKriging_centerY(object$ptr)
}

#' @method scaleY MLPKriging
#' @export
scaleY.MLPKriging <- function(object, ...) {
  mlpKriging_scaleY(object$ptr)
}

#' @method normalize MLPKriging
#' @export
normalize.MLPKriging <- function(object, ...) {
  mlpKriging_normalize(object$ptr)
}

#' @method regmodel MLPKriging
#' @export
regmodel.MLPKriging <- function(object, ...) {
  mlpKriging_regmodel(object$ptr)
}

#' @method F_ MLPKriging
#' @export
F_.MLPKriging <- function(object, ...) {
  mlpKriging_F(object$ptr)
}

#' @method T_ MLPKriging
#' @export
T_.MLPKriging <- function(object, ...) {
  mlpKriging_T(object$ptr)
}

#' @method M MLPKriging
#' @export
M.MLPKriging <- function(object, ...) {
  mlpKriging_M(object$ptr)
}

#' @method z MLPKriging
#' @export
z.MLPKriging <- function(object, ...) {
  mlpKriging_z(object$ptr)
}

#' @method beta MLPKriging
#' @export
beta.MLPKriging <- function(object, ...) {
  mlpKriging_beta(object$ptr)
}

#' @title Deep copy of MLPKriging model
#' @param object MLPKriging object
#' @param ... ignored
#' @return a new independent MLPKriging object
#' @method copy MLPKriging
#' @export
copy.MLPKriging <- function(object, ...) {
  ptr_copy <- mlpKriging_copy(object$ptr)
  obj <- list(ptr = ptr_copy)
  class(obj) <- "MLPKriging"
  obj
}

#' @title Save an MLPKriging model to file
#' @param object MLPKriging object
#' @param filename path to save file
#' @param ... ignored
#' @method save MLPKriging
#' @export
save.MLPKriging <- function(object, filename, ...) {
  if (!is.character(filename))
    stop("'filename' must be a string")
  mlpKriging_save(object$ptr, filename)
  invisible(NULL)
}

#' @title Load an MLPKriging model from file
#' @param filename path to saved file
#' @param ... ignored
#' @return MLPKriging object
#' @method load MLPKriging
#' @export
load.MLPKriging <- function(filename, ...) {
  if (!is.character(filename))
    stop("'filename' must be a string")
  ptr <- mlpkriging_load(filename)
  obj <- list(ptr = ptr)
  class(obj) <- "MLPKriging"
  obj
}
