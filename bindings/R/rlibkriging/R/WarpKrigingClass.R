## *************************************************************************
##  WarpKriging S3 class for rlibkriging
##
##  Kriging with per-variable input warping.
##  Warping specs are plain strings, parsed by C++ WarpSpec::from_string.
## *************************************************************************

# -----------------------------------------------------------------------
#  Warp specification helpers  (return strings)
#
#  These are convenience functions for tab-completion and documentation.
#  They simply build the canonical string that WarpSpec::from_string expects.
# -----------------------------------------------------------------------

#' @title No warping (identity)
#' @return string \code{"none"}
#' @export
warp_none <- function() "none"

#' @title Affine warping: w(x) = a*x + b
#' @return string \code{"affine"}
#' @export
warp_affine <- function() "affine"

#' @title Box-Cox warping
#' @return string \code{"boxcox"}
#' @export
warp_boxcox <- function() "boxcox"

#' @title Kumaraswamy CDF warping on [0,1]
#' @return string \code{"kumaraswamy"}
#' @export
warp_kumaraswamy <- function() "kumaraswamy"

#' @title Monotone neural network warping
#' @param n_hidden number of hidden units (default 8)
#' @return string e.g. \code{"neural_mono(16)"}
#' @export
warp_neural_mono <- function(n_hidden = 8) {
  paste0("neural_mono(", n_hidden, ")")
}

#' @title Per-variable MLP warping (unconstrained, multi-dim output)
#' @param hidden_dims integer vector of hidden layer sizes
#' @param d_out output dimensionality (default 2)
#' @param activation activation: "relu","selu","tanh","sigmoid","elu"
#' @return string e.g. \code{"mlp(16:8,3,selu)"}
#' @export
warp_mlp <- function(hidden_dims, d_out = 2, activation = "selu") {
  h <- paste(as.integer(hidden_dims), collapse = ":")
  paste0("mlp(", h, ",", d_out, ",", activation, ")")
}

#' @title Categorical embedding
#' @param n_levels number of levels (integer-coded 0..n_levels-1)
#' @param embed_dim embedding dimensionality (default 2)
#' @return string e.g. \code{"categorical(5,2)"}
#' @export
warp_categorical <- function(n_levels, embed_dim = 2) {
  paste0("categorical(", n_levels, ",", embed_dim, ")")
}

#' @title Ordinal warping (learned ordered positions)
#' @param n_levels number of ordered levels (0..n_levels-1)
#' @return string e.g. \code{"ordinal(4)"}
#' @export
warp_ordinal <- function(n_levels) {
  paste0("ordinal(", n_levels, ")")
}

# -----------------------------------------------------------------------
#  Constructor
# -----------------------------------------------------------------------

#' @title Create a WarpKriging model
#'
#' @description Kriging with per-variable input warping.  Each input
#' dimension is independently transformed before the GP kernel is
#' evaluated.  Supports continuous, categorical, ordinal variables
#' and joint deep kernel learning.
#'
#' @param y numeric vector of observations (n)
#' @param X numeric matrix of inputs (n x d)
#' @param warping character vector of warp specifications, one per column
#'   of X. Use \code{warp_*()} helpers or plain strings:
#'   \code{"none"}, \code{"affine"}, \code{"boxcox"}, \code{"kumaraswamy"},
#'   \code{"neural_mono(8)"}, \code{"mlp(16:8,2,selu)"},
#'   \code{"categorical(5,2)"}, \code{"ordinal(4)"}.
#' @param kernel covariance kernel: "gauss", "matern3_2", "matern5_2", "exp"
#' @param regmodel trend: "constant", "linear", "quadratic"
#' @param normalize logical; normalise continuous inputs?
#' @param optim optimiser (currently only one bi-level strategy)
#' @param objective "LL" (log-likelihood)
#' @param parameters optional named list of tuning parameters,
#'   e.g. \code{list(max_iter_adam = "300", adam_lr = "0.001",
#'                   max_iter_bfgs = "50")}
#'
#' @return An S3 object of class "WarpKriging".
#'
#' @examples
#' # Continuous with Kumaraswamy warping
#' X <- as.matrix(c(0.0, 0.25, 0.5, 0.75, 1.0))
#' f <- function(x) 1 - 1/2 * (sin(12*x)/(1+x) + 2*cos(7*x)*x^5 + 0.7)
#' y <- f(X)
#' k <- WarpKriging(y, X, warping = "kumaraswamy", kernel = "gauss")
#' print(k)
#'
#' # Mixed: 1 continuous + 1 categorical (3 levels)
#' k2 <- WarpKriging(y_mix, X_mix,
#'          warping = c("mlp(16:8,2,selu)", "categorical(3,2)"),
#'          kernel = "matern5_2")
#'
#' @export
WarpKriging <- function(y, X, warping,
                        kernel = "gauss",
                        regmodel = "constant",
                        normalize = FALSE,
                        optim = "BFGS+Adam",
                        objective = "LL",
                        parameters = NULL) {
  y <- as.numeric(y)
  X <- as.matrix(X)
  warping <- as.character(warping)

  ptr <- warpKriging_new(y, X, warping, kernel,
                         regmodel, normalize, optim, objective,
                         parameters)
  obj <- list(ptr = ptr)
  class(obj) <- "WarpKriging"
  return(obj)
}

# -----------------------------------------------------------------------
#  S3 methods
# -----------------------------------------------------------------------

#' @export
print.WarpKriging <- function(x, ...) {
  cat(warpKriging_summary(x$ptr))
  invisible(x)
}

#' @export
summary.WarpKriging <- function(object, ...) {
  warpKriging_summary(object$ptr)
}

#' @title Fit a WarpKriging model to data
#'
#' @description (Re-)fit an already-constructed WarpKriging object on new
#'   data.  The warping specification and kernel are kept from construction.
#'
#' @param object WarpKriging object (created with the constructor or an
#'   empty-kernel call)
#' @param y numeric vector of observations (n)
#' @param X numeric matrix of inputs (n x d)
#' @param regmodel trend: "constant", "linear", "quadratic"
#' @param normalize logical; normalise continuous inputs?
#' @param optim optimiser
#' @param objective "LL" (log-likelihood)
#' @param parameters optional named list of tuning parameters
#' @param ... ignored
#'
#' @return No return value. WarpKriging object argument is modified.
#'
#' @method fit WarpKriging
#' @export
fit.WarpKriging <- function(object, y, X,
                            regmodel = "constant",
                            normalize = FALSE,
                            optim = "BFGS+Adam",
                            objective = "LL",
                            parameters = NULL, ...) {
  warpKriging_fit(object$ptr,
                  as.numeric(y), as.matrix(X),
                  regmodel, normalize, optim, objective,
                  parameters)
  invisible(NULL)
}

#' @title Predict with a WarpKriging model
#' @param object WarpKriging object
#' @param x prediction matrix (m x d)
#' @param return_stdev return standard deviations?
#' @param return_cov return full covariance?
#' @param return_deriv return derivatives of mean and stdev wrt x?
#' @param ... ignored
#' @return list with \code{mean}, optionally \code{stdev}, \code{cov},
#'   \code{mean_deriv}, \code{stdev_deriv}
#' @export
predict.WarpKriging <- function(object, x, return_stdev = TRUE, return_cov = FALSE,
                                return_deriv = FALSE, ...) {
  warpKriging_predict(object$ptr, as.matrix(x), return_stdev, return_cov, return_deriv)
}

#' @title Simulate from a WarpKriging model
#' @param object WarpKriging object
#' @param nsim number of simulations
#' @param seed random seed
#' @param x simulation matrix (m x d)
#' @param will_update logical; if TRUE, cache data for update_simulate
#' @param ... ignored
#' @return matrix (m x nsim)
#' @export
simulate.WarpKriging <- function(object, nsim = 1, seed = 123, x,
                                 will_update = FALSE, ...) {
  warpKriging_simulate(object$ptr, as.integer(nsim),
                       as.integer(seed), as.matrix(x),
                       as.logical(will_update))
}

#' @title Update simulated paths with new observations (FOXY algorithm)
#' @param object WarpKriging object (must have called simulate with will_update=TRUE)
#' @param y_u new observations
#' @param X_u new input matrix
#' @param ... ignored
#' @return matrix (m x nsim) of updated simulated paths
#' @method update_simulate WarpKriging
#' @export
update_simulate.WarpKriging <- function(object, y_u, X_u, ...) {
  warpKriging_update_simulate(object$ptr, as.numeric(y_u), as.matrix(X_u))
}

#' @title Update a WarpKriging model with new observations
#' @param object WarpKriging object
#' @param y_u new observations
#' @param X_u new input matrix
#' @param refit logical; if TRUE (default), re-optimise hyperparameters
#' @param ... ignored
#' @export
update.WarpKriging <- function(object, y_u, X_u, refit = TRUE, ...) {
  warpKriging_update(object$ptr, as.numeric(y_u), as.matrix(X_u),
                     as.logical(refit))
  invisible(object)
}

#' @title Log-likelihood of the fitted model
#' @export
logLikelihood.WarpKriging <- function(object, ...) {
  warpKriging_logLikelihood(object$ptr)
}

#' @title Evaluate log-likelihood at given theta
#' @param object WarpKriging object
#' @param theta range parameter vector
#' @param return_grad return gradient?
#' @param return_hess return hessian?
#' @param ... ignored
#' @return list with \code{logLikelihood} and optionally \code{gradient}, \code{hessian}
#' @export
logLikelihoodFun.WarpKriging <- function(object, theta, return_grad = FALSE, return_hess = FALSE, ...) {
  warpKriging_logLikelihoodFun(object$ptr, theta, return_grad, return_hess)
}

#' @export
theta <- function(object, ...) UseMethod("theta")

#' @title Get GP range parameters
#' @export
theta.WarpKriging <- function(object, ...) {
  warpKriging_theta(object$ptr)
}

#' @export
sigma2 <- function(object, ...) UseMethod("sigma2")

#' @title Get process variance (concentrated MLE)
#' @export
sigma2.WarpKriging <- function(object, ...) {
  warpKriging_sigma2(object$ptr)
}

#' @export
kernel <- function(object, ...) UseMethod("kernel")

#' @title Get kernel name
#' @export
kernel.WarpKriging <- function(object, ...) {
  warpKriging_kernel(object$ptr)
}

#' @title Get warping specifications as strings
#' @export
warping <- function(object, ...) UseMethod("warping")

#' @export
warping.WarpKriging <- function(object, ...) {
  warpKriging_warping(object$ptr)
}

#' @title Get feature dimensionality of warped space
#' @export
feature_dim.WarpKriging <- function(object, ...) {
  warpKriging_featureDim(object$ptr)
}

#' @title Check if the model has been fitted
#' @export
is_fitted <- function(object, ...) UseMethod("is_fitted")

#' @export
is_fitted.WarpKriging <- function(object, ...) {
  warpKriging_isFitted(object$ptr)
}

#' @title Get training input matrix
#' @param object A fitted model object
#' @param ... ignored
#' @return matrix of training inputs
#' @export
X <- function(object, ...) UseMethod("X")

#' @export
X.WarpKriging <- function(object, ...) {
  warpKriging_X(object$ptr)
}

#' @title Get training output vector
#' @param object A fitted model object
#' @param ... ignored
#' @return vector of training outputs
#' @export
y <- function(object, ...) UseMethod("y")

#' @export
y.WarpKriging <- function(object, ...) {
  warpKriging_y(object$ptr)
}

#' @title Deep copy of WarpKriging model
#' @param object WarpKriging object
#' @param ... ignored
#' @return a new independent WarpKriging object
#' @method copy WarpKriging
#' @export
copy.WarpKriging <- function(object, ...) {
  ptr_copy <- warpKriging_copy(object$ptr)
  obj <- list(ptr = ptr_copy)
  class(obj) <- "WarpKriging"
  obj
}

#' @title Save a WarpKriging model to file
#' @param object WarpKriging object
#' @param filename path to save file
#' @param ... ignored
#' @export
save.WarpKriging <- function(object, filename, ...) {
  if (!is.character(filename))
    stop("'filename' must be a string")
  warpKriging_save(object$ptr, filename)
  invisible(NULL)
}

#' @title Load a WarpKriging model from file
#' @param filename path to saved file
#' @param ... ignored
#' @return WarpKriging object
#' @export
load.WarpKriging <- function(filename, ...) {
  if (!is.character(filename))
    stop("'filename' must be a string")
  ptr <- warpkriging_load(filename)
  obj <- list(ptr = ptr)
  class(obj) <- "WarpKriging"
  obj
}
