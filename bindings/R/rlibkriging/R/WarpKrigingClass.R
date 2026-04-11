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

#' @title Joint MLP warping (all inputs together, deep kernel learning)
#' @param hidden_dims integer vector of hidden layer sizes
#' @param d_out output dimensionality (default 2)
#' @param activation activation function (default "selu")
#' @return string e.g. \code{"mlp_joint(32:16,3,selu)"}
#' @export
warp_mlp_joint <- function(hidden_dims, d_out = 2, activation = "selu") {
  h <- paste(as.integer(hidden_dims), collapse = ":")
  paste0("mlp_joint(", h, ",", d_out, ",", activation, ")")
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
#'   of X (or a single \code{"mlp_joint(...)"} for joint mode).
#'   Use \code{warp_*()} helpers or plain strings:
#'   \code{"none"}, \code{"affine"}, \code{"boxcox"}, \code{"kumaraswamy"},
#'   \code{"neural_mono(8)"}, \code{"mlp(16:8,2,selu)"},
#'   \code{"mlp_joint(32:16,3,selu)"}, \code{"categorical(5,2)"},
#'   \code{"ordinal(4)"}.
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
#' # Deep kernel learning (joint MLP on all inputs)
#' k3 <- WarpKriging(y, X,
#'          warping = "mlp_joint(32:16,3,selu)",
#'          kernel = "gauss", normalize = TRUE)
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

#' @title Predict with a WarpKriging model
#' @param object WarpKriging object
#' @param x prediction matrix (m x d)
#' @param stdev return standard deviations?
#' @param cov return full covariance?
#' @param ... ignored
#' @return list with \code{mean}, \code{stdev}, \code{cov}
#' @export
predict.WarpKriging <- function(object, x, stdev = TRUE, cov = FALSE, ...) {
  warpKriging_predict(object$ptr, as.matrix(x), stdev, cov)
}

#' @title Simulate from a WarpKriging model
#' @param object WarpKriging object
#' @param nsim number of simulations
#' @param seed random seed
#' @param x simulation matrix (m x d)
#' @param ... ignored
#' @return matrix (m x nsim)
#' @export
simulate.WarpKriging <- function(object, nsim = 1, seed = 123, x, ...) {
  warpKriging_simulate(object$ptr, as.integer(nsim),
                       as.integer(seed), as.matrix(x))
}

#' @title Update a WarpKriging model with new observations
#' @param object WarpKriging object
#' @param y_new new observations
#' @param X_new new input matrix
#' @param ... ignored
#' @export
update.WarpKriging <- function(object, y_new, X_new, ...) {
  warpKriging_update(object$ptr, as.numeric(y_new), as.matrix(X_new))
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
#' @param grad return gradient?
#' @param ... ignored
#' @return list with \code{logLikelihood} and optionally \code{gradient}
#' @export
logLikelihoodFun.WarpKriging <- function(object, theta, grad = TRUE, ...) {
  warpKriging_logLikelihoodFun(object$ptr, theta, grad)
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
