## *************************************************************************
##  WarpKriging S3 class for rlibkriging
##
##  Kriging with per-variable input warping.
##  Warping specs are plain strings, parsed by C++ WarpSpec::from_string.
## *************************************************************************

## Register WarpKriging as a known S3 class to the S4 system so that
## packages that override generics (e.g. RobustGaSP overriding simulate)
## can still dispatch to the S3 simulate.WarpKriging method.
setOldClass("WarpKriging")

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
#  String / factor column encoding helper
# -----------------------------------------------------------------------

# Detect character or factor columns in X (data.frame, matrix, or vector),
# encode them as integers 0..L-1, and rewrite the warping spec to include
# level names.  Returns list(X = numeric_matrix, warping = updated_specs).
.encode_string_columns <- function(X, warping) {
  if (is.data.frame(X)) {
    d <- ncol(X)
  } else {
    X <- as.matrix(X)
    d <- ncol(X)
  }
  warping <- as.character(warping)
  if (length(warping) == 1 && d > 1)
    warping <- rep(warping, d)

  X_num <- matrix(NA_real_, nrow = nrow(X), ncol = d)
  warping_out <- warping

  for (j in seq_len(d)) {
    col <- if (is.data.frame(X)) X[[j]] else X[, j]
    if (is.factor(col) || is.character(col)) {
      col <- as.character(col)
      labels <- sort(unique(col))
      label_map <- setNames(seq_along(labels) - 1L, labels)
      X_num[, j] <- as.numeric(label_map[col])

      # Parse and rewrite warping spec
      spec <- trimws(warping_out[j])
      spec_lower <- tolower(spec)
      names_str <- paste0('[', paste0('"', labels, '"', collapse = ','), ']')

      if (grepl('^categorical', spec_lower)) {
        # Extract embed_dim if present
        embed_dim <- 2L
        m <- regmatches(spec, regexec('\\(([^)]*)\\)', spec))[[1]]
        if (length(m) == 2 && nchar(m[2]) > 0) {
          parts <- trimws(strsplit(m[2], ',')[[1]])
          if (length(parts) >= 2)
            embed_dim <- as.integer(parts[length(parts)])
        }
        warping_out[j] <- paste0('categorical(', names_str, ',', embed_dim, ')')
      } else if (grepl('^ordinal', spec_lower)) {
        warping_out[j] <- paste0('ordinal(', names_str, ')')
      } else {
        stop(sprintf(
          "Column %d contains strings/factors but warping spec '%s' is not 'categorical' or 'ordinal'",
          j, spec))
      }
    } else {
      X_num[, j] <- as.numeric(col)
    }
  }

  list(X = X_num, warping = warping_out)
}

# Check whether X has any string or factor columns.
.has_string_columns <- function(X) {
  if (is.data.frame(X)) {
    any(vapply(X, function(col) is.character(col) || is.factor(col), logical(1)))
  } else {
    is.character(X)
  }
}

# Encode X using level names already stored in warping specs.
# Used by predict/simulate/update after the model is fitted.
.encode_X_from_warping <- function(X, warping) {
  if (!.has_string_columns(X))
    return(as.matrix(X))
  res <- .encode_string_columns(X, warping)
  res$X
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
                        parameters = NULL,
                        noise = NULL) {
  y <- as.numeric(y)
  warping <- as.character(warping)

  if (.has_string_columns(X)) {
    enc <- .encode_string_columns(X, warping)
    X <- enc$X
    warping <- enc$warping
  } else {
    X <- as.matrix(X)
  }

  ptr <- warpKriging_new(y, X, warping, kernel,
                         regmodel, normalize, optim, objective,
                         parameters, noise)
  obj <- list(ptr = ptr)
  class(obj) <- "WarpKriging"
  return(obj)
}

# -----------------------------------------------------------------------
#  S3 methods
# -----------------------------------------------------------------------

#' @method print WarpKriging
#' @export
print.WarpKriging <- function(x, ...) {
  cat(warpKriging_summary(x$ptr))
  invisible(x)
}

#' @method summary WarpKriging
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
                            parameters = NULL,
                            noise = NULL, ...) {
  warping <- warpKriging_warping(object$ptr)
  X <- .encode_X_from_warping(X, warping)
  warpKriging_fit(object$ptr,
                  as.numeric(y), X,
                  regmodel, normalize, optim, objective,
                  parameters, noise)
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
#' @method predict WarpKriging
#' @export
predict.WarpKriging <- function(object, x, return_stdev = TRUE, return_cov = FALSE,
                                return_deriv = FALSE, ...) {
  x <- .encode_X_from_warping(x, warpKriging_warping(object$ptr))
  warpKriging_predict(object$ptr, x, return_stdev, return_cov, return_deriv)
}

#' @title Simulate from a WarpKriging model
#' @param object WarpKriging object
#' @param nsim number of simulations
#' @param seed random seed
#' @param x simulation matrix (m x d)
#' @param will_update logical; if TRUE, cache data for update_simulate
#' @param ... ignored
#' @return matrix (m x nsim)
#' @method simulate WarpKriging
#' @export
simulate.WarpKriging <- function(object, nsim = 1, seed = 123, x,
                                 will_update = FALSE, ...) {
  x <- .encode_X_from_warping(x, warpKriging_warping(object$ptr))
  warpKriging_simulate(object$ptr, as.integer(nsim),
                       as.integer(seed), x,
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
  X_u <- .encode_X_from_warping(X_u, warpKriging_warping(object$ptr))
  warpKriging_update_simulate(object$ptr, as.numeric(y_u), X_u)
}

#' @title Update a WarpKriging model with new observations
#' @param object WarpKriging object
#' @param y_u new observations
#' @param X_u new input matrix
#' @param refit logical; if TRUE (default), re-optimise hyperparameters
#' @param ... ignored
#' @method update WarpKriging
#' @export
update.WarpKriging <- function(object, y_u, X_u, refit = TRUE, ...) {
  X_u <- .encode_X_from_warping(X_u, warpKriging_warping(object$ptr))
  warpKriging_update(object$ptr, as.numeric(y_u), X_u,
                     as.logical(refit))
  invisible(object)
}

#' @title Log-likelihood of the fitted model
#' @method logLikelihood WarpKriging
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
#' @method logLikelihoodFun WarpKriging
#' @export
logLikelihoodFun.WarpKriging <- function(object, theta, return_grad = FALSE, return_hess = FALSE, ...) {
  warpKriging_logLikelihoodFun(object$ptr, theta, return_grad, return_hess)
}

#' @export
theta <- function(object, ...) UseMethod("theta")

#' @title Get GP range parameters
#' @method theta WarpKriging
#' @export
theta.WarpKriging <- function(object, ...) {
  warpKriging_theta(object$ptr)
}

#' @export
sigma2 <- function(object, ...) UseMethod("sigma2")

#' @title Get process variance (concentrated MLE)
#' @method sigma2 WarpKriging
#' @export
sigma2.WarpKriging <- function(object, ...) {
  warpKriging_sigma2(object$ptr)
}

#' @export
kernel <- function(object, ...) UseMethod("kernel")

#' @title Get kernel name
#' @method kernel WarpKriging
#' @export
kernel.WarpKriging <- function(object, ...) {
  warpKriging_kernel(object$ptr)
}

#' @title Get warping specifications as strings
#' @export
warping <- function(object, ...) UseMethod("warping")

#' @method warping WarpKriging
#' @export
warping.WarpKriging <- function(object, ...) {
  warpKriging_warping(object$ptr)
}

#' @title Get feature dimensionality of warped space
#' @method feature_dim WarpKriging
#' @export
feature_dim.WarpKriging <- function(object, ...) {
  warpKriging_featureDim(object$ptr)
}

#' @title Check if the model has been fitted
#' @export
is_fitted <- function(object, ...) UseMethod("is_fitted")

#' @method is_fitted WarpKriging
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

#' @method X WarpKriging
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

#' @method y WarpKriging
#' @export
y.WarpKriging <- function(object, ...) {
  warpKriging_y(object$ptr)
}

#' @title Get input centering vector
#' @export
centerX <- function(object, ...) UseMethod("centerX")

#' @method centerX WarpKriging
#' @export
centerX.WarpKriging <- function(object, ...) {
  warpKriging_centerX(object$ptr)
}

#' @title Get input scaling vector
#' @export
scaleX <- function(object, ...) UseMethod("scaleX")

#' @method scaleX WarpKriging
#' @export
scaleX.WarpKriging <- function(object, ...) {
  warpKriging_scaleX(object$ptr)
}

#' @title Get output centering value
#' @export
centerY <- function(object, ...) UseMethod("centerY")

#' @method centerY WarpKriging
#' @export
centerY.WarpKriging <- function(object, ...) {
  warpKriging_centerY(object$ptr)
}

#' @title Get output scaling value
#' @export
scaleY <- function(object, ...) UseMethod("scaleY")

#' @method scaleY WarpKriging
#' @export
scaleY.WarpKriging <- function(object, ...) {
  warpKriging_scaleY(object$ptr)
}

#' @title Get normalize flag
#' @export
normalize <- function(object, ...) UseMethod("normalize")

#' @method normalize WarpKriging
#' @export
normalize.WarpKriging <- function(object, ...) {
  warpKriging_normalize(object$ptr)
}

#' @title Get regression model type
#' @export
regmodel <- function(object, ...) UseMethod("regmodel")

#' @method regmodel WarpKriging
#' @export
regmodel.WarpKriging <- function(object, ...) {
  warpKriging_regmodel(object$ptr)
}

#' @title Get trend matrix F
#' @export
F_ <- function(object, ...) UseMethod("F_")

#' @method F_ WarpKriging
#' @export
F_.WarpKriging <- function(object, ...) {
  warpKriging_F(object$ptr)
}

#' @title Get Cholesky factor T
#' @export
T_ <- function(object, ...) UseMethod("T_")

#' @method T_ WarpKriging
#' @export
T_.WarpKriging <- function(object, ...) {
  warpKriging_T(object$ptr)
}

#' @title Get whitened trend matrix M
#' @export
M <- function(object, ...) UseMethod("M")

#' @method M WarpKriging
#' @export
M.WarpKriging <- function(object, ...) {
  warpKriging_M(object$ptr)
}

#' @title Get whitened residuals z
#' @export
z <- function(object, ...) UseMethod("z")

#' @method z WarpKriging
#' @export
z.WarpKriging <- function(object, ...) {
  warpKriging_z(object$ptr)
}

#' @title Get trend coefficients beta
#' @export
beta <- function(object, ...) UseMethod("beta")

#' @method beta WarpKriging
#' @export
beta.WarpKriging <- function(object, ...) {
  warpKriging_beta(object$ptr)
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
#' @method save WarpKriging
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
#' @method load WarpKriging
#' @export
load.WarpKriging <- function(filename, ...) {
  if (!is.character(filename))
    stop("'filename' must be a string")
  ptr <- warpkriging_load(filename)
  obj <- list(ptr = ptr)
  class(obj) <- "WarpKriging"
  obj
}
