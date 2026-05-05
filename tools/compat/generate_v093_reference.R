#!/usr/bin/env Rscript
# Generate reference values from the v0.9.3 rlibkriging API.
# Must be run with R_LIBS pointing to an rlibkriging v0.9.3 installation.
#
# Usage: Rscript generate_v093_reference.R [outfile.rds]
#   outfile defaults to /tmp/rlibkriging_v093_ref.rds

library(rlibkriging)

outfile <- commandArgs(trailingOnly = TRUE)
if (length(outfile) == 0) outfile <- "/tmp/rlibkriging_v093_ref.rds"

stopifnot(packageVersion("rlibkriging") < "0.10")  # guard: must be v0.9.x

set.seed(1234)
n <- 7
X <- matrix(seq(0, 1, length.out = n), ncol = 1)
f <- function(x) sin(2 * pi * x) + 0.5 * x
y <- f(X) + rnorm(n, sd = 0)  # noiseless for plain Kriging
y_noise <- f(X)

X_new <- matrix(c(0.15, 0.42, 0.73), ncol = 1)
noise_vec <- rep(0.05^2, n)

params_plain  <- list(theta = matrix(0.3), sigma2 = 0.5)
params_nugget <- list(theta = matrix(0.3), sigma2 = 0.5, nugget = 0.05^2)
params_noise  <- list(theta = matrix(0.3), sigma2 = 0.5)

ref <- list()

# ── Plain Kriging ──────────────────────────────────────────────────────────────
k <- Kriging(y = y, X = X, kernel = "gauss",
             regmodel = "linear", normalize = FALSE, optim = "none",
             parameters = params_plain)
p <- predict(k, X_new, return_stdev = TRUE)
ref$plain <- list(
  theta        = k$theta(),
  sigma2       = k$sigma2(),
  beta         = k$beta(),
  T            = k$T(),
  M            = k$M(),
  z            = k$z(),
  pred_mean    = p$mean,
  pred_stdev   = p$stdev,
  loglik       = k$logLikelihood(),
  loo          = k$leaveOneOut()
)

# ── NoiseKriging ───────────────────────────────────────────────────────────────
nk <- NoiseKriging(y = matrix(y_noise, ncol = 1),
                   noise = matrix(noise_vec, ncol = 1),
                   X = X, kernel = "gauss",
                   regmodel = "linear", normalize = FALSE, optim = "none",
                   parameters = params_noise)
p <- predict(nk, X_new, return_stdev = TRUE)
ref$noise <- list(
  theta        = nk$theta(),
  sigma2       = nk$sigma2(),
  beta         = nk$beta(),
  T            = nk$T(),
  M            = nk$M(),
  z            = nk$z(),
  pred_mean    = p$mean,
  pred_stdev   = p$stdev,
  loglik       = nk$logLikelihood()
)

# ── NuggetKriging ──────────────────────────────────────────────────────────────
uk <- NuggetKriging(y = matrix(y_noise, ncol = 1),
                    X = X, kernel = "gauss",
                    regmodel = "linear", normalize = FALSE, optim = "none",
                    parameters = params_nugget)
p <- predict(uk, X_new, return_stdev = TRUE)
ref$nugget <- list(
  theta        = uk$theta(),
  sigma2       = uk$sigma2(),
  nugget       = uk$nugget(),
  beta         = uk$beta(),
  T            = uk$T(),
  M            = uk$M(),
  z            = uk$z(),
  pred_mean    = p$mean,
  pred_stdev   = p$stdev
  # logmargpost intentionally omitted: platform-specific issue in macOS arm64 v0.9.3 binary
)

saveRDS(ref, outfile)
cat("Reference values written to:", outfile, "\n")
cat("rlibkriging version:", as.character(packageVersion("rlibkriging")), "\n")
