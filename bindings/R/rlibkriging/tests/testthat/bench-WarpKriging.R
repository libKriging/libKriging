## *************************************************************************
##  bench-WarpKriging.R
##
##  Compare WarpKriging against regular Kriging on limit cases:
##
##  1. warp_none   : identity transformation  -> should match Kriging exactly
##  2. warp_affine : a·x + b, degenerates to identity when a=1, b=0
##                   (verifies the affine path does not hurt)
##  3. Prediction accuracy across 1-D / 2-D test functions
##  4. Timing comparison: WarpKriging(none) vs Kriging
##
##  Run from the package root:
##    Rscript tests/testthat/bench-WarpKriging.R
## *************************************************************************

library(rlibkriging,lib.loc = "bindings/R/Rlibs/")

cat("================================================================\n")
cat(" bench-WarpKriging  —  limit cases vs regular Kriging\n")
cat("================================================================\n\n")

# -----------------------------------------------------------------------
#  Helper
# -----------------------------------------------------------------------
rmse <- function(pred, truth) sqrt(mean((pred - truth)^2))

# -----------------------------------------------------------------------
#  1-D Goldstein function (nonstationary, curved)
# -----------------------------------------------------------------------
f1d <- function(x) {
  1 - 0.5 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
}

# -----------------------------------------------------------------------
#  2-D Branin (rescaled to [0,1]^2)
# -----------------------------------------------------------------------
branin <- function(X) {
  x1 <- X[, 1] * 15 - 5
  x2 <- X[, 2] * 15
  a <- 1; b <- 5.1 / (4 * pi^2); cc <- 5 / pi
  r <- 6; s <- 10; tt <- 1 / (8 * pi)
  a * (x2 - b * x1^2 + cc * x1 - r)^2 + s * (1 - tt) * cos(x1) + s
}

# -----------------------------------------------------------------------
#  Section 1 — warp_none == Kriging  (LL and predictions must match)
# -----------------------------------------------------------------------
cat("--- Section 1: WarpKriging('none') vs Kriging ---\n\n")

for (kern in c("gauss", "matern5_2", "matern3_2")) {

  cat(sprintf("  kernel = %s\n", kern))

  set.seed(42)
  n  <- 15
  X  <- as.matrix(seq(0.02, 0.98, length.out = n))
  y  <- f1d(X)
  xp <- as.matrix(seq(0.01, 0.99, length.out = 100))

  # Standard Kriging
  t_krig <- system.time(
    k_std <- Kriging(y, X, kernel = kern, regmodel = "constant",
                     normalize = FALSE, optim = "BFGS", objective = "LL")
  )["elapsed"]

  # WarpKriging with no warping
  t_warp <- system.time(
    k_warp <- WarpKriging(y, X, warping = "none", kernel = kern,
                          regmodel = "constant", normalize = FALSE,
                          optim = "BFGS+Adam", objective = "LL",
                          parameters = list(max_iter_adam = "0"))
  )["elapsed"]

  ll_std  <- logLikelihoodFun(k_std,  as.list(k_std)$theta)$logLikelihood
  ll_warp <- logLikelihood(k_warp)

  p_std  <- predict(k_std,  xp, stdev = TRUE)
  p_warp <- predict(k_warp, xp, stdev = TRUE)

  diff_mean  <- max(abs(p_std$mean  - p_warp$mean))
  diff_stdev <- max(abs(p_std$stdev - p_warp$stdev))
  diff_ll    <- abs(ll_std - ll_warp)

  rmse_std  <- rmse(p_std$mean,  f1d(xp))
  rmse_warp <- rmse(p_warp$mean, f1d(xp))

  cat(sprintf("    LL  Kriging=%.4f   WarpKriging(none)=%.4f   |diff|=%.2e\n",
              ll_std, ll_warp, diff_ll))
  cat(sprintf("    RMSE Kriging=%.6f  WarpKriging(none)=%.6f\n",
              rmse_std, rmse_warp))
  cat(sprintf("    Max |mean diff|=%.2e   Max |stdev diff|=%.2e\n",
              diff_mean, diff_stdev))
  cat(sprintf("    Time  Kriging=%.3fs   WarpKriging(none)=%.3fs\n",
              t_krig, t_warp))

  if (diff_ll > 0.5)
    warning(sprintf("[%s] LL mismatch: %.4f vs %.4f", kern, ll_std, ll_warp))
  if (diff_mean > 1e-2)
    warning(sprintf("[%s] Mean prediction mismatch: max diff = %.2e", kern, diff_mean))

  cat("\n")
}

# -----------------------------------------------------------------------
#  Section 2 — warp_affine:  should not degrade compared to none
#             (affine can find identity as a special case)
# -----------------------------------------------------------------------
cat("--- Section 2: WarpKriging('affine') vs 'none' on 1-D ---\n\n")

set.seed(42)
n  <- 15
X  <- as.matrix(seq(0.02, 0.98, length.out = n))
y  <- f1d(X)
xp <- as.matrix(seq(0.01, 0.99, length.out = 100))

for (kern in c("gauss", "matern5_2")) {
  k_none   <- WarpKriging(y, X, "none",   kern, parameters = list(max_iter_adam = "0"))
  k_affine <- WarpKriging(y, X, "affine", kern, parameters = list(max_iter_adam = "200"))

  ll_none   <- logLikelihood(k_none)
  ll_affine <- logLikelihood(k_affine)

  p_none   <- predict(k_none,   xp)
  p_affine <- predict(k_affine, xp)

  cat(sprintf("  kernel=%s  LL none=%.4f  LL affine=%.4f  (affine >= none: %s)\n",
              kern, ll_none, ll_affine,
              ifelse(ll_affine >= ll_none - 0.1, "YES", "NO")))
  cat(sprintf("    RMSE none=%.6f  RMSE affine=%.6f\n",
              rmse(p_none$mean, f1d(xp)),
              rmse(p_affine$mean, f1d(xp))))
}
cat("\n")

# -----------------------------------------------------------------------
#  Section 3 — Simple warpings (boxcox, kumaraswamy) vs none on 1-D
#              Non-identity warpings should match or improve LL.
# -----------------------------------------------------------------------
cat("--- Section 3: Richer warpings vs 'none' on 1-D function ---\n\n")

set.seed(42)
n  <- 20
X  <- as.matrix(seq(0.02, 0.98, length.out = n))
y  <- f1d(X)
xp <- as.matrix(seq(0.01, 0.99, length.out = 200))

warpings <- list(
  none         = "none",
  affine       = "affine",
  boxcox       = "boxcox",
  kumaraswamy  = "kumaraswamy",
  neural_mono  = "neural_mono(8)",
  mlp          = "mlp(16:8,2,selu)"
)

results_1d <- data.frame(
  warping  = character(),
  LL       = numeric(),
  RMSE     = numeric(),
  time_s   = numeric(),
  stringsAsFactors = FALSE
)

for (wname in names(warpings)) {
  t <- system.time(
    k <- WarpKriging(y, X, warpings[[wname]], "gauss",
                     parameters = list(max_iter_adam = "300"))
  )["elapsed"]
  ll   <- logLikelihood(k)
  pred <- predict(k, xp)$mean
  results_1d <- rbind(results_1d, data.frame(
    warping = wname, LL = ll, RMSE = rmse(pred, f1d(xp)), time_s = t
  ))
}

cat("  1-D Goldstein  (gauss kernel, n=20)\n")
cat(sprintf("  %-14s  %10s  %10s  %8s\n", "warping", "LL", "RMSE", "time(s)"))
cat(sprintf("  %-14s  %10s  %10s  %8s\n",
            strrep("-", 14), strrep("-", 10), strrep("-", 10), strrep("-", 8)))
for (i in seq_len(nrow(results_1d))) {
  cat(sprintf("  %-14s  %10.4f  %10.6f  %8.3f\n",
              results_1d$warping[i],
              results_1d$LL[i],
              results_1d$RMSE[i],
              results_1d$time_s[i]))
}
cat("\n")

# -----------------------------------------------------------------------
#  Section 4 — 2-D Branin:  warp_none vs Kriging (limit case)
# -----------------------------------------------------------------------
cat("--- Section 4: 2-D Branin — WarpKriging('none') vs Kriging ---\n\n")

set.seed(77)
n   <- 30
X2  <- matrix(runif(n * 2), ncol = 2)
y2  <- branin(X2)

set.seed(88)
Xp2 <- matrix(runif(50 * 2), ncol = 2)
yp2 <- branin(Xp2)

t_krig2 <- system.time(
  k2_std <- Kriging(y2, X2, kernel = "matern5_2", regmodel = "constant",
                    normalize = TRUE, optim = "BFGS", objective = "LL")
)["elapsed"]

t_warp2 <- system.time(
  k2_warp <- WarpKriging(y2, X2, warping = c("none", "none"),
                         kernel = "matern5_2", regmodel = "constant",
                         normalize = TRUE, optim = "BFGS+Adam", objective = "LL",
                         parameters = list(max_iter_adam = "0"))
)["elapsed"]

ll2_std  <- logLikelihoodFun(k2_std, as.list(k2_std)$theta)$logLikelihood
ll2_warp <- logLikelihood(k2_warp)

p2_std  <- predict(k2_std,  Xp2, stdev = TRUE)
p2_warp <- predict(k2_warp, Xp2, stdev = TRUE)

cat(sprintf("  Kriging      LL=%.4f  RMSE=%.4f  time=%.3fs\n",
            ll2_std,  rmse(p2_std$mean,  yp2), t_krig2))
cat(sprintf("  Warp(none)   LL=%.4f  RMSE=%.4f  time=%.3fs\n",
            ll2_warp, rmse(p2_warp$mean, yp2), t_warp2))
cat(sprintf("  Max |mean diff|=%.2e  |LL diff|=%.2e\n",
            max(abs(p2_std$mean - p2_warp$mean)),
            abs(ll2_std - ll2_warp)))
cat("\n")

# -----------------------------------------------------------------------
#  Section 5 — Timing: scaling with n  (warp_none vs Kriging)
# -----------------------------------------------------------------------
cat("--- Section 5: Timing scaling with n (1-D, gauss, warp_none vs Kriging) ---\n\n")

ns <- c(10, 20, 50, 100)
cat(sprintf("  %-6s  %-12s  %-12s  %-12s\n",
            "n", "Kriging(s)", "Warp(none)(s)", "ratio"))
cat(sprintf("  %-6s  %-12s  %-12s  %-12s\n",
            "------", "------------", "------------", "------------"))

for (n_i in ns) {
  set.seed(n_i)
  Xi <- as.matrix(seq(0.01, 0.99, length.out = n_i))
  yi <- f1d(Xi)

  t_k <- system.time(
    Kriging(yi, Xi, "gauss", "constant", FALSE, "BFGS", "LL")
  )["elapsed"]

  t_w <- system.time(
    WarpKriging(yi, Xi, "none", "gauss", "constant", FALSE,
                "BFGS+Adam", "LL",
                parameters = list(max_iter_adam = "0"))
  )["elapsed"]

  cat(sprintf("  %-6d  %-12.4f  %-12.4f  %-12.2f\n",
              n_i, t_k, t_w, if (t_k > 0) t_w / t_k else NA))
}
cat("\n")

cat("================================================================\n")
cat(" bench-WarpKriging done.\n")
cat("================================================================\n")
