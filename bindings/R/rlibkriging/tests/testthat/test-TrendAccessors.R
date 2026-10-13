## F / T accessors are named alike for Kriging, WarpKriging and MLPKriging:
## `k$F()` / `k$T()`, and the S3 functions `F_(k)` / `T_(k)` (not `F` / `T`,
## which would mask base::F / base::T).
library(testthat)
library(rlibkriging)

f1d <- function(x) 1 - 0.5 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
X <- matrix(seq(0.01, 0.99, length.out = 10), ncol = 1)
y <- f1d(X[, 1])

test_that("Kriging, WarpKriging and MLPKriging expose $F(), $T(), F_() and T_()", {
  models <- list(
    Kriging(y, X, "gauss"),
    WarpKriging(y, X, warping = "none", kernel = "gauss"),
    MLPKriging(y, X, hidden_dims = c(4), d_out = 1, activation = "selu", kernel = "gauss")
  )
  for (k in models) {
    expect_equal(dim(k$F()), c(nrow(X), 1))
    expect_equal(dim(k$T()), c(nrow(X), nrow(X)))
    expect_equal(F_(k), k$F())
    expect_equal(T_(k), k$T())
  }
  # base::F / base::T are not masked by the package
  expect_false(F)
  expect_true(T)
})
