library(testthat)

context("Explicit Vecchia / Nystrom methods")

set.seed(6)
X <- matrix(runif(120), ncol = 2)
y <- sin(3 * X[, 1]) + X[, 2]
Xn <- matrix(runif(14), ncol = 2)

k <- Kriging(kernel = "matern5_2")

test_that("exact commit is the default", {
  expect_true(k$vecchia_exact_commit())
  k$set_vecchia_exact_commit(FALSE)
  expect_false(k$vecchia_exact_commit())
})

k$fit(y, X, objective = "LLVecchia(10)")

test_that("light fit: explicit method, no implicit switch", {
  expect_true(k$is_vecchia_light())
  expect_false(k$is_nystrom_light())
  expect_error(k$predict(Xn))                    # exact: no factorization
  expect_error(k$simulate(nsim = 2, seed = 1, x = Xn))
  p <- k$predict(Xn, return_stdev = TRUE, method = "vecchia")
  expect_equal(length(p$mean), 7)
  expect_true(all(is.finite(p$mean)))
  expect_true(all(p$stdev >= 0))
  s <- k$simulate(nsim = 4, seed = 123, x = Xn, method = "vecchia")
  expect_equal(dim(s), c(7, 4))
  expect_equal(dim(k$simulate(nsim = 4, seed = 123, x = Xn, method = "vecchia(5)")), c(7, 4))
  expect_error(k$predict(Xn, return_cov = TRUE, method = "vecchia"))
  expect_error(k$predict(Xn, method = "unknown"))
  expect_true(is.finite(k$logLikelihood(method = "vecchia")))
})

test_that("light fit: update and save / load", {
  Xu <- matrix(runif(10), ncol = 2)
  yu <- sin(3 * Xu[, 1]) + Xu[, 2]
  k$update(yu, Xu, refit = FALSE)
  expect_true(k$is_vecchia_light())
  expect_equal(nrow(k$X()), 65)
  expect_true(max(abs(k$predict(Xu, method = "vecchia")$mean - yu)) < 1e-3)
  f <- tempfile(fileext = ".json")
  k$save(f)
  kl <- load(f)
  expect_true(kl$is_vecchia_light())
  expect_equal(kl$vecchia_neighbors(), 10)
  expect_equal(kl$predict(Xn, method = "vecchia")$mean, k$predict(Xn, method = "vecchia")$mean,
               tolerance = 1e-12)
})

test_that("refit with the exact commit: exact and Vecchia methods available", {
  k$set_vecchia_exact_commit(TRUE)
  k$fit(y, X, objective = "LLVecchia(10)")
  expect_false(k$is_vecchia_light())
  expect_true(is.matrix(k$predict(Xn, return_cov = TRUE)$cov))
  expect_equal(length(k$predict(Xn, method = "vecchia")$mean), 7)
  expect_equal(k$logLikelihood(), k$logLikelihood(method = "exact"))
})

test_that("Nystrom fits need method nystrom", {
  kn <- Kriging(y, X, "matern5_2", objective = "LLNystrom(8)")
  expect_true(kn$is_nystrom_light())
  expect_error(kn$predict(Xn))
  expect_equal(length(kn$predict(Xn, method = "nystrom")$mean), 7)
  expect_equal(dim(kn$simulate(nsim = 3, seed = 1, x = Xn, method = "nystrom")), c(7, 3))
  expect_true(is.finite(kn$logLikelihood(method = "nystrom")))
  expect_error(kn$logLikelihood(method = "vecchia"))
  expect_false(Kriging(y, X, "matern5_2")$is_nystrom_light())
})
