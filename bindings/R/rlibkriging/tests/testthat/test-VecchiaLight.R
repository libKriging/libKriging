library(testthat)

context("Vecchia light fit")

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

test_that("light fit: predict / simulate go through Vecchia", {
  expect_true(k$is_vecchia_light())
  expect_false(k$is_nystrom_light())
  p <- k$predict(Xn, return_stdev = TRUE)
  expect_equal(length(p$mean), 7)
  expect_true(all(is.finite(p$mean)))
  expect_true(all(p$stdev >= 0))
  s <- k$simulate(nsim = 4, seed = 123, x = Xn)
  expect_equal(dim(s), c(7, 4))
  expect_error(k$predict(Xn, return_stdev = TRUE, return_cov = TRUE))
  expect_error(k$save(tempfile(fileext = ".json")))
})

test_that("refit with the exact commit is no longer light", {
  k$set_vecchia_exact_commit(TRUE)
  k$fit(y, X, objective = "LLVecchia(10)")
  expect_false(k$is_vecchia_light())
})

test_that("Nystrom fits are light", {
  expect_true(Kriging(y, X, "matern5_2", objective = "LLNystrom(8)")$is_nystrom_light())
  expect_false(Kriging(y, X, "matern5_2")$is_nystrom_light())
})
