context("R serialization (saveRDS / readRDS, serialize, PSOCK) of Kriging and NestedKriging (#327)")

f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
X <- as.matrix(c(0.0, 0.2, 0.4, 0.5, 0.7, 0.8, 1.0))
y <- f(X)
x <- as.matrix(seq(0, 1, , 21))

test_that("saveRDS / readRDS round-trips a fitted Kriging", {
  k <- Kriging(y, X, "gauss")
  file <- tempfile(fileext = ".rds")
  saveRDS(k, file)
  k2 <- readRDS(file)
  unlink(file)
  expect_s3_class(k2, "Kriging")
  expect_equal(k2$theta(), k$theta())
  expect_equal(k2$sigma2(), k$sigma2())
  p <- predict(k, x)
  p2 <- predict(k2, x)
  expect_equal(p2$mean, p$mean)
  expect_equal(p2$stdev, p$stdev)
  expect_equal(logLikelihood(k2), logLikelihood(k))
})

test_that("the unserialized object and its $ methods share one C++ model", {
  k <- Kriging(y, X, "gauss")
  k2 <- unserialize(serialize(k, NULL))
  k2$update(y_u = f(0.3), X_u = matrix(0.3), refit = FALSE)  # through the closure
  expect_equal(nrow(k2$X()), nrow(X) + 1)
  expect_equal(nrow(rlibkriging:::kriging_X(k2)), nrow(X) + 1)  # same model through the object itself
  expect_equal(nrow(k$X()), nrow(X))  # the original is not affected
})

test_that("the state at serialization time is saved, not the fitted one", {
  k <- Kriging(y, X, "gauss")
  k$update(y_u = f(0.3), X_u = matrix(0.3), refit = FALSE)
  k2 <- unserialize(serialize(k, NULL))
  expect_equal(nrow(k2$X()), nrow(X) + 1)
})

test_that("noise models round-trip", {
  set.seed(1)
  yn <- y + rnorm(length(y), sd = 0.01)
  kn <- Kriging(yn, X, "matern5_2", noise = "nugget")
  kn2 <- unserialize(serialize(kn, NULL))
  expect_equal(kn2$nugget(), kn$nugget())
  expect_equal(predict(kn2, x)$mean, predict(kn, x)$mean)
})

test_that("save() / load() of an R session image round-trip", {
  k <- Kriging(y, X, "gauss")
  file <- tempfile(fileext = ".RData")
  base::save(k, file = file)
  k_ref <- k
  rm(k)
  base::load(file)
  unlink(file)
  expect_equal(predict(k, x)$mean, predict(k_ref, x)$mean)
})

test_that("saveRDS / readRDS round-trips a NestedKriging", {
  set.seed(2)
  Xn <- matrix(runif(60), ncol = 2)
  yn <- sin(3 * Xn[, 1]) + Xn[, 2]^2
  nk <- NestedKriging(yn, Xn, "gauss", 3)
  nk2 <- unserialize(serialize(nk, NULL))
  xn <- matrix(runif(10), ncol = 2)
  expect_equal(predict(nk2, xn)$mean, predict(nk, xn)$mean)
})

test_that("a Kriging is shipped to a PSOCK worker", {
  skip_on_cran()
  k <- Kriging(y, X, "gauss")
  cl <- parallel::makeCluster(1, type = "PSOCK")
  on.exit(parallel::stopCluster(cl))
  libs <- .libPaths()
  parallel::clusterCall(cl, function(l) .libPaths(l), libs)
  res <- parallel::clusterCall(cl, function(m, xx) {
    library(rlibkriging)
    predict(m, xx)$mean
  }, k, x)[[1]]
  expect_equal(res, predict(k, x)$mean)
})
