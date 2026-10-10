library(testthat)
library(rlibkriging)

f <- function(X) apply(X, 1, function(x) sin(3 * x[1]) + cos(5 * x[2]) + x[1] * x[2])

set.seed(123)
X <- matrix(runif(2 * 200), ncol = 2)
y <- f(X)
Xt <- matrix(runif(2 * 100), ncol = 2)

test_that("NestedKriging fits and predicts with all aggregations", {
  for (agg in c("PoE", "gPoE", "BCM", "rBCM", "NK")) {
    k <- NestedKriging(y, X, kernel = "matern5_2", nb_groups = 4, aggregation = agg)
    p <- predict(k, Xt)
    expect_length(p$mean, nrow(Xt))
    expect_length(p$stdev, nrow(Xt))
    expect_true(all(is.finite(p$mean)))
    expect_true(all(p$stdev >= 0))
    rmse <- sqrt(mean((p$mean - f(Xt))^2))
    expect_lt(rmse, 0.5 * sd(y))
  }
})

test_that("NK aggregation interpolates the design", {
  k <- NestedKriging(y, X, kernel = "matern5_2", nb_groups = 4, aggregation = "NK")
  p <- predict(k, X)
  expect_lt(max(abs(p$mean - y)), 1e-3)
  expect_lt(max(p$stdev), 1e-2)
})

test_that("accessors are consistent", {
  k <- NestedKriging(y, X, kernel = "gauss", nb_groups = 4)
  expect_equal(k$kernel(), "gauss")
  expect_equal(k$aggregation(), "NK")
  expect_equal(k$nb_groups(), 4)
  expect_length(k$theta(), 2)
  expect_gt(k$sigma2(), 0)
  expect_equal(sort(unlist(k$groups())), 1:nrow(X))  # 1-based partition
})

test_that("close to full Kriging on moderate n", {
  kf <- Kriging(y, X, kernel = "matern5_2")
  pf <- predict(kf, Xt, return_stdev = FALSE)
  k <- NestedKriging(y, X, kernel = "matern5_2", nb_groups = 4, aggregation = "NK")
  p <- predict(k, Xt, return_stdev = FALSE)
  expect_lt(mean(abs(p$mean - pf$mean)), 0.05 * sd(y))
})

test_that("reproducibility given a seed", {
  k1 <- NestedKriging(y, X, kernel = "gauss", nb_groups = 5, partition = "random", seed = 42)
  k2 <- NestedKriging(y, X, kernel = "gauss", nb_groups = 5, partition = "random", seed = 42)
  expect_identical(predict(k1, Xt), predict(k2, Xt))
})

test_that("input validation", {
  expect_error(NestedKriging(y, X, kernel = "gauss", nb_groups = 4, aggregation = "median"))
  expect_error(NestedKriging(y, X, kernel = "gauss", nb_groups = 1000))
  expect_error(NestedKriging(y, X, kernel = "gauss", nb_groups = 4,
                             aggregation = "NK", regmodel = "linear"))
})

test_that("warped submodels (WarpKriging) are supported", {
  k <- NestedKriging(y, X, kernel = "gauss", nb_groups = 3,
                     warping = c("kumaraswamy", "kumaraswamy"))
  expect_equal(k$warping(), c("kumaraswamy", "kumaraswamy"))
  p <- predict(k, X)
  expect_lt(max(abs(p$mean - y)), 1e-3)  # NK interpolates under warping
  expect_true(all(p$stdev >= 0))
})

test_that("save/load roundtrip (plain and generic load)", {
  for (agg in c("NK", "gPoE")) {
    k <- NestedKriging(y, X, kernel = "matern5_2", nb_groups = 4, aggregation = agg)
    outfile <- tempfile(fileext = ".json")
    save(k, outfile)
    for (k2 in list(load(outfile), load.NestedKriging(outfile))) {
      expect_s3_class(k2, "NestedKriging")
      expect_equal(k2$aggregation(), agg)
      expect_equal(k2$nb_groups(), k$nb_groups())
      expect_equal(k2$theta(), k$theta())
      p1 <- predict(k, Xt)
      p2 <- predict(k2, Xt)
      expect_equal(p2$mean, p1$mean, tolerance = 1e-10)
      expect_equal(p2$stdev, p1$stdev, tolerance = 1e-10)
    }
    unlink(outfile)
  }
})

test_that("fit refits an existing object; tuning setters", {
  k <- NestedKriging(y, X, kernel = "matern5_2", nb_groups = 4)
  X2 <- matrix(runif(2 * 120), ncol = 2)
  y2 <- f(X2)
  k$fit(y2, X2, 3)
  expect_equal(k$nb_groups(), 3)
  expect_equal(k$X(), X2)
  p <- predict(k, X2)
  expect_lt(max(abs(p$mean - y2)), 1e-3)  # NK interpolates the new design

  k$set_predict_chunk(5)
  p5 <- predict(k, Xt)
  k$set_predict_chunk(128)
  p128 <- predict(k, Xt)
  # same result up to BLAS blocking / summation order (~1e-10 observed)
  expect_equal(p5$mean, p128$mean, tolerance = 1e-8)
  expect_error(k$set_predict_chunk(0))
  expect_error(k$set_warp_subsample(500), NA)
})
