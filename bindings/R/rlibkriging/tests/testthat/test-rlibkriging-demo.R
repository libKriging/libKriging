library(rlibkriging)

test_that("rlibkriging demo: Kriging predict, simulate, update work", {
  X <- as.matrix(c(0.0, 0.2, 0.5, 0.8, 1.0))
  f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
  y <- f(X)

  k_R <- Kriging(y, X, "gauss")
  expect_s3_class(k_R, "Kriging")

  x <- as.matrix(seq(0, 1, length.out = 100))
  p <- predict(k_R, x, TRUE, FALSE)
  expect_equal(length(p$mean), 100)
  expect_true(all(is.finite(p$mean)))
  expect_true(all(is.finite(p$stdev)))

  s <- simulate(k_R, nsim = 10, seed = 123, x = x)
  expect_equal(dim(s), c(100, 10))
  expect_true(all(is.finite(s)))

  Xn <- as.matrix(c(0.3, 0.4))
  yn <- f(Xn)
  update(k_R, yn, Xn)
  p2 <- predict(k_R, x, TRUE, FALSE)
  expect_equal(length(p2$mean), 100)
  expect_true(all(is.finite(p2$mean)))
})