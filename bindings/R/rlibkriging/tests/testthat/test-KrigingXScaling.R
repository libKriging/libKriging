# centerX() / scaleX() of a model with several inputs (a row vector in C++), and
# leaveOneOutVec() given a plain vector theta (#435)
set.seed(1)
n <- 10
X <- matrix(runif(2 * n), ncol = 2)
y <- sin(3 * X[, 1]) + X[, 2]

k <- Kriging(y, X, "gauss", normalize = TRUE)

test_that("centerX() and scaleX() give one value per input", {
  expect_equal(length(k$centerX()), 2)
  expect_equal(length(k$scaleX()), 2)
  expect_true(all(is.finite(k$centerX())))
  expect_true(all(k$scaleX() > 0))
})

test_that("leaveOneOutVec() accepts a vector, a column matrix and a 1-d array", {
  ref <- k$leaveOneOutVec(matrix(k$theta()))
  expect_equal(k$leaveOneOutVec(k$theta()), ref)
  expect_equal(k$leaveOneOutVec(array(k$theta())), ref)
  expect_equal(length(ref$mean), n)
})

test_that("leaveOneOutVec() rejects a wrong theta", {
  expect_error(k$leaveOneOutVec(c(0.5, 0.5, 0.5)), "must have 2 values")
  expect_error(k$leaveOneOutVec("a"), "must be numeric")
})
