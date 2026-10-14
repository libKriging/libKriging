library(testthat)

# Issue #249: introspecting an unfitted ("empty") Kriging model used to segfault
test_that("unfitted Kriging model can be introspected", {
  k <- Kriging("gauss")
  expect_no_error(l <- as.list(k))
  expect_equal(l$kernel, "gauss")
  expect_equal(nrow(l$X), 0)
  expect_no_error(capture.output(str(k)))
})
