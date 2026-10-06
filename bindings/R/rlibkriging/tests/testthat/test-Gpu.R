library(testthat)

test_that("GPU API is always defined", {
  expect_type(gpu_compiled_backends(), "character")
  expect_type(gpu_available(), "logical")
  expect_true(gpu_backend() %in% c("none", "cuda", "hip", "sycl", "metal"))
  expect_equal(gpu_enabled(), gpu_backend() != "none")
})

test_that("set_gpu_enabled round trip never enables a missing device", {
  initial <- gpu_enabled()
  set_gpu_enabled(FALSE)
  expect_false(gpu_enabled())
  expect_equal(gpu_backend(), "none")
  set_gpu_enabled(TRUE)
  expect_equal(gpu_enabled(), gpu_available())
  set_gpu_enabled(initial)
})
