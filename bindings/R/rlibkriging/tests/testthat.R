## load dependencies
library(testthat)

relative_error <- function(x, y) {
  err <- 0
  if (x != 0. || y != 0)
    err <- max(abs(x - y)) / max(abs(x), abs(y))
  cat("Relative error is",err,"\n")
  return(err)
}

## test package
test_check('rlibkriging')