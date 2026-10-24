# Lists the names exposed by rlibkriging, one "<scope>\t<name>" line each,
# for compare.py, which compares the bindings with each other.
# Usage: Rscript dump_r.R <output file>
#  - model classes: the `k$name()` members of a fitted object, the S3 methods
#    `generic.Class` (load(), summary(), as.list()...) and the free functions
#    standing for a static method (subsetOfData);
#  - Optim / LinearAlgebra: the namespace functions optim_* / linalg_*
#    (exported or not), prefix removed, except the wrappers of C++ internals
#    kept for the R tests (TEST_HELPERS).
suppressPackageStartupMessages(library(rlibkriging))

ns <- asNamespace("rlibkriging")
lines <- character(0)
add <- function(scope, names) {
  lines <<- c(lines, paste0(scope, "\t", sort(unique(names))))
}

set.seed(1)
X <- matrix(runif(20), ncol = 2)
y <- sin(3 * X[, 1]) + X[, 2]
models <- list(
  Kriging = Kriging(y, X, "gauss"),
  WarpKriging = WarpKriging(y, X, warping = c("none", "none"), kernel = "gauss"),
  MLPKriging = MLPKriging(y, X, hidden_dims = 2),
  NestedKriging = NestedKriging(y, X, "gauss", nb_groups = 2)
)
for (scope in names(models)) {
  obj <- models[[scope]]
  add(scope, Filter(function(n) is.function(obj[[n]]), names(obj)))
}

internals <- ls(ns, all.names = TRUE)
for (scope in names(models)) {
  s3 <- grep(paste0("^.+\\.", scope, "$"), internals, value = TRUE)
  s3 <- Filter(function(n) is.function(get(n, envir = ns)), s3)
  add(scope, sub(paste0("\\.", scope, "$"), "", s3))
}
# free functions of the namespace standing for a static method of a class
statics <- list(Kriging = "subsetOfData")
for (scope in names(statics)) {
  add(scope, Filter(function(n) exists(n, envir = ns, inherits = FALSE), statics[[scope]]))
}

# R test wrappers of C++ internals, not settings of the library: the
# Cholesky numerics below and the random generator (random_*, not listed)
TEST_HELPERS <- c("linalg_chol_safe", "linalg_chol_block", "linalg_rcond_chol", "linalg_rcond_approx_chol")

prefixes <- c(Optim = "optim_", LinearAlgebra = "linalg_")
for (scope in names(prefixes)) {
  fns <- setdiff(grep(paste0("^", prefixes[[scope]]), internals, value = TRUE), TEST_HELPERS)
  fns <- Filter(function(n) is.function(get(n, envir = ns)), fns)
  add(scope, sub(paste0("^", prefixes[[scope]]), "", fns))
}

writeLines(lines, commandArgs(trailingOnly = TRUE)[1])
