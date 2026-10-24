# Lists the names exposed by rlibkriging, one "<scope>\t<name>" line each,
# for compare.py, which compares the bindings with each other.
# Usage: Rscript dump_r.R <output file>
#  - model classes: the `k$name()` members of a fitted object;
#  - Optim / LinearAlgebra / Random: the namespace functions optim_* /
#    linalg_* / random_* (exported or not), prefix removed.
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
prefixes <- c(Optim = "optim_", LinearAlgebra = "linalg_", Random = "random_")
for (scope in names(prefixes)) {
  fns <- grep(paste0("^", prefixes[[scope]]), internals, value = TRUE)
  fns <- Filter(function(n) is.function(get(n, envir = ns)), fns)
  add(scope, sub(paste0("^", prefixes[[scope]]), "", fns))
}

writeLines(lines, commandArgs(trailingOnly = TRUE)[1])
