# Lists the names exposed by rlibkriging, one "<scope>\t<name>\t<args>" line
# each, for compare.py, which compares the bindings with each other.
# <args>: the arguments, the object excluded, "name" (required) or "name="
# (with a default) separated by ",", "*" for "...".
# Usage: Rscript dump_r.R <output file>
#  - model classes: the `k$name()` members of a fitted object (arguments of
#    the S3 method they forward to), the S3 methods `generic.Class` (load(),
#    summary(), as.list()...) and the free functions standing for a static
#    method (subsetOfData);
#  - Optim / LinearAlgebra: the namespace functions optim_* / linalg_*
#    (exported or not), prefix removed, except the wrappers of C++ internals
#    kept for the R tests (TEST_HELPERS).
suppressPackageStartupMessages(library(rlibkriging))

ns <- asNamespace("rlibkriging")
lines <- character(0)
args_of <- function(f, drop_first = FALSE) {
  fs <- formals(f)
  if (drop_first && length(fs) > 0) fs <- fs[-1]
  if (length(fs) == 0) return("")
  paste(vapply(names(fs), function(a) {
    if (a == "...") return("*")
    # no default: the empty symbol (not assignable to a variable)
    if (identical(fs[[a]], quote(expr = ))) a else paste0(a, "=")
  }, ""), collapse = ",")
}
add <- function(scope, name, args) {
  lines <<- c(lines, paste0(scope, "\t", name, "\t", args))
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
internals <- ls(ns, all.names = TRUE)
s3_method <- function(generic, scope) {
  n <- paste0(generic, ".", scope)
  if (n %in% internals && is.function(get(n, envir = ns))) get(n, envir = ns) else NULL
}

for (scope in names(models)) {
  obj <- models[[scope]]
  members <- sort(Filter(function(n) is.function(obj[[n]]), names(obj)))
  for (n in members) {
    f <- obj[[n]]
    # k$m <- function(...) m(k, ...): the arguments of the S3 method m.Class
    forwarded <- identical(names(formals(f)), "...") && !is.null(s3_method(n, scope))
    add(scope, n, if (forwarded) args_of(s3_method(n, scope), drop_first = TRUE) else args_of(f))
  }
  s3 <- grep(paste0("^.+\\.", scope, "$"), internals, value = TRUE)
  for (n in sort(setdiff(sub(paste0("\\.", scope, "$"), "", s3), members))) {
    f <- s3_method(n, scope)
    # static load(filename) has no object argument
    if (!is.null(f)) add(scope, n, args_of(f, drop_first = n != "load"))
  }
}
# free functions of the namespace standing for a static method of a class
statics <- list(Kriging = "subsetOfData")
for (scope in names(statics)) {
  for (n in statics[[scope]]) {
    if (exists(n, envir = ns, inherits = FALSE)) add(scope, n, args_of(get(n, envir = ns)))
  }
}

# R test wrappers of C++ internals, not settings of the library: the
# Cholesky numerics below and the random generator (random_*, not listed)
TEST_HELPERS <- c("linalg_chol_safe", "linalg_chol_block", "linalg_rcond_chol", "linalg_rcond_approx_chol")

prefixes <- c(Optim = "optim_", LinearAlgebra = "linalg_")
for (scope in names(prefixes)) {
  fns <- setdiff(grep(paste0("^", prefixes[[scope]]), internals, value = TRUE), TEST_HELPERS)
  fns <- Filter(function(n) is.function(get(n, envir = ns)), fns)
  for (n in sort(fns)) add(scope, sub(paste0("^", prefixes[[scope]]), "", n), args_of(get(n, envir = ns)))
}

writeLines(lines, commandArgs(trailingOnly = TRUE)[1])
