library(testthat)

# Lists the names exposed by rlibkriging, one "<scope>\t<name>" line each,
# for tools/checks/binding_api.py, which compares them to bindings/api.yaml.
# Written to $LK_API_DUMP_DIR/r.txt when that variable is set, only checked
# for non-emptiness otherwise.
#  - module: the exported names, except the members of a class below;
#  - model classes: the `k$name()` members of a fitted object;
#  - Optim / LinearAlgebra / Random: the namespace functions optim_* /
#    linalg_* / random_* (exported or not), prefix removed.

api_dump <- function() {
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

  # module: the exported names that are not members of a class above
  # (S3 generics such as fit or predict), S4 method tables (.__T__*) excluded
  members <- sub("^[^\t]*\t", "", lines)
  exports <- grep("^\\.", getNamespaceExports("rlibkriging"), value = TRUE, invert = TRUE)
  add("module", setdiff(exports, members))

  internals <- ls(ns, all.names = TRUE)
  prefixes <- c(Optim = "optim_", LinearAlgebra = "linalg_", Random = "random_")
  for (scope in names(prefixes)) {
    fns <- grep(paste0("^", prefixes[[scope]]), internals, value = TRUE)
    fns <- Filter(function(n) is.function(get(n, envir = ns)), fns)
    add(scope, sub(paste0("^", prefixes[[scope]]), "", fns))
  }
  lines
}

test_that("the binding API can be listed", {
  lines <- api_dump()
  scopes <- sub("\t.*", "", lines)
  for (scope in c("module", "Kriging", "WarpKriging", "MLPKriging", "NestedKriging",
                  "Optim", "LinearAlgebra", "Random"))
    expect_true(scope %in% scopes, info = scope)
  out_dir <- Sys.getenv("LK_API_DUMP_DIR")
  if (nzchar(out_dir)) {
    dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
    writeLines(lines, file.path(out_dir, "r.txt"))
  }
})
