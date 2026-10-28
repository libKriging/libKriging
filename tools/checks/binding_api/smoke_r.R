# Smoke calls of rlibkriging (issue #435): each family of methods is called
# once on a tiny model (n = 10, d = 2) with the argument types a user would
# pass (an integer and a numeric), and the settings of Optim / LinearAlgebra
# are checked by a get / set / get round trip (smoke_setters.tsv). Prints one
# "ok" / "FAIL" line per call, exits with status 1 on any failure.
# Usage: Rscript smoke_r.R [<markdown file to append to>]
suppressPackageStartupMessages(library(rlibkriging))

args <- commandArgs(trailingOnly = TRUE)
here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
failures <- character(0)
ncalls <- 0

check <- function(label, expr) {
  ncalls <<- ncalls + 1
  res <- tryCatch({ force(expr); NULL }, error = function(e) conditionMessage(e))
  if (is.null(res)) cat("ok    ", label, "\n", sep = "")
  else {
    failures <<- c(failures, label)
    cat("FAIL  ", label, ": ", strsplit(res, "\n")[[1]][1], "\n", sep = "")
  }
}

# optim_* / linalg_* functions of the namespace
fn <- function(scope, name) {
  prefix <- c(Optim = "optim_", LinearAlgebra = "linalg_")[[scope]]
  get(paste0(prefix, name), envir = asNamespace("rlibkriging"))
}

setter_roundtrip <- function(scope, setter, getter, kind, value, as_integer) {
  set <- fn(scope, setter)
  if (getter == "-") return(set(tolower(value) == "true"))
  get <- fn(scope, getter)
  old <- get()
  new <- switch(kind, int = if (as_integer) as.integer(value) else as.numeric(value),
                float = as.numeric(value), bool = !old)
  on.exit(set(old))
  set(new)
  if (!isTRUE(all.equal(get(), new))) stop(sprintf("set %s then get %s", format(new), format(get())))
}

for (line in readLines(file.path(here, "smoke_setters.tsv"))) {
  if (line == "" || startsWith(line, "#")) next
  f <- strsplit(line, "\t")[[1]]
  # an integer setting is given as an integer and as a numeric
  for (as_integer in if (f[4] == "int") c(TRUE, FALSE) else TRUE) {
    check(sprintf("%s.%s(%s)", f[1], f[2], if (f[4] != "int") f[4] else if (as_integer) "integer" else "numeric"),
          setter_roundtrip(f[1], f[2], f[3], f[4], f[5], as_integer))
  }
}

set.seed(1)
X <- matrix(runif(20), ncol = 2)
y <- sin(3 * X[, 1]) + X[, 2]
Xn <- matrix(runif(6), ncol = 2)
Xu <- matrix(runif(4), ncol = 2)
yu <- c(0.5, 0.7)
tmp <- tempfile("smoke")
dir.create(tmp)

models <- list()
build <- function(label, make) check(paste(label, "fit"), models[[label]] <<- make())
build("Kriging", function() Kriging(y, X, "gauss"))
build("WarpKriging", function() WarpKriging(y, X, warping = c("none", "none"), kernel = "gauss"))
build("MLPKriging", function() MLPKriging(y, X, hidden_dims = 2))
build("NestedKriging", function() NestedKriging(y, X, "gauss", nb_groups = 2))

for (label in names(models)) {
  m <- models[[label]]
  # zero-argument members of the model, whatever their name
  skip <- c("fit", "save", "load", "update", "update_simulate", "simulate", "copy", "print", "show", "str")
  for (name in sort(setdiff(names(m), skip))) {
    if (!is.function(m[[name]])) next
    check(sprintf("%s.%s()", label, name), tryCatch(m[[name]](), error = function(e) {
      if (!grepl("argument .* is missing|unused argument|missing", conditionMessage(e))) stop(e)
    }))
  }
  check(paste0(label, ".summary"), summary(m))
  check(paste0(label, ".predict"), predict(m, Xn))
  if (label == "NestedKriging") next
  check(paste0(label, ".simulate"), simulate(m, nsim = 2L, seed = 123L, x = Xn))
  check(paste0(label, ".update"), update(copy(m), yu, Xu))
  check(paste0(label, ".update_simulate"), update_simulate(copy(m), yu, Xu))
  path <- file.path(tmp, paste0(label, ".h5"))
  check(paste0(label, ".save/load"), { save(m, path); rlibkriging::load(path) })
}

if (!is.null(models$Kriging)) {
  k <- models$Kriging
  check("Kriging.logLikelihood", logLikelihood(k))
  check("Kriging.leaveOneOut", leaveOneOut(k))
  check("Kriging.logMargPost", logMargPost(k))
  check("Kriging.covMat", covMat(k, Xn, Xn))
  check("Kriging.logLikelihoodFun", k$logLikelihoodFun(c(0.5, 0.5), TRUE, FALSE))
  check("Kriging.leaveOneOutFun", k$leaveOneOutFun(c(0.5, 0.5), TRUE, FALSE))
  check("Kriging.leaveOneOutVec", k$leaveOneOutVec(c(0.5, 0.5)))
  check("Kriging.subsetOfData", subsetOfData(X, 5L))
}

cat(sprintf("\n%d call(s), %d failure(s)\n", ncalls, length(failures)))
if (length(args) > 0) {
  cat(sprintf("\n## Smoke calls: R\n\n%d call(s), **%d failure(s)**\n\n", ncalls, length(failures)),
      sprintf("- `%s`\n", failures), sep = "", file = args[1], append = TRUE)
}
quit(status = if (length(failures)) 1 else 0)
