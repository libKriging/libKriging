#!/usr/bin/env Rscript
# Benchmark rlibkriging against DiceKriging and RobustGaSP on the shared
# datasets produced by make_datasets.py (identical designs as the Python run).
#
# Common modelling choices: Matern 5/2 ARD kernel, constant trend,
# interpolation (no nugget), hyperparameters by MLE (package defaults).
# Per-fit wall-clock budget via setTimeLimit (default 300 s).
#
# Usage: Rscript run_r.R [data_dir] [out_csv] [budget_seconds]

args <- commandArgs(trailingOnly = TRUE)
data_dir <- ifelse(length(args) >= 1, args[1], "data")
out_csv  <- ifelse(length(args) >= 2, args[2], "results/r.csv")
budget   <- ifelse(length(args) >= 3, as.numeric(args[3]), 300)

dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)

read_mat <- function(p) as.matrix(read.csv(p, header = FALSE))
read_vec <- function(p) as.numeric(read.csv(p, header = FALSE)[[1]])

fitters <- list(
  rlibkriging = list(
    fit = function(X, y) rlibkriging::Kriging(
      y, X, kernel = "matern5_2", regmodel = "constant", objective = "LL"),
    pred = function(m, Xt) {
      p <- rlibkriging::predict(m, Xt, return_stdev = TRUE)
      list(mu = as.numeric(p$mean), sd = as.numeric(p$stdev))
    }),
  DiceKriging = list(
    fit = function(X, y) DiceKriging::km(
      formula = ~1, design = X, response = y, covtype = "matern5_2",
      nugget.estim = FALSE, control = list(trace = FALSE)),
    pred = function(m, Xt) {
      p <- DiceKriging::predict(m, newdata = Xt, type = "UK",
                                se.compute = TRUE, checkNames = FALSE)
      list(mu = p$mean, sd = p$sd)
    }),
  RobustGaSP = list(
    fit = function(X, y) RobustGaSP::rgasp(
      design = X, response = y, kernel_type = "matern_5_2"),
    pred = function(m, Xt) {
      p <- RobustGaSP::predict(m, Xt)
      list(mu = p$mean, sd = p$sd)
    })
)

metrics <- function(y, mu, sd) {
  rmse <- sqrt(mean((y - mu)^2))
  q2 <- 1 - sum((y - mu)^2) / sum((y - mean(y))^2)
  s2 <- pmax(sd, 1e-12)^2
  nlpd <- mean(0.5 * log(2 * pi * s2) + 0.5 * (y - mu)^2 / s2)
  c(rmse = rmse, q2 = q2, nlpd = nlpd)
}

rows <- c("func,d,n,rep,package,fit_time,pred_time,rmse,q2,nlpd,status")

for (xtr in sort(Sys.glob(file.path(data_dir, "*", "n*", "rep*", "X_train.csv")))) {
  rdir <- dirname(xtr); ndir <- dirname(rdir); fdir <- dirname(ndir)
  func <- basename(fdir)
  n <- as.integer(sub("^n", "", basename(ndir)))
  rep <- as.integer(sub("^rep", "", basename(rdir)))
  X  <- read_mat(xtr)
  y  <- read_vec(file.path(rdir, "y_train.csv"))
  Xt <- read_mat(file.path(fdir, "X_test.csv"))
  yt <- read_vec(file.path(fdir, "y_test.csv"))
  d <- ncol(X)
  colnames(X) <- colnames(Xt) <- paste0("x", seq_len(d))

  for (pkg in names(fitters)) {
    fit_time <- pred_time <- rmse <- q2 <- nlpd <- NA
    status <- "ok"
    res <- tryCatch({
      setTimeLimit(elapsed = budget, transient = TRUE)
      t0 <- proc.time()[["elapsed"]]
      m <- fitters[[pkg]]$fit(X, y)
      fit_time <- proc.time()[["elapsed"]] - t0
      t0 <- proc.time()[["elapsed"]]
      p <- fitters[[pkg]]$pred(m, Xt)
      pred_time <- proc.time()[["elapsed"]] - t0
      setTimeLimit(elapsed = Inf)
      metrics(yt, p$mu, p$sd)
    }, error = function(e) {
      setTimeLimit(elapsed = Inf)
      status <<- if (grepl("time limit", conditionMessage(e))) "timeout" else "error"
      message(sprintf("[%s %s n=%d rep=%d] %s", pkg, func, n, rep,
                      conditionMessage(e)))
      c(rmse = NA, q2 = NA, nlpd = NA)
    })
    rows <- c(rows, sprintf("%s,%d,%d,%d,%s,%s,%s,%s,%s,%s,%s",
                            func, d, n, rep, pkg,
                            fit_time, pred_time,
                            res[["rmse"]], res[["q2"]], res[["nlpd"]], status))
    cat(tail(rows, 1), "\n")
  }
}

writeLines(rows, out_csv)
