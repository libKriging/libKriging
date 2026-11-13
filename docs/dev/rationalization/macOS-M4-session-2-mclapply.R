# Scenario of #457 / #459: forked R workers (parallel::mclapply) doing large Accelerate calls.
# Parent warmed up with a multistart fit, then rounds of children: BFGS4 fit, predict (stdev, cov),
# simulate at 300 points (the first large call crashed before #459), update. Usage:
#   R_LIBS=<lib with rlibkriging> Rscript macOS-M4-session-2-mclapply.R
suppressMessages({library(rlibkriging); library(parallel)})
set.seed(2)
mk <- function(n, d = 4) { X <- matrix(runif(n * d), n, d); y <- apply(X, 1, function(x) sum(sin(3 * x)) + x[1] * x[2]); list(X = X, y = y) }
d <- mk(500); Kriging(d$y, d$X, "matern5_2", optim = "BFGS4")   # parent: loops, multistart threads
tot <- 0; rounds <- 5; kids <- 8
for (rep in 1:rounds) {
  r <- mclapply(1:kids, function(i) {
    d <- mk(400); k <- Kriging(d$y, d$X, "gauss", optim = "BFGS4")
    p <- predict(k, matrix(runif(400 * 4), 400, 4), TRUE, TRUE)
    s <- simulate(k, nsim = 3, seed = i, x = matrix(runif(300 * 4), 300, 4))
    u <- update(k, mk(10)$y, mk(10)$X); 1 }, mc.cores = kids)
  tot <- tot + sum(sapply(r, is.numeric))
}
cat("MCLAPPLY: ", tot, "/", rounds * kids, "children ok\n")
t0 <- proc.time()[3]; d <- mk(500); Kriging(d$y, d$X, "matern5_2"); cat("parent still fits n=500 in", round(proc.time()[3] - t0, 2), "s\n")
