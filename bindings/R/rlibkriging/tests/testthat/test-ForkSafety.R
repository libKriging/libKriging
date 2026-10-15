context("forked workers (parallel::mclapply) after a fit do not deadlock (#325)")

# A forked child that inherits a locked idle BLAS / OpenMP thread pool blocks
# forever in futex(): the pthread_atfork handlers of libKriging quiesce both
# before fork(). Children are collected with a timeout so that a deadlock
# fails the test instead of hanging the suite.

test_that("fit / predict in forked children after a multistart fit in the parent", {
  skip_on_os("windows")  # no fork()
  skip_on_cran()
  set.seed(1)
  n <- 60
  X <- matrix(runif(n * 4), ncol = 4)
  y <- rowSums(sin(3 * X))
  k <- Kriging(y, X, "matern5_2", optim = "BFGS10")  # spins up the BLAS / OpenMP pools
  x <- matrix(runif(40), ncol = 4)
  p_parent <- predict(k, x)$mean

  jobs <- lapply(1:4, function(i) parallel::mcparallel({
    ki <- Kriging(y, X, "matern5_2", optim = "BFGS4")
    list(refit = predict(ki, x)$mean, inherited = predict(k, x)$mean)
  }))
  pids <- vapply(jobs, function(j) j$pid, integer(1))
  res <- list()
  deadline <- Sys.time() + 120
  while (length(res) < length(jobs) && Sys.time() < deadline) {
    todo <- jobs[!(as.character(pids) %in% names(res))]
    got <- parallel::mccollect(todo, wait = FALSE, timeout = 1)
    if (!is.null(got))
      res[names(got)] <- got
  }
  done <- as.character(pids) %in% names(res)
  if (!all(done)) {  # deadlocked children: kill them before failing
    tools::pskill(pids[!done])
    parallel::mccollect(jobs[!done], wait = FALSE)
  }
  expect_true(all(done), info = "forked children did not finish within 120 s (deadlock?)")
  for (r in res) {
    expect_false(inherits(r, "try-error"))
    expect_equal(r$inherited, p_parent)
    expect_true(all(is.finite(r$refit)))
  }
})
