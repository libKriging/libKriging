context("forked workers (parallel::mclapply) after a fit do not deadlock (#325)")

# A forked child that inherits a locked idle BLAS / OpenMP thread pool blocks
# forever in futex(): the pthread_atfork handlers of libKriging quiesce both
# before fork(). Children are collected with a timeout so that a deadlock
# fails the test instead of hanging the suite.

# Collects mcparallel() jobs, waiting at most `timeout` seconds; deadlocked
# children are killed. Returns the results of the finished ones, by pid.
collect_or_kill <- function(jobs, timeout = 120) {
  pids <- vapply(jobs, function(j) j$pid, integer(1))
  res <- list()
  deadline <- Sys.time() + timeout
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
  expect_true(all(done), info = paste("forked children did not finish within", timeout, "s (deadlock?)"))
  res
}

# Thread counts seen by libKriging: budget, OpenBLAS count (0: not OpenBLAS),
# OpenMP limit of the calling thread (0: no OpenMP)
thread_counts <- function() {
  c(budget = rlibkriging:::optim_get_num_threads(),
    blas = rlibkriging:::optim_get_blas_num_threads(),
    omp = rlibkriging:::optim_get_omp_num_threads())
}

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
  res <- collect_or_kill(jobs)
  for (r in res) {
    expect_false(inherits(r, "try-error"))
    expect_equal(r$inherited, p_parent)
    expect_true(all(is.finite(r$refit)))
  }
})

# The child handler sets the OpenBLAS count to 1 and caps the thread budget at
# 1, so a child runs single-threaded, and a fit in it does not raise them. The
# OpenMP limit is not checked in the child: LLVM libomp's own child handler
# resets it (macOS, Clang builds); the budget bounds the library's OpenMP loops
# whatever it is. The parent handler restores the parent's counts after fork().
test_that("forked children run with 1 thread; the parent's counts are unchanged (#420)", {
  skip_on_os("windows")  # no fork()
  skip_on_cran()
  set.seed(2)
  n <- 60
  X <- matrix(runif(n * 3), ncol = 3)
  y <- rowSums(sin(3 * X))
  k <- Kriging(y, X, "matern5_2", optim = "BFGS10")  # spins up the BLAS / OpenMP pools
  parent_before <- thread_counts()

  jobs <- lapply(1:3, function(i) parallel::mcparallel({
    before <- thread_counts()
    ki <- Kriging(y, X, "matern5_2", optim = "BFGS4")
    list(before = before, after = thread_counts(), ll = logLikelihood(ki))
  }))
  res <- collect_or_kill(jobs)
  parent_after <- thread_counts()

  for (r in res) {
    expect_false(inherits(r, "try-error"))
    expect_true(is.finite(r$ll))
    for (counts in list(r$before, r$after)) {
      expect_equal(counts[["budget"]], 1)
      if (parent_before[["blas"]] > 0)  # OpenBLAS: its count can be read
        expect_equal(counts[["blas"]], 1)
    }
  }
  expect_equal(parent_after, parent_before)
})
