## zzz.R – package load / unload hooks

## When packages such as RobustGaSP replace stats::simulate with a new S4
## generic they typically do NOT provide an S4 method for every S3 class, so
## the S3 method simulate.WarpKriging is no longer reachable via S4 dispatch.
## We work around this by re-registering an explicit S4 method whenever a
## relevant package is loaded (or immediately if simulate is already an S4
## generic when rlibkriging itself loads).
##
## Note: setMethod is called with where = .GlobalEnv because the rlibkriging
## namespace is locked after loading and cannot be modified at that point.

.register_warpkriging_simulate_s4 <- function() {
  # isGeneric() can return TRUE for implicit generics (methods package lazy
  # registration of S3 base functions) where setMethod() still fails.
  # Use tryCatch so failures never propagate to .onLoad.
  tryCatch({
    if (methods::isGeneric("simulate") &&
        !is.null(tryCatch(methods::getGeneric("simulate"), error = function(e) NULL)) &&
        !methods::existsMethod("simulate", "WarpKriging")) {
      methods::setMethod(
        "simulate", "WarpKriging",
        function(object, nsim = 1, seed = 123, x, will_update = FALSE, ...) {
          rlibkriging:::simulate.WarpKriging(object,
                               nsim = nsim, seed = seed, x = x,
                               will_update = will_update, ...)
        },
        where = .GlobalEnv)
    }
  }, error = function(e) NULL)
}

.onLoad <- function(libname, pkgname) {
  ## Handle the (unlikely) case where RobustGaSP / DiceKriging already loaded.
  .register_warpkriging_simulate_s4()

  ## Re-register whenever a package that creates a new simulate S4 generic loads.
  setHook(packageEvent("RobustGaSP",  "onLoad"),
          function(...) .register_warpkriging_simulate_s4())
  setHook(packageEvent("DiceKriging", "onLoad"),
          function(...) .register_warpkriging_simulate_s4())
}
