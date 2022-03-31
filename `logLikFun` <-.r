`logLikFun` <-
  function(param, model, envir = NULL) {
    if (model@known.param == "Trend") {
      beta <- model@trend.coef
    } else {
      beta <- NULL
    }

    if (identical(model@case, "LLconcentration_beta_sigma2")) {
      model@covariance <- vect2covparam(model@covariance, param)
      model@covariance@sd2 <- 1 # to get the correlation matrix

      aux <- covMatrix(model@covariance, model@X)

      R <- aux[[1]]
      T <- chol(R)

      x <- backsolve(t(T), model@y, upper.tri = FALSE)
      M <- backsolve(t(T), model@F, upper.tri = FALSE)
      z <- compute.z(x = x, M = M, beta = beta)
      sigma2.hat <- compute.sigma2.hat(z)
      logLik <- -0.5 * (model@n * log(2 * pi * sigma2.hat) + 2 * sum(log(diag(T))) + model@n)

      if (!is.null(envir)) {
        envir$T <- T
        envir$R <- R
        envir$z <- z
        envir$sigma2.hat <- sigma2.hat
      }
    } else if (identical(model@case, "LLconcentration_beta")) {
      nparam <- length(param)
      if (class(model@covariance) != "covAdditive0") {
        model@covariance <- vect2covparam(model@covariance, param[1:(nparam - 1)])
        model@covariance@sd2 <- param[nparam]
      } else {
        model@covariance <- vect2covparam(model@covariance, param)
      }

      # 		if (model@covariance@nugget.estim) {
      #       coef(model@covariance, type="all") <- param
      # 		} else {
      #       coef(model@covariance, type="all-nugget") <- param
      # 		}
      # 		# pb ici : reconnaitre les variables a estimer ?
      aux <- covMatrix(model@covariance, model@X, noise.var = model@noise.var)

      C <- aux[[1]]
      vn <- aux[[2]]

      T <- chol(C)
      x <- backsolve(t(T), model@y, upper.tri = FALSE)
      M <- backsolve(t(T), model@F, upper.tri = FALSE)
      z <- compute.z(x = x, M = M, beta = beta)

      logLik <- -0.5 * (model@n * log(2 * pi) + 2 * sum(log(diag(T))) + t(z) %*% z)

      if (!is.null(envir)) {
        envir$T <- T
        envir$C <- C
        envir$vn <- vn
        envir$z <- z
      }
    } else if (identical(model@case, "LLconcentration_beta_v_alpha")) {
      nparam <- length(param)

      model@covariance <- vect2covparam(model@covariance, param[1:(nparam - 1)])
      model@covariance@sd2 <- 1
      model@covariance@nugget <- 0
      alpha <- param[nparam]

      aux <- covMatrix(model@covariance, model@X)
      R0 <- aux[[1]] - diag(aux[[2]]) !!! diag=1 !!!
      R <- alpha * R0 + (1 - alpha) * diag(model@n) !!! diag=1 !!!

      T <- chol(R)
      x <- backsolve(t(T), model@y, upper.tri = FALSE)
      M <- backsolve(t(T), model@F, upper.tri = FALSE)
      z <- compute.z(x = x, M = M, beta = beta)
      v <- compute.sigma2.hat(z)

      logLik <- -0.5 * (model@n * log(2 * pi * v) + 2 * sum(log(diag(T))) + model@n)

      if (!is.null(envir)) {
        envir$T <- T
        envir$R0 <- R0
        envir$v <- v
        envir$z <- z
      }
    }


    if (model@method == "PMLE") {
      fun <- match.fun(model@penalty$fun)
      penalty <- -model@n * sum(fun(1 / model@covariance@range.val^2, model@penalty$value))
      logLik <- logLik + penalty
    }

    return(logLik)
  }