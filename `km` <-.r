`km` <-
    function(formula = ~1, design, response, covtype = "matern5_2",
             coef.trend = NULL, coef.cov = NULL, coef.var = NULL,
             nugget = NULL, nugget.estim = FALSE, noise.var = NULL,
             estim.method = "MLE", penalty = NULL,
             optim.method = "BFGS", lower = NULL, upper = NULL,
             parinit = NULL, multistart = 1, control = NULL, gr = TRUE,
             iso = FALSE, scaling = FALSE, knots = NULL, kernel = NULL) {
        if (!is.null(kernel)) {
            covtype <- "covUser"
            nugget.estim <- FALSE
        }

        model <- new("km")
        model@call <- match.call()

        ## formula : remove automatically the response from it
        data <- data.frame(design)
        model@trend.formula <- formula <- drop.response(formula, data = data)
        F <- model.matrix(formula, data = data)

        # X <- as.matrix(design)
        X <- as.matrix(data)
        y <- as.matrix(response)
        model@X <- X
        model@y <- y
        model@d <- ncol(X)
        model@n <- nrow(X)
        model@F <- F
        model@p <- ncol(F)
        model@noise.flag <- (length(noise.var) != 0)
        model@noise.var <- as.numeric(noise.var)

        isTrend <- length(coef.trend) != 0
        isCov <- length(coef.cov) != 0
        isVar <- length(coef.var) != 0

        if ((scaling) & (length(knots) > 0)) {
            knots <- checkNamesList(X1 = X, l2 = knots)
        }

        if ((isTrend && isCov && isVar) || (covtype == "covUser")) {
            known.param <- "All"
            nugget.estim <- FALSE
        } else if ((isTrend) && ((!isCov) || (!isVar))) {
            known.param <- "Trend"
        } else if ((!isTrend) && isCov && isVar) {
            known.param <- "CovAndVar"
            nugget.estim <- FALSE
        } else { # In the other cases: All parameters are estimated (at this stage)
            known.param <- "None"
            coef.var <- coef.cov <- NULL
        }

        if (isCov) { # curious : why 'known.covparam' is not a boolean ??
            known.covparam <- "All"
        } else {
            known.covparam <- "None"
        }

        if (isTRUE(scaling) && is.null(knots)) { # because we need to setup knots to X boundaries if not set (formerl called affineScaling)
              knots <- as.list(data.frame(rbind(apply(FUN = min, 2, X = design), apply(FUN = max, 2, X = design))))
          } # by default will use min & max for each dimension

        model@covariance <- covStruct.create(
            covtype = covtype, d = model@d,
            known.covparam = known.covparam, var.names = colnames(X),
            coef.cov = coef.cov, coef.var = coef.var, nugget = nugget,
            nugget.estim = nugget.estim,
            nugget.flag = ((length(nugget) != 0) || nugget.estim),
            iso = iso, scaling = scaling, knots = knots, kernel = kernel
        )

        model@known.param <- known.param

        ## Now, at least some parameters are unknown
        if (known.param == "All") {
            model@trend.coef <- as.numeric(coef.trend)
            model@param.estim <- FALSE
            validObject(model)
            model <- computeAuxVariables(model)
            return(model)
        }

        if (known.param == "CovAndVar") {
            model@param.estim <- TRUE
            validObject(model)
            model <- computeAuxVariables(model)
            x <- backsolve(t(model@T), model@y, upper.tri = FALSE)
            beta <- compute.beta.hat(x = x, M = model@M)
            z <- compute.z(x = x, M = model@M, beta = beta)
            model@z <- z
            model@trend.coef <- beta
            return(model)
        }

        if (known.param == "Trend") {
            model@trend.coef <- as.numeric(coef.trend)
        }

        if (length(penalty) == 0) {
            if (is.element(estim.method, c("MLE", "LOO"))) {
                model@method <- estim.method
            } else {
                stop("estim.method must be: 'MLE' or 'LOO'")
            }
        } else {
            if (covtype != "gauss") {
                stop("At this stage, Penalized Maximum Likelihood is coded only for Gaussian covariance")
            }
            penalty.set <- c("SCAD")
            if (!is.element(penalty$fun, penalty.set)) {
                stop("At this stage, the penalty #function has to be one of : SCAD")
            }
            if (length(penalty$value) == 0) {
                penalty$value <- sqrt(2 * log(model@n) / model@n) * seq(from = 1, by = 0.5, length = 15)
            }
            penalty$fun.derivative <- paste(penalty$fun, ".derivative", sep = "")
            model@penalty <- penalty
            model@method <- "PMLE"
        }

        model@param.estim <- TRUE
        model@optim.method <- as.character(optim.method)

        if ((length(lower) == 0) || (length(upper) == 0)) {
            bounds <- covParametersBounds(model@covariance, design)
            if (length(lower) == 0) lower <- bounds$lower
            if (length(upper) == 0) upper <- bounds$upper
        }

        if ((multistart > 1) && (optim.method == "gen")) {
            warning("The 'multistart' argument is not used when 'optim.method' is 'gen'.")
            multistart <- 1
        }
        control$multistart <- multistart
        model@lower <- as.numeric(lower)
        model@upper <- as.numeric(upper)
        model@parinit <- as.numeric(parinit)

        if (optim.method == "BFGS") {
            if (length(control$pop.size) == 0) control$pop.size <- 20
            control$pop.size <- max(control$pop.size, multistart)
            if (identical(control$trace, FALSE)) control$trace <- 0
            if ((length(control$trace) == 0) || (identical(control$trace, TRUE))) {
                control$trace <- 3
            }
        }
        if (optim.method == "gen") {
            d <- ncol(design)
            if (length(control$pop.size) == 0) control$pop.size <- min(20, floor(4 + 3 * log(d)))
            if (length(control$max.generations) == 0) control$max.generations <- 5
            if (length(control$wait.generations) == 0) control$wait.generations <- 2
            if (length(control$BFGSburnin) == 0) control$BFGSburnin <- 0
            if (identical(control$trace, FALSE)) {
                control$trace <- 0
            } else {
                control$trace <- 1
            }
        }

        upper.alpha <- control$upper.alpha
        if (length(upper.alpha) == 0) {
            control$upper.alpha <- 1 - 1e-8
        } else if ((upper.alpha < 0) || (upper.alpha > 1)) {
            control$upper.alpha <- 1 - 1e-8
        }

        model@control <- control

        model@gr <- as.logical(gr)

        envir.logLik <- new.env()

        validObject(model, complete = TRUE)

        varStationaryClass <- c("covTensorProduct", "covScaling", "covIso")

        if (length(noise.var) != 0) { # noisy observations
            model@case <- "LLconcentration_beta"
        } else if (!is.element(class(model@covariance), varStationaryClass)) {
            model@case <- "LLconcentration_beta"
        } else { # variance-stationary kernels
            knownNugget <- ((length(nugget) > 0) && (!nugget.estim))
            if (nugget.estim) { # then concentrate / beta, v=sigma^2+tau^2 and alpha=sigma^2/v
                model@case <- "LLconcentration_beta_v_alpha"
            } else if (knownNugget) {
                model@case <- "LLconcentration_beta"
            } else {
                model@case <- "LLconcentration_beta_sigma2"
            }
        }

        #   if ((length(noise.var) != 0) || ((length(nugget) != 0) && (!nugget.estim))) {
        #     model@case <- "Nuggets"
        #   }
        #
        #   if ((length(nugget) == 0) && (!nugget.estim) && (length(noise.var) == 0)) {
        #     model@case <- "NoNugget"
        #   }
        #
        #   if ((length(noise.var) == 0) && (nugget.estim)) {
        #     model@case <- "1Nugget"
        #   }

        #  knownNugget <- (length(nugget)>0) & (!nugget.estim)
        if ((model@method == "LOO") & (model@case != "LLconcentration_beta_sigma2")) {
            stop("leave-One-Out is not available for this model")
        }

        f <- kmEstimate

        if (identical(model@method, "PMLE")) {
            cv <- function(lambda, object, f) {
                object@penalty$value <- lambda
                object@control$trace <- 0
                object <- f(object, envir = envir.logLik)
                criterion <- sum((object@y - leaveOneOut.km(object, type = "UK")$mean)^2)
                return(criterion)
            }

            lambda.val <- model@penalty$value
            nval <- length(lambda.val)
            u <- rep(0, nval)
            for (i in 1L:nval) {
                u[i] <- cv(lambda.val[i], object = model, f)
            }
            # plot(lambda.val, u)
            lambda <- lambda.val[which.min(u)]
            model@penalty$value <- lambda
            model <- f(model, envir = envir.logLik)
        } else {
            model <- f(model, envir = envir.logLik)
        }

        return(model)
    }