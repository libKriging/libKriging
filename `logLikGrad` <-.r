`logLikGrad` <-
    function(param, model, envir) {
        if (identical(model@case, "LLconcentration_beta_sigma2")) {
            nparam <- length(param)

            R <- envir$R
            T <- envir$T
            z <- envir$z
            sigma2.hat <- envir$sigma2.hat

            model@covariance <- vect2covparam(model@covariance, param)
            model@covariance@sd2 <- 1 # to get the correlation matrix



            logLik.derivative <- matrix(0, nparam, 1)

            x <- backsolve(T, z) # compute x := T^(-1)*z
            Rinv <- chol2inv(T) # compute inv(R) by inverting T

            Rinv.upper <- Rinv[upper.tri(Rinv)]
            xx <- x %*% t(x)
            xx.upper <- xx[upper.tri(xx)]

            for (k in 1:nparam) {
                gradR.k <- covMatrixDerivative(model@covariance, X = model@X, C0 = R, k = k)
                gradR.k.upper <- gradR.k[upper.tri(gradR.k)]

                terme1 <- sum(xx.upper * gradR.k.upper) / sigma2.hat
                # quick computation of t(x)%*%gradR.k%*%x /  ...
                terme2 <- -sum(Rinv.upper * gradR.k.upper)
                # quick computation of trace(Rinv%*%gradR.k)
                logLik.derivative[k] <- terme1 + terme2
            }







            if (model@method == "PMLE") {
                fun.derivative <- match.fun(model@penalty$fun.derivative)
                penalty <- -model@n * fun.derivative(1 / param^2, model@penalty$value) * ((-2) / param^3)
                logLik.derivative <- logLik.derivative + penalty
            }
        } else if (identical(model@case, "LLconcentration_beta")) {
            nparam <- length(param)

            model@covariance <- vect2covparam(model@covariance, param[1:(nparam - 1)])
            model@covariance@sd2 <- sigma2 <- param[nparam]

            logLik.derivative <- matrix(0, nparam, 1)

            C <- envir$C
            T <- envir$T
            vn <- envir$vn
            z <- envir$z

            x <- backsolve(T, z) # x := T^(-1)*z
            Cinv <- chol2inv(T) # Invert R from given T

            for (k in 1:(nparam)) {
                gradC.k <- covMatrixDerivative(model@covariance,
                    X = model@X,
                    C0 = C - diag(vn, nrow = nrow(C)), k = k
                )
                term1 <- -t(x) %*% gradC.k %*% x
                term2 <- sum(Cinv * gradC.k) # economic computation of trace(Cinv%*%gradC.k)
                logLik.derivative[k] <- -0.5 * (term1 + term2) # /sigma2
            }

            if (model@method == "PMLE") {
                fun.derivative <- match.fun(model@penalty$fun.derivative)
                param.pen <- model@covariance@range.val
                param.pen.pos <- 1:(nparam - 1)
                penalty <- -model@n * fun.derivative(1 / param.pen^2, model@penalty$value) * ((-2) / param.pen^3)
                logLik.derivative[param.pen.pos] <- logLik.derivative[param.pen.pos] + penalty
            }
        } else if (identical(model@case, "LLconcentration_beta_v_alpha")) {
            nparam <- length(param)

            R0 <- envir$R0
            T <- envir$T
            z <- envir$z
            v <- envir$v
            
            model@covariance <- vect2covparam(model@covariance, param[1:(nparam - 1)])
            alpha <- param[nparam]
            model@covariance@sd2 <- 1 # sigma2 <- param[nparam-1]
            model@covariance@nugget <- 0 # param[nparam]

            logLik.derivative <- matrix(0, nparam, 1)

            x <- backsolve(T, z) # x := T^(-1)*z
            Cinv <- chol2inv(T) # Invert R from given T

            Cinv.upper <- Cinv[upper.tri(Cinv)]
            xx <- x %*% t(x)
            xx.upper <- xx[upper.tri(xx)]

            # partial derivative with respect to parameters except sigma^2
            for (k in 1:(nparam - 1)) {
                gradC.k <- covMatrixDerivative(model@covariance, X = model@X, C0 = R0, k = k)
                gradC.k <- alpha * gradC.k
                gradC.k.upper <- gradC.k[upper.tri(gradC.k)]
                term1 <- sum(xx.upper * gradC.k.upper) / v
                # economic computation of - t(x)%*%gradC.k%*%x / v
                term2 <- -sum(Cinv.upper * gradC.k.upper)
                # economic computation of trace(Cinv%*%gradC.k)
                logLik.derivative[k] <- term1 + term2
            }
            # partial derivative with respect to v = sigma^2 + delta^2
            dCdv <- R0 - diag(model@n)
            term1 <- -t(x) %*% dCdv %*% x / v
            term2 <- sum(Cinv * dCdv) # economic computation of trace(Cinv%*%C0)
            logLik.derivative[nparam] <- -0.5 * (term1 + term2) # /sigma2

            # partial derivative with respect to delta2
            # term1 <- -t(x)%*%x
            # term2 <- sum(diag(Cinv))
            # logLik.derivative[nparam] <- -0.5*(term1 + term2)

            if (model@method == "PMLE") {
                fun.derivative <- match.fun(model@penalty$fun.derivative)
                param.pen <- model@covariance@range.val
                param.pen.pos <- 1:(nparam - 2)
                penalty <- -model@n * fun.derivative(1 / param.pen^2, model@penalty$value) * ((-2) / param.pen^3)
                logLik.derivative[param.pen.pos] <- logLik.derivative[param.pen.pos] + penalty
            }
        }


        return(logLik.derivative)
    }