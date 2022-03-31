`km1Nugget.init` <-
    function(model) {
        n <- nrow(model@X)
        parinit <- model@parinit

        # standard estimate of sigma^2 + nugget^2 (total variance)
        trend.estimate <- lm(model@y ~ model@F - 1)
        random.part.estimate <- trend.estimate$residuals
        sigma.total.standard <- sd(random.part.estimate)

        # total variance estimate using the variogram
        x.dist <- dist(model@X)
        y.dist <- dist(random.part.estimate)

        alpha <- max(min(0.5, 100 / (n * (n - 1))), 10 / 100)
        I <- (x.dist >= quantile(x.dist, alpha))
        sigma.total.vario <- sqrt(1 / 2 * mean(y.dist[I]^2))

        # both are negatively biased, take the max
        sigma.total <- max(sigma.total.standard, sigma.total.vario)

        # If realistic keep the user value for nugget, otherwise estimate it using the variogram
        # Deduce variance estimate from the total variance estimate and the nugget estimate
        sq.nugget.init <- sqrt(model@covariance@nugget)

        if (identical(sq.nugget.init, numeric(0))) sq.nugget.init <- -1

        if ((sq.nugget.init >= 0) & (sq.nugget.init <= sigma.total)) {
            sigma.init <- sqrt(sigma.total^2 - sq.nugget.init^2)
        } else {
            # variogram behaviour at 0
            alpha <- max(min(0.5, 20 / (n * (n - 1))), 1 / 100)
            I <- (x.dist <= quantile(x.dist, alpha))
            mreg <- lm(y.dist[I]^2 ~ I(x.dist[I]^2))
            nugget.init <- mreg$coef[1]

            # if too small, small ; if too big, big
            if (nugget.init <= 0) {
                sq.nugget.init <- sqrt(1 / 10) * sigma.total
                sigma.init <- sqrt(9 / 10) * sigma.total
            } else if (nugget.init >= sigma.total^2) {
                sq.nugget.init <- sqrt(9 / 10) * sigma.total
                sigma.init <- sqrt(1 / 10) * sigma.total
            } else {
                sq.nugget.init <- sqrt(nugget.init)
                sigma.init <- sqrt(sigma.total^2 - sq.nugget.init^2)
            }
        }

        lower <- model@lower
        upper <- model@upper
        ninit <- model@control$pop.size
        param.n <- model@covariance@param.n

        if (length(parinit) > 0) {
            matrixinit <- matrix(parinit, nrow = param.n, ncol = ninit)
        } else {
            if (existsMethod("paramSample", signature = class(model@covariance))) {
                matrixinit <- paramSample(model@covariance, n = ninit, lower = lower, upper = upper, y = model@y)
            } else {
                # sample ninit design points, generated from uniform [lower, upper]
                matrixinit <- matrix(runif(ninit * param.n), nrow = param.n, ncol = ninit)
                matrixinit <- lower + matrixinit * (upper - lower)
            }
        }

        # simulate nugget and sigma s.t.
        #  (0.5*sigma.total)^2 <= sigma^2 + nugget^2 <= (1.5*sigma.total)^2
        #  nugget.min < nugget < nugget.max
        # uniform simulation on a domain included in a disc
        angle.init <- atan(sigma.init / sq.nugget.init)
        radius.init <- sigma.total
        radius.min <- 1 / 2 * radius.init
        radius.max <- 3 / 2 * radius.init
        angle.sim <- runif(n = ninit, min = 1 / 2 * angle.init, max = min(3 / 2 * angle.init, pi / 2))
        radius.sim <- sqrt(2 * runif(n = ninit, min = 1 / 2 * radius.min^2, max = 1 / 2 * radius.max^2))
        sq.nugget.init.sim <- radius.sim * cos(angle.sim)
        sigma.init.sim <- radius.sim * sin(angle.sim)

        # take the best point(s)
        alphainit <- sigma.init.sim^2 / (sigma.init.sim^2 + sq.nugget.init.sim^2)
        matrixinit <- rbind(matrixinit, alphainit)
        fninit <- apply(matrixinit, 2, logLikFun, model)
        selection <- sort(fninit, decreasing = TRUE, index.return = TRUE)$ix
        selection <- selection[1:model@control$multistart]
        parinit <- matrixinit[, selection, drop = FALSE]
        # for one point : parinit <- matrixinit[, which.max(logLikinit)]

        lp <- nrow(parinit)
        covinit <- list()
        for (i in 1:model@control$multistart) {
            pari <- as.numeric(parinit[, i])
            covinit[[i]] <- vect2covparam(model@covariance, pari[1:(lp - 1)])
            covinit[[i]]@nugget <- pari[lp]
            covinit[[i]]@sd2 <- pari[lp - 1]
        }

        return(list(
            par = parinit,
            value = fninit[selection],
            cov = covinit,
            lower = c(lower, 0),
            upper = c(upper, model@control$upper.alpha)
        ))
    }