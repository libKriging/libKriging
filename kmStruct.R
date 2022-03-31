## ----------------
## CLASS definition
## ----------------

## km Class

setClass("km", 		
         representation( 
                        d = "integer",          ## spatial dimension
                        n = "integer",          ## observations number
			## data
                        X = "matrix",           ## the design of experiments, size nxd
                        y = "matrix",           ## the observations, size nx1
			## trend information
                        p = "integer",             ## 1+number of trend basis functions
                        F = "matrix",              ## the experimental matrix, size nxp
                        trend.formula = "formula", ## trend form
                        trend.coef = "numeric",    ## trend coefficients, size px1
			## covariance
                        covariance = "covKernel",  ## covariance structure (new S4 class, see covStruct.R)
			## noisy observations
                        noise.flag = "logical",    ## Are observations noisy ? 
                        noise.var = "numeric",     ## vector of length n
			## model information
                        known.param = "character", ## which parameters among trend, cov, var are known? ("None", "All", "Trend", "CovAndVar", etc.)
                        case = "character",        ## "NoNugget" : deterministic observations, no nugget effect
                        ## "1Nugget"  : homogenous nugget effect, to be estimated
                        ## "Nuggets"  : known nugget or (exclusive) noisy observations 
			## optimisation 
                        param.estim = "logical",   ## Are parameter estimated ??
                        method = "character",      ## Statistical criterion : "MLE" or "PMLE"
                        penalty = "list",          ## fun ("SCAD"), fun.derivative, value
                        optim.method = "character",  ## Optimisation algorithm : "BFGS" or "gen"
                        lower = "numeric",           ##
                        upper = "numeric",         ## boudaries for parameter estimation. Length covariance@param.n
                        control = "list",          ## pop.size, wait.generations, max.generations, BFGSburnin 
                        gr = "logical",	           ## is analytical gradient to be used ?
                        call = "language",         ## user call
                        parinit = "numeric",       ## initial values used (given by user or computed)
                        logLik = "numeric",        ## objective function value (concentrated log-Likelihood)
			## auxilliary variables 
                        T = "matrix",              ## Upper triang. factor of the Choleski dec. 
                        ## of the cov. matrix : t(T)%*%T = C. Size nxn
                        z = "numeric",             ## t(T)^(-1)*(y - F%*%beta). Size nx1
                        M = "matrix"	           ## t(T)^(-1)%*%F. Size nxp
                        ), 
         validity = function(object) {
           if (object@n <= object@d) {
             return("the number of experiments must be larger than the spatial dimension")
           }
           
           if (ncol(object@y) != 1) {
             return("the response must be 1-dimensional")
           }
           
           if (!identical(nrow(object@X), nrow(object@y))) {
             return("the number of observations is not equal to the number of experiments")
           }
          
           if (object@covariance@nugget.flag & object@noise.flag) {
             return("'nugget' and 'noise' cannot be specified together")
           }
           
           if (object@noise.flag) {	
             if (!identical(length(object@noise.var), object@n)) {
               return("the length of the vector 'noise.var' must be equal to the number of experiments")
             }
           }
           
           if (!identical(object@trend.coef, numeric(0))) {
             if (!identical(length(object@trend.coef), object@p)) {
               return("the number of trend coefficients is not compatible with the trend formula")
             }
           }
           TRUE
         }
         )

if(!isGeneric("show")) {
  setGeneric(name = "show",
             def = function(object) standardGeneric("show")
             )
}

setMethod("show", "km", 
          function(object){
            show.km(object)		
          }
          )

##*****************************************************************************
##                            C O E F  METHOD
##*****************************************************************************

setMethod("coef", "km", 
          function(object, type = "all", as.list = TRUE){
            if (is.element(type, c("all", "all-nugget", "all-sd2-nugget"))) {
              val <- c(trend=object@trend.coef, coef(object@covariance, type = type))
              if (!as.list) val <- unlist(val, use.names = FALSE)
            } else if (type=="trend") {
              val <- object@trend.coef
            } else {
              val <- coef(object@covariance, type = type, as.list=as.list)
            }
            return(val)
          }
)

##*****************************************************************************
##                            P L O T  METHOD
##*****************************************************************************

plot.km <- function(x, kriging.type = "UK", trend.reestim=FALSE, ...) {
  
  model <- x
  pred <- leaveOneOut.km(model, type=kriging.type, 
                         trend.reestim=trend.reestim)
  y <- as.matrix(model@y)
  yhat <- pred$mean
  sigma <- pred$sd
  
  resid <- (y-yhat)/sigma
                                        #par(ask=TRUE)
  xmin <- min(min(yhat), min(y))
  xmax <- max(max(yhat), max(y))
  
  par(mfrow=c(3,1))
  plot(x = y[,1], y = yhat,
       xlim = c(xmin, xmax), ylim = c(xmin, xmax),
       xlab = "Exact values", ylab = "Fitted values",
       main = "Leave-one-out", ...)
  lines(x = c(xmin, xmax), y = c(xmin, xmax))
  plot(resid, xlab = "Index", ylab = "Standardized residuals",
       main = "Standardized residuals", ...)
  qqnorm(resid, main = "Normal QQ-plot of standardized residuals") 
  qqline(resid)
  par(mfrow = c(1, 1))
  
  invisible(pred)
}

if(!isGeneric("plot")) {
  setGeneric(name = "plot",
             def = function(x, y, ...) standardGeneric("plot")
             )
}

setMethod("plot",
          signature(x = "km"), 
          function(x, y, kriging.type = "UK", trend.reestim=FALSE, ...) {
            if (!missing(y)) warning("Argument y is ignored (not used in this plot method)")
            plot.km(x = x, kriging.type = kriging.type, 
                           trend.reestim=trend.reestim, ...)
          }
          )

##*****************************************************************************
##                        P R E D I C T  METHOD
##*****************************************************************************


predict.km <- function(object, newdata, type,
                       se.compute = TRUE, cov.compute = FALSE, light.return = FALSE,
                       bias.correct = FALSE, checkNames = TRUE, ...) {
  ## newdata : n x d
  
  nugget.flag <- object@covariance@nugget.flag 
  
  X <- object@X
  y <- object@y
  
  if (checkNames) {
    newdata <- checkNames(X1 = X, X2 = newdata, X1.name = "the design", X2.name = "newdata")
  } else {
    newdata <- as.matrix(newdata)
    d.newdata <- ncol(newdata)
    if (!identical(d.newdata, object@d)) {
      stop("newdata must have the same numbers of columns than the experimental design")
    }
    if (!identical(colnames(newdata), colnames(X))) {
      ##  warning("column names mismatch between 'newdata' and the experimental design -
      ## the columns of 'newdata' are interpreted in the same order as the experimental design names")
      colnames(newdata) <- colnames(X)
    }
  }
  
  T <- object@T
  z <- object@z
  M <- object@M
  
  beta <- object@trend.coef
    
  F.newdata <- model.matrix(object@trend.formula, data = data.frame(newdata))
  y.predict.trend <- F.newdata%*%beta
  
  c.newdata <- covMat1Mat2(object@covariance, X1 = X, X2 = newdata,
                           nugget.flag = object@covariance@nugget.flag)
  ## compute c(x) for x = newdata ; remark that for prediction (or filtering), cov(Yi, Yj)=0
  ## even if Yi and Yj are the outputs related to the equal points xi and xj.
  
  Tinv.c.newdata <- backsolve(t(T), c.newdata, upper.tri=FALSE)
  y.predict.complement <- t(Tinv.c.newdata)%*%z
  y.predict <- y.predict.trend + y.predict.complement
  y.predict <- as.numeric(y.predict)
  
  output.list <- list()
  output.list$trend <- y.predict.trend
  output.list$mean <- y.predict
  
  if (!light.return) {
    output.list$c <- c.newdata
    output.list$Tinv.c <- Tinv.c.newdata
  } 
  
  
  ## A FAIRE : 
  ## REMPLACER total.sd2 par cov(Z(x),Z(x)) ou x = newdata
  ## partout dans les formules ci-dessous
  ## c'est utile dans le cas non stationnaire

  if ((se.compute) || (cov.compute)) {
    if ( !is(object@covariance, "covUser") ) {
      total.sd2 <- object@covariance@sd2
    } else {
      m <- nrow(newdata)
      total.sd2 <- rep(NA,m)
      for(i in 1:m) {
        total.sd2[i] <- object@covariance@kernel(newdata[i, ], newdata[i, ])
      }
    }
    if (object@covariance@nugget.flag) {
        total.sd2 <- total.sd2 + object@covariance@nugget
    }
  }
  
  
  if (se.compute) {		
    
    s2.predict.1 <- apply(Tinv.c.newdata, 2, crossprod)         # compute c(x)'*C^(-1)*c(x)   for x = newdata
        
    if (type == "SK") {
      s2.predict <- pmax(total.sd2 - s2.predict.1, 0)
      s2.predict <- as.numeric(s2.predict)
      q95 <- qnorm(0.975)
    }
    else if (type == "UK") {
      T.M <- chol(t(M)%*%M)   # equivalently : qrR <- qr.R(qr(M))
      s2.predict.mat <- backsolve(t(T.M), t(F.newdata - t(Tinv.c.newdata)%*%M) , upper.tri = FALSE)
      
      s2.predict.2 <- apply(s2.predict.mat, 2, crossprod)
      s2.predict <- pmax(total.sd2 - s2.predict.1 + s2.predict.2, 0)
      s2.predict <- as.numeric(s2.predict)
      if (bias.correct) s2.predict <- s2.predict * object@n/(object@n - object@p)
      q95 <- qt(0.975, object@n - object@p)
    }
    
    lower95 <- y.predict - q95*sqrt(s2.predict)
    upper95 <- y.predict + q95*sqrt(s2.predict)
    
    output.list$sd <- sqrt(s2.predict)
    output.list$lower95 <- lower95
    output.list$upper95 <- upper95
  }
  
  if (cov.compute) {		
    
    C.newdata <- covMatrix(object@covariance, newdata)[[1]]
    cond.cov <- C.newdata - crossprod(Tinv.c.newdata)
    
    if (type=="UK") {	
      T.M <- chol(t(M)%*%M)   # equivalently : qrR <- qr.R(qr(M))
      s2.predict.mat <- backsolve(t(T.M), t(F.newdata - t(Tinv.c.newdata)%*%M), upper.tri = FALSE)
      cond.cov <- cond.cov + crossprod(s2.predict.mat)
      if (bias.correct) cond.cov <- cond.cov * object@n/(object@n - object@p)
    }
    
    output.list$cov <- cond.cov
    
  }
  
  return(output.list)
  
}



if(!isGeneric("predict")) {
  setGeneric(name = "predict",
             def = function(object, ...) standardGeneric("predict")
             )
}

setMethod("predict", "km", 
          function(object, newdata, type, se.compute = TRUE,
                   cov.compute = FALSE, light.return = FALSE, bias.correct = FALSE,
                   checkNames = TRUE, ...) {
            predict.km(object = object, newdata = newdata, type = type,
                       se.compute = se.compute, cov.compute = cov.compute,
                       light.return = light.return,
                       bias.correct = bias.correct, checkNames = checkNames, ...)
          }
          )


##****************************************************************************
##			     S I M U L A T E  METHOD
##****************************************************************************

simulate.km <- function(object, nsim = 1, seed = NULL, newdata = NULL,
                        cond = FALSE, nugget.sim = 0, checkNames = TRUE, ...) {
	
  newdataMissing <- is.null(newdata)  
  if (!is.numeric(nugget.sim)) stop("'nugget.sim' must be a number")
  if (nugget.sim<0) stop("nugget.sim (homogenous to a variance) must not be negative")
  if (!is.logical(cond)) stop("'cond' must be TRUE/FALSE")
  if ((!newdataMissing) && (checkNames)) {
    newdata <- checkNames(X1 = object@X, X2 = newdata, X1.name = "the design", X2.name = "newdata")
  }

  
  if (newdataMissing) {
    newdata <- object@X
    F.newdata <- object@F
    T.newdata <- object@T
  } else {
    newdata <- as.matrix(newdata)
    m <- nrow(newdata)
    if (!identical(ncol(newdata), object@d)) 
      stop("newdata must have the same numbers of columns than the experimental design")
    if (!identical(colnames(newdata), colnames(object@X))) {
      colnames(newdata) <- colnames(object@X)
    }
    F.newdata <- model.matrix(object@trend.formula, data = data.frame(newdata))
    Sigma <- covMatrix(object@covariance, newdata)[[1]]
    T.newdata <- chol(Sigma + diag(nugget.sim, m, m))
  }
  
  y.trend <- F.newdata %*% object@trend.coef
  
  m <- nrow(newdata)
  
  if (!cond) {			# non conditional simulations
    white.noise <- matrix(rnorm(m*nsim), m, nsim)
    y.rand <- t(T.newdata) %*% white.noise
    y <- matrix(y.trend, m, nsim) + y.rand
  } else {				# simulations conditional to the observations
                                        #if (object@noise.flag) {
                                        #	stop("conditional simulations not available for heterogeneous observations")
                                        #} else {
    Sigma21 <- covMat1Mat2(object@covariance, X1 = object@X, X2 = newdata, nugget.flag = FALSE)          ## size n x m
    Tinv.Sigma21 <- backsolve(t(object@T), Sigma21, upper.tri = FALSE)     ## t(T22)^(-1) * Sigma21,  size  n x m
    y.trend.cond <- y.trend + t(Tinv.Sigma21) %*% object@z                 ## size m x 1
    
    if (!newdataMissing) {
      Sigma11 <- Sigma
    } else Sigma11 <- t(object@T) %*% object@T	
    
    Sigma.cond <- Sigma11 - t(Tinv.Sigma21) %*% Tinv.Sigma21          ## size m x m
    T.cond <- chol(Sigma.cond + diag(nugget.sim, m, m))			
    white.noise <- matrix(rnorm(m*nsim), m, nsim)
    y.rand.cond <- t(T.cond) %*% white.noise
    y <- matrix(y.trend.cond, m, nsim) + y.rand.cond	
  }
  
  return(t(y))
  
}


if(!isGeneric("simulate")) {
  setGeneric(name = "simulate",
             def = function(object, nsim = 1, seed = NULL, ...) standardGeneric("simulate")
             )
}

setMethod("simulate", "km", 
          function(object, nsim = 1, seed = NULL, newdata = NULL,
                   cond = FALSE, nugget.sim = 0, checkNames = TRUE, ...) {
            simulate.km(object = object, nsim = nsim, newdata = newdata,
                        cond = cond, nugget.sim = nugget.sim, checkNames = checkNames,...)
          }
          )

##****************************************************************************
##			     update  METHOD
##****************************************************************************

update.km <- function(object,
                      newX,
                      newy,
                      newX.alreadyExist =  FALSE,
                      cov.reestim = TRUE,trend.reestim = TRUE,nugget.reestim=FALSE,
                      newnoise.var = NULL, kmcontrol = NULL, newF = NULL){
  
  if (newX.alreadyExist == FALSE) {
    
    object@X <- rbind(object@X, as.matrix(newX))
    object@y <- rbind(object@y, as.matrix(newy))
    
    ######## Consistency tests between object@noise.var and newnoise.var ########
    
    if ( (length(object@noise.var) != 0) && (is.null(newnoise.var)) ) {
      ## noisy initial observations and noise free new observations
      if (is.null(nrow(newX))) {Thenewnoise.var=0} #only one new point, with no noise
      else {Thenewnoise.var <- rep(0,nrow(newX))}	 #many new points, with no noise							
      object@noise.var <- c(object@noise.var,Thenewnoise.var) #merge old noise vector with new noises
    }
    if ((length(object@noise.var) != 0) && (!is.null(newnoise.var))) {
      ## noisy initial observations and noisy new observations
      object@noise.var <- c(object@noise.var,newnoise.var) #merge old noise vector with new noises
    }
    if ((length(object@noise.var) == 0) && (!is.null(newnoise.var))) {
      ## noise free initial observations and noisy new observations
      if (any(newnoise.var!=0)){
        noise.var.init <- rep(0,object@n) #noise vector for initial observations (noise = 0)
        object@noise.var <- c(noise.var.init,newnoise.var)  #merge old noise vector with new noises
      }
    }
    
    
    if (cov.reestim | trend.reestim | nugget.reestim) {  
      ## case 1: new points, covariance and/or trend parameter re-estimation (provided object@param.estim == true)
      if (object@param.estim) {
        ## case 1a: here we re-estimate the covariance and/or trend parameter
        ## default values for the cov param estimation parameters - when they are not provided
        TheCov <- object@covariance
        TheClass <- class(TheCov)
        
        if (is.null(kmcontrol$penalty)) kmcontrol$penalty <- object@penalty
        if (length(object@penalty == 0)) kmcontrol$penalty <- NULL
        if (is.null(kmcontrol$optim.method)) kmcontrol$optim.method <- object@optim.method 
        if (length(kmcontrol$optim.method) == 0) kmcontrol$optim.method <- "BFGS"
        if (is.null(kmcontrol$control)) kmcontrol$control <- object@control
	if (is.null(kmcontrol$multistart)) kmcontrol$multistart <- object@control$multistart
        if(length(object@gr) == 0) object@gr <- TRUE
        knots <- NULL; if(TheClass == "covScaling") knots <- object@covariance@knots
        
        #filtre sur cov.reestim / trend.reestim
        if(cov.reestim){
          coef.cov <- NULL
          coef.var <- NULL
        }else{
          if((TheClass == "covTensorProduct") | (TheClass == "covIso")) coef.cov <- covparam2vect(object@covariance)
          if(TheClass == "covScaling") coef.cov <- object@covariance@eta
          
          coef.var <- object@covariance@sd2
        }
        
        if(trend.reestim) {coef.trend <- NULL
        } else { coef.trend <- object@trend.coef}
        
        if(length(object@covariance@nugget) == 0) {nugget <- NULL
        } else{nugget <- object@covariance@nugget}
        
        nugget.estim <- nugget.reestim
        
        ## if (is.null(kmcontrol$parinit)) kmcontrol$parinit <- covparam2vect(object@covariance) ## retire
        
        if ((TheClass == "covTensorProduct") | (TheClass == "covIso") | (TheClass =="covScaling")) {
          object <- km(formula = object@trend.formula, design = object@X, response = object@y,
                       covtype = object@covariance@name, 
                       coef.trend = coef.trend, coef.cov = coef.cov, coef.var = coef.var,
                       nugget=nugget,nugget.estim=nugget.estim,
                       noise.var = object@noise.var, penalty = kmcontrol$penalty,
		       optim.method = kmcontrol$optim.method, multistart = kmcontrol$multistart,
                       lower = object@lower, upper = object@upper,
                       control = kmcontrol$control,gr = object@gr,
                       iso =     (TheClass == "covIso"),
                       scaling = (TheClass == "covScaling"),
                       knots = knots)                
        } else {
          print("Unknown covariance type. Accepted types are \"covTensorProduct\", \"covIso\" and \"covScaling\"")
          return(0)
        }
      } else {
        ## case 1b: no re-estimation at all because object@param.estim == FALSE. 
        ## Keeping the current covariance parameters
        
        object@n <- nrow(object@X)
        if (is.null(newF)) {
          object@F <- trendMatrix.update(object, Xnew = data.frame(newX))
        } else object@F <- rbind(object@F, newF)
        
        object <- computeAuxVariables(object)
        
      }
    } else {
      ## case 2: new points, no re-estimation because cov.reestim==FALSE. 
      ## Keeping the current covariance parameters
      
      object@n <- nrow(object@X)
      if (is.null(newF)) {
        object@F <- trendMatrix.update(object, Xnew = data.frame(newX))
      } else object@F <- rbind(object@F, newF)
      
      object <- computeAuxVariables(object)
      
    }
  } else {
    ## case 3: existing points with a modified response, 
    ## No need to recalculate the Cholesky but the other variables are still needed (z, M) for prediction. 
    ## No covariance param re-estimation in this case
    ## we assume that the k new values given modify the k last points of the design object@X
    
    if (is.null(nrow(newX))) {
      numrow <- 1
    } else numrow <- nrow(newX)
    
    for (i in 1:numrow) {
      object@y[object@n - numrow + i] <- newy[i]
    }
    object <- computeAuxVariables_noChol(object)
  }
  return(object)
}

if(!isGeneric("update")) {
  setGeneric(name = "update",
             def = function(object, ...) standardGeneric("update")
  )
}

setMethod("update", "km", 
          function(object, newX, newy, newX.alreadyExist =  FALSE, 
                   cov.reestim = TRUE, trend.reestim = TRUE, nugget.reestim = FALSE,
                   newnoise.var = NULL, kmcontrol = NULL, newF = NULL, ...) {
            update.km(object=object, newX = newX, newy = newy, 
                      newX.alreadyExist = newX.alreadyExist, 
                      cov.reestim = cov.reestim, trend.reestim = trend.reestim, nugget.reestim = nugget.reestim,
                      newnoise.var = newnoise.var, kmcontrol = kmcontrol, newF = newF, ...) 
          }
)

