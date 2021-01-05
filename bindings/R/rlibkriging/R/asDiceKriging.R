# loadDiceKriging <- function() {
  if (! "DiceKriging" %in% installed.packages())
    warning("DiceKriging must be installed to use its wrapper from libKriging.")
  if (!"DiceKriging" %in% loadedNamespaces())
    try(library(DiceKriging))
  if (!isClass("as_km"))
    try(setClass("as_km",slots = "Kriging",contains="km"))
# }
  
as_km.Kriging <- function(k,.call=NULL) {
  # loadDiceKriging()
  if (! "DiceKriging" %in% installed.packages())
    stop("DiceKriging must be installed to use its wrapper from libKriging.")
  
  model <- new("as_km")
  model@Kriging = k
  
  if (is.null(.call))
    model@call <- match.call()
  else
    model@call <- .call
  
  m = kriging_model(k)
  data <- data.frame(m$X)
  model@trend.formula <- regmodel2formula(m$regmodel)
  model@trend.coef <- as.numeric(m$beta)
  model@X <- m$X
  model@y <- m$y
  model@d <- ncol(m$X)
  model@n <- nrow(m$X)
  model@F <- m$F
    colnames(model@F) <- colnames(model.matrix(model@trend.formula,data))
  model@p <- ncol(m$F)
  model@noise.flag <- FALSE
  model@noise.var <- 0
  
  model@case <- "LLconcentration_beta_sigma2"
  
  model@known.param <- "None"
  model@param.estim <- NA
  model@method = m$objective
  model@optim.method <- m$optim
  
  model@penalty = list()
  model@lower = 0
  model@upper = Inf
  model@control = list()
  
  model@gr = FALSE
  
  model@T = m$T
  model@z = as.numeric(m$z)
  model@M = m$M

  covStruct =  new("covTensorProduct", d=model@d, name=m$kernel, 
                                      sd2 = m$sigma2, var.names=names(data), 
                                      nugget = 0, nugget.flag=FALSE, nugget.estim=FALSE, known.covparam="") 
  covStruct@range.names  = "theta"
  covStruct@paramset.n <- as.integer(1)
  covStruct@param.n <- as.integer(model@d)
  covStruct@range.n <- as.integer(model@d)
  covStruct@range.val = as.numeric(m$theta)
  model@covariance = covStruct 
  
  return(model)
}


predict.as_km <- function(object, newdata, type,
                          se.compute = TRUE, cov.compute = FALSE, light.return = FALSE,
                          bias.correct = FALSE, checkNames = TRUE) {
  if (isTRUE(checkNames)) stop("checkNames=TRUE unsupported.")
  if (isTRUE(bias.correct)) stop("bias.correct=TRUE unsupported.")
  if (!isTRUE(light.return)) stop("light.return=FALSE unsupported.")
  if (type!="UK") stop("type!=UK unsupported.")
  
  y.predict = predict.Kriging(object@Kriging, x = newdata, stdev=se.compute, cov=cov.compute)
  
  output.list <- list()
  # output.list$trend <- y.predict.trend
  output.list$mean <- y.predict$mean
  
  if (se.compute) {		
    s2.predict <- y.predict$stdev^2
    q95 <- qt(0.975, object@n - object@p)
    
    lower95 <- y.predict$mean - q95*sqrt(s2.predict)
    upper95 <- y.predict$mean + q95*sqrt(s2.predict)
    
    output.list$sd <- sqrt(s2.predict)
    output.list$lower95 <- lower95
    output.list$upper95 <- upper95
  }
  
  if (cov.compute) {		
    output.list$cov <- y.predict$cov
  }
  
  F.newdata <- model.matrix(object@trend.formula, data = data.frame(newdata))
  output.list$trend <- F.newdata %*% object@trend.coef
  
  return(output.list)
}

if(!isGeneric("predict")) {
  setGeneric(name = "predict",
             def = function(object, ...) standardGeneric("predict")
  )
}

setMethod("predict", "as_km", 
          function(object, newdata, type, se.compute = TRUE,
                   cov.compute = FALSE, light.return = TRUE, bias.correct = FALSE,
                   checkNames = FALSE) {
            predict.as_km(object = object, newdata = newdata, type = type,
                       se.compute = se.compute, cov.compute = cov.compute,
                       light.return = light.return,
                       bias.correct = bias.correct, checkNames = checkNames)
          }
)


simulate.as_km <- function(object, nsim = 1, seed = NULL, newdata = NULL,
                           cond = FALSE, nugget.sim = 0, checkNames = TRUE) {
  if (isTRUE(checkNames)) stop("checkNames=TRUE unsupported.")
  if (!isTRUE(cond)) stop("cond=FALSE unsupported.")
  if (nugget.sim!=0) stop("nugget.sim!=0 unsupported.")
  
  return(simulate.Kriging(object = object@Kriging,x = newdata,nsim = nsim))
}


if(!isGeneric("simulate")) {
  setGeneric(name = "simulate",
             def = function(object, nsim = 1, seed = NULL, ...) standardGeneric("simulate")
  )
}

setMethod("simulate", "as_km", 
          function(object, nsim = 1, seed = NULL, newdata = NULL,
                   cond = FALSE, nugget.sim = 0, checkNames = TRUE) {
            simulate.as_km(object = object, nsim = nsim, newdata = newdata,
                        cond = cond, nugget.sim = nugget.sim, checkNames = checkNames)
          }
)

update.as_km <- function(object,
                         newX,
                         newy,
                         newX.alreadyExist =  FALSE,
                         cov.reestim = TRUE,trend.reestim = TRUE,nugget.reestim=FALSE,
                         newnoise.var = NULL, kmcontrol = NULL, newF = NULL){
  if (isTRUE(newX.alreadyExist)) stop("newX.alreadyExist=TRUE unsupported.")
  if (!is.null(newnoise.var)) stop("newnoise.var!=NULL unsupported.")
  if (!is.null(kmcontrol)) stop("kmcontrol!=NULL unsupported.")
  if (!is.null(newF)) stop("newF!=NULL unsupported.")
  
  update.Kriging(object@Kriging,newy,newX)
  
  return(object)
}

if(!isGeneric("update")) {
  setGeneric(name = "update",
             def = function(object, ...) standardGeneric("update")
  )
}

setMethod("update", "as_km", 
          function(object, newX, newy, newX.alreadyExist =  FALSE, 
                   cov.reestim = TRUE, trend.reestim = TRUE, nugget.reestim = FALSE,
                   newnoise.var = NULL, kmcontrol = NULL, newF = NULL) {
            update.as_km(object=object, newX = newX, newy = newy, 
                      newX.alreadyExist = newX.alreadyExist, 
                      cov.reestim = cov.reestim, trend.reestim = trend.reestim, nugget.reestim = nugget.reestim,
                      newnoise.var = newnoise.var, kmcontrol = kmcontrol, newF = newF) 
          }
)


as_km <- function(formula=~1, design, response, covtype="matern5_2",
                  coef.cov = NULL, coef.var = NULL,
                  estim.method="MLE", optim.method = "BFGS", parinit = NULL) {
  formula = formula2regmodel(formula)
  
  if (!is.matrix(design)) design = as.matrix(design)
  if (!is.matrix(response)) response = matrix(response, nrow=nrow(design))
    
  if (estim.method=="MLE")
    estim.method = "LL"
  else if (estim.method=="LOO")
    estim.method = "LL"
  else stop("Unsupported estim.method ",estim.method)
  
  if (!(covtype=="gauss" | covtype=="exp"))
    stop("Unsupported covtype ",covtype)
  
  if (optim.method!="BFGS")
    warning("Cannot setup optim.method ",optim.method,". Ignored.")
  
  parameters=list()
  if (!is.null(coef.var))
    parameters = c(parameters,list(sigma2=coef.var))
  if (!is.null(coef.cov)) {
    parameters = c(parameters,list(theta=matrix(coef.cov,ncol=ncol(design))))
    optim.method = "none"
  }
  if (!is.null(parinit)) {
    parameters = c(parameters,list(theta=matrix(parinit,ncol=ncol(design))))
  }
  if (length(parameters)==0) 
    parameters=NULL
  
  r = rlibkriging::kriging(y=response, X=design, kernel = covtype, regmodel = formula,
                           normalize = F, objective = estim.method,optim = "BFGS",parameters = parameters)
  
  return(as_km.Kriging(r,.call=match.call()))
}

formula2regmodel = function(form) {
  if (format(form) == "~1")
    return("constant")
  else if (format(form) == "~.")
    return("linear")
  else if (format(form) == "~.^2")
    return("interactive")
  else stop("Unsupported formula ",form)
}


regmodel2formula = function(regmodel) {
  if (regmodel == "constant")
    return(~1)
  else if (regmodel == "linear")
    return(~.)
  else if (regmodel == "interactive")
    return(~.^2)
  else stop("Unsupported regmodel ",regmodel)
}
