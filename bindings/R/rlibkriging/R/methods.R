# if(!isGeneric("print")) {
#   setGeneric(name = "print",
#              def = function(model, ...) standardGeneric("print")
#   )
# }
# 
print <- function (x, ...) {
  UseMethod("print", x)
}

print.Kriging <- function(x) {
  k=kriging_model(x)
  p = "Kriging model:\n"
  p = paste0(p,"\n  * data: ",paste0(collapse=" x ",dim(k$X))," -> ",paste0(collapse=" x ",dim(k$y)))
  p = paste0(p,"\n  * trend: ",k$regmodel)#,"(",paste0(collapse=",",k$F),")")
  p = paste0(p,"\n  * variance: ",k$sigma2)
  p = paste0(p,"\n  * covariance:")
  p = paste0(p,"\n    * kernel: ",k$kernel)
  p = paste0(p,"\n    * range: ",paste0(collapse=", ",k$theta))
  p = paste0(p,"\n    * fit: ")
  p = paste0(p,"\n      * objective: ",k$objective)
  p = paste0(p,"\n      * optim: ",k$optim)
  cat(p)
  # return(p)
}

setMethod("print", "Kriging", print.Kriging)


predict.Kriging <- function(object,x,stdev=T,cov=F) {
  k=kriging_model(object) 
  if (!is.matrix(x)) x=matrix(x,ncol=ncol(k$X))
  if (ncol(x)!=ncol(k$X))
    stop("Input x must have ",ncol(k$X), " columns (instead of ",ncol(x),")")
  return(kriging_predict(object, x,stdev,cov))
}

predict <- function (x, ...) {
  UseMethod("predict", x)
}

setMethod("predict", "Kriging", predict.Kriging)


simulate.Kriging <- function(object,nsim=1,x) {
  k=kriging_model(object) 
  if (!is.matrix(x)) x=matrix(x,ncol=ncol(k$X))
  if (ncol(x)!=ncol(k$X))
    stop("Input x must have ",ncol(k$X), " columns (instead of ",ncol(x),")")
  return(kriging_simulate(object, nsim, x))
}

simulate <- function (object, ...) {
  UseMethod("simulate", object)
}

setMethod("simulate", "Kriging", simulate.Kriging)


update.Kriging <- function(object,newy,newX,normalize=FALSE) {
  k=kriging_model(object) 
  if (!is.matrix(newX)) newX=matrix(newX,ncol=ncol(k$X))
  if (!is.matrix(newy)) newy=matrix(newy,ncol=ncol(k$y))
  if (ncol(newX)!=ncol(k$X))
    stop("Input newX must have ",ncol(k$X), " columns (instead of ",ncol(newX),")")
  if (nrow(newy)!=nrow(newX))
    stop("Input newX and newy must have the same number of rows.")
  kriging_update(object, newy,newX,normalize)
}

update <- function (object, ...) {
  UseMethod("update", object)
}

setMethod("update", "Kriging", update.Kriging)


logLikelihood.Kriging <- function(object,theta,grad=FALSE,hess=FALSE) {
  k=kriging_model(object) 
  if (!is.matrix(theta)) theta=matrix(theta,ncol=ncol(k$X))
  if (ncol(theta)!=ncol(k$X))
    stop("Input theta must have ",ncol(k$X), " columns (instead of ",ncol(theta),")")
  out=list(logLikelihood=matrix(NA,nrow=nrow(theta)),logLikelihoodGrad=matrix(NA,nrow=nrow(theta),ncol=ncol(theta)),logLikelihoodHess=array(NA,c(nrow(theta),ncol(theta),ncol(theta))))
  for (i in 1:nrow(theta)) {
    print(theta[i,])
    ll = kriging_logLikelihood(object,theta[i,],isTRUE(grad),isTRUE(hess))
    out$logLikelihood[i] = ll$logLikelihood
    if (isTRUE(grad)) out$logLikelihoodGrad[i,] = ll$logLikelihoodGrad
    if (isTRUE(hess)) out$logLikelihoodHess[i,,] = ll$logLikelihoodHess
  }
  if (!isTRUE(grad)) out$logLikelihoodGrad <- NULL
  if (!isTRUE(hess)) out$logLikelihoodHess <- NULL
  return(out)
}

logLikelihood <- function (object, ...) {
  UseMethod("logLikelihood", object)
}

setMethod("logLikelihood", "Kriging", logLikelihood.Kriging)


leaveOneOut.Kriging <- function(object,theta,grad=FALSE) {
  k=kriging_model(object) 
  if (!is.matrix(theta)) theta=matrix(theta,ncol=ncol(k$X))
  if (ncol(theta)!=ncol(k$X))
    stop("Input theta must have ",ncol(k$X), " columns (instead of ",ncol(theta),")")
  out=list(leaveOneOut=matrix(NA,nrow=nrow(theta)),leaveOneOutGrad=matrix(NA,nrow=nrow(theta),ncol=ncol(theta)))
  for (i in 1:nrow(theta)) {
    loo = kriging_leaveOneOut(object,theta[i,],isTRUE(grad))
    out$leaveOneOut[i] = loo$leaveOneOut
    if (isTRUE(grad)) out$leaveOneOutGrad[i,] = loo$leaveOneOutGrad
  }    
  if (!isTRUE(grad)) out$leaveOneOutGrad <- NULL
  return(out)
}

leaveOneOut <- function (object, ...) {
  UseMethod("leaveOneOut", object)
}

setMethod("leaveOneOut", "Kriging", leaveOneOut.Kriging)
