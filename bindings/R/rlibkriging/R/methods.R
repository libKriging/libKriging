#' Build a "Kriging" object from libKriging.
#'
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param y Array of response values
#' @param X Matrix of input design
#' @param kernel Covariance model: "gauss", "exp", ...
#' @param regmodel Universal Kriging linear trend: "constant", "linear", "interactive" ("constant" by default)
#' @param normalize Normalize X and y in [0,1] (FALSE by default)
#' @param optim Optimization method to fit hyper-parameters: "BFGS", "Newton" (uses objective Hessian), "none" (keep initial "parameters" values)
#' @param objective Objective function to optimize: "LL" (log-Likelihood, by default), "LOO" (leave one out), "LMP" (log Marginal Posterior)
#' @param parameters Initial hyper parameters: list(sigma2=..., theta=...). If theta has many rows, each is used as a starting point for optim.
#' 
#' @return S3 Kriging object. Should be used with its predict, simulate, update methods.
#' 
#' @export
#' @useDynLib rlibkriging, .registration=TRUE
#' @importFrom Rcpp sourceCpp
Kriging <- function(y, X, kernel, regmodel = "constant", normalize = FALSE, optim = "BFGS", objective = "LL", parameters = NULL) {
  new_Kriging(y, X, kernel, regmodel, normalize, optim, objective, parameters)
}


#' List Kriging object content
#'
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param x S3 Kriging object
#' @param ... Ignored
#'
#' @return list of Kriging object fields: kernel, optim, objective, theta, sigma2, X, centerX, scaleX, y, centerY, scaleY, regmodel, F, T, M, z, beta
#' 
#' @export as.list
#' @method as.list Kriging
#' @aliases as.list,Kriging,Kriging-method
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#' r <- Kriging(y, X, "gauss")
#' l = as.list(r)
#' cat(paste0(names(l)," =" ,l,collapse="\n"))
as.list.Kriging <- function(x, ...) {
  if (length(list(...))>0) warning("Arguments ",paste0(names(list(...)),"=",list(...),collapse=",")," are ignored.")
  kriging_model(x)
}

setMethod("as.list", "Kriging", as.list.Kriging)


#' Print Kriging object content
#'
#' @author Yann Richet (yann.richet@irsn.fr)
#'
#' @param x S3 Kriging object
#' @param ... Ignored
#'
#' @return NULL
#'
#' @method print Kriging
#' @export print
#' @aliases print,Kriging,Kriging-method
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#' r <- Kriging(y, X, "gauss")
#' print(r)
print.Kriging <- function(x, ...) {
  if (length(list(...))>0) warning("Arguments ",paste0(names(list(...)),"=",list(...),collapse=",")," are ignored.")
  k=kriging_model(x)
  p = "Kriging model:\n"
  p = paste0(p,"\n  * data: ",paste0(collapse=" x ",dim(k$X))," -> ",paste0(collapse=" x ",dim(k$y)))
  p = paste0(p,"\n  * trend ",k$regmodel, ifelse(k$estim_beta," (est.)",""), ": ", paste0(collapse=",",k$beta))#,"(",paste0(collapse=",",k$F),")")
  p = paste0(p,"\n  * variance",ifelse(k$estim_sigma2," (est.)",""),": ",k$sigma2)
  p = paste0(p,"\n  * covariance:")
  p = paste0(p,"\n    * kernel: ",k$kernel)
  p = paste0(p,"\n    * range",ifelse(k$estim_theta," (est.)",""),": ",paste0(collapse=", ",k$theta))
  p = paste0(p,"\n    * fit: ")
  p = paste0(p,"\n      * objective: ",k$objective)
  p = paste0(p,"\n      * optim: ",k$optim)
  p = paste0(p,"\n")
  cat(p)
  # return(p)
}

setMethod("print", "Kriging", print.Kriging)


#' Predict Kriging model at given points
#' 
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param object S3 Kriging object
#' @param x points in model input space where to predict
#' @param stdev return also standard deviation (default TRUE)
#' @param cov return covariance matrix between x points (default FALSE)
#' @param ... Ignored
#'
#' @return list containing: mean, stdev, cov
#' 
#' @method predict Kriging
#' @export predict
#' @aliases predict,Kriging,Kriging-method
#' 
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#'   plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#'   points(X,y,col='blue')
#' r <- Kriging(y, X, "gauss")
#' x = seq(0,1,,101)
#' p_x = predict(r, x)
#'   lines(x,p_x$mean,col='blue')
#'   lines(x,p_x$mean-2*p_x$stdev,col='blue')
#'   lines(x,p_x$mean+2*p_x$stdev,col='blue')
predict.Kriging <- function(object, x, stdev=T, cov=F, ...) {
  if (length(list(...))>0) warning("Arguments ",paste0(names(list(...)),"=",list(...),collapse=",")," are ignored.")
  k=kriging_model(object) 
  if (!is.matrix(x)) x=matrix(x,ncol=ncol(k$X))
  if (ncol(x)!=ncol(k$X))
    stop("Input x must have ",ncol(k$X), " columns (instead of ",ncol(x),")")
  return(kriging_predict(object, x,stdev,cov))
}

predict <- function (...) UseMethod("predict")
setMethod("predict", "Kriging", predict.Kriging)


#' Simulate (conditional) Kriging model at given points
#' 
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param object S3 Kriging object
#' @param nsim number of simulations to perform
#' @param seed random seed used
#' @param x points in model input space where to simulate
#' @param ... Ignored
#'
#' @return length(x) x nsim matrix containing simulated path at x points
#' 
#' @method simulate Kriging
#' @export simulate
#' @aliases simulate,Kriging,Kriging-method
#' 
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#'   plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#'   points(X,y,col='blue')
#' r <- Kriging(y, X, "gauss")
#' x = seq(0,1,,101)
#' s_x = simulate(r, nsim=3, x=x)
#'   lines(x,s_x[,1],col='blue')
#'   lines(x,s_x[,2],col='blue')
#'   lines(x,s_x[,3],col='blue')
simulate.Kriging <- function(object, nsim=1, seed=123, x, ...) {
  if (length(list(...))>0) warning("Arguments ",paste0(names(list(...)),"=",list(...),collapse=",")," are ignored.")
  k=kriging_model(object) 
  if (!is.matrix(x)) x=matrix(x,ncol=ncol(k$X))
  if (ncol(x)!=ncol(k$X))
    stop("Input x must have ",ncol(k$X), " columns (instead of ",ncol(x),")")
  if (is.null(seed)) seed = floor(runif(1)*99999)
  return(kriging_simulate(object, nsim=nsim, seed=seed, X=x))
}

simulate <- function (...) UseMethod("simulate")
setMethod("simulate", "Kriging", simulate.Kriging)


#' Update Kriging model with new points
#' 
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param object S3 Kriging object
#' @param newy new points in model output space
#' @param newX new points in model input space
#' @param normalize Normalize X and y in [0,1] (FALSE by default)
#' @param ... Ignored
#' 
#' @method update Kriging
#' @export update
#' @aliases update,Kriging,Kriging-method
#' 
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#'   plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#'   points(X,y,col='blue')
#' r <- Kriging(y, X, "gauss")
#' x = seq(0,1,,101)
#' p_x = predict(r, x)
#'   lines(x,p_x$mean,col='blue')
#'   lines(x,p_x$mean-2*p_x$stdev,col='blue')
#'   lines(x,p_x$mean+2*p_x$stdev,col='blue')
#' newX <- as.matrix(runif(3))
#' newy <- f(newX)
#'   points(newX,newy,col='red')
#' update(r,newy,newX)
#' x = seq(0,1,,101)
#' p2_x = predict(r, x)
#'   lines(x,p2_x$mean,col='red')
#'   lines(x,p2_x$mean-2*p2_x$stdev,col='red')
#'   lines(x,p2_x$mean+2*p2_x$stdev,col='red')
update.Kriging <- function(object, newy, newX, normalize=FALSE, ...) {
  if (length(list(...))>0) warning("Arguments ",paste0(names(list(...)),"=",list(...),collapse=",")," are ignored.")
  k=kriging_model(object) 
  if (!is.matrix(newX)) newX=matrix(newX,ncol=ncol(k$X))
  if (!is.matrix(newy)) newy=matrix(newy,ncol=ncol(k$y))
  if (ncol(newX)!=ncol(k$X))
    stop("Input newX must have ",ncol(k$X), " columns (instead of ",ncol(newX),")")
  if (nrow(newy)!=nrow(newX))
    stop("Input newX and newy must have the same number of rows.")
  kriging_update(object, newy, newX, normalize)
}

update <- function(...) UseMethod("update")
setMethod("update", "Kriging", update.Kriging)
#setGeneric(name = "update", def = function(...) standardGeneric("update"))


#' Compute log-Likelihood of Kriging model
#' 
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param object S3 Kriging object
#' @param theta range parameters to evaluate
#' @param grad return Gradient ? (default is TRUE)
#' @param hess return Hessian ? (default is FALSe)
#'
#' @return log-Likelihood computed for given theta
#' 
#' @method logLikelihood Kriging
#' @export logLikelihood
#' @aliases logLikelihood,Kriging,Kriging-method
#' 
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#' r <- Kriging(y, X, "gauss")
#' print(r)
#' ll = function(theta) logLikelihood(r,theta)$logLikelihood
#' t = seq(0.0001,2,,101)
#'   plot(t,ll(t),type='l')
#'   abline(v=as.list(r)$theta,col='blue')
logLikelihood.Kriging <- function(object, theta, grad=FALSE, hess=FALSE) {
  k=kriging_model(object) 
  if (!is.matrix(theta)) theta=matrix(theta,ncol=ncol(k$X))
  if (ncol(theta)!=ncol(k$X))
    stop("Input theta must have ",ncol(k$X), " columns (instead of ",ncol(theta),")")
  out=list(logLikelihood=matrix(NA,nrow=nrow(theta)),logLikelihoodGrad=matrix(NA,nrow=nrow(theta),ncol=ncol(theta)),logLikelihoodHess=array(NA,c(nrow(theta),ncol(theta),ncol(theta))))
  for (i in 1:nrow(theta)) {
    ll = kriging_logLikelihood(object,theta[i,],isTRUE(grad),isTRUE(hess))
    out$logLikelihood[i] = ll$logLikelihood
    if (isTRUE(grad)) out$logLikelihoodGrad[i,] = ll$logLikelihoodGrad
    if (isTRUE(hess)) out$logLikelihoodHess[i,,] = ll$logLikelihoodHess
  }
  if (!isTRUE(grad)) out$logLikelihoodGrad <- NULL
  if (!isTRUE(hess)) out$logLikelihoodHess <- NULL
  return(out)
}

#' Compute model log-Likelihood at given args
#'
#' @param ... args
#'
#' @return log-Likelihood
#' @export
logLikelihood <- function (...) UseMethod("logLikelihood")
setMethod("logLikelihood", "Kriging", logLikelihood.Kriging)
setGeneric(name = "logLikelihood", def = function(...) standardGeneric("logLikelihood"))


#' Compute leave-One-Out of Kriging model
#' 
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param object S3 Kriging object
#' @param theta range parameters to evaluate
#' @param grad return Gradient ? (default is TRUE)
#'
#' @return leave-One-Out computed for given theta
#' 
#' @method leaveOneOut Kriging
#' @export leaveOneOut
#' @aliases leaveOneOut,Kriging,Kriging-method
#' 
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#' r <- Kriging(y, X, "gauss",objective="LOO")
#' print(r)
#' loo = function(theta) leaveOneOut(r,theta)$leaveOneOut
#' t = seq(0.0001,2,,101)
#'   plot(t,loo(t),type='l')
#'   abline(v=as.list(r)$theta,col='blue')
leaveOneOut.Kriging <- function(object, theta, grad=FALSE) {
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


#' Compute model leave-One-Out error at given args
#'
#' @param ... args
#'
#' @return leave-One-Out
#' @export
leaveOneOut <- function (...) UseMethod("leaveOneOut")
setMethod("leaveOneOut", "Kriging", leaveOneOut.Kriging)
setGeneric(name = "leaveOneOut", def = function(...) standardGeneric("leaveOneOut"))


#' Compute log-Marginal-Posterior of Kriging model
#' 
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param object S3 Kriging object
#' @param theta range parameters to evaluate
#' @param grad return Gradient ? (default is TRUE)
#'
#' @return log-MargPost computed for given theta
#' 
#' @method logMargPost Kriging
#' @export logMargPost
#' @aliases logMargPost,Kriging,Kriging-method
#' 
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#' r <- Kriging(y, X, "gauss")
#' print(r)
#' lmp = function(theta) logMargPost(r,theta)$logMargPost
#' t = seq(0.0001,2,,101)
#'   plot(t,lmp(t),type='l')
#'   abline(v=as.list(r)$theta,col='blue')
logMargPost.Kriging <- function(object, theta, grad=FALSE) {
  k=kriging_model(object) 
  if (!is.matrix(theta)) theta=matrix(theta,ncol=ncol(k$X))
  if (ncol(theta)!=ncol(k$X))
    stop("Input theta must have ",ncol(k$X), " columns (instead of ",ncol(theta),")")
  out=list(logMargPost=matrix(NA,nrow=nrow(theta)),logMargPostGrad=matrix(NA,nrow=nrow(theta),ncol=ncol(theta)))
  for (i in 1:nrow(theta)) {
    lmp = kriging_logMargPost(object,theta[i,],isTRUE(grad))
    out$logMargPost[i] = lmp$logMargPost
    if (isTRUE(grad)) out$logMargPostGrad[i,] = lmp$logMargPostGrad
  }
  if (!isTRUE(grad)) out$logMargPostGrad <- NULL
  return(out)
}

#' Compute model log-Marginal-Posterior at given args
#'
#' @param ... args
#'
#' @return log-Marginal-Posterior
#' @export
logMargPost <- function (...) UseMethod("logMargPost")
setMethod("logMargPost", "Kriging", logMargPost.Kriging)
setGeneric(name = "logMargPost", def = function(...) standardGeneric("logMargPost"))