#' Build a "NuggetKriging" object from libKriging.
#'
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param y Array of response values
#' @param X Matrix of input design
#' @param kernel Covariance model: "gauss", "exp", ...
#' @param regmodel Universal NuggetKriging linear trend: "constant", "linear", "interactive" ("constant" by default)
#' @param normalize Normalize X and y in [0,1] (FALSE by default)
#' @param optim Optimization method to fit hyper-parameters: "BFGS", "Newton" (uses objective Hessian), "none" (keep initial "parameters" values)
#' @param objective Objective function to optimize: "LL" (log-Likelihood, by default), "LOO" (leave one out), "LMP" (log Marginal Posterior)
#' @param parameters Initial hyper parameters: list(sigma2=..., theta=...). If theta has many rows, each is used as a starting point for optim.
#' 
#' @return S3 NuggetKriging object. Should be used with its predict, simulate, update methods.
#' 
#' @export
#' @useDynLib rlibkriging, .registration=TRUE
#' @importFrom Rcpp sourceCpp
NuggetKriging <- function(y, X, kernel, regmodel = "constant", normalize = FALSE, optim = "BFGS", objective = "LL", parameters = NULL) {
  new_NuggetKriging(y, X, kernel, regmodel, normalize, optim, objective, parameters)
}


#' List NuggetKriging object content
#'
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param x S3 NuggetKriging object
#' @param ... Ignored
#'
#' @return list of NuggetKriging object fields: kernel, optim, objective, theta, sigma2, X, centerX, scaleX, y, centerY, scaleY, regmodel, F, T, M, z, beta
#' 
#' @export as.list
#' @method as.list NuggetKriging
#' @aliases as.list,NuggetKriging,NuggetKriging-method
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#' r <- NuggetKriging(y, X, "gauss")
#' l = as.list(r)
#' cat(paste0(names(l)," =" ,l,collapse="\n"))
as.list.NuggetKriging <- function(x, ...) {
  if (length(list(...))>0) warning("Arguments ",paste0(names(list(...)),"=",list(...),collapse=",")," are ignored.")
  nuggetkriging_model(x)
}

setMethod("as.list", "NuggetKriging", as.list.NuggetKriging)


#' Print NuggetKriging object content
#'
#' @author Yann Richet (yann.richet@irsn.fr)
#'
#' @param x S3 NuggetKriging object
#' @param ... Ignored
#'
#' @return NULL
#'
#' @method print NuggetKriging
#' @export print
#' @aliases print,NuggetKriging,NuggetKriging-method
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#' r <- NuggetKriging(y, X, "gauss")
#' print(r)
print.NuggetKriging <- function(x, ...) {
  if (length(list(...))>0) warning("Arguments ",paste0(names(list(...)),"=",list(...),collapse=",")," are ignored.")
  k=nuggetkriging_model(x)
  p = paste0("NuggetKriging model:\n\n",nuggetkriging_summary(x),"\n")
  cat(p)
  # return(p)
}

setMethod("print", "NuggetKriging", print.NuggetKriging)


#' Predict NuggetKriging model at given points
#' 
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param object S3 NuggetKriging object
#' @param x points in model input space where to predict
#' @param stdev return also standard deviation (default TRUE)
#' @param cov return covariance matrix between x points (default FALSE)
#' @param ... Ignored
#'
#' @return list containing: mean, stdev, cov
#' 
#' @method predict NuggetKriging
#' @export predict
#' @aliases predict,NuggetKriging,NuggetKriging-method
#' 
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#'   plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#'   points(X,y,col='blue')
#' r <- NuggetKriging(y, X, "gauss")
#' x = seq(0,1,,101)
#' p_x = predict(r, x)
#'   lines(x,p_x$mean,col='blue')
#'   lines(x,p_x$mean-2*p_x$stdev,col='blue')
#'   lines(x,p_x$mean+2*p_x$stdev,col='blue')
predict.NuggetKriging <- function(object, x, stdev=T, cov=F, ...) {
  if (length(list(...))>0) warning("Arguments ",paste0(names(list(...)),"=",list(...),collapse=",")," are ignored.")
  k=nuggetkriging_model(object) 
  if (!is.matrix(x)) x=matrix(x,ncol=ncol(k$X))
  if (ncol(x)!=ncol(k$X))
    stop("Input x must have ",ncol(k$X), " columns (instead of ",ncol(x),")")
  return(nuggetkriging_predict(object, x,stdev,cov))
}

predict <- function (...) UseMethod("predict")
setMethod("predict", "NuggetKriging", predict.NuggetKriging)


#' Simulate (conditional) NuggetKriging model at given points
#' 
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param object S3 NuggetKriging object
#' @param nsim number of simulations to perform
#' @param seed random seed used
#' @param x points in model input space where to simulate
#' @param ... Ignored
#'
#' @return length(x) x nsim matrix containing simulated path at x points
#' 
#' @method simulate NuggetKriging
#' @export simulate
#' @aliases simulate,NuggetKriging,NuggetKriging-method
#' 
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#'   plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#'   points(X,y,col='blue')
#' r <- NuggetKriging(y, X, "gauss")
#' x = seq(0,1,,101)
#' s_x = simulate(r, nsim=3, x=x)
#'   lines(x,s_x[,1],col='blue')
#'   lines(x,s_x[,2],col='blue')
#'   lines(x,s_x[,3],col='blue')
simulate.NuggetKriging <- function(object, nsim=1, seed=123, x, ...) {
  if (length(list(...))>0) warning("Arguments ",paste0(names(list(...)),"=",list(...),collapse=",")," are ignored.")
  k=nuggetkriging_model(object) 
  if (!is.matrix(x)) x=matrix(x,ncol=ncol(k$X))
  if (ncol(x)!=ncol(k$X))
    stop("Input x must have ",ncol(k$X), " columns (instead of ",ncol(x),")")
  if (is.null(seed)) seed = floor(runif(1)*99999)
  return(nuggetkriging_simulate(object, nsim=nsim, seed=seed, X=x))
}

simulate <- function (...) UseMethod("simulate")
setMethod("simulate", "NuggetKriging", simulate.NuggetKriging)


#' Update NuggetKriging model with new points
#' 
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param object S3 NuggetKriging object
#' @param newy new points in model output space
#' @param newX new points in model input space
#' @param normalize Normalize X and y in [0,1] (FALSE by default)
#' @param ... Ignored
#' 
#' @method update NuggetKriging
#' @export update
#' @aliases update,NuggetKriging,NuggetKriging-method
#' 
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#'   plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#'   points(X,y,col='blue')
#' r <- NuggetKriging(y, X, "gauss")
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
update.NuggetKriging <- function(object, newy, newX, normalize=FALSE, ...) {
  if (length(list(...))>0) warning("Arguments ",paste0(names(list(...)),"=",list(...),collapse=",")," are ignored.")
  k=nuggetkriging_model(object) 
  if (!is.matrix(newX)) newX=matrix(newX,ncol=ncol(k$X))
  if (!is.matrix(newy)) newy=matrix(newy,ncol=ncol(k$y))
  if (ncol(newX)!=ncol(k$X))
    stop("Input newX must have ",ncol(k$X), " columns (instead of ",ncol(newX),")")
  if (nrow(newy)!=nrow(newX))
    stop("Input newX and newy must have the same number of rows.")
  nuggetkriging_update(object, newy, newX, normalize)
}

update <- function(...) UseMethod("update")
setMethod("update", "NuggetKriging", update.NuggetKriging)
#setGeneric(name = "update", def = function(...) standardGeneric("update"))


#' Compute log-Likelihood of NuggetKriging model
#' 
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param object S3 NuggetKriging object
#' @param theta range parameters to evaluate
#' @param grad return Gradient ? (default is TRUE)
#'
#' @return log-Likelihood computed for given theta
#' 
#' @method logLikelihood NuggetKriging
#' @export logLikelihood
#' @aliases logLikelihood,NuggetKriging,NuggetKriging-method
#' 
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#' r <- NuggetKriging(y, X, "gauss")
#' print(r)
#' ll = function(theta) logLikelihood(r,c(theta,1))$logLikelihood
#' t = seq(0.0001,2,,101)
#'   plot(t,ll(t),type='l')
#'   abline(v=as.list(r)$theta,col='blue')
logLikelihood.NuggetKriging <- function(object, theta_alpha, grad=FALSE) {
  k=nuggetkriging_model(object) 
  if (!is.matrix(theta_alpha)) theta_alpha=matrix(theta_alpha,ncol=ncol(k$X)+1)
  if (ncol(theta_alpha)!=ncol(k$X)+1)
    stop("Input theta,alpha must have ",ncol(k$X)+1, " columns (instead of ",ncol(theta_alpha),")")
  out=list(logLikelihood=matrix(NA,nrow=nrow(theta_alpha)),logLikelihoodGrad=matrix(NA,nrow=nrow(theta_alpha),ncol=ncol(theta_alpha)))
  for (i in 1:nrow(theta_alpha)) {
    ll = nuggetkriging_logLikelihood(object,theta_alpha[i,],isTRUE(grad))
    out$logLikelihood[i] = ll$logLikelihood
    if (isTRUE(grad)) out$logLikelihoodGrad[i,] = ll$logLikelihoodGrad
  }
  if (!isTRUE(grad)) out$logLikelihoodGrad <- NULL
  return(out)
}

#' Compute model log-Likelihood at given args
#'
#' @param ... args
#'
#' @return log-Likelihood
#' @export
logLikelihood <- function (...) UseMethod("logLikelihood")
setMethod("logLikelihood", "NuggetKriging", logLikelihood.NuggetKriging)
setGeneric(name = "logLikelihood", def = function(...) standardGeneric("logLikelihood"))

#' Compute log-Marginal-Posterior of NuggetKriging model
#' 
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param object S3 NuggetKriging object
#' @param theta range parameters to evaluate
#' @param grad return Gradient ? (default is TRUE)
#'
#' @return log-MargPost computed for given theta
#' 
#' @method logMargPost NuggetKriging
#' @export logMargPost
#' @aliases logMargPost,NuggetKriging,NuggetKriging-method
#' 
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#' r <- NuggetKriging(y, X, "gauss")
#' print(r)
#' lmp = function(theta) logMargPost(r,theta)$logMargPost
#' t = seq(0.0001,2,,101)
#'   plot(t,lmp(t),type='l')
#'   abline(v=as.list(r)$theta,col='blue')
logMargPost.NuggetKriging <- function(object, theta, grad=FALSE) {
  k=nuggetkriging_model(object) 
  if (!is.matrix(theta)) theta=matrix(theta,ncol=ncol(k$X))
  if (ncol(theta)!=ncol(k$X))
    stop("Input theta must have ",ncol(k$X), " columns (instead of ",ncol(theta),")")
  out=list(logMargPost=matrix(NA,nrow=nrow(theta)),logMargPostGrad=matrix(NA,nrow=nrow(theta),ncol=ncol(theta)))
  for (i in 1:nrow(theta)) {
    lmp = nuggetkriging_logMargPost(object,theta[i,],isTRUE(grad))
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
setMethod("logMargPost", "NuggetKriging", logMargPost.NuggetKriging)
setGeneric(name = "logMargPost", def = function(...) standardGeneric("logMargPost"))