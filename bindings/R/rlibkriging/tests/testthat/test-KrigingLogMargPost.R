library(testthat)

kernel_type = function(kernel) {
  if (kernel=="matern3_2") return("matern_3_2")
  if (kernel=="matern5_2") return("matern_5_2")
  stop(paste0("Cannot use ",kernel))
}
kernel_type_num = function(kernel) {
  if (kernel=="matern3_2") return(2)
  if (kernel=="matern5_2") return(3)
  stop(paste0("Cannot use ",kernel))
}

for (kernel in c("matern5_2","matern3_2")) {
  context(paste0("Check Marginal Posterior for kernel ",kernel))
  
  f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
  plot(f)
  n <- 5
  set.seed(123)
  X <- as.matrix(runif(n))
  y = f(X)
  points(X,y)
  k = RobustGaSP::rgasp(design=X,response=y,kernel_type=kernel_type(kernel))

  lmp = function(theta) {
    #cat("theta: ",theta,"\n")
    lml = RobustGaSP::log_marginal_lik(param=log(1/theta),nugget=k@nugget,nugget_est=k@nugget.est,
      R0=k@R0,X=k@X,zero_mean=k@zero_mean,output=k@output,kernel_type=kernel_type_num(kernel),alpha=k@alpha)
    #cat("  lml: ",lml,"\n")
    larp = RobustGaSP::log_approx_ref_prior(param=log(1/theta),nugget=k@nugget,nugget_est=k@nugget.est,
      CL=k@CL,a=0.2,b=1/(length(y))^{1/dim(as.matrix(X))[2]}*(0.2+dim(as.matrix(X))[2]))
    #cat("  larp: ",larp,"\n")
    return(lml+larp)
  }

  plot(Vectorize(lmp),ylab="LMP",xlab="theta",xlim=c(0.01,2),ylim=c(-5,5))
  abline(v=1/k@beta_hat)

  lmp_deriv = function(theta) {
    #cat("theta: ",theta,"\n")
    lml_d = RobustGaSP::log_marginal_lik_deriv(param=log(1/theta),nugget=k@nugget,nugget_est=k@nugget.est,
      R0=k@R0,X=k@X,zero_mean=k@zero_mean,output=k@output,kernel_type=kernel_type_num(kernel),alpha=k@alpha)
    #cat("  lml_d: ",lml_d,"\n")
    larp_d = RobustGaSP::log_approx_ref_prior_deriv(param=log(1/theta),nugget=k@nugget,nugget_est=k@nugget.est,
      CL=k@CL,a=0.2,b=1/(length(y))^{1/dim(as.matrix(X))[2]}*(0.2+dim(as.matrix(X))[2]))
    #cat("  larp_d: ",larp_d,"\n")
    return((lml_d + larp_d)* 1/theta * (-1/theta))
  }

  for (x in seq(0.01,2,,11)){
    arrows(x,lmp(x),x+.1,lmp(x)+.1*lmp_deriv(x))
  }

  library(rlibkriging)
  r <- Kriging(y, X, kernel)
  ## Should be equal:
  #lmp(1.0); lmp_deriv(1.0);
  #logMargPost(r,1.0,grad = T)
  #lmp(0.1); lmp_deriv(0.1);
  #logMargPost(r,0.1,grad = T)
  #ll2 = function(theta) logMargPost(r,theta)$logMargPost
  # plot(Vectorize(ll2),col='red',add=T,xlim=c(0.01,2)) # FIXME fails with "error: chol(): decomposition failed"
  for (x in seq(0.01,2,,11)){
    ll2x = logMargPost(r,x)$logMargPost
    gll2x = logMargPost(r,x,grad = T)$logMargPostGrad
    arrows(x,ll2x,x+.1,ll2x+.1*gll2x,col='red')
  }
  
  precision <- 1e-8  # the following tests should work with it, since the computations are analytical
  x=.5
  test_that(desc="logMargPost is the same that RobustGaSP one", 
            expect_equal(logMargPost(r,x)$logMargPost[1],lmp(x),tolerance = precision))
  
  test_that(desc="logMargPost Grad is the same that RobustGaSP one", 
            expect_equal(logMargPost(r,x,grad = T)$logMargPostGrad[1],lmp_deriv(x),tolerance= precision))
}
