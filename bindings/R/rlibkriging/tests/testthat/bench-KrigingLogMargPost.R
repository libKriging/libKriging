library(testthat)

f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))
n <- 100
set.seed(123)
X <- cbind(runif(n),runif(n),runif(n))
y <- f(X)

xx <- seq(0.01, 1,, 21)
times <- list(R_ll=rep(NA, length(xx)), R_gll=rep(NA, length(xx)), cpp_ll=rep(NA, length(xx)), cpp_gll=rep(NA, length(xx)))
times.n = 100

k <- RobustGaSP::rgasp(design=X, response=y, kernel_type = "matern_3_2")
  lmp = function(theta) {
    #cat("theta: ",theta,"\n")
    lml = RobustGaSP::log_marginal_lik(param=log(1/theta),nugget=k@nugget,nugget_est=k@nugget.est,
      R0=k@R0,X=k@X,zero_mean=k@zero_mean,output=k@output,kernel_type=2,alpha=k@alpha)
    #cat("  lml: ",lml,"\n")
    larp = RobustGaSP::log_approx_ref_prior(param=log(1/theta),nugget=k@nugget,nugget_est=k@nugget.est,
      CL=k@CL,a=0.2,b=1/(length(y))^{1/dim(as.matrix(X))[2]}*(0.2+dim(as.matrix(X))[2]))
    #cat("  larp: ",larp,"\n")
    return(lml+larp)
  }

  lmp_deriv = function(theta) {
    #cat("theta: ",theta,"\n")
    lml_d = RobustGaSP::log_marginal_lik_deriv(param=log(1/theta),nugget=k@nugget,nugget_est=k@nugget.est,
      R0=k@R0,X=k@X,zero_mean=k@zero_mean,output=k@output,kernel_type=2,alpha=k@alpha)
    #cat("  lml_d: ",lml_d,"\n")
    larp_d = RobustGaSP::log_approx_ref_prior_deriv(param=log(1/theta),nugget=k@nugget,nugget_est=k@nugget.est,
      CL=k@CL,a=0.2,b=1/(length(y))^{1/dim(as.matrix(X))[2]}*(0.2+dim(as.matrix(X))[2]))
    #cat("  larp_d: ",larp_d,"\n")
    return((lml_d + larp_d)* 1/theta * (-1/theta))
  }

i <- 1
for (x in xx){
  times$R_ll[i]=system.time(for (j in 1:times.n) llx <- lmp(rep(x,3)))
  times$R_gll[i]=system.time(for (j in 1:times.n) gllx <- lmp_deriv(rep(x,3)))
  i <- i+1
}

 pack=list.files(file.path("bindings","R"),pattern = ".tar.gz",full.names = T)
 install.packages(pack,repos=NULL)
 library(rlibkriging)

r <- Kriging(y, X, "gauss")#"matern3_2")
i <- 1
for (x in xx){
  times$cpp_ll[i]=system.time(for (j in 1:times.n) ll2x <- logMargPost(r,rep(x,3))$logMargPost)
  times$cpp_gll[i]=system.time(for (j in 1:times.n) gll2x <- logMargPost(r,rep(x,3),grad=T)$logMargPostGrad)
  i <- i+1
}

plot(xx,log(times$R_ll),ylim=c(log(min(min(times$R_ll,na.rm = T),min(times$cpp_ll,na.rm = T))),log(max(max(times$R_ll,na.rm = T),max(times$cpp_ll,na.rm = T)))),xlab="nb points",ylab="log(user_time (s))", panel.first=grid())
text(20,-1,"RobustGaSP")
points(xx,log(times$cpp_ll),col='red')
text(80,-1,"libKriging",col = 'red')

hist(times$R_ll/times$cpp_ll,main="times R/cpp")
abline(v=1.0,col='red')

hist(times$R_gll/times$cpp_gll,main="times R/cpp")
abline(v=1.0,col='red')