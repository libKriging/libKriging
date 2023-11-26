library(testthat)
 Sys.setenv('OMP_THREAD_LIMIT'=2)
library(rlibkriging)

library(RobustGaSP)

context("RobustGaSP / Fit: 1D")

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#plot(f)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y = f(X)
#points(X,y)
k = RobustGaSP::rgasp(design=X,response=y, trend=cbind(matrix(1,length(y),1),X))
#library(rlibkriging)
r <- Kriging(y, X,
  kernel="matern5_2",
  regmodel = "linear", normalize = FALSE,
  optim = "BFGS",
  objective = "LMP")
# m = as.list(r)

# Check lmp function

lmp_rgasp = function(X, model=k) {if (!is.matrix(X)) X = matrix(X,ncol=1);
                  # print(dim(X));
                  apply(X,1,
                    function(x) {
                      #y=-logMargPostFun(r,matrix(unlist(x),ncol=2))$logMargPost
                      y=RobustGaSP:::neg_log_marginal_post_approx_ref(param=(x),nugget=0, nugget.est=model@nugget.est, 
                                        R0=model@R0,X=model@X, zero_mean=model@zero_mean,output=model@output, 
                                        CL=model@CL, 
                                        a=0.2,
                                        b=1/(length(model@output))^{1/dim(as.matrix(model@input))[2]}*(0.2+dim(as.matrix(model@input))[2]),
                                        kernel_type=rep(as.integer(3),ncol(X)),alpha=model@alpha
                                        )
                      y})}
lmp_rgasp(1)

plot(lmp_rgasp,xlim=c(0.01,6))
abline(v=(log(k@beta_hat)))

lmp_lk = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=1);
                  # print(dim(X));
                  apply(X,1,
                    function(x) {
                      y=-logMargPostFun(r,matrix(unlist(exp(-(x))),ncol=1))$logMargPost
                      y})}
lmp_lk(1)

lines(seq(0.1,6,,101),lmp_lk(seq(0.1,6,,101)),col='red')
abline(v=(log(1/as.list(r)$theta)),col='red')

precision <- 1e-3
test_that(desc=paste0("RobustGaSP / Fit: 1D / rgasp/lmp is the same that lk/lmp one"),
          expect_equal(lmp_rgasp(1),lmp_lk(1),tol = precision))
test_that(desc=paste0("RobustGaSP / Fit: 1D / fitted theta is the same that RobustGaSP one"),
          expect_equal(as.list(r)$theta[1],1/k@beta_hat,tol = precision))



dlmp_rgasp = function(X, model=k) {if (!is.matrix(X)) X = matrix(X,ncol=1);
                  # print(dim(X));
                  apply(X,1,
                    function(x) {

#    print(RobustGaSP:::log_marginal_lik_deriv(param=(x),nugget=0,nugget_est=model@nugget.est, 
#                                        R0=model@R0,X=model@X, zero_mean=model@zero_mean,
#                                        output=model@output, 
#                                        kernel_type=rep(as.integer(3),ncol(X)),alpha=model@alpha))
#
#    print(RobustGaSP:::log_approx_ref_prior_deriv(param=(x),nugget=0, nugget_est=model@nugget.est, 
#                                        CL=model@CL, 
#                                        a=0.2,
#                                        b=1/(length(model@output))^{1/dim(as.matrix(model@input))[2]}*(0.2+dim(as.matrix(model@input))[2])))


                      #y=-logMargPostFun(r,matrix(unlist(x),ncol=2))$logMargPost
                      y=RobustGaSP:::neg_log_marginal_post_approx_ref_deriv(param=(x),nugget=0, nugget.est=model@nugget.est, 
                                        R0=model@R0,X=model@X, zero_mean=model@zero_mean,output=model@output, 
                                        CL=model@CL, 
                                        a=0.2,
                                        b=1/(length(model@output))^{1/dim(as.matrix(model@input))[2]}*(0.2+dim(as.matrix(model@input))[2]),
                                        kernel_type=rep(as.integer(3),ncol(X)),alpha=model@alpha
                                        )
                      y})}
dlmp_rgasp(1)

dlmp_lk = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=1);
                  apply(X,1,
                    function(x) {
                      y=-logMargPostFun(r,matrix(unlist(exp(-(x))),ncol=1),TRUE)$logMargPostGrad
                      y})}
-exp(-1)*dlmp_lk(1)

precision <- 1e-3
test_that(desc=paste0("RobustGaSP / Fit: 1D / rgasp/lmp deriv is the same that lk/lmp deriv"),
          expect_equal(dlmp_rgasp(1),-exp(-1)*dlmp_lk(1),tol = precision))


