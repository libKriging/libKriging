library(RobustGaSP)

context("RobustGaSP / Fit: 1D")

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#plot(f)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y = f(X)
#points(X,y)
k = RobustGaSP::rgasp(design=X,response=y)
library(rlibkriging)
r <- Kriging(y, X,
  kernel="matern5_2",
  regmodel = "constant", normalize = FALSE,
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


# Check predict

ntest <- 100
Xtest <- seq(0,1,,ntest)
Ytest_rgasp <- predict(k,matrix(Xtest,ncol=1))
Ytest_libK <- predict(r,Xtest)

plot(f)
points(X,y)
lines(Xtest,Ytest_rgasp$mean,col='blue')
polygon(c(Xtest,rev(Xtest)),
        c(Ytest_rgasp$mean+2*Ytest_rgasp$sd,rev(Ytest_rgasp$mean-2*Ytest_rgasp$sd)),
        col=rgb(0,0,1,0.1), border=NA)

lines(Xtest,Ytest_libK$mean,col='red')
polygon(c(Xtest,rev(Xtest)),
        c(Ytest_libK$mean+2*Ytest_libK$stdev,rev(Ytest_libK$mean-2*Ytest_libK$stdev)),
        col=rgb(1,0,0,0.1), border=NA)

precision <- 1e-3
test_that(desc=paste0("pred mean is the same that RobustGaSP one"),
          expect_equal(predict(r,0.7)$mean[1],predict(k,matrix(0.7))$mean,tol = precision))
test_that(desc=paste0("pred sd is the same that RobustGaSP one"),
          expect_equal(predict(r,0.7)$stdev[1],predict(k,matrix(0.7))$sd,tol = precision))


## RobustGaSP examples

  #---------------------------------------
  # a 1 dimensional example 
  #---------------------------------------
context("RobustGaSP / 1 dimensional example")


  input=10*seq(0,1,1/14)
  output<-higdon.1.data(input)
  #the following code fit a GaSP with zero mean by setting zero.mean="Yes"
  model<- rgasp(design = input, response = output, zero.mean="No")
  model
  
  testing_input = as.matrix(seq(0,10,1/100))
  model.predict<-predict(model,testing_input)
  names(model.predict)
  
  #########plot predictive distribution
  testing_output=higdon.1.data(testing_input)
  plot(testing_input,model.predict$mean,type='l',col='blue',
       xlab='input',ylab='output')
  polygon( c(testing_input,rev(testing_input)),c(model.predict$lower95,
        rev(model.predict$upper95)),col =  "grey80", border = FALSE)
  lines(testing_input, testing_output)
  lines(testing_input,model.predict$mean,type='l',col='blue')
  lines(input, output,type='p')
  
  ## mean square erros
  mean((model.predict$mean-testing_output)^2)

model_libK = Kriging(matrix(output,ncol=1), matrix(input,ncol=1), 
  kernel="matern5_2", 
  regmodel = "constant", normalize = FALSE, 
  optim = "BFGS", 
  objective = "LMP", parameters = NULL)

    lines(testing_input,predict(model_libK,testing_input)$mean,type='l',col='red')
    polygon( 
      c(testing_input,rev(testing_input)),
      c(
        predict(model_libK,testing_input)$mean+2*predict(model_libK,testing_input)$stdev,
        rev(predict(model_libK,testing_input)$mean-2*predict(model_libK,testing_input)$stdev)),
      col = rgb(1,0,0,0.1), border = FALSE)

precision <- 1e-3
test_that(desc=paste0("RobustGaSP / 1 dimensional example / pred mean is the same that RobustGaSP one"),
          expect_equal(predict(model_libK,0.7)$mean[1],predict(model,matrix(0.7))$mean,tol = precision))
test_that(desc=paste0("RobustGaSP / 1 dimensional example / pred sd is the same that RobustGaSP one"),
          expect_equal(predict(model_libK,0.7)$stdev[1],predict(model,matrix(0.7))$sd,tol = precision))





context("RobustGaSP / Fit: 2D (Branin)")

f = function(X) apply(X,1,DiceKriging::branin)
n <- 15
set.seed(1234)
X <- cbind(runif(n),runif(n))
y = f(X)
model = NULL
r = NULL
library(RobustGaSP)
k = rgasp(design=X,response=y)
library(rlibkriging)
r <- Kriging(y, X, "matern5_2", objective="LMP", optim="BFGS10")

lmp_rgasp = function(X, model=k) {if (!is.matrix(X)) X = matrix(X,ncol=2);
                  # print(dim(X));
                  apply(X,1,
                    function(x) {
                      #y=-logMargPostFun(r,matrix(unlist(x),ncol=2))$logMargPost
    
#    print(RobustGaSP:::log_marginal_lik(param=(x),nugget=0,nugget_est=model@nugget.est, 
#                                        R0=model@R0,X=model@X, zero_mean=model@zero_mean,
#                                        output=model@output, 
#                                        kernel_type=rep(as.integer(3),ncol(X)),alpha=model@alpha))
#
#    print(RobustGaSP:::log_approx_ref_prior(param=(x),nugget=0, nugget_est=model@nugget.est, 
#                                        CL=model@CL, 
#                                        a=0.2,
#                                        b=1/(length(model@output))^{1/dim(as.matrix(model@input))[2]}*(0.2+dim(as.matrix(model@output))[2])))

                      y=RobustGaSP:::neg_log_marginal_post_approx_ref(param=(x),nugget=0, nugget.est=model@nugget.est, 
                                        R0=model@R0,X=model@X, zero_mean=model@zero_mean,output=model@output, 
                                        CL=model@CL, 
                                        a=0.2,
                                        b=1/(length(model@output))^{1/dim(as.matrix(model@input))[2]}*(0.2+dim(as.matrix(model@input))[2]),
                                        kernel_type=rep(as.integer(3),ncol(X)),alpha=model@alpha
                                        )
                      y})}
lmp_rgasp(c(1,1))

x=seq(-3,3,,51)
contour(x,x,matrix(lmp_rgasp(as.matrix(expand.grid(x,x))),nrow=length(x)),levels=seq(50,100,,51))
points(log(k@beta_hat[1]),log(k@beta_hat[2]))

lmp_lk = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=2);
                  apply(X,1,
                    function(x) {
                      y=-logMargPostFun(r,matrix(unlist(exp(-(x))),ncol=2))$logMargPost
                      y})}
lmp_lk(c(1,1))

#contour(x,x,matrix(lmp_lk(as.matrix(expand.grid(x,x))),nrow=length(x)),levels = seq(50,100,,51),col='red')
points(log(1/as.list(r)$theta[1]),log(1/as.list(r)$theta[2]),col='red')

precision <- 1e-1
test_that(desc=paste0("Fit: 2D (Branin) / rgasp/lmp is the same that lk/lmp one"),
          expect_equal(lmp_rgasp(c(1,1)),lmp_lk(c(1,1)),tol = precision))
test_that(desc=paste0("Fit: 2D (Branin) / fitted theta is the same that RobustGaSP one"),
          expect_equal(lmp_rgasp(log(k@beta_hat)),lmp_rgasp(t(log(1/as.list(r)$theta))),tol = precision))


dlmp_rgasp = function(X, model=k) {if (!is.matrix(X)) X = matrix(X,ncol=2);
                  # print(dim(X));
                  apply(X,1,
                    function(x) {
                      #y=-logMargPostFun(r,matrix(unlist(x),ncol=2))$logMargPost
                      y=RobustGaSP:::neg_log_marginal_post_approx_ref_deriv(param=(x),nugget=0, nugget.est=model@nugget.est, 
                                        R0=model@R0,X=model@X, zero_mean=model@zero_mean,output=model@output, 
                                        CL=model@CL, 
                                        a=0.2,
                                        b=1/(length(model@output))^{1/dim(as.matrix(model@input))[2]}*(0.2+dim(as.matrix(model@input))[2]),
                                        kernel_type=rep(as.integer(3),ncol(X)),alpha=model@alpha
                                        )
                      y})}
dlmp_rgasp(c(1,1))

dlmp_lk = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=2);
                  apply(X,1,
                    function(x) {
                      y=-logMargPostFun(r,matrix(unlist(exp(-(x))),ncol=2),TRUE)$logMargPostGrad
                      y})}
-exp(-1)*dlmp_lk(c(1,1))

precision <- 1e-1
test_that(desc=paste0("Fit: 2D (Branin) / rgasp/lmp deriv is the same that lk/lmp deriv"),
          expect_equal(dlmp_rgasp(c(1,1)),-exp(-1)*dlmp_lk(c(1,1)),tol = precision))