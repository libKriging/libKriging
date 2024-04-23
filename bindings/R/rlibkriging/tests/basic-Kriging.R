f <- function(x) {
    1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
}
plot(f)
n <- 5
X_o <- seq(from = 0, to = 1, length.out = n)
y_o <- f(X_o)
points(X_o, y_o)

library(rlibkriging, lib.loc="bindings/R/Rlibs")
lk <- NULL
lk <- Kriging(y = matrix(y_o, ncol = 1),
              X = matrix(X_o, ncol = 1),
              kernel = "matern5_2",
              objective="LOO")#,
              #regmodel = "none",
              #optim = "none", parameters = list(theta = matrix(0.1), sigma2=0.02))
# print aux var of model for checking
lK <- list(X=lk$X(),Y=lk$y(), theta = lk$theta(),
           F = lk$F(),
           T = lk$T(),
           M = lk$M(),
           beta = lk$beta(),
           z = lk$z(),
           sigma2 = lk$sigma2())
print(lK)

unlink("lk.json")
rlibkriging::save(lk, filename="lk.json")
lk2 <- rlibkriging::load(filename="lk.json")
print(lk2)

  library(DiceKriging)
  dk <- km(design=as.matrix(X_o), response = y_o, covtype = lk$kernel(),coef.cov = lk$theta()[1], coef.var = lk$sigma2())
 dk@case = if (lk$is_sigma2_estim()) "LLconcentration_beta_sigma2" else "LLconcentration_beta" # Force using same case as libK
 dk@known.param = if (lk$is_sigma2_estim()) "None" else "Var"
 dk@method = "" # Avoid error
 #dk@T - t(lK$T * sqrt(lK$sigma2))

#### Predict

xn = seq(0,1,,21)
lp = lk$predict(xn) # libK predict
lines(xn,lp$mean,col='red')
polygon(c(xn,rev(xn)),c(lp$mean+2*lp$stdev,rev(lp$mean-2*lp$stdev)),col=rgb(1,0,0,0.2),border=NA)
  dp = predict(dk,type="UK",newdata=xn,checkNames=FALSE)
  lines(xn,dp$mean,col='blue')
  polygon(c(xn,rev(xn)),c(dp$mean+2*dp$sd,rev(dp$mean-2*dp$sd)),col=rgb(0,0,1,0.2),border=NA)

#### Simulate

ls = lk$simulate(10, 123, xn) # libK simulate
  ds = simulate(dk,nsim=10,newdata=xn,seed=123,cond=TRUE,checkNames=FALSE,nugget.sim=1e-5)
for (i in 1:10) {
    lines(xn,ls[,i],col='red',lwd=4)
      lines(xn,ds[i,],col='blue',lwd=2)
}

# e=new.env()
# DiceKriging::logLikFun(param=0.1,model=dk,e)
# #e$z
# #e$T
# #e$R
# DiceKriging::logLikGrad(param=0.1,model=dk,e)
# #lk$logLikelihoodFun(0.1)$logLikelihood
# lk$logLikelihoodFun(0.1,grad=TRUE)$logLikelihoodGrad

#### LogLikelihood

ll <- function(theta) lk$logLikelihoodFun(theta)$logLikelihood
  ll_dk <- Vectorize(function(theta) DiceKriging::logLikFun(param=c(theta,if(lk$is_sigma2_estim())NULL else lk$sigma2()),model=dk))
gll <- function(theta) lk$logLikelihoodFun(theta,grad=TRUE)$logLikelihoodGrad
gll_approx <- function(theta) (ll(theta+0.001)-ll(theta))/0.001    
  gll_dk <- Vectorize(function(theta) {
       e <- new.env()
       theta = c(theta,if(lk$is_sigma2_estim())NULL else lk$sigma2())
       ll= DiceKriging::logLikFun(param=theta,model=dk,envir=e)
       DiceKriging::logLikGrad(param=theta,model=dk,envir=e)
       })
t <- seq(from = 0.001, to = 1, length.out = 101)
plot(t, ll(t), type = 'l',col='red')
  lines(t, ll_dk(t), col = 'blue')
abline(v = lk$theta(), col = "red",lty=2)
for (.t in t) {
    arrows(.t, ll(.t), .t+0.1, ll(.t) + gll(.t)*0.1, length = 0.1, angle = 15, code = 3, col = "red")
    arrows(.t, ll(.t), .t+0.1, ll(.t) + gll_approx(.t)*0.1, length = 0.1, angle = 15, code = 3, col = "orange")
      arrows(.t, ll_dk(.t), .t+0.1, ll_dk(.t) + gll_dk(.t)*0.1, length = 0.1, angle = 15, code = 3, col = "blue")
}

if (lk$kernel() == "gauss") { #: do not work for matern*
hll = function(theta) lk$logLikelihoodFun(theta, hess=TRUE)$logLikelihoodHess
hll_approx = function(theta) (gll(theta+0.000001)-gll(theta))/0.000001
plot(t, hll(t), type = 'l',col='red')
lines(t, hll_approx(t), col = 'blue')
}
#hll(0.5); hll_approx(0.5)
#plot(t, drop(hll(t))-hll_approx(t), type = 'l',col='red')

#### LeaveOneOut

loo <- function(theta) lk$leaveOneOutFun(theta)$leaveOneOut
  loo_dk <- Vectorize(function(theta) DiceKriging::leaveOneOutFun(param=c(theta,if(lk$is_sigma2_estim())NULL else lk$sigma2()),model=dk))
gloo <- function(theta) lk$leaveOneOutFun(theta,grad=TRUE)$leaveOneOutGrad
gloo_approx <- function(theta) (loo(theta+0.001)-loo(theta))/0.001    
  gloo_dk <- Vectorize(function(theta) {
      e <- new.env()
      theta = c(theta,if(lk$is_sigma2_estim())NULL else lk$sigma2())
      ll= DiceKriging::leaveOneOutFun(param=theta,model=dk,envir=e)
      DiceKriging::leaveOneOutGrad(param=theta,model=dk,envir=e)
      })
t <- seq(from = 0.001, to = 1, length.out = 101)
plot(t, loo(t), type = 'l',col='red')
  lines(t, loo_dk(t), col = 'blue')
abline(v = lk$theta(), col = "red",lty=2)
for (.t in t) {
    arrows(.t, loo(.t), .t+0.1, loo(.t) + gloo(.t)*0.1, length = 0.1, angle = 15, code = 3, col = "red")
    arrows(.t, loo(.t), .t+0.1, loo(.t) + gloo_approx(.t)*0.1, length = 0.1, angle = 15, code = 3, col = "orange")
      arrows(.t, loo_dk(.t), .t+0.1, loo_dk(.t) + gloo_dk(.t)*0.1, length = 0.1, angle = 15, code = 3, col = "blue")
}

#### LogMarginalPosterior

  rg = RobustGaSP::rgasp(design=X_o,response=y_o)
  lmp_rgasp = function(X, model=rg) {if (!is.matrix(X)) X = matrix(X,ncol=1);
                    # print(dim(X));
                    X = -log(X)
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
                        -y})}
  
  glmp_rgasp = function(X, model=rg) {if (!is.matrix(X)) X = matrix(X,ncol=1);
                    # print(dim(X));
                    X = -log(X)
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
                        1/exp(-X)*y})}


lmp <- function(theta) lk$logMargPostFun(theta)$logMargPost
  lmp_rgasp(0.5)-lmp(0.5)
glmp <- function(theta) lk$logMargPostFun(theta,grad=TRUE)$logMargPostGrad
  glmp_rgasp(0.5)-glmp(0.5)

glmp_approx <- function(theta) (lmp(theta+0.001)-lmp(theta))/0.001
t <- seq(from = 0.01, to = 1, length.out = 101)
plot(t, lmp(t), type = 'l',col='red')
  lines(t, lmp_rgasp(t), col = 'blue')
abline(v = lk$theta(), col = "red",lty=2)
for (.t in t) {
    arrows(.t, lmp(.t), .t+0.1, lmp(.t) + glmp(.t)*0.1, length = 0.1, angle = 15, code = 3, col = "red")
    arrows(.t, lmp(.t), .t+0.1, lmp(.t) + glmp_approx(.t)*0.1, length = 0.1, angle = 15, code = 3, col = "orange")
      arrows(.t, lmp_rgasp(.t), .t+0.1, lmp_rgasp(.t) + glmp_rgasp(.t)*0.1, length = 0.1, angle = 15, code = 3, col = "blue")
}
