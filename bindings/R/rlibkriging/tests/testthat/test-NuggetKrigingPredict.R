#library(rlibkriging, lib.loc="bindings/R/Rlibs")
#library(testthat)
#kernel="gauss"

for (kernel in c("gauss","exp","matern3_2","matern5_2")) {
  context(paste0("Check predict 1D for kernel ",kernel))

  f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
  #plot(f)
  n <- 5
  set.seed(123)
  X <- as.matrix(runif(n))
  y = f(X)
  #points(X,y)
  k = DiceKriging::km(design=X,response=y,covtype = "gauss",control = list(trace=F), nugget=0,nugget.estim = TRUE)
  #library(rlibkriging)
  r <- NuggetKriging(y,X,"gauss","constant",FALSE,"none","LL",
               parameters=list(sigma2=k@covariance@sd2,has_sigma2=TRUE, is_sigma2_estim=FALSE,
               theta=matrix(k@covariance@range.val),has_theta=TRUE, is_theta_estim=FALSE,
               nugget=k@covariance@nugget,has_nugget=TRUE, is_nugget_estim=FALSE))
  # m = as.list(r)
  
  ntest <- 100
  Xtest <- as.matrix(runif(ntest))
  ptest <- DiceKriging::predict(k,Xtest,type="UK",cov.compute = TRUE,checkNames=F)
  Yktest <- ptest$mean
  sktest <- ptest$sd
  cktest <- c(ptest$cov)
  Ytest <- predict(r,Xtest,TRUE,TRUE)
  
  plot(f)
  points(X,y)
  points(Xtest,Yktest,col='blue')
  points(Xtest,Ytest$mean,col='red')
  
  precision <- 1e-5
  test_that(desc=paste0("pred mean is the same that DiceKriging one:\n ",paste0(collapse=",",Yktest),"\n ",paste0(collapse=",",Ytest$mean)),
            expect_equal(array(Yktest),array(Ytest$mean),tol = precision))
  
  test_that(desc="pred sd is the same that DiceKriging one", 
            expect_equal(array(sktest),array(Ytest$stdev) ,tol = precision))
              
  test_that(desc="pred cov is the same that DiceKriging one", 
            expect_equal(cktest,c(Ytest$cov) ,tol = precision))
  
  plot(f)
  points(X,y)
  x=sort(c(seq(0,1,,101),X))
  lines(x,predict(r,x)$mean,lty=2)
  polygon(
      c(x,rev(x)),
      c(predict(r,x)$mean-predict(r,x)$stdev,
        predict(r,rev(x))$mean+predict(r,rev(x))$stdev),
        col=rgb(0,0,0,.1),border=NA)
  #
  #for (i in 1:10)
  #    lines(x,simulate(r,x=x,seed=i),col='grey')
  
  .x = seq(0.1,0.9,,11)
  p_allx = predict(r,.x,TRUE, cov=FALSE, deriv=TRUE)
  for (i in 1:length(.x)) {
    # ref from DiceOptim::EI.grad
    newdata = .x[i]
    model = k
    T <- model@T
    X <- model@X
    z <- model@z
    u <- model@M
    covStruct <- model@covariance
    predx <- predict(object=model, newdata=newdata, type="UK", checkNames = FALSE,se.compute=TRUE,cov.compute=FALSE)
    kriging.mean <- predx$mean
    kriging.sd <- predx$sd
    v <- predx$Tinv.c
    c <- predx$c
    dc <- DiceKriging::covVector.dx(x=newdata, X=X, object=covStruct, c=c)  
    #d = model@d
    #h=sqrt(.Machine$double.eps)
    #A <- matrix(newdata, nrow=d, ncol=d, byrow=TRUE)
    #Apos <- A+h*diag(d)
    #Aneg <- A-h*diag(d)
    #newpoints <- data.frame(rbind(Apos, Aneg))
    #f.newdata <- model.matrix(model@trend.formula, data = newpoints)
    #f.deltax <- (f.newdata[1:d,]-f.newdata[(d+1):(2*d),])/(2*h)
    f.deltax <- DiceKriging::trend.deltax(x=newdata, model=model)
    W <- backsolve(t(T), dc, upper.tri=FALSE)
    kriging.mean.grad <- t(W)%*%z + t(model@trend.coef%*%f.deltax)
    tuuinv <- solve(t(u)%*%u)
    F.newdata <- model.matrix(model@trend.formula, data=as.data.frame(newdata))
    kriging.sd2.grad <-  t( -2*t(v)%*%W +
                                  2*(F.newdata - t(v)%*%u )%*% tuuinv %*%
                                  (f.deltax - t(t(W)%*%u) ))
    kriging.sd.grad <- kriging.sd2.grad / (2*kriging.sd)
                       
    p = predict(r,.x[i],TRUE, cov=FALSE, deriv=TRUE)
  
    test_that(desc=paste0("vect pred mean deriv is ok:\n ",paste0(collapse=",",p_allx$mean_deriv[i]),"\n ",paste0(collapse=",",p$mean_deriv)),
             expect_equal(array(p_allx$mean_deriv[i]),array(p$mean_deriv),tol = precision))
    test_that(desc=paste0("vect pred sd deriv is ok:\n ",paste0(collapse=",",p_allx$stdev_deriv[i]),"\n ",paste0(collapse=",",p$stdev_deriv)),
             expect_equal(array(p_allx$stdev_deriv[i]),array(p$stdev_deriv),tol = precision))

  arrows(.x[i],p$mean, .x[i]+0.1, p$mean+0.1*p$mean_deriv)
  arrows(.x[i],p$mean+p$stdev, .x[i]+0.1, p$mean+p$stdev+0.1*p$mean_deriv+0.1*p$stdev_deriv, col='darkgrey')
  
    test_that(desc=paste0("pred mean deriv is the same that DiceKriging one:\n ",paste0(collapse=",",kriging.mean.grad),"\n ",paste0(collapse=",",p$mean_deriv)),
              expect_equal(array(kriging.mean.grad),array(p$mean_deriv),tol = precision))
    
    test_that(desc=paste0("pred sd deriv is the same that DiceKriging one:\n ",paste0(collapse=",",kriging.sd.grad),"\n ",paste0(collapse=",",p$stdev_deriv)),
              expect_equal(array(kriging.sd.grad),array(p$stdev_deriv),tol = precision))
              
  }
}

test_that(desc="pred cov is the same that DiceKriging one", 
          expect_equal(cktest,c(Ytest$cov) ,tol = precision))
