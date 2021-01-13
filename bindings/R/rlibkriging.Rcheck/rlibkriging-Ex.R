pkgname <- "rlibkriging"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('rlibkriging')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("as.list.Kriging")
### * as.list.Kriging

flush(stderr()); flush(stdout())

### Name: as.list.Kriging
### Title: List Kriging object content
### Aliases: as.list.Kriging as.list,Kriging,Kriging-method

### ** Examples

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
set.seed(123)
X <- as.matrix(runif(5))
y <- f(X)
r <- Kriging(y, X, "gauss")
l = as.list(r)
cat(paste0(names(l)," =" ,l,collapse="\n"))



cleanEx()
nameEx("as_km.Kriging")
### * as_km.Kriging

flush(stderr()); flush(stdout())

### Name: as_km.Kriging
### Title: Convert a "Kriging" object to a DiceKriging::km one.
### Aliases: as_km.Kriging as_km,Kriging,Kriging-method

### ** Examples

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
set.seed(123)
X <- as.matrix(runif(5))
y <- f(X)
r <- Kriging(y, X, "gauss")
print(r)
k <- as_km(r)
print(k)



cleanEx()
nameEx("as_km.default")
### * as_km.default

flush(stderr()); flush(stdout())

### Name: as_km.default
### Title: Build a DiceKriging "km" like object.
### Aliases: as_km.default

### ** Examples

# a 16-points factorial design, and the corresponding response
d <- 2; n <- 16
design.fact <- expand.grid(x1=seq(0,1,length=4), x2=seq(0,1,length=4))
y <- apply(design.fact, 1, DiceKriging::branin) 

#library(DiceKriging)
# kriging model 1 : matern5_2 covariance structure, no trend, no nugget effect
#m1 <- km(design=design.fact, response=y,covtype = "gauss",parinit = c(.5,1),control = list(trace=F))
as_m1 <- as_km(design=design.fact, response=y,covtype = "gauss",parinit = c(.5,1))



cleanEx()
nameEx("leaveOneOut.Kriging")
### * leaveOneOut.Kriging

flush(stderr()); flush(stdout())

### Name: leaveOneOut.Kriging
### Title: Compute leave-One-Out of Kriging model
### Aliases: leaveOneOut.Kriging leaveOneOut,Kriging,Kriging-method

### ** Examples

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
set.seed(123)
X <- as.matrix(runif(5))
y <- f(X)
r <- Kriging(y, X, "gauss",objective="LOO")
print(r)
loo = function(theta) leaveOneOut(r,theta)$leaveOneOut
t = seq(0.0001,2,,101)
  plot(t,loo(t),type='l')
  abline(v=as.list(r)$theta,col='blue')



cleanEx()
nameEx("logLikelihood.Kriging")
### * logLikelihood.Kriging

flush(stderr()); flush(stdout())

### Name: logLikelihood.Kriging
### Title: Compute log-Likelihood of Kriging model
### Aliases: logLikelihood.Kriging logLikelihood,Kriging,Kriging-method

### ** Examples

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
set.seed(123)
X <- as.matrix(runif(5))
y <- f(X)
r <- Kriging(y, X, "gauss")
print(r)
ll = function(theta) logLikelihood(r,theta)$logLikelihood
t = seq(0.0001,2,,101)
  plot(t,ll(t),type='l')
  abline(v=as.list(r)$theta,col='blue')



cleanEx()
nameEx("predict.Kriging")
### * predict.Kriging

flush(stderr()); flush(stdout())

### Name: predict.Kriging
### Title: Predict Kriging model at given points
### Aliases: predict.Kriging predict,Kriging,Kriging-method

### ** Examples

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
  plot(f)
set.seed(123)
X <- as.matrix(runif(5))
y <- f(X)
  points(X,y,col='blue')
r <- Kriging(y, X, "gauss")
x = seq(0,1,,101)
p_x = predict(r, x)
  lines(x,p_x$mean,col='blue')
  lines(x,p_x$mean-2*p_x$stdev,col='blue')
  lines(x,p_x$mean+2*p_x$stdev,col='blue')



cleanEx()
nameEx("predict.as_km")
### * predict.as_km

flush(stderr()); flush(stdout())

### Name: predict.as_km
### Title: Overload DiceKriging::predict.km for as_km objects (expected
###   faster).
### Aliases: predict.as_km predict,as_km,as_km-method

### ** Examples

# a 16-points factorial design, and the corresponding response
d <- 2; n <- 16
design.fact <- expand.grid(x1=seq(0,1,length=4), x2=seq(0,1,length=4))
y <- apply(design.fact, 1, DiceKriging::branin) 

#library(DiceKriging)
# kriging model 1 : matern5_2 covariance structure, no trend, no nugget effect
#m1 <-      km(design=design.fact, response=y,covtype = "gauss",parinit = c(.5,1),control = list(trace=F))
as_m1 <- as_km(design=design.fact, response=y,covtype = "gauss",parinit = c(.5,1))
as_p = predict(as_m1,newdata=matrix(.5,ncol=2),type="UK",checkNames=FALSE,light.return=TRUE)



cleanEx()
nameEx("print.Kriging")
### * print.Kriging

flush(stderr()); flush(stdout())

### Name: print.Kriging
### Title: Print Kriging object content
### Aliases: print.Kriging print,Kriging,Kriging-method

### ** Examples

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
set.seed(123)
X <- as.matrix(runif(5))
y <- f(X)
r <- Kriging(y, X, "gauss")
print(r)



cleanEx()
nameEx("simulate.Kriging")
### * simulate.Kriging

flush(stderr()); flush(stdout())

### Name: simulate.Kriging
### Title: Simulate (conditional) Kriging model at given points
### Aliases: simulate.Kriging simulate,Kriging,Kriging-method

### ** Examples

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
  plot(f)
set.seed(123)
X <- as.matrix(runif(5))
y <- f(X)
  points(X,y,col='blue')
r <- Kriging(y, X, "gauss")
x = seq(0,1,,101)
s_x = simulate(r, nsim=3, x=x)
  lines(x,s_x[,1],col='blue')
  lines(x,s_x[,2],col='blue')
  lines(x,s_x[,3],col='blue')



cleanEx()
nameEx("simulate.as_km")
### * simulate.as_km

flush(stderr()); flush(stdout())

### Name: simulate.as_km
### Title: Overload DiceKriging::simulate.km for as_km objects (expected
###   faster).
### Aliases: simulate.as_km simulate,as_km,as_km-method

### ** Examples

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
  plot(f)
set.seed(123)
X <- as.matrix(runif(5))
y <- f(X)
  points(X,y,col='blue')
k <- as_km(design=X, response=y,covtype = "gauss")
x = seq(0,1,,101)
s_x = simulate(k, nsim=3, newdata=x)
  lines(x,s_x[,1],col='blue')
  lines(x,s_x[,2],col='blue')
  lines(x,s_x[,3],col='blue')



cleanEx()
nameEx("update.Kriging")
### * update.Kriging

flush(stderr()); flush(stdout())

### Name: update.Kriging
### Title: Update Kriging model with new points
### Aliases: update.Kriging update,Kriging,Kriging-method

### ** Examples

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
  plot(f)
set.seed(123)
X <- as.matrix(runif(5))
y <- f(X)
  points(X,y,col='blue')
r <- Kriging(y, X, "gauss")
x = seq(0,1,,101)
p_x = predict(r, x)
  lines(x,p_x$mean,col='blue')
  lines(x,p_x$mean-2*p_x$stdev,col='blue')
  lines(x,p_x$mean+2*p_x$stdev,col='blue')
newX <- as.matrix(runif(3))
newy <- f(newX)
  points(newX,newy,col='red')
update(r,newy,newX)
x = seq(0,1,,101)
p2_x = predict(r, x)
  lines(x,p2_x$mean,col='red')
  lines(x,p2_x$mean-2*p2_x$stdev,col='red')
  lines(x,p2_x$mean+2*p2_x$stdev,col='red')



cleanEx()
nameEx("update.as_km")
### * update.as_km

flush(stderr()); flush(stdout())

### Name: update.as_km
### Title: Overload DiceKriging::update.km methd for as_km objects
###   (expected faster).
### Aliases: update.as_km update,as_km,as_km-method

### ** Examples

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
  plot(f)
set.seed(123)
X <- as.matrix(runif(5))
y <- f(X)
  points(X,y,col='blue')
k <- as_km(design=X, response=y,covtype = "gauss")
x = seq(0,1,,101)
p_x = predict(k, x)
  lines(x,p_x$mean,col='blue')
  lines(x,p_x$lower95,col='blue')
  lines(x,p_x$upper95,col='blue')
newX <- as.matrix(runif(3))
newy <- f(newX)
  points(newX,newy,col='red')
update(k,newy,newX)
x = seq(0,1,,101)
p2_x = predict(k, x)
  lines(x,p2_x$mean,col='red')
  lines(x,p2_x$lower95,col='red')
  lines(x,p2_x$upper95,col='red')



### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
