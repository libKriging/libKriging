#library(rlibkriging, lib.loc="bindings/R/Rlibs")

# 2D (Branin) function, small design
set.seed(123)
X <- cbind(runif(15),runif(15))
f <- function(x) {
  if (!is.matrix(x)) x = matrix(x,nrow=1)
  x1 <- x[,1] * 15 - 5
  x2 <- x[,2] * 15
  (x2 - 5/(4 * pi^2) * (x1^2) + 5/pi * x1 - 6)^2 + 10 * (1 - 1/(8 * pi)) * cos(x1) + 10
}
y <- f(X) # + 10*rnorm(nrow(X))

library(DiceView)
contourview(f, dim=2, col = 'black', nlevels=21)
points(X)

# Build Kriging (https://libkriging.readthedocs.io/en/latest/math/KrigingModels.html)
k <- Kriging(y, X, kernel="gauss", optim = "BFGS10") #, objective = "LL", optim = "BFGS", regmodel = "constant", parameters = ...)
# kernel: https://libkriging.readthedocs.io/en/latest/math/kernel.html
# regmodel: https://libkriging.readthedocs.io/en/latest/math/trend.html
# parameters: https://libkriging.readthedocs.io/en/latest/math/parameters.html
print(k)

# Save / load /inspect
save(k,"k.json")

# Predict
#.x <- as.matrix(seq(0, 1, , 11))
#.xx = as.matrix(expand.grid(.x,.x))
#p.xx <- predict(k, .xx, TRUE, FALSE)

par(mfrow=c(1,3))
options(repr.plot.width=15, repr.plot.height=5)

contourview(f, dim=2, col_surf = 'black', nlevels=21)
points(X)
contourview(function(.xx) predict(k,.xx,TRUE,FALSE)$mean, dim=2, col_surf = 'blue', nlevels=21, add=TRUE)

contourview(f, dim=2, col_surf = 'black', nlevels=21)
points(X)
contourview(function(.xx) {p.xx=predict(k,.xx,TRUE,FALSE); p.xx$mean + 2*p.xx$stdev}, dim=2, col_surf = 'blue', nlevels=21, add=TRUE)

contourview(f, dim=2, col_surf = 'black', nlevels=21)
points(X)
contourview(function(.xx) {p.xx=predict(k,.xx,TRUE,FALSE); p.xx$mean - 2*p.xx$stdev}, dim=2, col_surf = 'blue', nlevels=21, add=TRUE)

options(repr.plot.width=7, repr.plot.height=7)

sectionview3d(f, dim=2, col_surf = 'black', engine3d = "scatterplot3d")
sectionview3d(k, add=TRUE)

# Simulate
#s.xx <- simulate(k, nsim = 1, seed = 123, x=.xx)

contourview(f, dim=2, col_surf = 'black', nlevels=21)
points(X)

contourview(function(.xx) simulate(k,nsim=1,seed=123, .xx), dim=2, col_surf = 'blue', nlevels=21, add=TRUE)

sectionview3d(f, dim=2, col_surf = 'black', engine3d = "scatterplot3d")
sectionview3d(function(.xx) simulate(k,nsim=1,seed=123, .xx), dim=2, add=TRUE)
sectionview3d(function(.xx) simulate(k,nsim=1,seed=1234, .xx), dim=2, add=TRUE)
sectionview3d(function(.xx) simulate(k,nsim=1,seed=12345, .xx), dim=2, add=TRUE)
sectionview3d(function(.xx) simulate(k,nsim=1,seed=123456, .xx), dim=2, add=TRUE)

# log-Likelihood (https://libkriging.readthedocs.io/en/latest/math/likelihood.html)
print(k$logLikelihood())

contourview( function(t) k$logLikelihoodFun(t)$logLikelihood , dim=2 , Xlab=c(expression(theta[1]),expression(theta[2])), ylab="LL")
abline(v=k$theta()[1],col='blue')
abline(h=k$theta()[2],col='blue')

# leave-one-out (https://libkriging.readthedocs.io/en/latest/math/leaveOneOut.html)
print(k$leaveOneOut())

contourview( function(t) k$leaveOneOutFun(t)$leaveOneOut , dim=2 , Xlab=c(expression(theta[1]),expression(theta[2])), ylab="LOO")
abline(v=k$theta()[1],col='blue')
abline(h=k$theta()[2],col='blue')

# log-marginal-posterior (https://libkriging.readthedocs.io/en/latest/math/Bayesian.html)
print(k$logMargPost())

contourview( function(t) k$logMargPostFun(t)$logMargPost , dim=2 , Xlab=c(expression(theta[1]),expression(theta[2])), ylab="LMP")
abline(v=k$theta()[1],col='blue')
abline(h=k$theta()[2],col='blue')
