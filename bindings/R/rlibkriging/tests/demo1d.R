#library(rlibkriging, lib.loc="bindings/R/Rlibs")

# 1D function, small design
X <- as.matrix(c(0.0, 0.25, 0.5, 0.75, 1.0))
f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
y <- f(X) # + 10*rnorm(nrow(X))

plot(f)
points(X, y)

# Build Kriging (https://libkriging.readthedocs.io/en/latest/math/KrigingModels.html)
k <- Kriging(y, X, kernel="gauss") #, objective = "LL", optim = "BFGS", regmodel = "constant", parameters = ...)
# kernel: https://libkriging.readthedocs.io/en/latest/math/kernel.html
# regmodel: https://libkriging.readthedocs.io/en/latest/math/trend.html
# parameters: https://libkriging.readthedocs.io/en/latest/math/parameters.html
print(k)

# Save / load /inspect
save(k,"k.json")

# Predict
.x <- as.matrix(seq(0, 1, , 101))
p.x <- predict(k, .x, TRUE, FALSE)

plot(f)
points(X, y)
lines(.x, p.x$mean, col = 'blue')
polygon(c(.x, rev(.x)), c(p.x$mean - 2 * p.x$stdev, rev(p.x$mean + 2 * p.x$stdev)), border = NA, col = rgb(0, 0, 1, 0.2))

# Simulate
s.x <- simulate(k, nsim = 100, seed = 123, x=.x)

plot(f)
points(X,y)
matplot(.x,s.x,col=rgb(0,0,1,0.2),type='l',lty=1,add=T)

# log-Likelihood (https://libkriging.readthedocs.io/en/latest/math/likelihood.html)
print(k$logLikelihood())

plot( function(t) k$logLikelihoodFun(t)$logLikelihood ,xlab=expression(theta),ylab="LL")
abline(v=k$theta(),col='blue')
abline(h=k$logLikelihood(),col='blue')

# leave-one-out (https://libkriging.readthedocs.io/en/latest/math/leaveOneOut.html)
print(k$leaveOneOut())

plot( function(t) k$leaveOneOutFun(t)$leaveOneOut ,xlab=expression(theta),ylab="LOO")
abline(v=k$theta(),col='blue')
abline(h=k$leaveOneOut(),col='blue')

.t = seq(0,1,,101)

matplot(.t, t(sapply(.t, function(t) k$leaveOneOutVec(matrix(t))$mean-k$y()) ),xlab=expression(theta),ylab="LOOv")

# log-marginal-posterior (https://libkriging.readthedocs.io/en/latest/math/Bayesian.html)
print(k$logMargPost())

plot( function(t) k$logMargPostFun(t)$logMargPost ,xlab=expression(theta),ylab="LMP")
abline(v=k$theta(),col='blue')
abline(h=k$logMargPost(),col='blue')
