library(DiceKriging)
d <- 2; n <- 16
design.fact <- expand.grid(x1=seq(0,1,length=4), x2=seq(0,1,length=4))
Xtest <- as.data.frame(matrix(runif(100*2),100,2))
colnames(Xtest) <- colnames(design.fact)
y <- apply(design.fact, 1, branin)
m1 <- km(formula=~1,design=design.fact, response=y, covtype="gauss")
cc.theta <- m1@covariance@range.val
cc.var <- m1@covariance@sd2
m1 <- km(formula=~1,design=design.fact, response=y, covtype="gauss", coef.cov = cc.theta, coef.var = cc.var)
m1@z
Ytest <- predict(m1,Xtest,type="UK")$mean

# Replicate with RCPP
library(Rcpp)

sourceCpp("test.cpp")
ll <- Krigingfit(as.matrix(y), as.matrix(design.fact), as.matrix(cc.theta),as.matrix(Xtest))


plot(ll$prediction,Ytest)

