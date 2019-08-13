library(testBinding)
demo_binding1()
demo_binding2()


set.seed(42)
X <- matrix(rnorm(4*4), 4, 4)
Z <- X %*% t(X)
getEigenValues(Z)