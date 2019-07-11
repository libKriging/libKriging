library(testBinding)
test_binding1()
test_binding2()


 set.seed(42)
X <- matrix(rnorm(4*4), 4, 4)
Z <- X %*% t(X)
getEigenValues(Z)