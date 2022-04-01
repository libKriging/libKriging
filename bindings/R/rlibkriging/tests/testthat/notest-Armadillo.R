#demo_binding1()
#demo_binding2()

set.seed(42)
X <- matrix(rnorm(4*4), 4, 4)
Z <- X %*% t(X)
obj <- buildDemoArmadilloClass("Z",Z)
getEigenValues(obj)

# Remove all internal references
rm(obj)
gc()
