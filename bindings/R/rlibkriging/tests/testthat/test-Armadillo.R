manual_demo_binding1()
manual_demo_binding2()

demo_binding()
demo_binding()

set.seed(42)
X <- matrix(rnorm(4*4), 4, 4)
Z <- X %*% t(X)
obj <- buildDemoArmadilloClass("Z",Z)
getEigenValues(obj)

# Remove all internal references
rm(obj)
gc()
