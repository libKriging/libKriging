# wrapper function to invoke test_binding
demo_binding1 <- function() {
  result <- .Call("demo_binding1")
  return(result)
}

demo_binding2 <- function() {
  result <- .Call("demo_binding2")
  return(result)
}

predict.LinearRegression <- function(object,x) {
    return(linear_regression_predict(object, x))
}

# n <- 10
# X <- as.matrix(runif(n))
# y = 4*X+rnorm(n,0,.1)
# r <- linear_regression(y, X)
# plot(X,y)
# x=as.matrix(seq(0,1,,100))
# px = predict(r,x)
# lines(x,px$y)

predict.OrdinaryKriging <- function(object,x) {
  return(ordinary_kriging_predict(object, x))
}

predict <- function (x, ...) {
  UseMethod("predict", x)
}
