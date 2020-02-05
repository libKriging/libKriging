# wrapper function to invoke test_binding
demo_binding1 <- function() {
  result <- .Call("demo_binding1")
  return(result)
}

demo_binding2 <- function() {
  result <- .Call("demo_binding2")
  return(result)
}

predict.LinearRegression <- function(object,X)
    return(linear_regression_predict(object, X))