predict.LinearRegression <- function(object,x) {
    return(linear_regression_predict(object, x))
}

predict.Kriging <- function(object,x) {
  return(kriging_predict(object, x))
}

predict <- function (x, ...) {
  UseMethod("predict", x)
}
