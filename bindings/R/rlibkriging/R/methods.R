predict.LinearRegression <- function(object,x) {
    return(linear_regression_predict(object, x))
}

predict.OrdinaryKriging <- function(object,x) {
  return(ordinary_kriging_predict(object, x))
}

predict <- function (x, ...) {
  UseMethod("predict", x)
}
