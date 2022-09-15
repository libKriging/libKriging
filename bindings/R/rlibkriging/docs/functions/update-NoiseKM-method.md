# `update,NoiseKM-method`

Update a `NoiseKM` Object with New Points


## Description

The `update` method is used when new observations are added
 to a fitted kriging model. Rather than fitting the model from
 scratch with the updated observations added, the results of the
 fit as stored in `object` are used to achieve some savings.


## Usage

```r
list(list("update"), list("NoiseKM"))(
  object,
  newX,
  newy,
  newnoise.var,
  newX.alreadyExist = FALSE,
  cov.reestim = TRUE,
  trend.reestim = cov.reestim,
  nugget.reestim = FALSE,
  kmcontrol = NULL,
  newF = NULL,
  ...
)
```


## Arguments

Argument      |Description
------------- |----------------
`object`     |     A NoiseKM object.
`newX`     |     A numeric matrix containing the new design points. It must have `object@d` columns in correspondence with those of the design matrix used to fit the model which is stored as `object@X` .
`newy`     |     A numeric vector of new response values, in correspondence with the rows of `newX` .
`newnoise.var`     |     Variance of an additional noise on the new response.
`newX.alreadyExist`     |     Logical. If TRUE, `newX` can contain some input points that are already in `object@X` .
`cov.reestim`     |     Logical. If `TRUE` , the vector `theta` of correlation ranges will be re-estimated using the new observations as well as the observations already used when fitting `object` . Only `TRUE` can be used for now.
`trend.reestim`     |     Logical. If `TRUE` the vector `beta` of trend coefficients will be re-estimated using all the observations. Only `TRUE` can be used for now.
`nugget.reestim`     |     Logical. If `TRUE` the nugget effect will be re-estimated using all the observations. Only `FALSE` can be used for now.
`kmcontrol`     |     A list of options to tune the fit. Not available yet.
`newF`     |     New trend matrix. XXXY?
`...`     |     Ignored.


## Details

Without a dedicated `update` method for the class
 `"NoiseKM"` , this would have been inherited from the class
 `"km"` . The dedicated method is expected to run faster.  A
 comparison can be made by coercing a `NoiseKM` object to a
 `km` object with [`as.km`](#as.km) before calling
 `update` .


## Value

The updated `NoiseKM` object.


## Seealso

[`as.km`](#as.km) to coerce a `NoiseKM` object to the
 class `"km"` .


## Author

Yann Richet yann.richet@irsn.fr


## Examples

```r
f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
plot(f)
set.seed(123)
X <- as.matrix(runif(5))
y <- f(X) + 0.01*rnorm(nrow(X))
points(X, y, col = "blue")
KMobj <- NoiseKM(design = X, response = y, noise=rep(0.01^2,5), covtype = "gauss")
x <-  seq(from = 0, to = 1, length.out = 101)
p_x <- predict(KMobj, x)
lines(x, p_x$mean, col = "blue")
lines(x, p_x$lower95, col = "blue")
lines(x, p_x$upper95, col = "blue")
newX <- as.matrix(runif(3))
newy <- f(newX) + 0.01*rnorm(nrow(newX))
points(newX, newy, col = "red")

## replace the object by its udated version
KMobj <- update(KMobj, newy, rep(0.01^2,3), newX)

x <- seq(from = 0, to = 1, length.out = 101)
p2_x <- predict(KMobj, x)
lines(x, p2_x$mean, col = "red")
lines(x, p2_x$lower95, col = "red")
lines(x, p2_x$upper95, col = "red")
```


