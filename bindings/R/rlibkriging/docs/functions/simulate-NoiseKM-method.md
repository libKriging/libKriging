# `simulate,NoiseKM-method`

Simulation from a `NoiseKM` Object


## Description

The `simulate` method is used to simulate paths from the
 kriging model described in `object` .


## Usage

```r
list(list("simulate"), list("NoiseKM"))(
  object,
  nsim = 1,
  seed = NULL,
  newdata,
  cond = TRUE,
  nugget.sim = 0,
  checkNames = FALSE,
  ...
)
```


## Arguments

Argument      |Description
------------- |----------------
`object`     |     A `NoiseKM` object.
`nsim`     |     Integer: number of response vectors to simulate.
`seed`     |     Random seed.
`newdata`     |     Numeric matrix with it rows giving the points where the simulation is to be performed.
`cond`     |     Logical telling wether the simulation is conditional or not. Only `TRUE` is accepted for now.
`nugget.sim`     |     Numeric. A postive nugget effect used to avoid numerical instability.
`checkNames`     |     Check consistency between the design data `X` within `object` and `newdata` . The default is `FALSE` . XXXY Not used!!!
`...`     |     Ignored.


## Details

Without a dedicated `simulate` method for the class
 `"NoiseKM"` , this method would have been inherited from the
 `"km"` class. The dedicated method is expected to run faster.
 A comparison can be made by coercing a `NoiseKM` object to a
 `km` object with [`as.km`](#as.km) before calling
 `simulate` .


## Value

A numeric matrix with `nrow(newdata)` rows and
  `nsim` columns containing as its columns the simulated
 paths at the input points given in `newdata` .
 
 XXX method simulate NoiseKM


## Author

Yann Richet yann.richet@irsn.fr


## Examples

```r
f <-  function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
plot(f)
set.seed(123)
X <- as.matrix(runif(5))
y <- f(X) + 0.01*rnorm(nrow(X))
points(X, y, col = 'blue')
k <- NoiseKM(design = X, response = y, covtype = "gauss", noise=rep(0.01^2,nrow(X)))
x <- seq(from = 0, to = 1, length.out = 101)
s_x <- simulate(k, nsim = 3, newdata = x)
lines(x, s_x[ , 1], col = 'blue')
lines(x, s_x[ , 2], col = 'blue')
lines(x, s_x[ , 3], col = 'blue')
```


