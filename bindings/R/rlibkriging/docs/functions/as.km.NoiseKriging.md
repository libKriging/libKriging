# `as.km.NoiseKriging`

Coerce a `NoiseKriging` Object into the Class `"km"`


## Description

Coerce a `NoiseKriging` object into the `"km"` class of the
 DiceKriging package.


## Usage

```r
list(list("as.km"), list("NoiseKriging"))(x, .call = NULL, ...)
```


## Arguments

Argument      |Description
------------- |----------------
`x`     |     An object with S3 class `"NoiseKriging"` .
`.call`     |     Force the `call` slot to be filled in the returned `km` object.
`...`     |     Not used.


## Value

An object of having the S4 class `"KM"` which extends
 the `"km"` class of the DiceKriging package and
 contains an extra `NoiseKriging` slot.


## Author

Yann Richet yann.richet@irsn.fr


## Examples

```r
f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
set.seed(123)
X <- as.matrix(runif(5))
y <- f(X) + 0.01*rnorm(nrow(X))
r <- NoiseKriging(y, rep(0.01^2,nrow(X)), X, "gauss")
print(r)
k <- as.km(r)
print(k)
```


