# `as.km.Kriging`

Coerce a `Kriging` Object into the Class `"km"`


## Description

Coerce a `Kriging` object into the `"km"` class of the
 DiceKriging package.


## Usage

```r
list(list("as.km"), list("Kriging"))(x, .call = NULL, ...)
```


## Arguments

Argument      |Description
------------- |----------------
`x`     |     An object with S3 class `"Kriging"` .
`.call`     |     Force the `call` slot to be filled in the returned `km` object.
`...`     |     Not used.


## Value

An object of having the S4 class `"KM"` which extends
 the `"km"` class of the DiceKriging package and
 contains an extra `Kriging` slot.


## Author

Yann Richet yann.richet@irsn.fr


## Examples

```r
f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
set.seed(123)
X <- as.matrix(runif(5))
y <- f(X)
r <- Kriging(y, X, "gauss")
print(r)
k <- as.km(r)
print(k)
```


