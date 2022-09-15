# `as.km.NuggetKriging`

Coerce a `NuggetKriging` Object into the Class `"km"`


## Description

Coerce a `NuggetKriging` object into the `"km"` class of the
 DiceKriging package.


## Usage

```r
list(list("as.km"), list("NuggetKriging"))(x, .call = NULL, ...)
```


## Arguments

Argument      |Description
------------- |----------------
`x`     |     An object with S3 class `"NuggetKriging"` .
`.call`     |     Force the `call` slot to be filled in the returned `km` object.
`...`     |     Not used.


## Value

An object of having the S4 class `"KM"` which extends
 the `"km"` class of the DiceKriging package and
 contains an extra `NuggetKriging` slot.


## Author

Yann Richet yann.richet@irsn.fr


## Examples

```r
f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
set.seed(123)
X <- as.matrix(runif(5))
y <- f(X) + 0.01*rnorm(nrow(X))
r <- NuggetKriging(y, X, "gauss")
print(r)
k <- as.km(r)
print(k)
```


