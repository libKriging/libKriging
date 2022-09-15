# `predict,NoiseKM-method`

Prediction Method for a `NoiseKM` Object


## Description

Compute predictions for the response at new given input
 points. These conditional mean, the conditional standard deviation
 and confidence limits at the 95% level. Optionnally the
 conditional covariance can be returned as well.


## Usage

```r
list(list("predict"), list("NoiseKM"))(
  object,
  newdata,
  type = "UK",
  se.compute = TRUE,
  cov.compute = FALSE,
  light.return = TRUE,
  bias.correct = FALSE,
  checkNames = FALSE,
  ...
)
```


## Arguments

Argument      |Description
------------- |----------------
`object`     |     `NoiseKM` object.
`newdata`     |     Matrix of "new" input points where to perform prediction.
`type`     |     character giving the kriging type. For now only `"UK"` is possible.
`se.compute`     |     Logical. Should the standard error be computed?
`cov.compute`     |     Logical. Should the covariance matrix between newdata points be computed?
`light.return`     |     Logical. If `TRUE` , no auxiliary results will be returned (such as the Cholesky root of the correlation matrix).
`bias.correct`     |     Logical. If `TRUE` the UK variance and covariance are .
`checkNames`     |     Logical to check the consistency of the column names between the design stored in `object@X` and the new one given `newdata` .
`...`     |     Ignored.


## Details

Without a dedicated `predict` method for the class
 `"NoiseKM"` , this method would have been inherited from the
 `"km"` class. The dedicated method is expected to run faster.
 A comparison can be made by coercing a `NoiseKM` object to a
 `km` object with [`as.km`](#as.km) before calling
 `predict` .


## Value

A named list. The elements are the conditional mean and
 standard deviation ( `mean` and `sd` ), the predicted
 trend ( `trend` ) and the confidence limits ( `lower95` 
 and `upper95` ). Optionnally, the conditional covariance matrix
 is returned in `cov` .


## Author

Yann Richet yann.richet@irsn.fr


## Examples

```r
## a 16-points factorial design, and the corresponding response
d <- 2; n <- 16
design.fact <- expand.grid(x1 = seq(0, 1, length = 4), x2 = seq(0, 1, length = 4))
y <- apply(design.fact, 1, DiceKriging::branin) + rnorm(nrow(design.fact))

## library(DiceKriging)
## kriging model 1 : matern5_2 covariance structure, no trend, no nugget
## m1 <- km(design = design.fact, response = y, covtype = "gauss",
##          noise.var=rep(1,nrow(design.fact)),
##          parinit = c(.5, 1), control = list(trace = FALSE))
KM1 <- NoiseKM(design = design.fact, response = y, covtype = "gauss",
noise=rep(1,nrow(design.fact)),
parinit = c(.5, 1))
Pred <- predict(KM1, newdata = matrix(.5,ncol = 2), type = "UK",
checkNames = FALSE, light.return = TRUE)
```


