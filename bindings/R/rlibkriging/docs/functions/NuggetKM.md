# `NuggetKM`

Create an `NuggetKM` Object


## Description

Create an object of S4 class `"NuggetKM"` similar to a
 `km` object in the DiceKriging package.


## Usage

```r
NuggetKM(
  formula = ~1,
  design,
  response,
  covtype = c("matern5_2", "gauss", "matern3_2", "exp"),
  coef.trend = NULL,
  coef.cov = NULL,
  coef.var = NULL,
  nugget = NULL,
  nugget.estim = TRUE,
  noise.var = NULL,
  estim.method = c("MLE", "LOO"),
  penalty = NULL,
  optim.method = "BFGS",
  lower = NULL,
  upper = NULL,
  parinit = NULL,
  multistart = 1,
  control = NULL,
  gr = TRUE,
  iso = FALSE,
  scaling = FALSE,
  knots = NULL,
  kernel = NULL,
  ...
)
```


## Arguments

Argument      |Description
------------- |----------------
`formula`     |     R formula object to setup the linear trend in Universal NuggetKriging. Supports `~ 1` , ~. and `~ .^2` .
`design`     |     Data frame. The design of experiments.
`response`     |     Vector of output values.
`covtype`     |     Covariance structure. For now all the kernels are tensor product kernels.
`coef.trend`     |     Optional value for a fixed vector of trend coefficients.  If given, no optimization is done.
`coef.cov`     |     Optional value for a fixed correlation range value. If given, no optimization is done.
`coef.var`     |     Optional value for a fixed variance. If given, no optimization is done.
`nugget.estim, nugget`     |     Should nugget be estimated? (defaults TRUE) or given values.
`noise.var`     |     Not implemented.
`estim.method`     |     Estimation criterion. `"MLE"` for Maximum-Likelihood or `"LOO"` for Leave-One-Out cross-validation.
`penalty`     |     Not implemented yet.
`optim.method`     |     Optimization algorithm used in the optimization of the objective given in `estim.method` . Supports `"BFGS"` .
`lower, upper`     |     Not implemented yet.
`parinit`     |     Initial values for the correlation ranges which will be optimized using `optim.method` .
`multistart, control, gr, iso`     |     Not implemented yet.
`scaling, knots, kernel, `     |     Not implemented yet.
`...`     |     Ignored.


## Details

The class `"NuggetKM"` extends the `"km"` class of the
 DiceKriging package, hence has all slots of `"km"` . It
 also has an extra slot `"NuggetKriging"` slot which contains a copy
 of the original object.


## Value

A NuggetKM object. See Details .


## Seealso

[`km`](#km) in the DiceKriging 
 package for more details on the slots.


## Author

Yann Richet yann.richet@irsn.fr


## Examples

```r
# a 16-points factorial design, and the corresponding response
d <- 2; n <- 16
design.fact <- as.matrix(expand.grid(x1 = seq(0, 1, length = 4),
x2 = seq(0, 1, length = 4)))
y <- apply(design.fact, 1, DiceKriging::branin) + rnorm(nrow(design.fact))

# Using `km` from DiceKriging and a similar `NuggetKM` object
# kriging model 1 : matern5_2 covariance structure, no trend, no nugget effect
km1 <- DiceKriging::km(design = design.fact, response = y, covtype = "gauss",
nugget.estim=TRUE,
parinit = c(.5, 1), control = list(trace = FALSE))
KM1 <- NuggetKM(design = design.fact, response = y, covtype = "gauss",
parinit = c(.5, 1))
```


