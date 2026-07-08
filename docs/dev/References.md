# References

## Scientific references

Gaussian process / Kriging modelling:

* Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for
  Machine Learning*. MIT Press.
* Roustant, O., Ginsbourger, D. and Deville, Y. (2012). DiceKriging,
  DiceOptim: Two R Packages for the Analysis of Computer Experiments by
  Kriging-Based Metamodeling and Optimization. *Journal of Statistical
  Software*, 51(1), 1–55. doi:10.18637/jss.v051.i01

Large-scale approximations (as used by libKriging's Nested and Vecchia paths):

* Rullière, D., Durrande, N., Bachoc, F. and Chevalier, C. (2018). Nested
  Kriging predictions for datasets with a large number of observations.
  *Statistics and Computing*, 28, 849–867. doi:10.1007/s11222-017-9766-2
* Vecchia, A. V. (1988). Estimation and Model Identification for Continuous
  Spatial Processes. *Journal of the Royal Statistical Society: Series B*,
  50(2), 297–312.
* Katzfuss, M. and Guinness, J. (2021). A General Framework for Vecchia
  Approximations of Gaussian Processes. *Statistical Science*, 36(1), 124–141.
  doi:10.1214/19-STS755

Input warpings and non-stationarity (the warpings exposed by libKriging: `boxcox`, `kumaraswamy`, `knots`, `mlp`, `categorical`, `ordinal`):

* Box, G. E. P. and Cox, D. R. (1964). An Analysis of Transformations (`boxcox`). *Journal of the Royal Statistical Society: Series B*, 26(2), 211–252. doi:10.1111/j.2517-6161.1964.tb00553.x
* Kumaraswamy, P. (1980). A generalized probability density function for double-bounded random processes (`kumaraswamy`). *Journal of Hydrology*, 46(1–2), 79–88. doi:10.1016/0022-1694(80)90036-0
* Snoek, J., Swersky, K., Zemel, R. and Adams, R. P. (2014). Input Warping for Bayesian Optimization of Non-Stationary Functions (Beta/Kumaraswamy CDF input warping). *Proceedings of the 31st ICML*, PMLR 32(2), 1674–1682. arXiv:1402.0929
* Snelson, E., Rasmussen, C. E. and Ghahramani, Z. (2004). Warped Gaussian Processes. *Advances in Neural Information Processing Systems 16* (NIPS).
* Sampson, P. D. and Guttorp, P. (1992). Nonparametric Estimation of Nonstationary Spatial Covariance Structure (spatial deformation / warping). *Journal of the American Statistical Association*, 87(417), 108–119. doi:10.1080/01621459.1992.10475181
* Calandra, R., Peters, J., Rasmussen, C. E. and Deisenroth, M. P. (2016). Manifold Gaussian Processes for Regression (neural-network / `mlp` feature mapping). *IEEE IJCNN 2016*, 3338–3345. doi:10.1109/IJCNN.2016.7727626
* Roustant, O., Padonou, E., Deville, Y., et al. (2020). Group kernels for Gaussian process metamodels with categorical inputs (`categorical`). *SIAM/ASA Journal on Uncertainty Quantification*, 8(2), 775–806. doi:10.1137/18M1209386
* Qian, P. Z. G., Wu, H. and Wu, C. F. J. (2008). Gaussian Process Models for Computer Experiments with Qualitative and Quantitative Factors (`ordinal` / qualitative factors). *Technometrics*, 50(3), 383–396. doi:10.1198/004017008000000262

Optimization and linear algebra:

* Byrd, R. H., Lu, P., Nocedal, J. and Zhu, C. (1995). A Limited Memory
  Algorithm for Bound Constrained Optimization. *SIAM Journal on Scientific
  Computing*, 16(5), 1190–1208. doi:10.1137/0916069
* Sanderson, C. and Curtin, R. (2016). Armadillo: a template-based C++ library
  for linear algebra. *Journal of Open Source Software*, 1(2), 26.
  doi:10.21105/joss.00026

## Development tools

* [Armadillo documentation](http://arma.sourceforge.net/docs.html)
* [Catch2 documentation](https://github.com/catchorg/Catch2/blob/v2.x/docs/Readme.md)
* [Doxygen manual](https://www.doxygen.nl/manual/index.html)
