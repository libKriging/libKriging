# install.packages doc
# https://www.rdocumentation.org/packages/utils/versions/3.6.2/topics/install.packages

# To install a specified version, you can use package 'devtools' (https://cran.r-project.org/web/packages/devtools) using
#$ install.packages("devtools", repos=repos)
#$ require(devtools)
#$ install_version("ggplot2", version = "0.9.1", repos = "http://cran.us.r-project.org")

# repos <- 'http://cran.us.r-project.org'
repos <- 'http://cran.irsn.fr'

type <- switch(Sys.info()[['sysname']],
               Windows= {"binary"},
               Linux  = {"source"},
               Darwin = {"binary"})

install.packages("Rcpp", repos=repos, type=type)
install.packages("RcppArmadillo", repos=repos, type=type)
install.packages("RhpcBLASctl", repos=repos, type=type)
install.packages("testthat", repos=repos, type=type)
install.packages("DiceKriging", repos=repos, type=type)
install.packages("RobustGaSP", repos=repos, type=type)
install.packages("roxygen2", repos=repos, type=type)

