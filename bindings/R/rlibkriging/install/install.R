# install.packages doc
# https://www.rdocumentation.org/packages/utils/versions/3.6.2/topics/install.packages

# To install a specified version, you can use package 'devtools' (https://cran.r-project.org/web/packages/devtools) using
#$ install.packages("devtools", repos=repos)
#$ require(devtools)
#$ install_version("ggplot2", version = "0.9.1", repos = "http://cran.us.r-project.org")

# Use RStudio Public Package Manager (RSPM) for faster binary installations
# Check if repos is already configured (e.g., by r-lib/actions/setup-r)
repos <- getOption("repos")
if (is.null(repos) || identical(repos, c(CRAN = "@CRAN@")) || repos["CRAN"] == "@CRAN@") {
  # Configure RSPM based on OS
  repos <- switch(Sys.info()[['sysname']],
    Windows = "https://packagemanager.posit.co/cran/latest",
    Darwin  = "https://packagemanager.posit.co/cran/latest",
    Linux   = {
      # For Linux, use the distribution-specific binary repo if available
      # Default to Ubuntu 22.04 (jammy) which is common in CI
      if (file.exists("/etc/os-release")) {
        os_release <- readLines("/etc/os-release")
        if (any(grepl("Ubuntu", os_release))) {
          version_line <- grep("VERSION_ID", os_release, value = TRUE)
          if (length(version_line) > 0 && grepl("22\\.04", version_line)) {
            "https://packagemanager.posit.co/cran/__linux__/jammy/latest"
          } else if (length(version_line) > 0 && grepl("20\\.04", version_line)) {
            "https://packagemanager.posit.co/cran/__linux__/focal/latest"
          } else {
            "https://packagemanager.posit.co/cran/latest"
          }
        } else {
          "https://packagemanager.posit.co/cran/latest"
        }
      } else {
        "https://packagemanager.posit.co/cran/latest"
      }
    },
    # Fallback to standard CRAN
    "https://cran.r-project.org"
  )
}

type <- switch(Sys.info()[['sysname']],
               Windows= {"binary"},
               Linux  = {"binary"},  # RSPM provides binaries for Linux
               Darwin = {"binary"})

# Print repository configuration for debugging
message("Using repository: ", repos)
message("Package type: ", type)

install.packages("Rcpp", repos=repos, type=type)
install.packages("RcppArmadillo", repos=repos, type=type)
install.packages("RhpcBLASctl", repos=repos, type=type)
install.packages("testthat", repos=repos, type=type)
install.packages("DiceKriging", repos=repos, type=type)
install.packages("foreach", repos=repos, type=type)
install.packages("RobustGaSP", repos=repos, type=type)
install.packages("roxygen2", repos=repos, type=type)
install.packages("pkgbuild", repos=repos, type=type)
