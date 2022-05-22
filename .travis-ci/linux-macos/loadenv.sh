if [ "$(uname -s)" == "Darwin" ]; then
  export PATH=$(brew --prefix gfortran)/bin:$PATH
fi