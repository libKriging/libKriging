if ( command -v gfortran >/dev/null 2>&1 ); then
    # echo "gfortran found in PATH"
    : # do nothing
elif ( command -v R >/dev/null 2>&1 ); then
    # echo "R will get its fortran compiler"
    Ftmp=$(R CMD config FC)
    export PATH=$PATH:$(cd $(dirname -- $Ftmp) && pwd -P)
elif [ "$(uname -s)" == "Darwin" ]; then
    # echo "Ask to brew package system
    export PATH=$(brew --prefix gfortran)/bin:$PATH
fi
