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

ROOT_DIR=$(git rev-parse --show-toplevel)
case $(uname -s) in
  Linux|Darwin)
    BIN_VENV=${ROOT_DIR}/venv/bin
    ;;
  MSYS_NT*|MINGW64_NT*)
    BIN_VENV=${ROOT_DIR}/venv/Scripts
    ;;
  *)
    echo "Unknown OS [$ARCH]"
    exit 1
    ;;
esac
if [[ -f "${BIN_VENV}"/activate ]]; then
  echo "Loading virtual environment from ${BIN_VENV}"
  . "${BIN_VENV}"/activate
fi