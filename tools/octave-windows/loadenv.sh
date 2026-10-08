# if [[ "$ENABLE_PYTHON_BINDING" == "on" ]]; then
  # Using choco installer : should be loaded by env in CI workflow
  # export PATH=/c/Python37:/c/Python37/Scripts:$PATH
    
  ## Using miniconda (Python 3.7 is already included in Miniconda3)
  #export PATH=${HOME}/Miniconda3:$PATH
# fi

# Octave MinGW (put first to ensure we use Octave's compiler toolchain)
export PATH=/c/ProgramData/Chocolatey/lib/octave.portable/tools/octave/mingw64/bin:${PATH}

# make
export PATH="/c/Program Files/make/make-4.3/bin":${PATH}

# Belt and braces for OpenBLAS 0.3.34 (OpenMathLib/OpenBLAS#6013/#6021, see
# the 0.3.33 pin in tools/windows/install.sh): its Zen 4 GEMM blocking
# override only fires when L2 reads as 1024 KB, and overflows the stack of
# dgemm_kernel_ZEN/HASWELL on AVX-512 Zen 4/5 hosts. Claiming 2 MB disables it.
export OPENBLAS_L2_SIZE=2048
