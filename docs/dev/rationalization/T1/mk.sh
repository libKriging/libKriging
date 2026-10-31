#!/bin/bash
# mk.sh <src-root> <build-dir> <out>
SRC=$1; B=$2; OUT=$3
c++ -DARMA_32BIT_WORD -DARMA_ALIEN_MEM_ALLOC_FUNCTION=lkalloc::malloc -DARMA_ALIEN_MEM_FREE_FUNCTION=lkalloc::free -DARMA_DONT_USE_OPENMP -DCARMA_DO_NOT_EXPORT_ALIEN_MEM_FUNCTIONS \
  -I$SRC/src/lib/include -I$B/src/lib -I$B/dependencies/armadillo-code/tmp/include -I$SRC/dependencies/lbfgsb_cpp/include -isystem $SRC/dependencies/armadillo-code/include \
  -O3 -DNDEBUG -std=gnu++17 -fopenmp $(dirname $0)/gradcheck.cpp -o $OUT -Wl,-rpath,$B/src/lib:$B/dependencies/armadillo-code $B/src/lib/libKriging.so.1.2.2 $B/dependencies/armadillo-code/libarmadillo.so.15.0.1
