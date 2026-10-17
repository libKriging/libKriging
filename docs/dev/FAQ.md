This page collects questions about libKriging development

# Armadillo uses `-fpic` in Rcpp and `-fPIC` in C++ code, is it important ?

 Options `-fpic` and `-fPIC` are about shared libraries, to generate position independent code. 
 Whether to use -fPIC or -fpic to generate position independent code is target-dependent. 
 The -fPIC choice always works, but may produce larger code than -fpic 
 (mnenomic to remember this is that PIC is in a larger case, so it may produce larger amounts of code). 
 Using -fpic option usually generates smaller and faster code, but will have platform-dependent limitations, 
 such as the number of globally visible symbols or the size of the code. 
 The linker will tell you whether it fits when you create the shared library. 
 When in doubt, choose -fPIC, because it always works.
 
 More info: http://tldp.org/HOWTO/Program-Library-HOWTO/shared-libraries.html

 
 
 
# Is a fit reproducible with a fixed seed?

 With a fixed seed (`Random::reset_seed`, 123 by default) the multistart starting points are
 always the same, and two fits on the same machine, with the same BLAS library and the same
 BLAS thread count, give bit-identical results.

 The result may change with the BLAS thread count (`OPENBLAS_NUM_THREADS`, the number of cores),
 the BLAS library or its version, or the CPU (OpenBLAS selects AVX2 / AVX-512 kernels at run
 time): a multithreaded BLAS splits its sums by thread, so the rounding of the objective and of
 its gradient changes, by ~1e-15 relative. The OpenMP thread count (`OMP_NUM_THREADS`) has no
 effect. These rounding differences stay at that level on a well-conditioned covariance matrix,
 but are amplified by its condition number: on noise-free, smooth data the optimum often lies
 at large `theta`, where `cond(R)` reaches 1e14 or more, the objective is then only known to
 ~1e-2 and L-BFGS-B may follow a different path and end at another, equivalent, local optimum.
 E.g. n = 500, d = 4, `y = sum(sin(3 x))`, `"BFGS20"`: log-likelihood 3376.36 with 1 BLAS thread,
 3375.21 with 4.

 To get the same result on several machines, set the same BLAS thread count everywhere (e.g.
 `OPENBLAS_NUM_THREADS=1`, at the cost of speed) and use the same BLAS build; a nugget or noise
 model, or tighter `theta` bounds, also reduce the sensitivity.
