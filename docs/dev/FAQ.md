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

 
 
 