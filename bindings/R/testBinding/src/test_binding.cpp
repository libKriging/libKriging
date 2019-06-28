/***************************************************************************
 R callable function
 ***************************************************************************/

#include <R.h>
#include <Rdefines.h>

// FIXME: collision with Rinternals
//#include <iostream>

#include <libKriging/libKriging.h>

extern "C" SEXP test_binding() {

    //    std::cout << "libKriging class tests" << std::endl;
    printf("libKriging class tests\n");

    std::unique_ptr<TestClass> x(new TestClass());
    int tmp = [&x]() { return x->f(); }();

    SEXP result;

    PROTECT(result = NEW_INTEGER(2));
    INTEGER(result)[0] = (int) 1;
    INTEGER(result)[1] = (int) tmp;
    UNPROTECT(1);

    return (result);
}
