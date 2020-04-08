//
// Created by Pascal Havé on 08/04/2020.
//

#ifndef LIBKRIGING_BINDINGS_OCTAVE_TOYS_FUNC_NEW_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_TOYS_FUNC_NEW_HPP

#include <mex.h>

void func_new(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
void func_delete(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);

#endif  // LIBKRIGING_BINDINGS_OCTAVE_TOYS_FUNC_NEW_HPP
