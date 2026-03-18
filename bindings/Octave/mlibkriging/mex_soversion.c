/* This file defines the SOVERSION symbol required by Octave 10+ */
/* It is included in the build of MEX files to ensure compatibility */

/* OCTAVE_MEX_SOVERSION should be defined at compile time via -D flag */
#ifndef OCTAVE_MEX_SOVERSION
#error "OCTAVE_MEX_SOVERSION must be defined"
#endif

const int __octave_mex_soversion__ = OCTAVE_MEX_SOVERSION;
