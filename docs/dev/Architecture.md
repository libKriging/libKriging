# libKriging Architecture

libKriging is built on a single C++ kernel implementing the Kriging / Gaussian
process algorithms (fit, prediction, simulation, update), plus input warpings.

Around this kernel, thin bindings expose the library to other languages:
**Python** (`pylibkriging`), **R** (`rlibkriging`), **Octave/Matlab**
(`mlibkriging`) and **Julia** (`jlibkriging`). Each binding is a light mapping
layer that forwards calls to the C++ kernel through the relevant objects; it
performs no intensive computation. All algorithms live in the C++ kernel, so
the numerical behaviour is identical across languages.

See [bindings/README.md](../../bindings/README.md) for the full per-language
method reference.
