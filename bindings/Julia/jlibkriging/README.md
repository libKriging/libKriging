# jlibkriging

Julia binding for [libKriging](https://github.com/libKriging/libKriging), a fast, portable Kriging library written in C++.

This directory is the development source of the binding. The installable, registrable package is
[JLibKriging.jl](https://github.com/libKriging/JLibKriging.jl), which packages it. See
[`bindings/Julia/README.md`](../README.md) for prerequisites, build instructions
(requires building libKriging with `-DENABLE_JULIA_BINDING=on`), and usage examples.

## License

Apache-2.0, see [`LICENSE`](../../../LICENSE) at the repository root.
