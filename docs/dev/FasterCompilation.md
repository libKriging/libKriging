In order to speed-up compilations, you can use *compiler cache*.

There are various compiler caches :
* `ccache` (https://ccache.dev) : the most common in Unix world
* `sccache` (https://github.com/mozilla/sccache) : a new *challenger* written in Rust and with the support of MSVC
* `buildcache` (https://github.com/mbitsnbites/buildcache) : 
* `fastbuild` (https://fastbuild.org)

There are two ways to enable a compiler cache:
* `cmake -DUSE_COMPILER_CACHE=<your-compiler-cache> <CMAKE_OPTIONS>`

    Example with `ccache`: `cmake -DUSE_COMPILER_CACHE=ccache ..`
    * The simplest config
    * Portable even in Windows world
    * No additional option
    
* `CXX="<your-compiler-cache> <your-cxx-compiler>" CC="<your-compiler-cache> <your-c-compiler>" cmake <CMAKE_OPTIONS>`

    Example with `ccache` and GCC: `CXX="ccache g++" CC="ccache gcc" cmake ..`    
    * You have to specify it as a new compiler
    * You can set additional options in command line  


To speed-up R compilation, you can override R default compiler.

The following script generates a config using `ccache`
```
#!/usr/bin/env bash
set -eo pipefail

echo "# Custom ~/.R/Makevars configuration"
CCACHE=ccache
for comp in CC CXX CXX11 CXX14 CXX17 FC; do
  echo "$comp=${CCACHE} $(R CMD config $comp)"
done
```
