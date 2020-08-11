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
    * Yo have to specify it as a new compiler
    * You can set additional options in command line  
