Developer tips.

# Common errors
 
## `Exit code 0xc0000135` at Windows runtime

There's missing a dynamic libray (.dll) while the program is loaded: check dependencies using [Dependenc Walker](http://www.dependencywalker.com).

## `fatal error: 'math.h' file not found` on macOS

More precisely:
```
$ clang++ test.cc
In file included from test.cc:1:
In file included from /usr/local/opt/llvm/bin/../include/c++/v1/cmath:305:
/usr/local/opt/llvm/bin/../include/c++/v1/math.h:301:15: fatal error: 'math.h' file not found
#include_next <math.h>
              ^~~~~~~~
1 error generated.
```

Usually, this should appear after major macOS upgrade. 

Diagnotics:
* It should be missing Xcode's Command Line Tools. To install it, you can use:  
```shell
xcode-select --install
# to enable tools, you have to accept the associated license
sudo xcodebuild -license accept
```

* It should be brew outdated local compiler. Then, you can upgrade it using:
```shell
$ brew upgrade llvm
````

(output example)
```
Updating Homebrew...
==> Upgrading 1 outdated package:
llvm 8.0.0_1 -> 9.0.0_1
==> Upgrading llvm
==> Installing dependencies for llvm: swig
==> Installing llvm dependency: swig
==> Downloading https://homebrew.bintray.com/bottles/swig-4.0.1.catalina.bottle.tar.gz
==> Downloading from https://akamai.bintray.com/d4/d44eb5e2ae81970d131618c4ddd508d3d798e9465979a125454db75c3b9125e1?__gda__=exp=1574870202~
######################################################################## 100.0%
==> Pouring swig-4.0.1.catalina.bottle.tar.gz
🍺  /usr/local/Cellar/swig/4.0.1: 723 files, 5.4MB
==> Installing llvm
==> Downloading https://homebrew.bintray.com/bottles/llvm-9.0.0_1.catalina.bottle.tar.gz
==> Downloading from https://akamai.bintray.com/a8/a8e2475a1fc5a81f0da83a73d17fd54cc2a686f7b5d8e7ace9ea18885971415f?__gda__=exp=1574870210~
######################################################################## 100.0%
==> Pouring llvm-9.0.0_1.catalina.bottle.tar.gz
```

## `CMake 3.13 or higher is required.`

Octave binding requires a more recent version of CMake than the other sections.

Specially for Ubuntu 18, the default CMake version is 3.10.2 and you need to upgrade it:
```
apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common # common package and security tools
curl -s https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null # get security keys
apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' # add kitware repo
apt-get install -y cmake # install latest version of CMake
```

(this procedure is also done in `linux-macos/install.sh` script)

# Before changing compiler check ABI compatibility

* GCC:
  [https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html](https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html)
* breaking changes every C++ developer should know: 
  [https://www.acodersjourney.com/20-abi-breaking-changes/](https://www.acodersjourney.com/20-abi-breaking-changes/)
* Clang <-> MSVC:
  [https://clang.llvm.org/docs/MSVCCompatibility.html](https://clang.llvm.org/docs/MSVCCompatibility.html)

## Memory allocation on passing Armadillo between C++ and R

On Rcpp side, `sizeof(arma::mat)` is 160 bytes. If on C++ side `sizeof(arma::mat)` is 176 bytes, this would be that integer encoding is not the same.
Use `ARMA_32BIT_WORD` directive also in C++. This is now the default option also in libKriging in all bindings.  

# To update subrepo branch of armadillo

You can update C++ tracked version using: 
```shell
git subrepo clone --force --branch 9.600.x https://gitlab.com/conradsnicta/armadillo-code.git dependencies/armadillo-code
```

# To run debugger with R

```shell
$ R -d lldb
run
# ctrl-c to come back to lldb
# and 'continue' to resume to R 
```

More info: [GDB to LLDB command map](https://lldb.llvm.org/use/map.html)

If you have to debug field offset in object, you can use:
```
print (int)&((class YourClass*)0)->your_field
```
More info about a GDB script: https://stackoverflow.com/questions/9788679/how-to-get-the-relative-address-of-a-field-in-a-structure-dump-c?answertab=votes#tab-top

To binary dump objects, you can use:
```
define xxd
dump binary memory dump.bin $arg0 $arg0+$arg1
shell xxd dump.bin
end
```
(tested in GDB; will create a `dump.bin` file) 

