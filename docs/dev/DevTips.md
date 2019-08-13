Developer tips.

# Common errors
 
## `Exit code 0xc0000135` at Windows runtime

There's missing a dynamic libray (.dll) while the program is loaded: check dependencies using [Dependenc Walker](http://www.dependencywalker.com).

# Before changing compiler check ABI compatibility

* GCC:
  [https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html](https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html)
* breaking changes every C++ developer should know: 
  [https://www.acodersjourney.com/20-abi-breaking-changes/](https://www.acodersjourney.com/20-abi-breaking-changes/)
* Clang <-> MSVC:
  [https://clang.llvm.org/docs/MSVCCompatibility.html](https://clang.llvm.org/docs/MSVCCompatibility.html)
