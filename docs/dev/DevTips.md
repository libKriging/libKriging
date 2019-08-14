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

