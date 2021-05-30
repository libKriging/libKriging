# macOS minimal setup

## Package Manager

We will use [brew](https://brew.sh) as package manager 

Install it using:
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## Tools

### Compiler

Ensure C++ compiler is available. 

On macOS systems you will need to install Xcode
with additional Command Line Tools using:  
```
xcode-select --install
sudo xcodebuild -license accept
```
(should be done after each macOS upgrade)

### R

https://cran.r-project.org/bin/macosx/

### Additional tools
(*this list could be incomplete: feedbacks are welcome*)
```
brew install cmake octave python@3.9 lapack openblas
```