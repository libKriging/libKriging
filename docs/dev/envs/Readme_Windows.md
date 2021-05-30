# Windows minimal setup

There are many ways to provide the required tools in Windows environment.

The way explained here is driven by the ability to be scriptable (as it is done in CI pipelines).

First of all, we need [Git Bash](https://gitforwindows.org) to be able to run Shell script on Windows (we do not recommend to use Cygwin).

## Package Managers

We use two different package managers: [Chocolatey](https://chocolatey.org) and [Anaconda](https://docs.conda.io/en/latest/miniconda.html).

### [Chocolatey](https://chocolatey.org) package manager

`choco` is the default package in Travis-CI and in GitHub Actions.

However `choco` has only few packages available about scientific computing.

Package list is available here: [https://chocolatey.org/search](https://chocolatey.org/search)

NB: On Windows, `choco` package manager requires admin rights, so that it could be simpler to do it by hand.

### Anaconda package manager

`conda` is an interesting alternative but not available by default in Travis-CI.

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) is one of its client.

Package list is available here: [https://anaconda.org/search](https://anaconda.org/search).

To install it, you can use [`install.sh`](../../../.travis-ci/windows/install.sh) script for more information.

This script will also install all the other required dependencies.

## Compiler

We recommend to use [Visual Studio Community](https://visualstudio.microsoft.com/fr/vs/community/)

Nevertheless, if you want to build libKriging for R, we recommend using a complete R environment (R + Rtools) and to use the associated compiler (See R toolchain).

# Development for Windows without native Windows

To do so, you can use:
### Remote Windows in the Cloud

Remote Windows server with Visual Studio are available in [Azure](https://azuremarketplace.microsoft.com/fr-fr/marketplace/apps/category/compute?filters=virtual-machine-images%3Bmicrosoft%3Bwindows&page=1&subcategories=application-infrastructure&search=visual%20studio) 

### Virtual Machine

Microsoft provides an [official Virtual Machine Image with Visual Studio](https://developer.microsoft.com/fr-fr/windows/downloads/virtual-machines/)

NB: more info about VirtualBox installation:
  * https://www.extremetech.com/computing/198427-how-to-install-windows-10-in-a-virtual-machine
  * https://www.howtogeek.com/657464/how-to-install-a-windows-10-virtualbox-vm-on-macos/
  
  IMPORTANT: use the previously downloaded disk image as initial disk (do not use the Windows installation procedure)

Don't forget to configure a port redirection from host:2022 to VM:22 if you want to use SSHD.

#### SSHD
* We recommend to also install sshd server to be able to run a shell from the host (easy and fast interaction using command line)

  > *Apps > Optional features, clicking Add a feature, selecting OpenSSH Server, and clicking Install* ([ref](https://virtualizationreview.com/articles/2020/05/21/ssh-server-on-windows-10.aspx))

* Next command to configure `sshd` should be done using Administrative mode of PowerShell ([ref](https://superuser.com/questions/1584086/cant-start-the-openssh-sshd-service-via-powershell-start-service-sshd))

* Set Startup type for sshd ([ref](https://medium.com/dev-genius/set-up-your-ssh-server-in-windows-10-native-way-1aab9021c3a6) and [ref](https://medium.com/dev-genius/set-up-your-ssh-server-in-windows-10-native-way-1aab9021c3a6))
  ```
  Set-Service -Name sshd -StartupType Automatic
  ```

* Set bash as default shell (after installation of Git Bash; run in Powershell as administrator)
  ```
  New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Program Files\Git\bin\bash.exe" -PropertyType String -Force
  ```
  or (without git bash)
  ```
  New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Windows\System32\bash.exe" -PropertyType String -Force
  ```

* Edit sshd config to allow Pubkey login

  `sshd` configuration is located in `C:\PROGRAMDATA\ssh\sshd_config`

  ```
  PubkeyAuthentication yes
  PasswordAuthentication no
  PermitEmptyPasswords no
  
  # disable this
  #Match Group administrators
  #  AuthorizedKeysFile
  #  __PROGRAMDATA__/ssh/administrators_authorized_keys
  ```

* From host, copy your public key:
  ```
  cd ~/.ssh && ssh-copy-id -p 2022 -i id_ed25519 user@localhost
  ```

* Update Visual Studio Community to support C++/CMake (not by default)

## Install common tools
* To install chocolatey, use following __cmd__ shell command (cf [https://chocolatey.org/docs/installation](https://chocolatey.org/docs/installation))
  ```
  REM cmd script (once per installation)
  @"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
  ```
* Install tools 
  ```
  # bash script (once per installation)
  choco install -y cmake
  ```
* Get libKriging repository
  ```
  git --recurse-submodules clone https://github.com/libKriging/libKriging.git
  cd libKriging
  ```
* Install project dependencies  
  ```
  # bash script from libKriging repository clone (once per installation)
  .travis-ci/windows/install.sh
  ```
* Load tools path
  ```
  # bash script (once per shell)
  export PATH='C:\Program Files\CMake\bin':$PATH
  ```
* Build libKriging
  ```
  .travis-ci/windows/build.sh
  ```

  For R development, use scripts in `.travis-ci/r-windows` instead of `.travis-ci/windows`.
  
  To get paths to load to launch command by hand, use:
  ```
  export DEBUG_CI=true
  ```

## Convenient tools

* Change password (not mandatory)
  ```
  net user <username> <new-password>
  ```

* Get current IP
  ```
  ipconfig
  ```
  
* For file edition
Using bash shell:
  ```
  # bash script
  # Notepad++
  curl -o ${HOME}/Downloads/npp.7.7.1.Installer.exe https://notepad-plus-plus.org/repository/7.x/7.7.1/npp.7.7.1.Installer.exe
  ${HOME}/Downloads/npp.7.7.1.Installer.exe
  ```  
  
* For dependency analysis
  * dependency walker
    ```
    curl -Lo depends22_x64.zip http://www.dependencywalker.com/depends22_x64.zip
    # command line option: https://dependencywalker.com/help/html/hidr_command_line_help.htm
    # ex:  ./depends.exe -f:1 -c -ot:output.txt file.exe
    ```
    
  * Dependencies (a modern replacement for dependency walker)
    ```
    curl -LO https://github.com/lucasg/Dependencies/releases/download/v1.10/Dependencies_x64_Release.zip
    # https://github.com/lucasg/Dependencies
    # https://stackoverflow.com/questions/49196859/how-to-output-dependency-walker-to-the-console
    ```
  


