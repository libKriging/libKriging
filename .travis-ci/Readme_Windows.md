
Windows environment for Travis-CI is still in early stages.

Help may be found via the forum : [https://travis-ci.community/c/environments/windows](https://travis-ci.community/c/environments/windows)

Pre-installed packages: [https://docs.travis-ci.com/user/reference/windows/](https://docs.travis-ci.com/user/reference/windows/)

# [Chocolatey](https://chocolatey.org) package manager

Use `choco` command as main package command and is installed by default in Travis-CI windows environment. However `choco` has only few packages available about scientific computing.

To install chocolatey, use following cmd shell command (cf [https://chocolatey.org/docs/installation](https://chocolatey.org/docs/installation))
```
@"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
```

Package list is available here: [https://chocolatey.org/search](https://chocolatey.org/search)

# Anaconda package manager

`conda` is an interesting alternative but not available by default in Travis-CI. See `install.sh` script for more information.

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) is one of its client.

Package list is available here: [https://anaconda.org/search](https://anaconda.org/search).

# Tools for a confortable Windows experience

* use remote [Visual Studio](https://azuremarketplace.microsoft.com/fr-fr/marketplace/apps/category/compute?filters=virtual-machine-images%3Bmicrosoft%3Bwindows&page=1&subcategories=application-infrastructure&search=visual%20studio) in Azure
* don't forget to install common tools
```
choco install cmake
```
* For file edition
Using bash shell:
```     
curl -o npp.7.7.1.Installer.exe https://notepad-plus-plus.org/repository/7.x/7.7.1/npp.7.7.1.Installer.exe
./npp.7.7.1.Installer.exe
``` 

