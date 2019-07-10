:: @echo off
:: convenient scrits from https://stackoverflow.com/questions/37464070/install-r-rtools-from-windows-terminal

If NOT exist "C:\Rtools\VERSION.txt"\ (
:: deprecated command replaced by curl
:: bitsadmin  /transfer mydownloadjob  /download  /priority normal  ^
::    https://cran.r-project.org/bin/windows/Rtools/Rtools35.exe %UserProfile%\Downloads\Rtools.exe

:: http://www.jrsoftware.org/ishelp/index.php?topic=setupcmdline
%UserProfile%\Downloads\Rtools.exe /VERYSILENT /SP- /NORESTART /DIR="C:\Rtools"
)