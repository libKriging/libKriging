# Copilot Instructions

## Allowed tools

The following tools are pre-approved and may be used freely without asking for user authorization:

- **Build/compile:** `cmake`, `ctest`, `ninja`, `make`, `gcc`, `g++`, `cc`, `c++`
- **Python:** `python`, `python3`, `pip`, `pip3`
- **R:** `R`, `Rscript`
- **Shell:** `cd`, `export`, `LD_LIBRARY_PATH`
- **File inspection:** `ls`, `find`, `cat`, `head`, `tail`, `grep`, `rg`
- **Text processing:** `sed`, `awk`, `echo`, `printf`
- **File operations:** `mkdir`, `cp`, `mv`, `rm`, `touch`, `chmod`
- **Version control:** `git` (except `git push` — see below), `gh`

## Git usage

Never run `git push` or any variant that pushes commits to a remote (e.g. `git push --force`, `git push origin`). If a task requires pushing, stop and ask the user to do it manually.
