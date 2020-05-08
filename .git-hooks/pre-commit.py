#!/usr/bin/env python
"""
Referencing current branch in github readme.md[1]

This pre-commit hook[2] updates the README.md file's
Travis badge with the current branch. Gist at[4].

[1] http://stackoverflow.com/questions/18673694/referencing-current-branch-in-github-readme-md
[2] http://www.git-scm.com/book/en/v2/Customizing-Git-Git-Hooks
[3] https://docs.travis-ci.com/user/status-images/
[4] https://gist.github.com/DrSAR/eb50d9b2b993384267db4ee9fd5e210f
"""
import subprocess
import re

# Hard-Coded for your repo (ToDo: get from remote?)
REPO_URL = subprocess.check_output(['git', 'config', '--local', 'remote.origin.url']).decode()
GITHUB_USER = re.match('.*[:/]([a-zA-Z0-9]*)\/', REPO_URL).groups()[0]
REPO = re.match('.*\/([a-zA-Z0-9]*).git', REPO_URL).groups()[0]

print("Starting pre-commit hook for {0} on {1}...".format(GITHUB_USER, REPO))

BRANCH = subprocess.check_output(["git",
                                  "rev-parse",
                                  "--abbrev-ref",
                                  "HEAD"]).strip()

# String with hard-coded values
# See Embedding Status Images[3] for alternate formats (private repos, svg, etc)

# Output String with Variable substitution
travis_sentinel_str = "[![Build Status]"
travis = "{SENTINEL}(https://travis-ci.com/" \
         "{GITHUB_USER}/{REPO}.svg?" \
         "branch={BRANCH})]" \
         "(https://travis-ci.com/" \
         "{GITHUB_USER}/{REPO})\n".format(SENTINEL=travis_sentinel_str,
                                          BRANCH=BRANCH.decode(),
                                          GITHUB_USER=GITHUB_USER,
                                          REPO=REPO)
coveralls_sentinel_str = "[![Coverage Status]"
coveralls = "{SENTINEL}(https://coveralls.io/repos/github/" \
            "{GITHUB_USER}/{REPO}/badge.svg?" \
            "branch={BRANCH})]" \
            "(https://coveralls.io/github/" \
            "{GITHUB_USER}/{REPO}?branch={BRANCH})\n".format(SENTINEL=coveralls_sentinel_str,
                                                             BRANCH=BRANCH.decode(),
                                                             GITHUB_USER=GITHUB_USER,
                                                             REPO=REPO)

readme_lines = open("README.md").readlines()
with open("README.md", "w") as fh:
    for line in readme_lines:
        if travis_sentinel_str in line and travis != line:
            print("Replacing:\n\t{line}\nwith:\n\t{travis}".format(
                line=line,
                travis=travis))
            fh.write(travis)
        elif coveralls_sentinel_str in line and coveralls != line:
            print("Replacing:\n\t{line}\nwith:\n\t{coveralls}".format(
                line=line,
                coveralls=coveralls))
            fh.write(coveralls)
        else:
            fh.write(line)

subprocess.check_output(["git", "add", "README.md"])

print("pre-commit hook complete.")
