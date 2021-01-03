#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

if [ "${GITHUB_ACTIONS:=false}" == "true" ]; then
  # see https://github.com/rlespinasse/github-slug-action
  if [ "${GITHUB_EVENT_NAME:=push}" != "pull-request" ]; then
    # Not in a pull request, so compare against parent commit
    # FIXME: need to use first commit of this push
    # see https://stackoverflow.com/questions/61860732/how-can-i-get-the-previous-commit-before-a-push-or-merge-in-github-action-workfl
    BASE_COMMIT="HEAD^"
  else
    BASE_COMMIT="$GITHUB_REF_SLUG"
  fi
elif [ "${TRAVIS:=false}" == "true" ]; then 
  if [ "${TRAVIS_PULL_REQUEST:=false}" == "false" ] ; then
    # Not in a pull request, so compare against parent commit
    # FIXME: need to use first commit of this push
    BASE_COMMIT="HEAD^"
  else
    BASE_COMMIT="$TRAVIS_BRANCH"
  fi
else
  echo "Undefined CI environment"
  exit 1 
fi

BASE_COMMIT_REVPARSE=$(git rev-parse "$BASE_COMMIT")
if [ "${BASE_COMMIT}" == "HEAD^" ]; then
  echo "Running clang-format against parent commit ${BASE_COMMIT_REVPARSE}"  
else
  echo "Running clang-format against branch $BASE_COMMIT, with hash ${BASE_COMMIT_REVPARSE}"
fi


output="$(.travis-ci/checks/git-clang-format --binary clang-format --commit "$BASE_COMMIT" --diff --exclude ^dependencies/)"
if [ "$output" == "no modified files to format" ] || [ "$output" == "clang-format did not modify any files" ] ; then
  echo "clang-format passed."
  exit 0
else
  echo "clang-format failed:"
  echo "$output"
  exit 1
fi
