This file describes good practices to follow if you want to contribue to this project.

# Commit Message Format
(inspired from git commit message from angular style and git-commit-message-convention repository)

```
[<type>[(<scope>)]: ]<subject>[ (#task)]
<empty ling>
<body>
<empty ling>
<footer>
```

## Types

| Type          | Description |
|:-------------:|-------------|
| `feature`     | for new feature implementing commit |
| `fix`         | for bug fix commit |
| `security`    | for security issue fix commit |
| `performance` | for performance issue fix commit |
| `improvement` | for backwards-compatible enhancement commit |
| `breaking`    | for backwards-incompatible enhancement commit |
| `deprecated`  | for deprecated feature commit |
| `i18n`        | for i18n (internationalization) commit |
| `refactor`    | for refactoring commit |
| `style`       | for coding style commit |
| `docs`        | for documentation commit |
| `example`     | for example code commit |
| `test`        | for testing commit |
| `dependency`  | for dependencies upgrading or downgrading commit |
| `config`      | for configuration commit |
| `build`       | for packaging or bundling commit |
| `ci`          | for continuous integration commit |
| `release`     | for publishing commit |
| `update`      | for update commit |
| `revert`      | for revert commit |
| `wip`         | for work in progress commit |

## Scope
The scope could be anything specifying place or category of the commit change.
For example tests, core, R, python, octave, etc...

## Subject
The subject contains succinct description of the change:

* use the imperative, present tense: "change" not "changed" nor "changes"
* don't capitalize first letter
* no dot (.) at the end

To skip Travis CI build, the HEAD commit message on push must contain `[skip ci]`.

## Message Body
Just as in the **Subject**, use the imperative, present tense: "change" not "changed" nor "changes". 
The body should include the motivation for the change and contrast this with previous behavior.

## Message Footer
The Message Footer should contain any information about **Notes** and also Message Footer 
should be **recommended** [GitHub Issue](https://github.com/features#issues) ID Reference, 
Ex. `Issue #27`, `Fixes #1`, `Closes #2`, `Resolves #3`.

**Notes** should start with the word `NOTE:` with a space or two newlines. 
The rest of the commit message is then used for this.


## Revert
If the commit reverts a previous commit, it should begin with **revert:**, 
followed by the header of the reverted commit. In the body it should say: 
`This reverts commit <hash>.`, where <hash> is the SHA of the commit being reverted.

## Examples

new feature:
```
feature(python): add 'custom expression' option
```

bug fix:
```
fix(R): correct memory corrumption while passing R objects

Closes #28
```

improve performance:
```
performance(core): improve thread management

Default OpenMP thread dispatch is now tuned to be compatible with outer threads from R 
```

revert:
```
revert: feature(python): add 'custom expression' option

This reverts commit 667ecc1654a317a13331b17617d973392f415f02.
```

skip ci:
```
doc: fix minor mispelling [skip ci]

CI build is a waste of time for a so tiny mispelling fix. 
```