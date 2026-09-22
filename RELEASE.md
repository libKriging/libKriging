# Releasing libKriging

How a version is prepared, checked and published. Maintainers only.

## Version numbers

`MAJOR.MINOR.PATCH` ([Semantic Versioning](https://semver.org/)):

- **patch**: fixes, documentation, packaging, CI; no API or numerical change;
- **minor**: new features or new supported platforms, backward compatible;
- **major**: incompatible changes.

A released number is never reused: PyPI refuses to upload a version twice, and a tag that was already built has published
assets. If a release is wrong, fix it forward with the next patch, as 1.2.1 did for 1.2.0.

## 1. Prepare the release

Work on a `release/X.Y.Z` branch and open a pull request into `master`. Never push to `master` directly.

1. **CHANGELOG.md**
   - rename `## [Unreleased]` into `## [X.Y.Z] - YYYY-MM-DD` and add a fresh empty `## [Unreleased]` above it. The
     GitHub release notes are extracted from the section whose heading starts with `## [X.Y.Z]`, so keep that form;
   - add the `[X.Y.Z]: https://github.com/libKriging/libKriging/compare/vPREVIOUS...vX.Y.Z` link at the bottom and
     point `[Unreleased]` at `vX.Y.Z...master`;
   - add the row of the version to the "Released versions" table;
   - list breaking changes explicitly, with the migration.
2. **Version and date**, in all of these files (the release date must be the same in the three that carry one):

   | File | Field |
   |---|---|
   | `cmake/version.cmake` | `KRIGING_VERSION_MAJOR`, `_MINOR`, `_PATCH` (the reference) |
   | `CITATION.cff` | `version`, `date-released` |
   | `.claude-plugin/plugin.json` | `version` |
   | `bindings/Julia/jlibkriging/Project.toml` | `version` |
   | `bindings/R/rlibkriging/DESCRIPTION` | `Version` (written `X.Y-Z`, e.g. `1.2-2`), `Date` |

3. Run the check; CI runs it on every push (job `guideline-checks`) and again on the tag:

   ```shell
   python3 tools/release/check_versions.py
   ```

4. Merge the pull request once CI is green.

## 2. Publish

Tag the merge commit on `master` and push the tag:

```shell
git tag vX.Y.Z
git push origin vX.Y.Z
```

Every `v*` tag starts the `release-*` workflows (they can also be started by hand with a `release-tag` input). Each one
first checks the tag against the code (`check_versions.py --tag`): a `vX.Y.Z` tag must equal the version exactly. A tag whose
patch is not a number (such as `v1.2.h`) only has to match `MAJOR.MINOR` and is published as a draft release.

| Workflow | Result |
|---|---|
| `release-cpp.yml` | C++ library archives (Linux, macOS, Windows) |
| `release-python.yml` | `pylibkriging` wheels, attached to the release and uploaded to PyPI (refused if the version is already there; needs the `TWINE_PASSWORD` secret) |
| `release-octave.yml`, `release-matlab.yml` | `mLibKriging` packages (the Matlab build needs a MathWorks license token) |
| `release-r.yml` | `rlibkriging` archives |
| `release-julia.yml` | tests the Julia binding on the tag; it does not publish anything |

The `cpp`, `r`, `octave` and `matlab` workflows create the GitHub release if it does not exist yet, with the changelog
section as notes.

The other packages follow `master` through `sync-as-submodule.yml`, on every push to `master`:

- `rlibkriging` (R package repository) gets the new submodule commits;
- `JLibKriging.jl` (the installable Julia package) gets the new libKriging sources, and registers itself on Julia's
  General registry from its own CI when the libKriging version is new.

## 3. After the release

- Check that the release page shows the changelog section, and that the wheels are on
  [PyPI](https://pypi.org/project/pylibkriging/).
- Update the example version in the install instructions once the assets exist: the wheel URL, `VERSION=` and
  `vX.Y.Z` in `README.md`, and `VERSION=` in `bindings/Octave/README.md`.
- Check that `JLibKriging.jl` picked up the version, and the R package repository.
