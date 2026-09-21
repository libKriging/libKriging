#!/usr/bin/env python3
"""Check that every file carrying the libKriging version agrees with it.

The reference is cmake/version.cmake (KRIGING_VERSION_MAJOR/MINOR/PATCH). The
version is also written by hand in:

  - CITATION.cff                              (version, date-released)
  - .claude-plugin/plugin.json                (version)
  - bindings/Julia/jlibkriging/Project.toml   (version)
  - bindings/R/rlibkriging/DESCRIPTION        (Version as X.Y-Z, Date)
  - CHANGELOG.md                              (section, compare link, table row)

and forgetting one of them has already broken a release (1.2.0, fixed by 1.2.1).
The release dates of CITATION.cff, DESCRIPTION and the CHANGELOG section must
also agree.

Usage (from anywhere):
  python3 tools/release/check_versions.py                # files agree with each other
  python3 tools/release/check_versions.py --tag v1.2.2   # ... and with the release tag

A tag vX.Y.Z must match the version exactly. A pre-release tag such as v1.2.h
(published as a draft) only has to match major.minor.
"""
import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DATE = r"(\d{4}-\d{2}-\d{2})"


def read(root, rel):
    return (root / rel).read_text(encoding="utf-8")


def search(pattern, text, what):
    m = re.search(pattern, text, re.M)
    if not m:
        raise ValueError(f"cannot find {what}")
    return m.group(1)


def code_version(root):
    text = read(root, "cmake/version.cmake")
    parts = [search(rf"^set\(KRIGING_VERSION_{k} (\d+)\)$", text, f"KRIGING_VERSION_{k}")
             for k in ("MAJOR", "MINOR", "PATCH")]
    return ".".join(parts)


def collect(root, version):
    """Return ([(label, found_version_or_None, error_or_None)], {label: date})."""
    versions, dates = [], {}

    def add(label, getter):
        try:
            found = getter()
            versions.append((label, found, None))
        except (ValueError, OSError, KeyError) as exc:
            versions.append((label, None, str(exc)))

    add("CITATION.cff", lambda: search(r"^version:\s*\"?([^\s\"]+)", read(root, "CITATION.cff"), "version"))
    add(".claude-plugin/plugin.json", lambda: json.loads(read(root, ".claude-plugin/plugin.json"))["version"])
    add("bindings/Julia/jlibkriging/Project.toml",
        lambda: search(r"^version\s*=\s*\"([^\"]+)\"", read(root, "bindings/Julia/jlibkriging/Project.toml"), "version"))
    # R writes its version as X.Y-Z
    add("bindings/R/rlibkriging/DESCRIPTION",
        lambda: search(r"^Version:\s*(\S+)", read(root, "bindings/R/rlibkriging/DESCRIPTION"), "Version").replace("-", "."))

    for label, getter in (
        ("CITATION.cff", lambda: search(rf"^date-released:\s*\"?{DATE}", read(root, "CITATION.cff"), "date-released")),
        ("bindings/R/rlibkriging/DESCRIPTION",
         lambda: search(rf"^Date:\s*{DATE}", read(root, "bindings/R/rlibkriging/DESCRIPTION"), "Date")),
        ("CHANGELOG.md", lambda: search(rf"^## \[{re.escape(version)}\] - {DATE}", read(root, "CHANGELOG.md"),
                                        f"a '## [{version}] - <date>' section")),
    ):
        try:
            dates[label] = getter()
        except (ValueError, OSError) as exc:
            dates[label] = f"MISSING ({exc})"
    return versions, dates


def changelog_errors(root, version):
    text = read(root, "CHANGELOG.md")
    errors = []
    if not re.search(rf"^\[{re.escape(version)}\]: https://", text, re.M):
        errors.append(f"CHANGELOG.md: no '[{version}]: https://...' compare link at the bottom")
    if not re.search(rf"^\| \[{re.escape(version)}\]\(", text, re.M):
        errors.append(f"CHANGELOG.md: no row for {version} in the 'Released versions' table")
    return errors


def tag_error(tag, version):
    m = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag)
    if m:
        if ".".join(m.groups()) != version:
            return f"tag {tag} does not match the code version {version}"
        return None
    major_minor = ".".join(version.split(".")[:2])
    if not re.match(rf"v{re.escape(major_minor)}\.", tag):
        return f"tag {tag} does not start with v{major_minor}. (code version {version})"
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", help="release tag to check against, e.g. v1.2.2")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="repository root (default: this checkout)")
    args = parser.parse_args(argv)
    root = args.root

    try:
        version = code_version(root)
    except (ValueError, OSError) as exc:
        print(f"FAIL cmake/version.cmake: {exc}")
        return 1
    print(f"libKriging version (cmake/version.cmake): {version}")

    errors = []
    versions, dates = collect(root, version)
    for label, found, err in versions:
        if err:
            errors.append(f"{label}: {err}")
            print(f"  FAIL {label}: {err}")
        elif found != version:
            errors.append(f"{label}: version is {found}, expected {version}")
            print(f"  FAIL {label}: {found}")
        else:
            print(f"  ok   {label}: {found}")

    try:
        errors += changelog_errors(root, version)
    except OSError as exc:
        errors.append(f"CHANGELOG.md: {exc}")

    print("release dates:")
    for label, date in dates.items():
        print(f"  {'FAIL' if date.startswith('MISSING') else 'ok  '} {label}: {date}")
    real = {d for d in dates.values() if not d.startswith("MISSING")}
    errors += [f"{label}: date {date}" for label, date in dates.items() if date.startswith("MISSING")]
    if len(real) > 1:
        errors.append("release dates differ between CITATION.cff, DESCRIPTION and CHANGELOG.md: " + ", ".join(sorted(real)))

    if args.tag:
        err = tag_error(args.tag, version)
        print(f"  {'FAIL' if err else 'ok  '} tag {args.tag}")
        if err:
            errors.append(err)

    if errors:
        print("\nVersion consistency check FAILED:")
        for e in errors:
            print(f"  - {e}")
        print("\nSee RELEASE.md for the files to update together.")
        return 1
    print("Version consistency check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
