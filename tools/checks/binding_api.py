#!/usr/bin/env python3
"""Checks the API exposed by the Python, R, Octave and Julia bindings
against the manifest bindings/api.yaml (issue #419).

  binding_api.py manifest          validate the manifest itself
  binding_api.py tests             every binding test file is run by ctest
                                   or by the binding's test runner
  binding_api.py dump FILE...      compare introspection dumps (python.txt,
                                   r.txt, octave.txt, julia.txt, written by
                                   the api_dump tests) to the manifest
  binding_api.py table             print the manifest as a Markdown table

A dump is a text file of "<scope>\\t<name>" lines, the binding being given
by the file name. Exit status 1 on any finding.
"""

import argparse
import os
import re
import sys

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("binding_api.py needs PyYAML: pip install pyyaml")

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_MANIFEST = os.path.join(ROOT, "bindings", "api.yaml")
BINDINGS = ("python", "r", "octave", "julia")
NOT_EXPOSED = "not exposed:"


def rel(path):
    path = os.path.abspath(path)
    return os.path.relpath(path, ROOT) if path.startswith(ROOT + os.sep) else path


def is_absent(value):
    return isinstance(value, str) and value.startswith(NOT_EXPOSED)


class Manifest:
    def __init__(self, path):
        with open(path) as f:
            self.data = yaml.safe_load(f)
        self.path = path
        self.errors = []
        self._validate()

    def _error(self, msg):
        self.errors.append("%s: %s" % (rel(self.path), msg))

    def _validate(self):
        if not isinstance(self.data, dict) or not isinstance(self.data.get("scopes"), dict):
            self._error("top-level 'scopes' mapping required")
            self.data = {"scopes": {}}
            return
        for scope, spec in self.data["scopes"].items():
            if not isinstance(spec, dict) or not isinstance(spec.get("members"), dict):
                self._error("scope %s: 'members' mapping required" % scope)
                continue
            for lang, value in (spec.get("bindings") or {}).items():
                if lang not in BINDINGS:
                    self._error("scope %s: unknown binding '%s'" % (scope, lang))
                elif not is_absent(value):
                    self._error("scope %s: bindings.%s must be '%s <reason>'" % (scope, lang, NOT_EXPOSED))
            for lang in BINDINGS:
                if not self.scope_exposed(scope, lang):
                    continue
                seen = {}
                for member in spec["members"]:
                    try:
                        name = self.member_name(scope, member, lang)
                    except ValueError as e:
                        self._error("%s.%s: %s" % (scope, member, e))
                        continue
                    if name is None:
                        continue
                    if name in seen:
                        self._error("%s: %s and %s are both '%s' in %s"
                                    % (scope, seen[name], member, name, lang))
                    seen[name] = member

    def scopes(self):
        return list(self.data["scopes"])

    def scope_exposed(self, scope, lang):
        return not is_absent((self.data["scopes"][scope].get("bindings") or {}).get(lang))

    def member_name(self, scope, member, lang):
        """Name of `member` in `lang`, or None when declared not exposed."""
        spec = self.data["scopes"][scope]["members"][member]
        if spec is None:
            return member
        if not isinstance(spec, dict):
            raise ValueError("value must be empty or a mapping")
        unknown = set(spec) - set(BINDINGS) - {"only", "reason", "optional"}
        if unknown:
            raise ValueError("unknown keys %s" % sorted(unknown))
        if "optional" in spec and not (isinstance(spec["optional"], str) and spec["optional"].strip()):
            raise ValueError("'optional' must give the reason")
        if "only" in spec:
            only = spec["only"]
            if isinstance(only, str):
                only = [only]
            if not only or set(only) - set(BINDINGS):
                raise ValueError("'only' must list bindings among %s" % (BINDINGS,))
            if not spec.get("reason"):
                raise ValueError("'only' requires a 'reason'")
            if lang not in only:
                return None
        value = spec.get(lang, member)
        if is_absent(value):
            if not value[len(NOT_EXPOSED):].strip():
                raise ValueError("%s: '%s' needs a reason" % (lang, NOT_EXPOSED))
            return None
        if not isinstance(value, str) or not re.fullmatch(r"\S+", value):
            raise ValueError("%s: %r is neither a name nor '%s <reason>'" % (lang, value, NOT_EXPOSED))
        return value

    def is_optional(self, scope, member):
        """True for a member that a build may or may not expose (the reason
        is the value of `optional`, e.g. Debug-only functions)."""
        spec = self.data["scopes"][scope]["members"][member]
        return isinstance(spec, dict) and bool(spec.get("optional"))

    def expected(self, scope, lang):
        names = {}
        for member in self.data["scopes"][scope]["members"]:
            name = self.member_name(scope, member, lang)
            if name is not None:
                names[name] = member
        return names


def read_dump(path):
    dump = {}
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            scope, sep, name = line.partition("\t")
            if not sep or not name:
                raise SystemExit("%s:%d: expected '<scope>\\t<name>', got %r" % (path, n, line))
            dump.setdefault(scope, set()).add(name)
    return dump


def check_dump(manifest, path):
    lang = os.path.splitext(os.path.basename(path))[0].lower()
    if lang not in BINDINGS:
        return ["%s: file name must be one of %s" % (path, ", ".join(b + ".txt" for b in BINDINGS))]
    if not os.path.exists(path):
        return ["%s: %s not found (did the %s api_dump test run?)" % (lang, path, lang)]
    dump = read_dump(path)
    errors = []
    for scope in sorted(set(dump) - set(manifest.scopes())):
        errors.append("%s: scope '%s' is not in the manifest" % (lang, scope))
    for scope in manifest.scopes():
        exposed = manifest.scope_exposed(scope, lang)
        actual = dump.get(scope)
        if not exposed:
            if actual:
                errors.append("%s: %s is declared not exposed but the dump lists %d names"
                              % (lang, scope, len(actual)))
            continue
        if actual is None:
            errors.append("%s: %s is missing from the dump (declare it not exposed in the manifest?)"
                          % (lang, scope))
            continue
        expected = manifest.expected(scope, lang)
        for name in sorted(set(expected) - actual):
            if manifest.is_optional(scope, expected[name]):
                continue
            errors.append("%s: %s.%s (manifest: %s) is not exposed"
                          % (lang, scope, name, expected[name]))
        for name in sorted(actual - set(expected)):
            errors.append("%s: %s.%s is exposed but not in the manifest" % (lang, scope, name))
    return errors


# --- test registration --------------------------------------------------

def _registered(cmakelists, macro):
    """FILENAME arguments of the macro calls in a CMakeLists.txt."""
    with open(cmakelists) as f:
        text = "\n".join(line.split("#", 1)[0] for line in f)
    files = set()
    for args in re.findall(r"\b%s\s*\(([^)]*)\)" % macro, text):
        m = re.search(r"\bFILENAME\s+(\S+)", args)
        if m:
            files.add(m.group(1))
    return files


# binding -> (tests directory, file pattern of a test, how the files are run)
TEST_RUNNERS = {
    "python": ("bindings/Python/pylibkriging/tests", r"(test_.*|.*_test)\.py$",
               ("add_python_test", "CMakeLists.txt")),
    "octave": ("bindings/Octave/mlibkriging/tests", r"(test_.*|.*test)\.m$",
               ("add_mex_test", "CMakeLists.txt")),
    "julia": ("bindings/Julia/jlibkriging/tests", r"(test_.*|.*_test)\.jl$",
              ("add_julia_test", "CMakeLists.txt")),
    # testthat runs every test*.R file of the directory (and helper*/setup*/
    # teardown* as support files): any other .R file never runs
    "r": ("bindings/R/rlibkriging/tests/testthat", r".*\.[rR]$", None),
}
TESTTHAT_RUN = re.compile(r"^(test|helper|setup|teardown).*\.[rR]$")


def check_tests(manifest):
    allowed = (manifest.data.get("tests") or {}).get("not_run") or {}
    errors = []
    for path, reason in allowed.items():
        if not reason:
            errors.append("tests.not_run.%s needs a reason" % path)
        if not os.path.exists(os.path.join(ROOT, path)):
            errors.append("tests.not_run.%s does not exist" % path)
    for lang, (tests_dir, pattern, runner) in TEST_RUNNERS.items():
        full_dir = os.path.join(ROOT, tests_dir)
        files = sorted(f for f in os.listdir(full_dir) if re.fullmatch(pattern, f))
        if runner is None:
            run = {f for f in files if TESTTHAT_RUN.match(f)}
        else:
            macro, cmakelists = runner
            run = _registered(os.path.join(full_dir, cmakelists), macro)
            for f in sorted(run):
                if not os.path.exists(os.path.join(full_dir, f)):
                    errors.append("%s: %s/%s registered by %s does not exist" % (lang, tests_dir, f, macro))
        for f in files:
            rel = "%s/%s" % (tests_dir, f)
            if f in run and rel in allowed:
                errors.append("%s: %s is run, remove it from tests.not_run" % (lang, rel))
            elif f not in run and rel not in allowed:
                how = "%s(...) in %s/%s" % (runner[0], tests_dir, runner[1]) if runner \
                    else "a test*.R name"
                errors.append("%s: %s is never run (needs %s, or a tests.not_run entry)" % (lang, rel, how))
    return errors


def table(manifest):
    out = []
    for scope in manifest.scopes():
        out.append("\n### %s\n" % scope)
        out.append("| member | " + " | ".join(BINDINGS) + " |")
        out.append("|---" * (len(BINDINGS) + 1) + "|")
        for member in manifest.data["scopes"][scope]["members"]:
            cells = []
            for lang in BINDINGS:
                name = manifest.member_name(scope, member, lang) if manifest.scope_exposed(scope, lang) else None
                cells.append("—" if name is None else ("`%s`" % name))
            out.append("| %s | %s |" % (member, " | ".join(cells)))
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest")
    sub.add_parser("tests")
    dump = sub.add_parser("dump")
    dump.add_argument("files", nargs="+")
    sub.add_parser("table")
    args = parser.parse_args(argv)

    manifest = Manifest(args.manifest)
    errors = list(manifest.errors)
    if not errors:
        if args.command == "tests":
            errors += check_tests(manifest)
        elif args.command == "dump":
            for path in args.files:
                errors += check_dump(manifest, path)
        elif args.command == "table":
            print(table(manifest))
    for e in errors:
        print("error: " + e)
    if errors:
        print("%d finding(s); the manifest is %s" % (len(errors), rel(args.manifest)))
        return 1
    if args.command != "table":
        print("binding API %s check: OK" % args.command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
