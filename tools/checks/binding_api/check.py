#!/usr/bin/env python3
"""Checks that the methods of a binding (Python, R, Octave or Julia) are
the ones listed in bindings/api.yaml (issue #419).

  check.py DUMP...

A dump is a text file of "<scope>\\t<name>" lines written by the dump_*
script of this directory, the binding being given by its file name
(python.txt, r.txt, octave.txt or julia.txt). A name of the manifest missing
from the binding, or exposed by the binding but not in the manifest, is an
error. Exit status 1 on any error.
"""

import argparse
import os
import re
import sys

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("check.py needs PyYAML: pip install pyyaml")

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
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
        return ["%s: %s not found" % (lang, path)]
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("dumps", nargs="+")
    args = parser.parse_args(argv)

    manifest = Manifest(args.manifest)
    errors = list(manifest.errors)
    if not errors:
        for path in args.dumps:
            errors += check_dump(manifest, path)
    for e in errors:
        print("error: " + e)
    if errors:
        print("%d error(s); the manifest is %s" % (len(errors), rel(args.manifest)))
        return 1
    print("binding API check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
