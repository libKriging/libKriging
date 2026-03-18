#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import sys
import re

if sys.version_info >= (3, 8):
    from importlib.metadata import version, PackageNotFoundError
else:
    from pkg_resources import require as version, DistributionNotFound as PackageNotFoundError


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    sys.stderr.flush()


class InlineClass:
    def __init__(self, dict):
        self.__dict__ = dict


def parse_requirements(file):
    def strip_comments(line):
        # Remove everything after the first # (comments)
        return line.split('#', 1)[0].strip()

    return [
        strip_comments(line)
        for line in file
        if strip_comments(line) and not line.startswith('#')
    ]


def has_requirements(filename, options=None):
    if options is None:
        options = InlineClass({
            'pretty': True,
            'verbose': False,
            'soft': False,
            'indent': ''
        })

    try:
        if filename.startswith(".") or filename.startswith("/"):
            filename = Path(filename)
        else:
            filename = Path(__file__).parent.absolute() / filename

        with filename.open() as file:
            requirements = parse_requirements(file)
            hasError = False
            for requirement in requirements:
                package_name = re.split('[<>=]', requirement)[0].strip()
                try:
                    if options.verbose:
                        print(f"{options.indent}Checking requirement package {requirement}".encode('utf8').decode(
                            sys.stdout.encoding))
                    version(package_name)
                    if options.pretty:
                        print(f"{options.indent}✔ Module {requirement} is available".encode('utf8').decode(
                            sys.stdout.encoding))
                except PackageNotFoundError:
                    if options.soft:
                        print(f"{options.indent}✘ Module {requirement} is NOT available".encode('utf8').decode(
                            sys.stdout.encoding))
                    else:
                        eprint(f"{options.indent}✘ Module {requirement} is NOT available".encode('utf8').decode(
                            sys.stdout.encoding))
                    hasError = True
            return not hasError
    except FileNotFoundError:
        eprint(f"ERROR: file '{filename}' not found")
        return False


def get_requirements(filename):
    with (Path(__file__).parent.absolute() / filename).open() as file:
        return parse_requirements(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check requirements from files.')
    parser.add_argument('filenames', metavar='filename', type=str, nargs='+',
                        help='requirements filename to check.\n\n'
                             'NB: Filenames not starting with . or / are considered as local to this script')
    parser.add_argument("--pretty", action="store_true",
                        help="Pretty display while processing")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Pretty display while processing")
    parser.add_argument("--indent", type=str, default='',
                        help="Prefix all output with given string")
    parser.add_argument("--soft", action="store_true",
                        help="Soft error are written in stdout not stderr")
    args = parser.parse_args()
    hasError = False
    for file in args.filenames:
        if args.verbose:
            print(f"{args.indent}Checking requirements file '{file}'")
        hasError |= not has_requirements(file, args)
    sys.exit(hasError)
