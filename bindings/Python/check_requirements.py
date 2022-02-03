#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pkg_resources
from pathlib import Path
import argparse
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    sys.stderr.flush()


class InlineClass(object):
    def __init__(self, dict):
        self.__dict__ = dict


def has_requirements(filename, options=None):
    if options is None:
        options = InlineClass({
            'pretty': True,
            'verbose': False,
            'soft': False,
            'indent': ''
        })

    try:
        with (Path(__file__).parent.absolute() / filename).open() as file:
            requirements = pkg_resources.parse_requirements(file)
            hasError = False
            for requirement in requirements:
                requirement = str(requirement)
                try:
                    if options.verbose:
                        print(f"{options.indent}Checking requirement package {requirement}".encode('utf-8'))
                    pkg_resources.require(requirement)
                    if options.pretty:
                        print(f"{options.indent}✔ Module {requirement} is available".encode('utf-8'))
                except pkg_resources.DistributionNotFound:
                    if options.soft:
                        print(f"{options.indent}✘ Module {requirement} is NOT available".encode('utf-8'))
                    else:
                        eprint(f"{options.indent}✘ Module {requirement} is NOT available".encode('utf-8'))
                    hasError = True
            return not hasError
    except FileNotFoundError:
        eprint("ERROR: file not found")
        return False


def get_requirements(filename):
    with (Path(__file__).parent.absolute() / filename).open() as file:
        requirements = pkg_resources.parse_requirements(file)
        return [str(requirement) for requirement in requirements]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check requirements from files.')
    parser.add_argument('filenames', metavar='filename', type=str, nargs='+',
                        help='requirements filename to check.')
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
    exit(hasError)
