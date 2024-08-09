#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    sys.stderr.flush()


class InlineClass(object):
    def __init__(self, dict):
        self.__dict__ = dict


if __name__ == '__main__':
    options = InlineClass({
            'pretty': True,
            'verbose': False,
            'soft': False,
            'indent': ''
        })
    print(f"{options.indent}!!! NOT Checking requirements (Python>=3.12)".encode('utf8').decode(sys.stdout.encoding))
    exit(0)