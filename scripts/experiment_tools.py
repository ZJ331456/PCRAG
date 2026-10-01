#!/usr/bin/env python
"""Command-line entry point for the Python helpers used by shell runners."""

import argparse

from utils import ablations, embedding, prefetch, safe_prefetch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for module in (ablations, embedding, prefetch, safe_prefetch):
        module.register_commands(subparsers)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
