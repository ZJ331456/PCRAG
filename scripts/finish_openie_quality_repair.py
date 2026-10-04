#!/usr/bin/env python3
"""Finish an existing repair, verify both retrievers, and report completion."""

import sys

sys.dont_write_bytecode = True

from utils.openie_repair_finish import main

if __name__ == "__main__":
    raise SystemExit(main())
