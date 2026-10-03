#!/usr/bin/env python3
"""Verify native HippoRAG2 and PathCondRAG retrieval on the repaired index."""

import sys

sys.dont_write_bytecode = True

from utils.openie_repair_smoke import main

if __name__ == '__main__':
    raise SystemExit(main())
