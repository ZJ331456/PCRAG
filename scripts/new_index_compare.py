#!/usr/bin/env python3
"""Build a fresh quality index and compare three retrieval methods."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from utils.new_index_compare import main

if __name__ == '__main__':
    raise SystemExit(main())
