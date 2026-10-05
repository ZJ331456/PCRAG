#!/usr/bin/env python3
"""Build shared indexes and evaluate all three multi-hop benchmarks."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from utils.multi_dataset_retrieval import main

if __name__ == '__main__':
    raise SystemExit(main())
