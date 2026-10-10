#!/usr/bin/env python3
"""Compare dependency-aware scoring with its legacy parent on a paired subset."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from utils.nv2_dependency_scoring_subset import main

if __name__ == '__main__':
    raise SystemExit(main())
