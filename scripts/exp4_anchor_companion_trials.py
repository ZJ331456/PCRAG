#!/usr/bin/env python3
"""Small paired tests of opt-in evidence selection after the frozen DAG."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from utils.exp4_anchor_companion_trials import main

if __name__ == '__main__':
    raise SystemExit(main())
