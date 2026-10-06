#!/usr/bin/env python3
"""Screen retrieval improvements, then run the selected frozen-index comparison."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from utils.exp4_improvements import main

if __name__ == '__main__':
    raise SystemExit(main())
