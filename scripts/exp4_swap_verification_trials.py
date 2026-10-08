#!/usr/bin/env python3
"""Run paired support-retention and independent swap-verification trials."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from utils.exp4_swap_verification_trials import main

if __name__ == '__main__':
    raise SystemExit(main())
