#!/usr/bin/env python3
"""Entry point for concise independent evidence review experiments."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from utils.exp4_source_unit_verification_trials import main

if __name__ == '__main__':
    raise SystemExit(main())
