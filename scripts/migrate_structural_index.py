#!/usr/bin/env python3
"""Preserve an unpublished shared index while changing its extraction contract."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from utils.openie_profile_migration import main

if __name__ == '__main__':
    raise SystemExit(main())
