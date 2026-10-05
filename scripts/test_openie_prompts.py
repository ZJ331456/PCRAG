#!/usr/bin/env python3
"""Compare OpenIE prompts on saved real extraction failures."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from utils.openie_prompt_probe import main

if __name__ == '__main__':
    raise SystemExit(main())
