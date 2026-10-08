#!/usr/bin/env python3
"""Measure intact NV-Embed-v2 batches on a GPU shared with vLLM."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from utils.nvembed_memory_probe import main

if __name__ == '__main__':
    raise SystemExit(main())
