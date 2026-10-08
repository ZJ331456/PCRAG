#!/usr/bin/env python3
"""Build three NV-Embed-v2 indexes and run fifteen retrieval comparisons."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from utils.multi_dataset_retrieval_nv2 import main

if __name__ == '__main__':
    raise SystemExit(main())
