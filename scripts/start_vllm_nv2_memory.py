#!/usr/bin/env python3
"""Restart the existing Qwen3 API with one specified GPU memory ratio."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.nv2_vllm_service import main

if __name__ == '__main__':
    raise SystemExit(main())
