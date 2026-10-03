#!/usr/bin/env python3
"""Repair a shared OpenIE index through PathCondRAG without editing HippoRAG."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
for name in ("PATHCONDRAG_LLM_MAX_IN_FLIGHT", "HIPPORAG_LLM_MAX_IN_FLIGHT",
             "HIPPO_OPENIE_MAX_WORKERS", "HIPPO_OPENIE_NER_WORKERS", "HIPPO_OPENIE_TRIPLE_WORKERS"):
    os.environ[name] = "8"
os.environ["HIPPO_OPENIE_NER_MAX_TOKENS"] = "512"
os.environ["HIPPO_OPENIE_TRIPLE_MAX_TOKENS"] = "2048"

from utils.openie_repair import main

if __name__ == "__main__":
    raise SystemExit(main())
