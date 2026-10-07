#!/usr/bin/env python3
"""Run full retrieval or the two-question check with the same DAG configuration."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from utils.multi_dataset_retrieval_plan_prune_dag_package import main

if __name__ == '__main__':
    raise SystemExit(main())
