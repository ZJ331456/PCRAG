#!/usr/bin/env python3
"""Build the shared index with PathCondRAG's strict OpenIE quality contract."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pathcondrag.index.shared_index_builder import run_shared_index_cli

if __name__ == '__main__':
    run_shared_index_cli()
