#!/usr/bin/env python3
"""Run the read-only HippoRAG baseline with memory-bounded NV embeddings."""

import os
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
BASELINE = Path(os.environ.get('HIPPO_ROOT', '/root/baseline/HippoRAG')).resolve()
sys.path.insert(0, str(BASELINE / 'src'))

from pathcondrag.embedding_model.nvembed_runtime import baseline_nvembed_runtime


def main():
    source = BASELINE / 'main.py'
    if not source.is_file():
        raise FileNotFoundError(f'HippoRAG baseline entry is missing: {source}')
    sys.argv[0] = str(source)
    with baseline_nvembed_runtime():
        return runpy.run_path(str(source), run_name='__main__')


if __name__ == '__main__':
    main()
