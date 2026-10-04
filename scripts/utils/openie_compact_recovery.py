"""Compatibility exports for the canonical PathCondRAG source module.

Private helpers remain exported for existing repair commands and CPU tests.
"""

from pathcondrag.index import openie_compact_recovery as _implementation

globals().update({name: value for name, value in vars(_implementation).items()
                  if not name.startswith('__')})
__all__ = [name for name in vars(_implementation) if not name.startswith('__')]
