"""Durable, per-stage OpenIE progress independent of HTTP response caching."""

import hashlib
import json
from pathlib import Path
import sqlite3


class OpenIECheckpoint:
    """One writer owns the journal; every completed stage is committed separately."""

    def __init__(self, path, identity):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('CREATE TABLE IF NOT EXISTS identity (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS progress (chunk_id TEXT PRIMARY KEY, passage_sha256 TEXT NOT NULL, payload TEXT NOT NULL)')
        serialized = json.dumps(identity, sort_keys=True, ensure_ascii=False)
        old = self.connection.execute('SELECT value FROM identity WHERE key = ?', ('contract',)).fetchone()
        if old and old[0] != serialized:
            self.connection.close()
            raise RuntimeError(f'OpenIE progress belongs to another corpus/producer/quality contract: {self.path}')
        self.connection.execute('INSERT OR IGNORE INTO identity VALUES (?, ?)', ('contract', serialized))
        self.connection.commit()

    def save(self, row):
        passage, key = row.get('passage'), row.get('idx')
        if not isinstance(passage, str) or not isinstance(key, str):
            raise ValueError('OpenIE checkpoint requires the complete original passage and chunk ID')
        digest = hashlib.sha256(passage.encode()).hexdigest()
        payload = json.dumps(row, ensure_ascii=False)
        with self.connection:
            self.connection.execute('INSERT OR REPLACE INTO progress VALUES (?, ?, ?)', (key, digest, payload))

    def overlay(self, rows, chunks):
        """Replay only matching original passages; never import another corpus."""
        by_id = {row['idx']: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError('Duplicate source rows in OpenIE checkpoint replay')
        for key, digest, serialized in self.connection.execute('SELECT chunk_id, passage_sha256, payload FROM progress'):
            if key not in chunks:
                raise RuntimeError(f'OpenIE progress has an out-of-corpus chunk: {key}')
            passage = chunks[key]['content']
            if digest != hashlib.sha256(passage.encode()).hexdigest():
                raise RuntimeError(f'OpenIE progress source changed: {key}')
            row = json.loads(serialized)
            if row.get('idx') != key or row.get('passage') != passage:
                raise RuntimeError(f'OpenIE progress row identity is corrupt: {key}')
            by_id[key] = row
        return [by_id[key] for key in chunks if key in by_id]

    def close(self):
        if self.connection is not None:
            self.connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            self.connection.close()
            self.connection = None
