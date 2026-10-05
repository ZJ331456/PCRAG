"""Explicitly resume an unpublished shared index with structural extraction.

The old journal is backed up. Passage vectors and successful NER are retained;
only original generated triples can become structural graph inputs.
"""

import argparse
import copy
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

from .openie_checkpoint_migration import migrate_source_verified_checkpoint
from pathcondrag.index.shared_index_builder import quality_profile

MODEL_DIR = 'qwen3-8b__root_models_Qwen3-Embedding-8B'


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_structural_resume(out_root, dataset, corpus_path):
    """Migrate one stopped, unpublished dataset without discarding its work."""
    out = Path(out_root).resolve()
    index = out / 'shared_indexes' / dataset
    model = index / MODEL_DIR
    metadata = out / 'metadata' / dataset
    journal = model / 'openie_progress.sqlite'
    manifest_path = model / 'index_manifest.json'
    manifest = _read(manifest_path)
    profile = manifest['openie']['quality_profile']
    if profile.get('validation_mode') == 'structural':
        report_path = metadata / 'structural_migration_report.json'
        if not report_path.is_file():
            raise ValueError('Structural index has no explicit migration report')
        return _read(report_path)
    for name in ('graph.pickle', 'entity_embeddings/vdb_entity.parquet',
                 'fact_embeddings/vdb_fact.parquet'):
        if (model / name).exists():
            raise ValueError(f'Cannot migrate a published or derived index: {name}')
    if any((out / 'cases' / dataset).glob('*/validated.ok')):
        raise ValueError('Cannot change the index behind completed retrieval cases')
    with closing(sqlite3.connect(journal.as_uri() + '?mode=ro', uri=True)) as connection:
        source_identity = json.loads(connection.execute(
            'SELECT value FROM identity WHERE key = ?', ('contract',)).fetchone()[0])
    if manifest['openie'] != source_identity['provenance']:
        raise ValueError('Index manifest and source checkpoint disagree')
    strict = profile.get('openie_strict', True)
    prompt = profile.get('prompt_version', 'optimized')
    expected_source = quality_profile(strict=strict, prompt_version=prompt,
                                      validation_mode='source_verified')
    if profile != expected_source:
        raise ValueError('Source checkpoint quality contract differs from the supported producer')
    target_profile = quality_profile(strict=strict, prompt_version=prompt,
                                     validation_mode='structural')
    target_identity = copy.deepcopy(source_identity)
    target_identity['provenance']['quality_profile'] = target_profile
    target_manifest = copy.deepcopy(manifest)
    target_manifest['openie'] = copy.deepcopy(target_identity['provenance'])
    chunks = {}
    for row in _read(corpus_path):
        text = row['title'] + '\n' + (row.get('text') or row.get('paragraph_text', ''))
        key = 'chunk-' + hashlib.md5(text.encode()).hexdigest()
        chunks[key] = {'content': text}
    vector = model / 'chunk_embeddings/vdb_chunk.parquet'
    vector_hash = _sha256(vector)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    backup = metadata / ('source_verified_backup_' + stamp)
    backup.mkdir(parents=True, exist_ok=False)
    with closing(sqlite3.connect(journal.as_uri() + '?mode=ro', uri=True)) as source:
        with closing(sqlite3.connect(str(backup / 'openie_progress.sqlite'))) as target:
            source.backup(target)
    _write(backup / 'index_manifest.json', manifest)
    protocol_path = out / 'protocol.json'
    protocol = _read(protocol_path) if protocol_path.is_file() else None
    if protocol is not None:
        _write(backup / 'protocol.json', protocol)
    target_journal = model / ('openie_progress.structural_' + stamp + '.sqlite')
    report = migrate_source_verified_checkpoint(
        journal, target_journal, source_identity=source_identity,
        target_identity=target_identity, chunks=chunks)
    if _sha256(vector) != vector_hash:
        target_journal.unlink(missing_ok=True)
        raise RuntimeError('Passage vectors changed during checkpoint migration')
    report.update(backup_dir=str(backup), source_index=str(index),
                  retained_passage_vector_sha256=vector_hash,
                  source_profile=profile, target_profile=target_profile)
    moved = []
    installed = manifest_updated = protocol_updated = False
    try:
        for suffix in ('', '-wal', '-shm'):
            original = Path(str(journal) + suffix)
            if original.exists():
                archived = backup / ('original_openie_progress.sqlite' + suffix)
                original.rename(archived)
                moved.append((original, archived))
        target_journal.rename(journal)
        installed = True
        _write(manifest_path, target_manifest)
        manifest_updated = True
        if protocol is not None:
            updated = dict(protocol, openie_validation_mode='structural')
            _write(protocol_path, updated)
            protocol_updated = True
        # Publishing a manifest without its migration report would make an
        # interrupted rerun impossible to identify. Report failure rolls back
        # the journal and the metadata together.
        _write(metadata / 'structural_migration_report.json', report)
    except BaseException:
        # Before the first rename fails, ``journal`` is still the only live
        # source. Never remove it unless the new journal was installed.
        if installed:
            journal.unlink(missing_ok=True)
        for original, archived in moved:
            archived.rename(original)
        if manifest_updated:
            _write(manifest_path, manifest)
        if protocol_updated:
            _write(protocol_path, protocol)
        raise
    finally:
        target_journal.unlink(missing_ok=True)
        for path in (manifest_path, protocol_path, metadata / 'structural_migration_report.json'):
            path.with_suffix(path.suffix + '.tmp').unlink(missing_ok=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--dataset', choices=('hotpotqa', '2wikimultihopqa', 'musique'), required=True)
    parser.add_argument('--corpus-path', required=True)
    args = parser.parse_args(argv)
    report = prepare_structural_resume(args.out_root, args.dataset, args.corpus_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0
