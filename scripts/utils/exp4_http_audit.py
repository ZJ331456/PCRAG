"""Locate a local vLLM log and retain immutable per-stage HTTP audit segments."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import re
import stat
from urllib.parse import urlsplit

from .common import read_json, write_json


ROOT = Path(__file__).resolve().parents[2]
_STATUS = re.compile(r'POST /v1/chat/completions HTTP/1\.1" (\d{3})')


def locate(base_url):
    url = urlsplit(base_url)
    if url.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Provide VLLM_LOG explicitly for a remote endpoint')
    port = str(url.port or (443 if url.scheme == 'https' else 80))
    matches = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / 'cmdline').read_bytes().decode('utf-8', 'replace').split('\0')
            if 'vllm.entrypoints.openai.api_server' not in args or '--port' not in args:
                continue
            if args[args.index('--port') + 1] != port:
                continue
            output = process / 'fd/1'
            if stat.S_ISREG(output.stat().st_mode):
                # The descriptor remains readable even when its log was deleted.
                matches.append(str(output))
        except (OSError, ValueError, IndexError):
            continue
    if len(matches) != 1:
        raise ValueError(f'Expected one readable local service log, found {len(matches)}; provide VLLM_LOG')
    return matches[0]


def archive(out, namespace):
    metadata = out / 'metadata' / namespace
    records, attempts, statuses = [], 0, Counter()
    for path in sorted((metadata / 'stage_reports').glob('*/*.json')):
        report = read_json(path)
        for name in ('proposer', 'verifier'):
            stage = report[name]
            segment = stage['http_log_segment']
            target = metadata / 'http_audit' / report['dataset'] / f'{report["phase"]}_{name}.log'
            index_path = target.with_suffix('.json')
            prior = read_json(index_path) if index_path.is_file() else None
            stage_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if prior and prior['stage_report_sha256'] == stage_hash:
                blob = target.read_bytes()
                if hashlib.sha256(blob).hexdigest() != prior['segment_sha256']:
                    raise ValueError('Retained HTTP audit bytes changed')
            else:
                with open(segment['path'], 'rb') as stream:
                    stream.seek(segment['start'])
                    blob = stream.read(segment['end'] - segment['start'])
            observed = Counter(_STATUS.findall(blob.decode('utf-8', 'replace')))
            stats = stage['observed_request_stats']
            if (dict(observed) != stage['http_status_in_log'] or set(observed) - {'200'}
                    or sum(observed.values()) != stats['http_attempts'] or stats['failures'] or stats['retries']):
                raise ValueError(f'HTTP segment failed audit: {report["dataset"]}/{report["phase"]}/{name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.tmp')
            temporary.write_bytes(blob)
            temporary.replace(target)
            record = {'dataset': report['dataset'], 'phase': report['phase'], 'stage': name,
                'stage_report': str(path), 'stage_report_sha256': stage_hash,
                'original_log_segment': segment, 'archive_path': str(target),
                'segment_sha256': hashlib.sha256(blob).hexdigest(), 'http_status': dict(observed),
                'observed_http_attempts': stats['http_attempts']}
            write_json(index_path, record)
            records.append(record)
            attempts += stats['http_attempts']
            statuses.update(observed)
    index = {'namespace': namespace, 'segments': records,
        'observed_http_attempts_counted_once': attempts, 'http_status_counted_once': dict(statuses),
        'note': 'Pilot and screening can reuse the same cached response; actual HTTP attempts counted by unique stage reports.'}
    write_json(metadata / 'http_audit/archive_index.json', index)
    print(f'[HTTP archive] {len(records)} segments; {attempts} attempts; {dict(statuses)}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('locate', 'archive'))
    parser.add_argument('--base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--out-root', default=str(ROOT / 'outputs/3multi_hop_datasets_results_10_5'))
    parser.add_argument('--namespace', default='exp4_swap_verification_v2_selection')
    args = parser.parse_args(argv)
    if args.action == 'locate':
        print(locate(args.base_url))
        return 0
    out = Path(args.out_root).resolve()
    if not out.is_relative_to(ROOT / 'outputs') or out == ROOT / 'outputs' or '/' in args.namespace:
        raise ValueError('Archive target must be a dedicated experiment metadata directory')
    archive(out, args.namespace)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
