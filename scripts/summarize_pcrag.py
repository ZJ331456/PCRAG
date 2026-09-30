#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load(path: Path):
    d = json.loads(path.read_text(encoding='utf-8'))
    r = d.get('retrieval_metrics', {})
    q = d.get('qa_metrics', {})
    diag = d.get('retrieval_diagnostics', {})
    return {
        'file': str(path),
        'Recall@5': r.get('Recall@5'),
        'Recall@10': r.get('Recall@10'),
        'Recall@20': r.get('Recall@20'),
        'Recall@50': r.get('Recall@50'),
        'EM': q.get('ExactMatch'),
        'F1': q.get('F1'),
        'no_facts_rate': diag.get('no_facts_rate'),
        'fallback_to_dpr_rate': diag.get('fallback_to_dpr_rate'),
        'no_facts_reason_counter': diag.get('no_facts_reason_counter'),
        'fallback_reason_counter': diag.get('fallback_reason_counter'),
        'cache_hit_count': diag.get('cache_hit_count'),
        'cache_miss_count': diag.get('cache_miss_count'),
        'avg_bridge_entities': diag.get('avg_bridge_entities'),
        'avg_path_candidates': diag.get('avg_path_candidates'),
        'avg_selected_paths': diag.get('avg_selected_paths'),
        'hop_counter': diag.get('hop_counter'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    rows = [load(Path(x)) for x in args.files]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
