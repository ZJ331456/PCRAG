"""Small uncached Qwen3 prompt comparison; never constructs an index or embeds."""

import argparse
import copy
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / 'outputs/pathcondrag_new_index_10_4'
DEFAULT_OUTPUT = ROOT / 'outputs/openie_prompt_probe_20261005'
HISTORICAL_IDS = (
    'chunk-cf6bae3832242da40ece26af3227919e',
    'chunk-1e64fc5b929a2e89e6ea3a3c51ad6687',
    'chunk-d8b09513b1fd6cdf0cb91bf988bcef5e',
)
PROPERTY_ID = 'chunk-45cbf085501d5962e4e775534948fa5d'
NER_SCHEMA = {'type': 'object', 'properties': {'named_entities': {
    'type': 'array', 'items': {'type': 'string', 'minLength': 1}}},
    'required': ['named_entities'], 'additionalProperties': False}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def prompt_hashes():
    prompt_root = ROOT / 'src/pathcondrag/prompts/templates'
    return {filename: hashlib.sha256((prompt_root / filename).read_bytes()).hexdigest()
            for filename in ('ner.py', 'triple_extraction.py', 'origin_ner_prompt.py',
                             'origin_triple_extraction_prompt.py')}


def extraction_hashes():
    sources = ('index/ner/source_verified.py', 'index/openie/source_verified_openie.py',
               'index/openie/openie_source_evidence.py', 'index/openie/openie_structured_output.py')
    return {name: hashlib.sha256((ROOT / 'src/pathcondrag' / name).read_bytes()).hexdigest()
            for name in sources}


def failure_cases(source_root):
    """Preserve unchanged paragraphs and original failure diagnostics."""
    before = read_json(source_root / 'failed_chunk_before_resume.json')
    cases = [{
        'chunk_id': before['idx'], 'source': before['passage'],
        'title': before['passage'].split('\n')[0],
        'kind': 'real_triple_failure',
        'original_failure': before['openie_metadata']['triples'],
    }]
    historical = read_json(ROOT / 'outputs/openie_failed_triple_chunks.json')['chunks']
    by_id = {row['idx']: row for row in historical}
    for identifier in HISTORICAL_IDS:
        row = by_id[identifier]
        cases.append({
            'chunk_id': identifier, 'source': row['passage'],
            'title': row['passage'].split('\n')[0], 'kind': 'real_triple_failure',
            'original_failure': [{key: record.get(key) for key in
                                  ('run', 'problem', 'openie_skipped', 'triple_metadata')}
                                 for record in row['sources']],
        })
    state = source_root / ('shared_hipporag2_index/'
                           'qwen3-8b__root_models_Qwen3-Embedding-8B/openie_state.json')
    property_row = next(row for row in read_json(state)['docs'] if row['idx'] == PROPERTY_ID)
    cases.append({
        'chunk_id': PROPERTY_ID, 'source': property_row['passage'],
        'title': property_row['passage'].split('\n')[0], 'kind': 'real_property_regression',
        'original_failure': property_row.get('openie_metadata', {}).get('triples', {}),
    })
    return cases


class DirectClient:
    """Production verifier adapter with zero result cache and no hidden retries."""

    def __init__(self, args):
        from transformers import AutoTokenizer
        self.llm_name = args.model
        self.seed = args.seed
        self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', 'EMPTY'),
                             base_url=args.base_url, timeout=args.timeout, max_retries=0)
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        self.tokenizer_load_seconds = time.perf_counter() - started
        self.requests = []

    @lru_cache(maxsize=512)
    def _count(self, serialized):
        return len(self.tokenizer.apply_chat_template(
            json.loads(serialized), tokenize=True, add_generation_prompt=True,
            enable_thinking=False))

    def count_prompt_tokens(self, messages):
        return self._count(json.dumps(messages, ensure_ascii=False, separators=(',', ':')))

    def infer(self, messages, **settings):
        settings = copy.deepcopy(settings)
        budget = settings.pop('max_completion_tokens', settings.pop('max_new_tokens', 2048))
        settings.setdefault('seed', self.seed)
        settings.setdefault('temperature', 0.0)
        extra = settings.setdefault('extra_body', {})
        extra.setdefault('chat_template_kwargs', {})['enable_thinking'] = False
        started = time.perf_counter()
        count = self.count_prompt_tokens(messages)
        tokenization_seconds = time.perf_counter() - started
        if count + budget > 8192:
            raise ValueError('Prompt and unchanged output budget exceed the 8192-token context')
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.llm_name, messages=messages, max_completion_tokens=budget, **settings)
        except Exception as error:
            self.requests.append({'messages': copy.deepcopy(messages), 'settings': settings,
                'raw_response': None, 'metadata': {
                    'seconds': time.perf_counter() - started, 'cache_hit': False,
                    'thinking': False, 'max_completion_tokens': budget,
                    'error': f'{type(error).__name__}: {error}',
                    'http_status': getattr(error, 'status_code', None)}})
            raise
        seconds = time.perf_counter() - started
        choice = response.choices[0]
        content = choice.message.content or ''
        usage = response.usage.model_dump() if response.usage else {}
        metadata = {**usage, 'finish_reason': choice.finish_reason, 'model': response.model,
                    'response_id': response.id, 'seconds': seconds,
                    'local_prompt_tokens': count, 'tokenization_seconds': tokenization_seconds,
                    'max_completion_tokens': budget, 'cache_hit': False, 'thinking': False,
                    'http_status': 200}
        self.requests.append({'messages': copy.deepcopy(messages), 'settings': settings,
                              'raw_response': content, 'metadata': metadata})
        return content, metadata, False


def request(client, messages, schema, budget):
    from pathcondrag.index.openie.openie_structured_output import guided_json_parameters
    started = time.perf_counter()
    grammar = guided_json_parameters(schema)
    preparation_seconds = time.perf_counter() - started
    try:
        raw, metadata, _ = client.infer(messages=messages, max_completion_tokens=budget,
                                        temperature=0.0, extra_body=grammar)
        return {'raw_response': raw, 'metadata': metadata,
                'grammar_preparation_seconds': preparation_seconds, 'error': None}
    except Exception as error:
        return {'raw_response': None, 'metadata': {},
                'grammar_preparation_seconds': preparation_seconds,
                'error': f'{type(error).__name__}: {error}'}


def validate_ner(result):
    try:
        payload = json.loads(result['raw_response'])
        entities = payload['named_entities']
        if (set(payload) != {'named_entities'} or not isinstance(entities, list)
                or any(not isinstance(value, str) or not value.strip() for value in entities)):
            raise ValueError('NER must contain only nonempty named-entity strings')
        complete = result['metadata'].get('finish_reason') == 'stop'
        return {'complete': complete, 'entities': list(dict.fromkeys(entities)),
                'raw_count': len(entities), 'duplicates': len(entities) - len(set(entities))}
    except Exception as error:
        return {'complete': False, 'entities': [], 'raw_count': 0, 'duplicates': 0,
                'validation_error': f'{type(error).__name__}: {error}'}


def validate_relations(result):
    from pathcondrag.index.openie.openie_quality import validate_triples
    try:
        payload = json.loads(result['raw_response'])
        if not isinstance(payload, dict) or set(payload) != {'triples'}:
            raise ValueError('Output must contain exactly the triples key')
        report = validate_triples(payload['triples'])
        return {'complete': result['metadata'].get('finish_reason') == 'stop'
                and not report.invalid_triples, 'raw_count': report.raw_count,
                'valid_count': len(report.valid_triples), 'invalid_count': len(report.invalid_triples),
                'duplicates': report.raw_count - len(report.invalid_triples) - len(report.valid_triples),
                'invalid_triples': report.invalid_triples, 'issues': report.issues,
                'triples': payload['triples'], 'valid_triples': report.valid_triples}
    except Exception as error:
        return {'complete': False, 'raw_count': 0, 'valid_count': 0, 'invalid_count': 0,
                'duplicates': 0, 'triples': [], 'valid_triples': [],
                'validation_error': f'{type(error).__name__}: {error}'}


def key_relation_review(case, triples):
    """Literal diagnostics locate candidates; these are not entailment approvals."""
    hits = {}
    if case['title'] == 'Randy and Sharon Marsh':
        for child in ('Stan', 'Shelly'):
            for parent in ('Randy', 'Sharon'):
                hits[f'{parent}_{child}'] = [triple for triple in triples
                    if parent.casefold() in triple[0].casefold() and child.casefold() in triple[2].casefold()
                    or child.casefold() in triple[0].casefold() and parent.casefold() in triple[2].casefold()]
        hits['dingbat_attribution_candidates'] = [triple for triple in triples
                                                  if 'dingbat' in ' '.join(triple).casefold()]
    elif case['title'] == 'List of Back to the Future characters':
        hits['parentage_candidates'] = [triple for triple in triples
            if any(word in triple[1].casefold() for word in ('son', 'daughter', 'child', 'parent'))]
        hits['portrayal_candidates'] = [triple for triple in triples
                                        if 'fox' in triple[2].casefold()]
    elif case['title'] == 'Scar (The Lion King)':
        hits['training_candidates'] = [triple for triple in triples
                                       if any(word in ' '.join(triple).casefold()
                                              for word in ('training', 'trained', 'theater'))]
    return {'literal_key_relation_candidates': hits,
            'warning': 'Candidate presence is diagnostic, not evidence of correct entailment or complete coverage.'}


def run_case(case, ordinal, args):
    from pathcondrag.prompts import PromptTemplateManager
    from pathcondrag.index.openie.openie_quality import TRIPLE_JSON_SCHEMA
    manager = PromptTemplateManager()
    client = DirectClient(args)
    versions = ('origin', 'optimized') if ordinal % 2 == 0 else ('optimized', 'origin')
    result = {**case, 'order': list(versions), 'versions': {}}
    for version in versions:
        names = ('origin_ner_prompt', 'origin_triple_extraction_prompt') if version == 'origin' else ('ner', 'triple_extraction')
        ner = request(client, manager.render(names[0], passage=case['source']), NER_SCHEMA, 512)
        ner['validation'] = validate_ner(ner)
        if ner['validation']['complete']:
            triples = request(client, manager.render(names[1], passage=case['source'],
                named_entity_json=json.dumps({'named_entities': ner['validation']['entities']}, ensure_ascii=False)),
                TRIPLE_JSON_SCHEMA, 2048)
        else:
            triples = {'raw_response': None, 'metadata': {}, 'error': 'NER incomplete; chain not fabricated'}
        triples['validation'] = validate_relations(triples)
        candidates = triples['validation']['valid_triples']
        result['versions'][version] = {'ner': ner, 'triples': triples,
                                       'key_relation_review': key_relation_review(case, candidates)}
        print(f"[probe] {case['title']} {version} NER={ner['validation']['complete']} "
              f"triples={triples['validation']['valid_count']} "
              f"complete={triples['validation']['complete']} "
              f"seconds={triples['metadata'].get('seconds')}", flush=True)
    result['requests'] = client.requests
    result['tokenizer_load_seconds'] = client.tokenizer_load_seconds
    return result


def audit_cases(rows, args):
    from pathcondrag.index.openie.openie_source_evidence import verify_source_relations, SourceEvidenceError
    client = DirectClient(args)
    for row in rows:
        if row['title'] not in ('Randy and Sharon Marsh', 'List of Back to the Future characters', 'Scar (The Lion King)'):
            continue
        for version, result in row['versions'].items():
            candidates = result['triples']['validation']['valid_triples']
            if not candidates:
                result['source_audit'] = {'complete': False, 'skipped': True,
                    'reason': 'No complete extracted candidates to audit; does not approve the failed chain.'}
                continue
            key = key_relation_review(row, candidates)['literal_key_relation_candidates']
            keys = list(key)
            if 'dingbat_attribution_candidates' in keys:
                keys.remove('dingbat_attribution_candidates')
                keys.insert(0, 'dingbat_attribution_candidates')
            prioritized = [triple for name in keys for triple in key[name]]
            subset = [list(value) for value in dict.fromkeys(tuple(triple) for triple in prioritized + candidates)]
            subset = subset[:args.audit_max_triples]
            started = time.perf_counter()
            try:
                accepted, audit = verify_source_relations(client, row['source'], subset)
                result['source_audit'] = {'complete': True, 'selected_candidates': subset,
                    'accepted': accepted, 'accepted_count': len(accepted), 'audit': audit}
            except SourceEvidenceError as error:
                result['source_audit'] = {'complete': False, 'selected_candidates': subset,
                    'error': str(error), 'audit': error.audit_metadata}
            result['source_audit'].update(seconds=time.perf_counter() - started,
                scope='selected key candidates only; not an audit of all extracted triples')
            print(f"[audit] {row['title']} {version} complete={result['source_audit']['complete']}", flush=True)
            write_json(Path(args.output_dir) / 'results.json', rows)
    write_json(Path(args.output_dir) / 'audit_requests.json', client.requests)


def summarize(rows, args):
    summary = {'n_cases': len(rows), 'model': args.model, 'seed': args.seed,
        'temperature': 0.0, 'thinking': False, 'ner_budget': 512, 'triple_budget': 2048,
        'workers': args.workers, 'client_result_cache': False, 'client_transport_retries': 0,
        'all_variants_use_same_structured_grammar': True,
        'warning': 'Small targeted failures sample; structural success is not semantic correctness. '
                   'Server prefix cache and GPU batching can affect timing; no retrieval metric is measured.',
        'versions': {},
        'http_attempts': sum(len(row['requests']) for row in rows),
        'http_statuses': dict(Counter(str(request['metadata'].get('http_status', 'unknown'))
            for row in rows for request in row['requests'])),
        'tokenizer_load_seconds_sum': sum(row['tokenizer_load_seconds'] for row in rows),
        'cpu_tokenization_seconds_sum': sum(request['metadata'].get('tokenization_seconds', 0)
            for row in rows for request in row['requests'])}
    for version in ('origin', 'optimized'):
        parts = [row['versions'][version] for row in rows]
        summary['versions'][version] = {
            'ner_complete': sum(part['ner']['validation']['complete'] for part in parts),
            'triples_complete': sum(part['triples']['validation']['complete'] for part in parts),
            'triples_total': sum(part['triples']['validation']['valid_count'] for part in parts),
            'duplicates': sum(part['triples']['validation']['duplicates'] for part in parts),
            'invalid_triples': sum(part['triples']['validation']['invalid_count'] for part in parts),
            'http_seconds_sum': sum(part[stage]['metadata'].get('seconds', 0)
                                    for part in parts for stage in ('ner', 'triples')),
            'completion_tokens': sum(part[stage]['metadata'].get('completion_tokens', 0)
                                    for part in parts for stage in ('ner', 'triples')),
            'prompt_tokens': sum(part[stage]['metadata'].get('prompt_tokens', 0)
                                    for part in parts for stage in ('ner', 'triples')),
            'finish_reasons': dict(Counter(part[stage]['metadata'].get('finish_reason', 'missing')
                                          for part in parts for stage in ('ner', 'triples'))),
        }
    return summary


def full_pipeline(case, args):
    """One historical failure runs the real strict queue and checkpoint replay."""
    from pathcondrag.index.shared_index_builder import SharedQualityOpenIE
    from pathcondrag.index.openie_checkpoint import OpenIECheckpoint
    hashes = prompt_hashes()
    implementation = extraction_hashes()
    client = DirectClient(args)
    chunks = {case['chunk_id']: {'content': case['source']}}
    extractor = SharedQualityOpenIE(client, max_workers=8, prompt_version='optimized')
    progress = OpenIECheckpoint(Path(args.output_dir) / 'full_pipeline_checkpoint.sqlite',
                               {'probe': 'optimized_real_failure', 'chunk_id': case['chunk_id']})
    started = time.perf_counter()
    try:
        extractor.checkpoint = progress.save
        ner, triples = extractor.batch_openie(chunks)
        key = case['chunk_id']
        result = {'case': case, 'ner': asdict(ner[key]), 'triples': asdict(triples[key]),
                  'verified_complete': extractor.is_verified_complete(triples[key]),
                  'seconds': time.perf_counter() - started, 'requests': client.requests,
                  'prompt_manifest': hashes, 'implementation_manifest': implementation}
        requests_before = len(client.requests)
        extractor.initial_rows = progress.overlay([], chunks)
        resumed_ner, resumed_triples = extractor.batch_openie(chunks)
        result.update(checkpoint_resume_new_requests=len(client.requests) - requests_before,
                      checkpoint_resume_equal=resumed_triples[key].triples == triples[key].triples,
                      checkpoint_resume_verified=extractor.is_verified_complete(resumed_triples[key]))
        write_json(Path(args.output_dir) / 'full_pipeline.json', result)
        print(f"[full pipeline] {case['title']} verified={result['verified_complete']} "
              f"requests={requests_before} resume_requests={result['checkpoint_resume_new_requests']}", flush=True)
        return result
    finally:
        extractor.checkpoint = None
        progress.close()


def ner_recovery(case, args):
    """Exercise production bounded NER retries and source-unit recovery only."""
    from pathcondrag.index.shared_index_builder import SharedQualityOpenIE
    hashes = prompt_hashes()
    implementation = extraction_hashes()
    client = DirectClient(args)
    extractor = SharedQualityOpenIE(client, max_workers=8, prompt_version='optimized')
    started = time.perf_counter()
    initial = extractor.ner(case['chunk_id'], case['source'])
    recovered = initial
    if initial.metadata.get('complete') is not True:
        recovered = extractor.recover_pending_ner(case['chunk_id'], case['source'], initial, 1)
    requests_before_resume = len(client.requests)
    resumed = None
    if recovered.metadata.get('complete') is True:
        resumed = extractor.recover_pending_ner(case['chunk_id'], case['source'], recovered, 2)
    result = {'case': case, 'initial': asdict(initial), 'recovered': asdict(recovered),
              'complete': recovered.metadata.get('complete') is True,
              'seconds': time.perf_counter() - started, 'requests': client.requests,
              'prompt_manifest': hashes,
              'implementation_manifest': implementation,
              'scope': 'NER only; no triple extraction or source relation approval'}
    if resumed is not None:
        result.update(recovery_resume_new_requests=len(client.requests) - requests_before_resume,
                      recovery_resume_complete=resumed.metadata.get('complete') is True,
                      recovery_resume_equal=resumed.unique_entities == recovered.unique_entities)
    write_json(Path(args.output_dir) / 'ner_recovery.json', result)
    print(f"[NER recovery] {case['title']} complete={result['complete']} "
          f"requests={len(client.requests)}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--model', default='qwen3-8b')
    parser.add_argument('--model-path', default='/root/models/Qwen3-8B')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--audit-max-triples', type=int, default=4)
    parser.add_argument('--no-audit', action='store_true')
    parser.add_argument('--full-pipeline-only', action='store_true')
    parser.add_argument('--full-pipeline', action='store_true')
    parser.add_argument('--ner-recovery-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--case-id', action='append', default=[],
                        help='Select only these saved chunk IDs; repeat to select more than one.')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.audit_max_triples < 1:
        parser.error('workers must be 1..8 and audit-max-triples must be positive')
    if not args.output_dir.resolve().is_relative_to(ROOT / 'outputs'):
        parser.error('probe outputs must be within PathCondRAG/outputs')
    cases = failure_cases(args.source_root)
    if args.case_id:
        cases = [case for case in cases if case['chunk_id'] in args.case_id]
        if len(cases) != len(set(args.case_id)):
            parser.error('case-id must name one of the five saved probe paragraphs')
    selection_name = ('full_pipeline_selected_cases.json' if args.full_pipeline_only else
                      'ner_recovery_selected_cases.json' if args.ner_recovery_only else 'selected_cases.json')
    write_json(args.output_dir / selection_name, cases)
    if args.prepare_only:
        print(f'[prepared] {len(cases)} unchanged real paragraphs', flush=True)
        return 0
    if args.full_pipeline_only:
        result = full_pipeline(cases[0], args)
        return int(not result['verified_complete'])
    if args.ner_recovery_only:
        result = ner_recovery(cases[0], args)
        return int(not result['complete'])
    write_json(args.output_dir / 'prompt_manifest.json', prompt_hashes())
    write_json(args.output_dir / 'implementation_manifest.json', extraction_hashes())
    # Compile both task grammars and warm the server before measured A/B cases.
    from pathcondrag.index.openie.openie_quality import TRIPLE_JSON_SCHEMA
    warmup = DirectClient(args)
    request(warmup, [{'role': 'user', 'content':
                      'Return only {"named_entities":["Ada"]} for Ada lives in Rome.'}], NER_SCHEMA, 512)
    request(warmup, [{'role': 'user', 'content':
                      'Return only {"triples":[["Ada","lives in","Rome"]]} for Ada lives in Rome.'}],
            TRIPLE_JSON_SCHEMA, 2048)
    write_json(args.output_dir / 'warmup_requests.json', warmup.requests)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda item: run_case(item[1], item[0], args), enumerate(cases)))
    write_json(args.output_dir / 'results.json', rows)
    if not args.no_audit:
        audit_cases(rows, args)
    if args.full_pipeline:
        full_pipeline(cases[0], args)
    report = summarize(rows, args)
    report['wall_seconds'] = time.perf_counter() - started
    write_json(args.output_dir / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0
