"""Prepare longest real inputs, then probe raw NV batches without OOM splitting.

Preparation loads only the local tokenizer. The GPU phase uses the production
NV initialization, dtype, normalization and 2048 token limit. Reports contain
input metadata; complete source text is retained only in the fixture file.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import runpy
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid


DATASETS = ('hotpotqa', '2wikimultihopqa', 'musique')
MAX_LENGTH = 2048
BATCH_SIZE = 4
VERSION = 'nvembed_raw_intact_batch4_memory_probe_v1'
LOG = logging.getLogger('nvembed_memory_probe')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def record_metadata(record):
    return {key: value for key, value in record.items() if key != 'text'}


def input_lengths(tokenizer, texts, instruction=''):
    """Match native input_transform_func: instruction + text + EOS, then BOS."""
    lengths = []
    for start in range(0, len(texts), 128):
        batch = [instruction + text + tokenizer.eos_token for text in texts[start:start + 128]]
        tokens = tokenizer(batch, padding=False, truncation=False, return_token_type_ids=False)
        lengths.extend(len(row) for row in tokens['input_ids'])
    return lengths


def make_record(tokenizer, text, **metadata):
    length = input_lengths(tokenizer, [text], metadata.get('instruction', ''))[0]
    return {**metadata, 'text': text, 'text_sha256': sha256_text(text),
            'characters': len(text), 'full_input_tokens': length,
            'effective_input_tokens': min(length, MAX_LENGTH),
            'truncated_by_native_2048_limit': length > MAX_LENGTH}


def prepare_fixtures(args):
    from transformers import AutoTokenizer

    # Importing pathcondrag.__init__ also imports the API stack. Read this small
    # pure instruction module directly so CPU preparation stays local.
    instruction_module = runpy.run_path(str(
        Path(__file__).resolve().parents[2] / 'src/pathcondrag/prompts/linking.py'))
    get_query_instruction = instruction_module['get_query_instruction']

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    # Report real lengths, including lengths above the production truncation
    # limit, without the tokenizer's irrelevant model_max_length warning.
    tokenizer.deprecation_warnings['sequence-length-is-longer-than-the-specified-maximum'] = True
    query_instruction = f"Instruct: {get_query_instruction('query_to_passage')}\nQuery: "
    fixtures = {'version': VERSION, 'created_at': utc_now(),
                'embedding_model': str(Path(args.embedding_model).resolve()),
                'max_length': MAX_LENGTH, 'batch_size': BATCH_SIZE,
                'tokenizer_class': tokenizer.__class__.__name__,
                'tokenizer_add_eos': True, 'padding_side': 'right',
                'passage_format': "title + '\\n' + original text",
                'sources': {}, 'datasets': {}, 'cases': []}
    longest_queries = []
    for dataset in DATASETS:
        corpus_path = Path(args.datasets_dir) / f'{dataset}_corpus.json'
        question_path = Path(args.datasets_dir) / f'{dataset}.json'
        corpus = json.loads(corpus_path.read_text(encoding='utf-8'))
        texts = [row['title'] + '\n' + (row.get('text') or row.get('paragraph_text', ''))
                 for row in corpus]
        lengths = input_lengths(tokenizer, texts)
        indices = sorted(range(len(texts)), key=lambda i: (-lengths[i], i))[:BATCH_SIZE]
        top = [{
            'dataset': dataset, 'source_row': i, 'title': corpus[i]['title'],
            'text': texts[i], 'text_sha256': sha256_text(texts[i]),
            'characters': len(texts[i]), 'full_input_tokens': lengths[i],
            'effective_input_tokens': min(lengths[i], MAX_LENGTH),
            'truncated_by_native_2048_limit': lengths[i] > MAX_LENGTH,
        } for i in indices]
        if len(top) != BATCH_SIZE:
            raise ValueError(f'{dataset} requires at least four source passages')
        questions = json.loads(question_path.read_text(encoding='utf-8'))
        query_texts = [row['question'] for row in questions]
        query_lengths = input_lengths(tokenizer, query_texts, query_instruction)
        query_indices = sorted(range(len(questions)), key=lambda i: (-query_lengths[i], i))[:BATCH_SIZE]
        longest_queries.extend(make_record(tokenizer, query_texts[i], dataset=dataset,
                                          source_row=i, kind='real_question',
                                          instruction=query_instruction) for i in query_indices)
        fixtures['sources'][dataset] = {
            'corpus_path': str(corpus_path.resolve()), 'corpus_sha256': sha256_file(corpus_path),
            'question_path': str(question_path.resolve()), 'question_sha256': sha256_file(question_path),
            'corpus_rows': len(corpus), 'question_rows': len(questions),
        }
        fixtures['datasets'][dataset] = {
            'top4_longest_passages': top,
            'maximum_full_input_tokens': max(lengths),
            'passages_over_2048_tokens': sum(length > MAX_LENGTH for length in lengths),
        }
        fixtures['cases'].append({'name': f'{dataset}_longest_original_top4',
                                  'kind': 'real_passages', 'instruction': '',
                                  'repetitions': 1, 'records': top})
        LOG.info('prepared %s rows=%d top4=%s', dataset, len(corpus),
                 [record_metadata(record) for record in top])

    top_global = sorted((record for dataset in DATASETS
                         for record in fixtures['datasets'][dataset]['top4_longest_passages']),
                        key=lambda row: (-row['full_input_tokens'], row['dataset'], row['source_row']))
    longest = top_global[0]
    fixtures['global_longest_original'] = record_metadata(longest)
    fixtures['cases'].append({'name': 'global_longest_original_repeated4',
                              'kind': 'real_passage_repeated_for_worst_padding',
                              'instruction': '', 'repetitions': 3,
                              'records': [longest] * BATCH_SIZE})
    queries = sorted(longest_queries, key=lambda row: (-row['full_input_tokens'],
                                                     row['dataset'], row['source_row']))[:BATCH_SIZE]
    fixtures['cases'].append({'name': 'query_to_passage_longest_original_top4',
                              'kind': 'real_questions_with_native_instruction',
                              'instruction': query_instruction, 'repetitions': 1,
                              'records': queries})
    entities = [make_record(tokenizer, record['title'], dataset=record['dataset'],
                            source_row=record['source_row'], kind='source_title_entity')
                for record in top_global[:BATCH_SIZE]]
    # Source title and source excerpt simulate the tuple serialization used by
    # fact embeddings, without making any API call or changing an index.
    facts = []
    for record in top_global[:BATCH_SIZE]:
        source_body = record['text'].partition('\n')[2]
        first_sentence = re.split(r'(?<=[.!?])\s+', source_body, maxsplit=1)[0][:400]
        fact = str((record['title'], 'source excerpt', first_sentence))
        facts.append(make_record(tokenizer, fact, dataset=record['dataset'],
                                 source_row=record['source_row'], kind='synthetic_fact_from_source_excerpt'))
    fixtures['cases'].extend([
        {'name': 'entity_short_inputs4', 'kind': 'source_title_entities',
         'instruction': '', 'repetitions': 1, 'records': entities},
        {'name': 'fact_short_inputs4', 'kind': 'synthetic_source_excerpt_facts',
         'instruction': '', 'repetitions': 1, 'records': facts},
    ])
    repeated_paragraph = longest['text'] + '\n\n' + longest['text']
    while input_lengths(tokenizer, [repeated_paragraph])[0] < MAX_LENGTH:
        repeated_paragraph += '\n\n' + longest['text']
    boundary = make_record(tokenizer, repeated_paragraph, dataset=longest['dataset'],
                           source_row=longest['source_row'], title=longest['title'],
                           kind='synthetic_repetition_of_intact_real_paragraph')
    assert boundary['effective_input_tokens'] == MAX_LENGTH
    fixtures['cases'].append({'name': 'native_2048_token_boundary_repeated4',
                              'kind': 'synthetic_token_limit_pressure_only',
                              'instruction': '', 'repetitions': 3,
                              'records': [boundary] * BATCH_SIZE})
    # A query instruction adds tokens to the longest passage. This checks the
    # instruction path at the same 2048 limit, including its pooling mask.
    boundary_query = make_record(tokenizer, repeated_paragraph, dataset=longest['dataset'],
                                 source_row=longest['source_row'], title=longest['title'],
                                 kind='synthetic_instruction_boundary', instruction=query_instruction)
    fixtures['cases'].append({'name': 'query_to_passage_2048_token_boundary4',
                              'kind': 'synthetic_instruction_token_limit_pressure_only',
                              'instruction': query_instruction, 'repetitions': 1,
                              'records': [boundary_query] * BATCH_SIZE})
    fixtures['batch_comparison_case'] = f"{longest['dataset']}_longest_original_top4"
    write_json(args.fixtures, fixtures)
    LOG.info('fixture saved=%s global_longest=%s cases=%d', args.fixtures,
             fixtures['global_longest_original'], len(fixtures['cases']))
    return fixtures


def gpu_snapshot(torch):
    devices = []
    for device in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(device)
        devices.append({
            'device': device, 'name': torch.cuda.get_device_name(device),
            'free_bytes': free, 'total_bytes': total,
            'allocated_bytes': torch.cuda.memory_allocated(device),
            'reserved_bytes': torch.cuda.memory_reserved(device),
            'max_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'max_reserved_bytes': torch.cuda.max_memory_reserved(device),
        })
    snapshot = {'timestamp': utc_now(), 'torch_cuda_devices': devices}
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=True)
        snapshot['nvidia_smi_gpu_csv'] = result.stdout.strip().splitlines()
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=True)
        snapshot['nvidia_smi_process_csv'] = result.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        snapshot['nvidia_smi_error'] = f'{type(error).__name__}: {error}'
    return snapshot


def synchronize(torch):
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)


def reset_peaks(torch):
    synchronize(torch)
    for device in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device)


def error_details(error):
    message = str(error)
    requested = re.search(r'Tried to allocate ([\d.]+\s*[A-Za-z]+)', message)
    free = re.search(r'([\d.]+\s*[A-Za-z]+) is free', message)
    if free is None:
        free = re.search(r'([\d.]+\s*[A-Za-z]+) free', message)
    return {'type': type(error).__name__, 'message': message,
            'requested_allocation': requested.group(1) if requested else None,
            'reported_free_memory': free.group(1) if free else None}


def validate_embedding(torch, output, norm):
    if not isinstance(output, torch.Tensor):
        output = torch.as_tensor(output)
    if output.ndim != 2 or tuple(output.shape) != (BATCH_SIZE, 4096):
        raise ValueError(f'Expected finite (4,4096) embeddings, received {tuple(output.shape)}')
    if not torch.isfinite(output).all().item():
        raise ValueError('Raw NV embeddings contain nonfinite values')
    output = output.detach().cpu()
    if norm:
        # Match memory_bounded_batch_encode's native NumPy normalization.
        import numpy as np
        array = output.numpy()
        array = (array.T / np.linalg.norm(array, axis=1)).T
        output = torch.from_numpy(array)
    if not torch.isfinite(output).all().item():
        raise ValueError('Normalized NV embeddings contain nonfinite values')
    norms = output.float().norm(dim=1)
    return output, {'shape': list(output.shape), 'dtype': str(output.dtype),
                    'all_finite': True, 'row_norms': norms.tolist()}


def prepare_llm_requests(args, fixtures):
    """Create eight unique near-long-context requests, with short JSON answers."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.llm_tokenizer, local_files_only=True)
    source = fixtures['cases'][0]['records'][0]['text']
    text = source
    while len(tokenizer.encode(text, add_special_tokens=False)) < 6000:
        text += '\n\n' + source
    source_tokens = tokenizer.encode(text, add_special_tokens=False)
    requests = []
    for position in range(8):
        request_id = f'nv-memory-{position}-{uuid.uuid4().hex[:12]}'
        system = 'Reply with the requested short JSON object only. Do not explain or summarize the source.'

        def messages_for(token_count):
            context = tokenizer.decode(source_tokens[:token_count], skip_special_tokens=True)
            return [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': 'Source context:\n' + context + '\n\n'
                     + f'Request ID: {request_id}. Return exactly '
                     + json.dumps({'id': request_id, 'ok': True}) + '.'}]

        # Include the exact no-thinking chat template in the 5700 token bound.
        low, high = 0, min(5700, len(source_tokens))
        while low < high:
            middle = (low + high + 1) // 2
            token_count = len(tokenizer.apply_chat_template(
                messages_for(middle), tokenize=True, add_generation_prompt=True,
                enable_thinking=False))
            if token_count <= 5700:
                low = middle
            else:
                high = middle - 1
        messages = messages_for(low)
        token_count = len(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))
        if token_count + 2048 > 8192:
            raise ValueError('LLM health prompt plus max_tokens must fit 8192 context')
        requests.append({'request_id': request_id, 'local_template_input_tokens': token_count,
                         'payload': {'model': 'qwen3-8b', 'messages': messages,
                                     'temperature': 0, 'max_tokens': 2048,
                                     'chat_template_kwargs': {'enable_thinking': False},
                                     'response_format': {'type': 'json_object'}}})
    return requests


def request_llm(args, request, barrier):
    barrier.wait(timeout=20)
    start = time.monotonic()
    result = {'request_id': request['request_id'], 'started_monotonic': start,
              'local_template_input_tokens': request['local_template_input_tokens'],
              'max_tokens': 2048, 'enable_thinking': False, 'status': 'failed'}
    try:
        headers = {'Content-Type': 'application/json',
                   'Authorization': 'Bearer ' + os.environ.get('OPENAI_API_KEY', 'EMPTY')}
        http_request = urllib.request.Request(
            args.llm_base_url.rstrip('/') + '/chat/completions',
            data=json.dumps(request['payload']).encode('utf-8'), headers=headers)
        with urllib.request.urlopen(http_request, timeout=120) as response:
            result['http_status'] = response.status
            payload = json.loads(response.read().decode('utf-8'))
        if result['http_status'] != 200 or payload.get('error'):
            raise ValueError(f'LLM response status={result["http_status"]} error={payload.get("error")}')
        message = payload['choices'][0]['message']
        content = message.get('content') or ''
        result['contains_think_tag'] = bool(re.search(r'</?think\b', content, re.IGNORECASE))
        result['reasoning_content_present'] = bool(message.get('reasoning_content'))
        if result['contains_think_tag'] or result['reasoning_content_present']:
            raise ValueError('No-thinking health response contained thinking output')
        answer = json.loads(content)
        if answer.get('id') != request['request_id'] or answer.get('ok') is not True:
            raise ValueError('LLM did not return the requested unique short JSON object')
        result['response_json'] = answer
        result['usage'] = payload.get('usage', {})
        result['input_tokens'] = result['usage'].get('prompt_tokens')
        result['generated_tokens'] = result['usage'].get('completion_tokens')
        result['finish_reason'] = payload['choices'][0].get('finish_reason')
        result['status'] = 'passed'
    except Exception as error:
        result['error'] = error_details(error)
        if isinstance(error, urllib.error.HTTPError):
            result['http_status'] = error.code
            result['http_error_body'] = error.read().decode('utf-8', errors='replace')[:2000]
    finally:
        result['finished_monotonic'] = time.monotonic()
        result['seconds'] = result['finished_monotonic'] - start
    return result


def run_concurrent_llm_probe(args, fixtures, wrapper, torch, report):
    requests = prepare_llm_requests(args, fixtures)
    result = {'status': 'running', 'llm_base_url': args.llm_base_url,
              'request_count': 8, 'request_concurrency': 8, 'nv_batch_size': BATCH_SIZE,
              'nv_max_length': MAX_LENGTH, 'window_seconds': args.concurrent_window_seconds,
              'llm_max_tokens': 2048, 'llm_enable_thinking': False,
              'before_requests': gpu_snapshot(torch), 'forward_passes': [], 'requests': []}
    report['concurrent_llm_health'] = result
    write_json(args.report, report)
    case = next(case for case in fixtures['cases']
                if case['name'] == 'native_2048_token_boundary_repeated4')
    texts = [record['text'] for record in case['records']]
    barrier = threading.Barrier(8)
    started = time.monotonic()
    forward_failure = None
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(request_llm, args, request, barrier) for request in requests]
        # Guarantee one intact batch4 forward. Continue while the same eight
        # requests are pending, capped at the requested 10-30 second window.
        while not result['forward_passes'] or (
                any(not future.done() for future in futures)
                and time.monotonic() - started < args.concurrent_window_seconds):
            reset_peaks(torch)
            forward = {'started_monotonic': time.monotonic(),
                       'pending_api_requests_at_start': sum(not future.done() for future in futures),
                       'before_forward': gpu_snapshot(torch), 'status': 'running'}
            result['forward_passes'].append(forward)
            try:
                forward['actual_encode_started_monotonic'] = time.monotonic()
                forward['pending_api_requests_at_encode_start'] = sum(not future.done() for future in futures)
                with torch.no_grad():
                    output = wrapper.embedding_model.encode(
                        prompts=texts, instruction='', max_length=MAX_LENGTH, num_workers=32)
                synchronize(torch)
                forward['actual_encode_finished_monotonic'] = time.monotonic()
                forward['after_forward_with_output'] = gpu_snapshot(torch)
                cpu_output, forward['embedding_check'] = validate_embedding(
                    torch, output, wrapper.embedding_config.norm)
                del output, cpu_output
                forward['status'] = 'passed'
            except Exception as error:
                forward['actual_encode_finished_monotonic'] = time.monotonic()
                forward['status'] = 'cuda_oom' if isinstance(error, torch.cuda.OutOfMemoryError) else 'failed'
                forward['error'] = error_details(error)
                forward['after_failed_forward'] = gpu_snapshot(torch)
                forward_failure = forward['error']
                break
            finally:
                forward['finished_monotonic'] = time.monotonic()
                forward['seconds'] = forward['finished_monotonic'] - forward['started_monotonic']
                write_json(args.report, report)
        result['requests'] = [future.result() for future in futures]
    result['forward_overlaps_api'] = [
        any(request['started_monotonic'] < forward['actual_encode_finished_monotonic']
            and request['finished_monotonic'] > forward['actual_encode_started_monotonic']
            for request in result['requests'])
        for forward in result['forward_passes']]
    result['seconds'] = time.monotonic() - started
    result['after_requests'] = gpu_snapshot(torch)
    result['status'] = 'passed' if (not forward_failure and
                                   all(request['status'] == 'passed' for request in result['requests'])
                                   and any(result['forward_overlaps_api'])) else 'failed'
    write_json(args.report, report)
    if forward_failure:
        if forward_failure['type'] == 'OutOfMemoryError':
            raise torch.cuda.OutOfMemoryError(forward_failure['message'])
        raise RuntimeError(forward_failure['message'])
    if result['status'] != 'passed':
        raise RuntimeError('Concurrent LLM health requires eight valid responses and an overlapping intact NV forward')
    LOG.info('concurrent API health passed requests=8 NV_forwards=%d overlaps=%s',
             len(result['forward_passes']), result['forward_overlaps_api'])


def run_probe(args, fixtures):
    import torch
    from pathcondrag.embedding_model.NVEmbedV2 import NVEmbedV2EmbeddingModel
    from pathcondrag.utils.config_utils import BaseConfig

    report = {'version': VERSION, 'started_at': utc_now(), 'status': 'running',
              'vllm_gpu_memory_utilization': args.utilization,
              'fixture_path': str(Path(args.fixtures).resolve()),
              'fixture_sha256': sha256_file(args.fixtures),
              'config': {'embedding_model': args.embedding_model, 'embedding_batch_size': BATCH_SIZE,
                         'embedding_max_seq_len': MAX_LENGTH, 'embedding_model_dtype': 'auto',
                         'embedding_return_as_normalized': True, 'num_workers': 32,
                         'encoder_use_cache': False, 'cuda_oom_splitting': False},
              'global_longest_original': fixtures['global_longest_original'],
              'dataset_statistics': {dataset: {
                  key: value for key, value in info.items() if key != 'top4_longest_passages'
              } for dataset, info in fixtures['datasets'].items()}, 'cases': []}
    write_json(args.report, report)
    wrapper = None
    comparison_reference = None
    status = 1
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('GPU probe requires CUDA; --prepare-only does not')
        reset_peaks(torch)
        report['before_model_loading'] = gpu_snapshot(torch)
        write_json(args.report, report)
        load_start = time.monotonic()
        config = BaseConfig(embedding_model_name=args.embedding_model,
                            embedding_batch_size=BATCH_SIZE, embedding_max_seq_len=MAX_LENGTH,
                            embedding_model_dtype='auto', embedding_return_as_normalized=True)
        wrapper = NVEmbedV2EmbeddingModel(global_config=config)
        wrapper.embedding_model.embedding_model.config.use_cache = False
        wrapper.embedding_model.eval()
        synchronize(torch)
        report['model_loading_seconds'] = time.monotonic() - load_start
        report['after_model_loading'] = gpu_snapshot(torch)
        report['model_parameter_dtypes'] = sorted({str(parameter.dtype)
                                                   for parameter in wrapper.embedding_model.parameters()})
        report['model_device_map'] = {str(key): str(value) for key, value in
                                     getattr(wrapper.embedding_model, 'hf_device_map', {}).items()}
        report['embedding_dim'] = wrapper.embedding_dim
        if wrapper.embedding_dim != 4096:
            raise ValueError(f'Expected NV dimension 4096, got {wrapper.embedding_dim}')
        write_json(args.report, report)
        for case in fixtures['cases']:
            texts = [record['text'] for record in case['records']]
            if len(texts) != BATCH_SIZE:
                raise ValueError(f'{case["name"]}: probe requires exactly four intact texts')
            for repeat in range(case['repetitions']):
                reset_peaks(torch)
                result = {'name': case['name'], 'kind': case['kind'], 'repeat': repeat + 1,
                          'batch_size': BATCH_SIZE, 'instruction': case['instruction'],
                          'inputs': [record_metadata(record) for record in case['records']],
                          'before_forward': gpu_snapshot(torch), 'status': 'running'}
                report['cases'].append(result)
                write_json(args.report, report)
                forward_start = time.monotonic()
                try:
                    # Deliberately call raw encode. There is no batch splitter,
                    # no inference cache, and no reduction of max_length.
                    with torch.no_grad():
                        output = wrapper.embedding_model.encode(
                            prompts=texts, instruction=case['instruction'],
                            max_length=MAX_LENGTH, num_workers=32)
                    synchronize(torch)
                    result['after_forward_with_output'] = gpu_snapshot(torch)
                    cpu_output, result['embedding_check'] = validate_embedding(
                        torch, output, wrapper.embedding_config.norm)
                    del output
                    synchronize(torch)
                    result['after_cpu_output_transfer'] = gpu_snapshot(torch)
                    if case['name'] == fixtures['batch_comparison_case']:
                        comparison_reference = cpu_output
                    result['status'] = 'passed'
                    del cpu_output
                except torch.cuda.OutOfMemoryError as error:
                    result['status'] = 'cuda_oom'
                    result['error'] = error_details(error)
                    result['after_failed_forward'] = gpu_snapshot(torch)
                    report['status'] = 'cuda_oom'
                    raise
                finally:
                    result['seconds'] = time.monotonic() - forward_start
                    write_json(args.report, report)
                LOG.info('probe %s repeat=%d passed max_allocated_bytes=%s', case['name'], repeat + 1,
                         [device['max_allocated_bytes'] for device in
                          result['after_forward_with_output']['torch_cuda_devices']])

        if args.llm_base_url:
            run_concurrent_llm_probe(args, fixtures, wrapper, torch, report)

        if comparison_reference is None:
            raise RuntimeError('No intact batch4 reference was recorded for comparison')
        case = next(case for case in fixtures['cases']
                    if case['name'] == fixtures['batch_comparison_case'])
        texts = [record['text'] for record in case['records']]
        comparison = {'name': case['name'], 'status': 'running',
                      'comparison': 'same intact texts batch4 versus two raw batch2 calls',
                      'bitwise_equality_required': False, 'microbatches': []}
        report['batch_size_comparison'] = comparison
        write_json(args.report, report)
        pieces = []
        for start in range(0, BATCH_SIZE, 2):
            reset_peaks(torch)
            microbatch = {'batch_size': 2, 'before_forward': gpu_snapshot(torch)}
            with torch.no_grad():
                output = wrapper.embedding_model.encode(
                    prompts=texts[start:start + 2], instruction=case['instruction'],
                    max_length=MAX_LENGTH, num_workers=32)
            synchronize(torch)
            microbatch['after_forward_with_output'] = gpu_snapshot(torch)
            if tuple(output.shape) != (2, 4096) or not torch.isfinite(output).all().item():
                raise ValueError('Comparison batch2 must return finite (2,4096) embeddings')
            pieces.append(output.detach().cpu())
            del output
            comparison['microbatches'].append(microbatch)
        split_output, comparison['embedding_check'] = validate_embedding(
            torch, torch.cat(pieces, dim=0), wrapper.embedding_config.norm)
        reference = comparison_reference.float()
        candidate = split_output.float()
        differences = (reference - candidate).abs()
        cosines = torch.nn.functional.cosine_similarity(reference, candidate, dim=1)
        comparison.update({'status': 'passed', 'cosine_similarity_per_row': cosines.tolist(),
                           'minimum_cosine_similarity': cosines.min().item(),
                           'maximum_absolute_difference': differences.max().item(),
                           'mean_absolute_difference': differences.mean().item(),
                           'bitwise_equal': bool(torch.equal(comparison_reference, split_output))})
        batch4_result = next(result for result in report['cases']
                             if result['name'] == fixtures['batch_comparison_case'])
        comparison['memory_comparison_per_device'] = []
        for device in batch4_result['after_forward_with_output']['torch_cuda_devices']:
            number = device['device']
            batch4_baseline = next(item for item in batch4_result['before_forward']['torch_cuda_devices']
                                   if item['device'] == number)
            batch2_peaks = [next(item for item in microbatch['after_forward_with_output']['torch_cuda_devices']
                                 if item['device'] == number)['max_allocated_bytes']
                            for microbatch in comparison['microbatches']]
            batch2_deltas = [
                next(item for item in microbatch['after_forward_with_output']['torch_cuda_devices']
                     if item['device'] == number)['max_allocated_bytes']
                - next(item for item in microbatch['before_forward']['torch_cuda_devices']
                       if item['device'] == number)['allocated_bytes']
                for microbatch in comparison['microbatches']]
            comparison['memory_comparison_per_device'].append({
                'device': number, 'batch4_peak_allocated_bytes': device['max_allocated_bytes'],
                'batch2_largest_peak_allocated_bytes': max(batch2_peaks),
                'peak_allocated_saving_bytes': device['max_allocated_bytes'] - max(batch2_peaks),
                'batch4_forward_increment_bytes': device['max_allocated_bytes'] - batch4_baseline['allocated_bytes'],
                'batch2_largest_forward_increment_bytes': max(batch2_deltas),
                'note': 'Allocated peaks are comparable; reserved peaks can retain earlier allocator cache.',
            })
        report['status'] = 'passed'
        status = 0
    except Exception as error:
        report['error'] = error_details(error)
        if isinstance(error, torch.cuda.OutOfMemoryError):
            report['status'] = 'cuda_oom'
        else:
            report['status'] = 'failed'
        LOG.error('probe failed status=%s error=%s', report['status'], report['error'])
    finally:
        # Clean up after leaving the exception scope so failed forward frames
        # cannot retain CUDA tensors into the next utilization trial.
        comparison_reference = None
        wrapper = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            report['after_model_release'] = gpu_snapshot(torch)
        report['finished_at'] = utc_now()
        write_json(args.report, report)
        LOG.info('probe report=%s status=%s', args.report, report['status'])
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets-dir', default='/root/datasets')
    parser.add_argument('--embedding-model', default='/root/models/NV-Embed-v2')
    parser.add_argument('--fixtures', required=True, help='JSON fixture file retaining full original text')
    parser.add_argument('--report', help='GPU JSON report (required unless --prepare-only)')
    parser.add_argument('--utilization', type=float, help='vLLM utilization of this externally managed trial')
    parser.add_argument('--llm-base-url', help='Optionally run eight real API requests overlapping intact NV forwards')
    parser.add_argument('--llm-tokenizer', default='/root/models/Qwen3-8B')
    parser.add_argument('--concurrent-window-seconds', type=float, default=20,
                        help='Run NV forwards while the eight API requests are pending (10 to 30 seconds)')
    parser.add_argument('--prepare-only', action='store_true', help='CPU tokenization only; never load model weights')
    args = parser.parse_args(argv)
    if not args.prepare_only and not args.report:
        parser.error('--report is required for a GPU probe')
    if not 10 <= args.concurrent_window_seconds <= 30:
        parser.error('--concurrent-window-seconds must be between 10 and 30')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.prepare_only or not Path(args.fixtures).is_file():
        fixtures = prepare_fixtures(args)
    else:
        fixtures = json.loads(Path(args.fixtures).read_text(encoding='utf-8'))
        if fixtures['version'] != VERSION or fixtures['max_length'] != MAX_LENGTH:
            raise ValueError('Fixture version/token limit differs from this probe')
        if fixtures['embedding_model'] != str(Path(args.embedding_model).resolve()):
            raise ValueError('Fixture tokenizer model differs from the GPU model')
    return 0 if args.prepare_only else run_probe(args, fixtures)
