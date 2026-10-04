"""Build a shared HippoRAG index with PathCondRAG's OpenIE quality contract.

The baseline checkout is imported read-only. Its graph schema, preprocessing,
embedding implementation, CLI, and native provenance identities are retained.
The changed extraction contract is recorded separately in ``quality_profile``;
this overlay is not a claim that the original v1 prompts were used unchanged.
"""

import argparse
import copy
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
from string import Template

from ..BaseRAG import BaseRAG
from ..information_extraction.source_verified_openie import SourceVerifiedOpenIE
from .openie_compact_recovery import RECOVERY_VERSION
from .openie_semantic_validation import VERIFIER_VERSION
from ..prompts.templates.triple_extraction import prompt_template
from .misc_utils import openie_row_needs_retry
from .openie_quality import TRIPLE_JSON_SCHEMA


QUALITY_SCHEMA = 'pathcondrag_openie_quality_v2'


def quality_profile():
    """Identity of the extra extraction contract, independent of native config."""
    digest = lambda value: hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')
    ).hexdigest()
    return {
        'schema_version': 2,
        'name': QUALITY_SCHEMA,
        'prompt_schema': 'pathcondrag_source_grounded_triples_v2',
        'prompt_sha256': digest([
            {'role': row['role'], 'content': row['content'].template
             if isinstance(row['content'], Template) else row['content']}
            for row in prompt_template
        ]),
        'triple_json_schema_sha256': digest(TRIPLE_JSON_SCHEMA),
        'extractor': 'pathcondrag.information_extraction.source_verified_openie.SourceVerifiedOpenIE',
        'validation': 'three_nonempty_unicode_strings_v2',
        'recovery': 'source_grounded_feedback_and_windows_v2',
        'fresh_recovery': RECOVERY_VERSION,
        'fresh_recovery_semantic_verifier': VERIFIER_VERSION,
        'semantic_scope': 'fresh_recovered_and_empty_only',
        'failure_policy': 'checkpoint_then_abort_before_graph_publication',
        'ner_max_tokens': 512,
        'triple_max_tokens': 2048,
        'native_identity_scope': 'baseline_configuration_compatibility',
    }


class SharedQualityOpenIE(SourceVerifiedOpenIE):
    """Accept the baseline constructor while preserving fixed token budgets."""

    def __init__(self, llm_model, max_workers=8, ner_max_tokens=512,
                 triple_max_tokens=2048, **kwargs):
        if ner_max_tokens != 512 or triple_max_tokens != 2048:
            raise ValueError('The shared quality builder requires NER=512 and triples=2048 tokens.')
        self.ner_max_tokens = ner_max_tokens
        self.triple_max_tokens = triple_max_tokens
        super().__init__(llm_model=llm_model, max_workers=max_workers,
                         respect_env_workers=False, **kwargs)

    def triple_extraction(self, *args, **kwargs):
        result = super().triple_extraction(*args, **kwargs)
        metadata = result.metadata
        if (metadata.get('window_recovery_complete')
                and metadata.get('quality_status') in ('success', 'empty_valid')):
            # The combined result completed in multiple requests, even if the
            # initial whole-passage request was truncated. Keep both facts.
            metadata.setdefault('whole_chunk_finish_reason', metadata.get('finish_reason'))
            metadata['finish_reason'] = 'stop'
            metadata['finish_source'] = 'all_windows_completed'
        return result


def quality_hipporag_class(native_class):
    """Create a runtime adapter without changing files in the baseline repo."""

    class QualitySharedHippoRAG(native_class):
        def add_synonymy_edges(self, query_node_keys=None):
            if os.environ.get('PATHCONDRAG_SHARED_KNN_DEVICE', '').strip().lower() != 'cuda':
                return super().add_synonymy_edges(query_node_keys)
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('PATHCONDRAG_SHARED_KNN_DEVICE=cuda requires an available CUDA device.')
            # Native index() encodes all chunk/entity/fact vectors before this
            # call. A build-only entry can release the encoder while KNN uses
            # the already stored vectors; no subsequent encoding is performed.
            if hasattr(self.embedding_model, 'model'):
                self.embedding_model.model = None
            gc.collect()
            torch.cuda.empty_cache()
            config = self.global_config
            previous_batches = (config.synonymy_edge_query_batch_size,
                                config.synonymy_edge_key_batch_size)
            previous_device = os.environ.get('HIPPORAG_KNN_DEVICE')
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            config.synonymy_edge_query_batch_size = 1000
            config.synonymy_edge_key_batch_size = 16384
            os.environ['HIPPORAG_KNN_DEVICE'] = 'cuda'
            torch.backends.cuda.matmul.allow_tf32 = False
            self._shared_knn_execution = {
                'device': 'cuda', 'dtype': 'float32', 'allow_tf32': False,
                'query_batch_size': 1000, 'key_batch_size': 16384,
                'embedding_model_released_after_encoding': True,
                'native_cosine_topk_and_threshold_unchanged': True,
            }
            try:
                result = super().add_synonymy_edges(query_node_keys)
                self._shared_knn_execution['complete'] = True
                if hasattr(self, 'working_dir'):
                    target = Path(self.working_dir) / 'shared_build_execution.json'
                    temporary = target.with_suffix('.json.tmp')
                    temporary.write_text(json.dumps(self._shared_knn_execution, indent=2), encoding='utf-8')
                    os.replace(temporary, target)
                return result
            finally:
                (config.synonymy_edge_query_batch_size,
                 config.synonymy_edge_key_batch_size) = previous_batches
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
                if previous_device is None:
                    os.environ.pop('HIPPORAG_KNN_DEVICE', None)
                else:
                    os.environ['HIPPORAG_KNN_DEVICE'] = previous_device

        def _current_openie_provenance(self):
            provenance = copy.deepcopy(super()._current_openie_provenance())
            provenance['quality_profile'] = quality_profile()
            return provenance

        def _validate_openie_provenance(self, provenance, source_path):
            validated = super()._validate_openie_provenance(provenance, source_path)
            if validated.get('quality_profile') != quality_profile():
                raise RuntimeError(f'OpenIE quality profile is incompatible: {source_path}')
            return validated

        def merge_openie_results(self, all_openie_info, chunks_to_save,
                                 ner_results_dict, triple_results_dict):
            expected = set(chunks_to_save)
            if set(ner_results_dict) != expected or set(triple_results_dict) != expected:
                raise RuntimeError('Incomplete or extraneous OpenIE stage results in shared index batch.')
            # The Path merge retains raw responses and diagnostics, including
            # failed results. Saving those first enables targeted recovery.
            merged = BaseRAG.merge_openie_results(
                self, all_openie_info, chunks_to_save, ner_results_dict, triple_results_dict,
            )
            incomplete = [row['idx'] for row in merged if openie_row_needs_retry(row)]
            for key in chunks_to_save:
                for stage, outputs in (('ner', ner_results_dict), ('triples', triple_results_dict)):
                    result = outputs[key]
                    if result.metadata.get('finish_reason') != 'stop' and key not in incomplete:
                        result.metadata['quality_status'] = 'failed'
                        result.metadata['error'] = f'Incomplete {stage} response: finish_reason != stop'
                        for row in merged:
                            if row['idx'] == key:
                                row['openie_metadata'][stage] = copy.deepcopy(result.metadata)
                        incomplete.append(key)
            self._openie_info = merged
            self._openie_provenance = self._current_openie_provenance()
            self._save_openie_state(merged)
            if incomplete:
                raise RuntimeError(
                    f'OpenIE incomplete for {len(set(incomplete))} chunks; '
                    f'diagnostics checkpoint saved at {self.openie_state_path}. '
                    'No completed graph was published.'
                )
            return merged

    QualitySharedHippoRAG.__name__ = 'QualitySharedHippoRAG'
    return QualitySharedHippoRAG


def run_shared_index_cli(argv=None, *, hippo_root=None):
    """Forward the baseline CLI in-process, restoring patched symbols on exit."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--eval_mode', default='rag_qa')
    parser.add_argument('--openie_mode', default='online')
    parser.add_argument('--rag_type', default='hipporag')
    parsed, _ = parser.parse_known_args(argv)
    if '--help' not in argv and '-h' not in argv:
        if parsed.eval_mode != 'index_only' or parsed.openie_mode != 'online' or parsed.rag_type != 'hipporag':
            raise ValueError('This entry requires --eval_mode index_only --openie_mode online --rag_type hipporag.')
    project_root = Path(__file__).resolve().parents[3]
    baseline_root = Path(hippo_root or os.environ.get('HIPPO_ROOT')
                         or project_root.parent / 'baseline' / 'HippoRAG').resolve()
    main_path = baseline_root / 'main.py'
    if not main_path.is_file():
        raise FileNotFoundError(f'Baseline main.py not found: {main_path}; set HIPPO_ROOT.')
    previous_path, previous_argv = list(sys.path), list(sys.argv)
    previous_bytecode = sys.dont_write_bytecode
    module = None
    original_class = original_openie = None
    try:
        # Imports must not create/update __pycache__ in the baseline checkout.
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(baseline_root / 'src'))
        module = importlib.import_module('hipporag.HippoRAG')
        original_class, original_openie = module.HippoRAG, module.OpenIE
        module.OpenIE = SharedQualityOpenIE
        module.HippoRAG = quality_hipporag_class(original_class)
        sys.argv = [str(main_path), *argv]
        return runpy.run_path(str(main_path), run_name='__main__')
    finally:
        if module is not None and original_class is not None:
            module.HippoRAG, module.OpenIE = original_class, original_openie
        sys.path[:] = previous_path
        sys.argv[:] = previous_argv
        sys.dont_write_bytecode = previous_bytecode
