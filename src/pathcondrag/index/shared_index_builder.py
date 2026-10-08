"""Build a shared HippoRAG index with PathCondRAG's OpenIE quality contract.

The baseline checkout is imported read-only. Its graph schema, preprocessing,
embedding implementation, CLI, and native provenance identities are retained.
The changed extraction contract is recorded separately in ``quality_profile``;
this overlay is not a claim that the original v1 prompts were used unchanged.
"""

import argparse
from contextlib import nullcontext
import copy
import gc
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import runpy
import sys
from string import Template

from ..BaseRAG import BaseRAG
from .openie.source_verified_openie import SourceVerifiedOpenIE, SOURCE_VERIFIED_VERSION
from .openie.structural_openie import StructuralOpenIE, STRUCTURAL_VERSION, RECOVERY_BUDGET
from .openie.openie_compact_recovery import RECOVERY_VERSION
from .openie.openie_source_evidence import VERIFIER_VERSION
from .openie_checkpoint import OpenIECheckpoint
from .openie_build_queue import openie_row_is_verified_complete
from ..utils.misc_utils import openie_row_needs_retry
from .openie.openie_quality import TRIPLE_JSON_SCHEMA, validate_triples
from .publication_policy import (
    apply_publication_policy, parse_bool, STRICT_FAILURE_POLICY, LENIENT_FAILURE_POLICY,
)
from .extraction_utils import resolve_prompt_version


QUALITY_SCHEMA = 'pathcondrag_openie_quality_v3'
VALIDATION_MODES = ('source_verified', 'structural')
LOG = logging.getLogger(__name__)


def quality_profile(*, strict=True, prompt_version='optimized', legacy=False,
                    validation_mode='source_verified'):
    """Identity of the extra extraction contract, independent of native config."""
    digest = lambda value: hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')
    ).hexdigest()
    prompt_version = 'origin' if legacy else resolve_prompt_version(prompt_version)
    if prompt_version == 'origin':
        from ..prompts.templates.origin_triple_extraction_prompt import prompt_template as selected_triples
        from ..prompts.templates.origin_ner_prompt import prompt_template as selected_ner
    else:
        from ..prompts.templates.triple_extraction import prompt_template as selected_triples
        from ..prompts.templates.ner import prompt_template as selected_ner
    serialize = lambda template: [
        {'role': row['role'], 'content': row['content'].template
         if isinstance(row['content'], Template) else row['content']}
        for row in template
    ]
    if validation_mode not in VALIDATION_MODES:
        raise ValueError(f'Unknown OpenIE validation mode: {validation_mode}')
    if validation_mode == 'structural':
        if legacy:
            raise ValueError('Structural extraction has no legacy semantic-verification profile.')
        from .openie_build_queue import QUEUE_VERSION
        from .ner.source_verified import NER_RECOVERY_VERSION
        return {
            'schema_version': 1, 'name': 'pathcondrag_openie_structural_quality_v1',
            'extractor': 'pathcondrag.index.openie.structural_openie.StructuralOpenIE',
            'prompt_version': prompt_version,
            'prompt_schema': 'pathcondrag_source_grounded_triples_v2',
            'prompt_sha256': digest(serialize(selected_triples)),
            'ner_prompt_sha256': digest(serialize(selected_ner)),
            'ner_recovery': NER_RECOVERY_VERSION,
            'triple_json_schema_sha256': digest(TRIPLE_JSON_SCHEMA),
            'validation': 'three_nonempty_unicode_strings_v2',
            'validation_mode': 'structural', 'validation_scope': 'structural',
            'semantic_scope': 'none', 'semantic_verified': False,
            'structural_schema': STRUCTURAL_VERSION,
            'recovery': 'bounded_structural_format_retry_v1',
            'recovery_budget': RECOVERY_BUDGET,
            'recovery_budget_scope': 'triple_stage_new_profile_logical_infer_calls',
            'ner_max_tokens': 512, 'triple_max_tokens': 2048,
            'pending_queue': QUEUE_VERSION, 'stage_checkpoint': 'sqlite_per_stage_v1',
            'openie_strict': strict,
            'failure_policy': STRICT_FAILURE_POLICY if strict else LENIENT_FAILURE_POLICY,
            'native_identity_scope': 'baseline_configuration_compatibility',
        }
    result = {
        'schema_version': 2 if legacy else 3,
        'name': 'pathcondrag_openie_quality_v2' if legacy else QUALITY_SCHEMA,
        'prompt_schema': 'pathcondrag_source_grounded_triples_v2',
        'prompt_sha256': digest(serialize(selected_triples)),
        'triple_json_schema_sha256': digest(TRIPLE_JSON_SCHEMA),
        # Stable producer label shared with existing index manifests.
        'extractor': 'pathcondrag.information_extraction.source_verified_openie.SourceVerifiedOpenIE',
        'validation': 'three_nonempty_unicode_strings_v2',
        'recovery': 'source_grounded_feedback_and_windows_v2' if legacy else 'whole_context_source_units_v3',
        'fresh_recovery': RECOVERY_VERSION,
        'fresh_recovery_semantic_verifier': 'pathcondrag_source_only_entailment_v1' if legacy else VERIFIER_VERSION,
        'semantic_scope': 'fresh_recovered_and_empty_only' if legacy else 'all_final_relations',
        'failure_policy': STRICT_FAILURE_POLICY,
        'ner_max_tokens': 512,
        'triple_max_tokens': 2048,
        'native_identity_scope': 'baseline_configuration_compatibility',
    }
    if not legacy:
        from .ner.source_verified import NER_RECOVERY_VERSION
        result.update(prompt_version=prompt_version,
                      ner_prompt_sha256=digest(serialize(selected_ner)),
                      structured_initial_triples=True,
                      ner_recovery=NER_RECOVERY_VERSION,
                      retry_feedback='attempt_scoped_v1')
        from .openie import openie_source_evidence as evidence
        from .openie import openie_atomic_recovery as atomic
        from .openie_build_queue import QUEUE_VERSION
        from .openie.openie_structured_output import STRUCTURED_OUTPUT_VERSION
        result.update(source_verified_schema=SOURCE_VERIFIED_VERSION, atomic_recovery=atomic.ATOMIC_RECOVERY_VERSION,
                      pending_queue=QUEUE_VERSION, stage_checkpoint='sqlite_per_stage_v1')
        result.update(evidence_implementation=evidence.EVIDENCE_IMPLEMENTATION,
                      atomic_implementation=atomic.ATOMIC_RECOVERY_IMPLEMENTATION,
                      structured_output=STRUCTURED_OUTPUT_VERSION,
                      evidence_prompt_sha256=digest(evidence.SYSTEM),
                      atomic_prompt_sha256=digest(atomic.SYSTEM),
                      atomic_schema_sha256=digest(atomic.SCHEMA))
        if not strict:
            result.update(openie_strict=False, failure_policy=LENIENT_FAILURE_POLICY)
    return result


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


class SharedStructuralOpenIE(StructuralOpenIE):
    """Use the native constructor with a bounded, structural-only extractor."""

    def __init__(self, llm_model, max_workers=8, ner_max_tokens=512,
                 triple_max_tokens=2048, **kwargs):
        if ner_max_tokens != 512 or triple_max_tokens != 2048:
            raise ValueError('The shared quality builder requires NER=512 and triples=2048 tokens.')
        self.ner_max_tokens, self.triple_max_tokens = ner_max_tokens, triple_max_tokens
        super().__init__(llm_model=llm_model, max_workers=max_workers,
                         respect_env_workers=False, **kwargs)


def quality_hipporag_class(native_class, *, openie_strict=True, prompt_version='optimized',
                          validation_mode='source_verified'):
    """Create a runtime adapter without changing files in the baseline repo."""

    class QualitySharedHippoRAG(native_class):
        def _quality_profile(self, *, legacy=False):
            return quality_profile(strict=openie_strict, prompt_version=prompt_version, legacy=legacy,
                                   validation_mode=validation_mode)

        def _prepare_openie_progress(self, rows, chunks):
            progress = OpenIECheckpoint(Path(self.working_dir) / 'openie_progress.sqlite', {
                'provenance': self._current_openie_provenance(),
                'corpus_ids_sha256': hashlib.sha256('\n'.join(sorted(chunks)).encode()).hexdigest(),
            })
            replayed = progress.overlay(rows, chunks)
            rows[:] = replayed
            self.openie.initial_rows = {row['idx']: row for row in rows}
            self.openie.checkpoint = progress.save
            self._strict_openie_progress = progress
            return rows

        def _openie_provenance_for_manifest(self):
            if getattr(self, '_legacy_quality_manifest_probe', False):
                provenance = copy.deepcopy(self._current_openie_provenance())
                provenance['quality_profile'] = self._quality_profile(legacy=True)
                return provenance
            return super()._openie_provenance_for_manifest()

        def _validate_or_create_index_manifest(self):
            path = Path(self.index_manifest_path)
            allowed = os.environ.get('HIPPO_ALLOW_INDEX_RESUME', '').strip().lower() in {'1', 'true', 'yes', 'on'}
            if (validation_mode == 'source_verified' and allowed and path.is_file()
                    and self.global_config.force_index_from_scratch
                    and self.global_config.force_openie_from_scratch):
                stored = json.loads(path.read_text())
                old_profile = (stored.get('openie') or {}).get('quality_profile')
                if old_profile == self._quality_profile(legacy=True):
                    source_path = next((Path(candidate) for candidate in (
                        getattr(self, 'openie_state_path', None), getattr(self, 'openie_results_path', None))
                        if candidate and Path(candidate).is_file()), None)
                    if source_path is not None:
                        source_state = json.loads(source_path.read_text())
                        if source_state.get('docs') and source_state.get('provenance') != stored['openie']:
                            raise RuntimeError('Quality upgrade refused: source provenance differs from the native manifest.')
                    if (self.graph.vcount() or Path(self._graph_pickle_filename).exists()
                            or self.entity_embedding_store.get_all_ids() or self.fact_embedding_store.get_all_ids()):
                        raise RuntimeError('Quality upgrade requires an unpublished index without derived graph/entity/fact state.')
                    # Let the real native validator build and compare every
                    # embedding/component/graph/producer field. The sole
                    # temporary difference is its quality-contract field.
                    self._legacy_quality_manifest_probe = True
                    fresh_openie = self.global_config.force_openie_from_scratch
                    self.global_config.force_openie_from_scratch = False
                    try:
                        super()._validate_or_create_index_manifest()
                    finally:
                        self.global_config.force_openie_from_scratch = fresh_openie
                        self._legacy_quality_manifest_probe = False
                    current = copy.deepcopy(stored)
                    current['openie'] = self._current_openie_provenance()
                    backup = path.with_name('index_manifest.before_quality_upgrade.json')
                    if not backup.exists():
                        backup.write_text(json.dumps(stored, ensure_ascii=False, indent=2))
                    temporary = path.with_suffix('.json.tmp')
                    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2))
                    os.replace(temporary, path)
                    LOG.info('Upgrading unpublished index to all-relation source evidence checks; existing extraction remains historical until re-audited.')
            return super()._validate_or_create_index_manifest()

        def load_existing_openie(self, chunk_keys, force_reextract=False):
            """Resume only an explicitly authorized, unpublished checkpoint.

            Native ``index_only`` still requires both fresh-build flags. The
            resume permission therefore has to be interpreted here, before
            native ``force_reextract`` would discard successful checkpoint
            rows. A failed row remains pending even though its ID is present.
            """
            allowed = os.environ.get('HIPPO_ALLOW_INDEX_RESUME', '').strip().lower() in {
                '1', 'true', 'yes', 'on',
            }
            checkpoint = next((Path(path) for path in (
                getattr(self, 'openie_state_path', None),
                getattr(self, 'openie_results_path', None),
            ) if path and Path(path).is_file()), None)
            if not (allowed and force_reextract and checkpoint is not None):
                rows, pending = super().load_existing_openie(chunk_keys, force_reextract=force_reextract)
                if force_reextract and hasattr(self, 'openie'):
                    keys = list(chunk_keys)
                    rows = self._prepare_openie_progress(rows, self.chunk_embedding_store.get_all_id_to_rows())
                    by_key = {row['idx']: row for row in rows}
                    pending = [key for key in keys if key not in by_key
                               or not openie_row_is_verified_complete(self.openie, by_key[key])]
                    if not pending:
                        self._strict_openie_progress.close()
                        self.openie.checkpoint = None
                return rows, pending
            config = self.global_config
            if not (config.force_index_from_scratch and config.force_openie_from_scratch):
                raise RuntimeError('Checkpoint resume requires both explicit fresh-index flags.')
            graph_path = getattr(self, '_graph_pickle_filename', None)
            if (self.graph.vcount() or (graph_path and Path(graph_path).exists())
                    or self.entity_embedding_store.get_all_ids()
                    or self.fact_embedding_store.get_all_ids()):
                raise RuntimeError('Checkpoint resume refused: graph or entity/fact state already exists.')

            keys = list(chunk_keys)
            requested = set(keys)
            if len(requested) != len(keys):
                raise RuntimeError('Checkpoint resume received duplicate corpus chunk IDs.')
            # This invokes native identity/producer checks plus our unchanged
            # quality-profile check; no checkpoint provenance is bypassed.
            rows, _ = super().load_existing_openie(keys, force_reextract=False)
            if self._openie_provenance.get('producer') != self._current_openie_provenance()['producer']:
                raise RuntimeError('Checkpoint resume refused: extraction producer differs.')
            by_key = {row['idx']: row for row in rows}
            if len(by_key) != len(rows) or set(by_key).difference(requested):
                raise RuntimeError('Checkpoint resume refused: duplicate or out-of-corpus rows.')

            def incomplete(row):
                if openie_row_needs_retry(row):
                    return True
                metadata = row.get('openie_metadata') or {}
                if any((metadata.get(stage) or {}).get('finish_reason') != 'stop'
                       for stage in ('ner', 'triples')):
                    return True
                triples = row.get('extracted_triples')
                if not isinstance(triples, list) or validate_triples(triples).invalid_triples:
                    return True
                if hasattr(self, 'openie'):
                    if not openie_row_is_verified_complete(self.openie, row):
                        return True
                if not triples and validation_mode != 'structural':
                    stage = metadata.get('triples') or {}
                    if not (stage.get('source_no_supported_relations') is True
                            and stage.get('semantic_verified') is True
                            and stage.get('complete') is True):
                        return True
                return False

            missing = [key for key in keys if key not in by_key]
            failed = [key for key in keys if key in by_key and incomplete(by_key[key])]
            pending_set = set(missing).union(failed)
            pending = [key for key in keys if key in pending_set]
            if hasattr(self, 'openie'):
                chunk_rows = self.chunk_embedding_store.get_all_id_to_rows()
                self._prepare_openie_progress(rows, chunk_rows)
                by_key = {row['idx']: row for row in rows}
                pending = [key for key in keys if key not in by_key or incomplete(by_key[key])]
                missing = [key for key in keys if key not in by_key]
                failed = [key for key in keys if key in by_key and incomplete(by_key[key])]
            self._openie_resume_diagnostics = {
                'schema': 'pathcondrag_unpublished_checkpoint_resume_v1',
                'checkpoint': str(checkpoint),
                'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'requested_chunk_count': len(keys), 'checkpoint_row_count': len(rows),
                'retained_success_count': len(rows) - len(failed),
                'missing_chunk_ids': missing, 'failed_chunk_ids': failed,
                'pending_chunk_ids': pending, 'reextract_count': len(pending),
                'explicit_resume_permission': True,
                'completed_graph_present': False, 'derived_entity_fact_state_present': False,
                'quality_profile': self._quality_profile(),
            }
            target = Path(self.working_dir) / 'openie_resume_diagnostics.json'
            temporary = target.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(self._openie_resume_diagnostics, indent=2), encoding='utf-8')
            os.replace(temporary, target)
            LOG.info('Resuming unpublished OpenIE checkpoint: verified_current_contract=%d, missing=%d, '
                     'awaiting_current_verification=%d, pending_count=%d, first_pending=%s',
                     len(rows) - len(failed), len(missing), len(failed), len(pending), pending[:10])
            if not pending and hasattr(self, '_strict_openie_progress'):
                self._strict_openie_progress.close()
                self.openie.checkpoint = None
            return rows, pending

        def add_synonymy_edges(self, query_node_keys=None):
            requested_device = os.environ.get('PATHCONDRAG_SHARED_KNN_DEVICE', '').strip().lower()
            if requested_device not in {'cpu', 'cuda'}:
                return super().add_synonymy_edges(query_node_keys)
            import torch
            if requested_device == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('PATHCONDRAG_SHARED_KNN_DEVICE=cuda requires an available CUDA device.')
            # Native index() encodes all chunk/entity/fact vectors before this
            # call. A build-only entry can release the encoder while KNN uses
            # the already stored vectors; no subsequent encoding is performed.
            released_attributes = []
            for attribute in ('model', 'embedding_model'):
                if getattr(self.embedding_model, attribute, None) is not None:
                    setattr(self.embedding_model, attribute, None)
                    released_attributes.append(attribute)
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            config = self.global_config
            previous_batches = (config.synonymy_edge_query_batch_size,
                                config.synonymy_edge_key_batch_size)
            previous_device = os.environ.get('HIPPORAG_KNN_DEVICE')
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            if requested_device == 'cuda':
                config.synonymy_edge_query_batch_size = 1000
                config.synonymy_edge_key_batch_size = 16384
            os.environ['HIPPORAG_KNN_DEVICE'] = requested_device
            torch.backends.cuda.matmul.allow_tf32 = False
            self._shared_knn_execution = {
                'device': requested_device, 'dtype': 'float32', 'allow_tf32': False,
                'query_batch_size': config.synonymy_edge_query_batch_size,
                'key_batch_size': config.synonymy_edge_key_batch_size,
                'embedding_model_released_after_encoding': bool(released_attributes),
                'released_embedding_attributes': released_attributes,
                'native_cosine_topk_and_threshold_unchanged': True,
            }
            embedding_execution = getattr(self.embedding_model, '_nvembed_execution_stats', None)
            if embedding_execution is not None:
                self._shared_knn_execution['embedding_execution'] = copy.deepcopy(embedding_execution)
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
            provenance['quality_profile'] = self._quality_profile()
            return provenance

        def _validate_openie_provenance(self, provenance, source_path):
            validated = super()._validate_openie_provenance(provenance, source_path)
            profile = validated.get('quality_profile')
            authorized_upgrade = (validation_mode == 'source_verified'
                                  and os.environ.get('HIPPO_ALLOW_INDEX_RESUME', '').strip().lower()
                                  in {'1', 'true', 'yes', 'on'}
                                  and self.global_config.force_index_from_scratch
                                  and self.global_config.force_openie_from_scratch
                                  and profile == self._quality_profile(legacy=True))
            if profile != self._quality_profile() and not authorized_upgrade:
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
            if hasattr(self, 'openie'):
                for row in merged:
                    verified = openie_row_is_verified_complete(self.openie, row)
                    if not verified and row['idx'] not in incomplete:
                        incomplete.append(row['idx'])
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
            self._openie_publication_report = apply_publication_policy(
                merged, incomplete, strict=openie_strict,
                report_path=Path(self.openie_state_path).with_name('openie_publication_report.json'))
            self._openie_info = merged
            self._openie_provenance = self._current_openie_provenance()
            self._save_openie_state(merged)
            progress = getattr(self, '_strict_openie_progress', None)
            if progress is not None:
                progress.close()
                self.openie.checkpoint = None
            if incomplete and openie_strict:
                raise RuntimeError(
                    f'OpenIE incomplete for {len(set(incomplete))} chunks; '
                    f'diagnostics checkpoint saved at {self.openie_state_path}. '
                    'No completed graph was published.'
                )
            if incomplete:
                LOG.warning('openie_strict=False: skipped relations for %d failed chunks; '
                            'all passages remain indexed. First chunks: %s',
                            len(set(incomplete)), incomplete[:5])
            return merged

    QualitySharedHippoRAG.__name__ = 'QualitySharedHippoRAG'
    return QualitySharedHippoRAG


def run_shared_index_cli(argv=None, *, hippo_root=None):
    """Forward the baseline CLI in-process, restoring patched symbols on exit."""
    argv = list(sys.argv[1:] if argv is None else argv)
    path_parser = argparse.ArgumentParser(add_help=False)
    path_parser.add_argument('--openie_strict', type=parse_bool, default=True,
                             help='Abort on failed chunks (true); skip their relations and continue (false).')
    path_parser.add_argument('--openie_prompt_version', choices=('origin', 'optimized'), default='optimized')
    path_parser.add_argument('--openie_validation_mode', choices=VALIDATION_MODES, default='structural',
                             help='Local structural checks or independent source-verification LLM audits.')
    path_options, argv = path_parser.parse_known_args(argv)
    if '--help' in argv or '-h' in argv:
        path_parser.print_help()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--eval_mode', default='rag_qa')
    parser.add_argument('--openie_mode', default='online')
    parser.add_argument('--rag_type', default='hipporag')
    parser.add_argument('--embedding_name', default=os.environ.get(
        'HIPPO_EMBEDDING_MODEL_NAME', 'nvidia/NV-Embed-v2'))
    parser.add_argument('--embedding_provider', default=None)
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
        selected_openie = SharedStructuralOpenIE if path_options.openie_validation_mode == 'structural' else SharedQualityOpenIE
        module.OpenIE = selected_openie
        if path_options.openie_prompt_version != 'optimized':
            class ConfiguredOpenIE(selected_openie):
                def __init__(self, *args, **kwargs):
                    kwargs['prompt_version'] = path_options.openie_prompt_version
                    super().__init__(*args, **kwargs)
            module.OpenIE = ConfiguredOpenIE
        module.HippoRAG = quality_hipporag_class(
            original_class, openie_strict=path_options.openie_strict,
            prompt_version=path_options.openie_prompt_version,
            validation_mode=path_options.openie_validation_mode)
        sys.argv = [str(main_path), *argv]
        uses_nvembed = (parsed.embedding_provider == 'nvembed' or (
            parsed.embedding_provider is None and 'NV-Embed-v2' in parsed.embedding_name))
        embedding_context = nullcontext()
        if uses_nvembed:
            from ..embedding_model.nvembed_runtime import baseline_nvembed_runtime
            embedding_context = baseline_nvembed_runtime()
        with embedding_context:
            if not path_options.openie_strict or path_options.openie_validation_mode == 'structural':
                from .build_report import tolerant_index_build_report
                entry = runpy.run_path(str(main_path), run_name='__pathcondrag_shared_main__')
                # The report is owned by the baseline entry, not its imported class.
                # Replace only this invocation's globals; leave the checkout intact.
                entry['main'].__globals__['index_build_report'] = tolerant_index_build_report
                return entry['main']()
            return runpy.run_path(str(main_path), run_name='__main__')
    finally:
        if module is not None and original_class is not None:
            module.HippoRAG, module.OpenIE = original_class, original_openie
        sys.path[:] = previous_path
        sys.argv[:] = previous_argv
        sys.dont_write_bytecode = previous_bytecode
