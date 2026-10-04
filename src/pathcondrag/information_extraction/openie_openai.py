import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.openie_quality import (
    REPAIR_JSON_SCHEMA, TRIPLE_JSON_SCHEMA, extract_triple_payload, merge_triples,
    entity_argument_issues, normalize_repair_payload, support_quote_error_feedback, validate_triples,
)
from ..utils.misc_utils import TripleRawOutput, NerRawOutput
from ..llm.openai_gpt import CacheOpenAI, LLM_MAX_IN_FLIGHT

logger = get_logger(__name__)
_LENGTH_RETRY_FREQUENCY_PENALTIES = (0.2, 0.5)
_DEFAULT_QUALITY_MAX_RETRIES = 5


def _safe_env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    try:
        val = int(raw)
        if val <= 0:
            return default
        return val
    except Exception:
        return default


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_json_list_field(response: str, field_name: str) -> List:
    decoder = json.JSONDecoder()
    for start_index, character in enumerate(response):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(response[start_index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or field_name not in payload:
            continue
        value = payload[field_name]
        if not isinstance(value, list):
            raise ValueError(f"OpenIE response field {field_name!r} must be a list.")
        return value
    raise ValueError(f"OpenIE response does not contain a valid JSON object with {field_name!r}.")


def _extract_ner_from_response(real_response):
    return _extract_json_list_field(real_response, "named_entities")


def _base_generate_seed(llm_model: CacheOpenAI) -> int:
    config = getattr(llm_model, "llm_config", None)
    params = getattr(config, "generate_params", {}) or {}
    base_seed = params.get("seed")
    if base_seed is None:
        base_seed = 0
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise ValueError(f"OpenIE retry requires an integer or null seed, got {base_seed!r}")
    return base_seed


def _length_retry_seed(llm_model: CacheOpenAI) -> int:
    """Change the cache key once without changing the decoding settings."""
    return _base_generate_seed(llm_model) + 1


def _resolve_quality_max_retries(explicit):
    if explicit is not None:
        if explicit < 0:
            raise ValueError("quality_max_retries cannot be negative.")
        return explicit
    raw = os.environ.get("HIPPO_OPENIE_QUALITY_MAX_RETRIES", "").strip()
    if raw:
        value = int(raw)
        if value < 0:
            raise ValueError("HIPPO_OPENIE_QUALITY_MAX_RETRIES cannot be negative.")
        return value
    return _DEFAULT_QUALITY_MAX_RETRIES


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI, max_workers: int = 8, respect_env_workers: bool = True,
                 quality_max_retries=None, guided_recovery: bool = False):
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1.")
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        self.max_workers = max_workers
        self.respect_env_workers = respect_env_workers
        self.quality_max_retries = _resolve_quality_max_retries(quality_max_retries)
        # Optional vLLM 0.8.x guided_json is used only for error recovery. The
        # client continues to enforce enable_thinking=False for Qwen3.
        self.guided_recovery = guided_recovery

    def worker_limits(self) -> Tuple[int, int]:
        """Resolve NER/triple limits; explicit config takes priority over legacy env."""
        if not getattr(self, "respect_env_workers", True):
            return self.max_workers, self.max_workers
        openie_workers = _safe_env_int('HIPPO_OPENIE_MAX_WORKERS', self.max_workers)
        return (
            _safe_env_int('HIPPO_OPENIE_NER_WORKERS', openie_workers),
            _safe_env_int('HIPPO_OPENIE_TRIPLE_WORKERS', openie_workers),
        )

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        raw_response = ""
        metadata = {}
        # Align with HippoRAG: default 512, escalate to 1024 on parse failure or truncation.
        ner_max_tokens = getattr(self, 'ner_max_tokens', _safe_env_int('HIPPO_OPENIE_NER_MAX_TOKENS', 512))
        token_budgets = [ner_max_tokens]
        if ner_max_tokens < 1024:
            token_budgets.append(1024)
        final_budget = token_budgets[-1]
        attempts = [(budget, None) for budget in token_budgets]
        attempts.append((final_budget, "seed"))
        attempts.extend(
            (final_budget, penalty)
            for penalty in _LENGTH_RETRY_FREQUENCY_PENALTIES
        )
        length_observed_count = 0
        length_retry_count = 0
        length_retry_penalties_attempted = []
        openie_attempt_settings = []
        previous_was_length = False
        attempt_count = 0
        try:
            unique_entities = []
            for attempt, (max_new_tokens, retry_setting) in enumerate(attempts):
                # Decoding changes are reached only after the final budget was truncated.
                kwargs = {"messages": ner_input_message, "max_completion_tokens": max_new_tokens}
                if retry_setting == "seed":
                    kwargs["seed"] = _length_retry_seed(self.llm_model)
                elif retry_setting is not None:
                    kwargs["frequency_penalty"] = retry_setting
                    length_retry_penalties_attempted.append(retry_setting)
                if previous_was_length:
                    length_retry_count += 1
                attempt_count += 1
                openie_attempt_settings.append({key: value for key, value in kwargs.items()
                                                if key != "messages"})
                raw_response, response_metadata, cache_hit = self.llm_model.infer(**kwargs)
                metadata = dict(response_metadata)
                metadata.update({
                    'cache_hit': cache_hit,
                    'ner_max_tokens_used': max_new_tokens,
                    'openie_attempt_count': attempt_count,
                    'length_retry_count': length_retry_count,
                    'length_observed_count': length_observed_count,
                    'length_retry_penalties_attempted': list(length_retry_penalties_attempted),
                    'openie_attempt_settings': list(openie_attempt_settings),
                })
                if retry_setting == "seed":
                    metadata['length_retry_seed'] = kwargs['seed']
                elif retry_setting is not None:
                    metadata['length_retry_frequency_penalty'] = retry_setting
                if metadata.get('finish_reason') == 'length':
                    length_observed_count += 1
                    metadata['length_observed_count'] = length_observed_count
                    previous_was_length = True
                    if attempt + 1 == len(attempts):
                        raise RuntimeError(
                            f"NER chunk {chunk_key} remains truncated (finish_reason=length) "
                            f"after {attempt + 1} attempts"
                        )
                    logger.warning(
                        "NER response truncated for %s at max_new_tokens=%s; retrying",
                        chunk_key, max_new_tokens,
                    )
                    continue
                previous_was_length = False
                try:
                    extracted_entities = _extract_ner_from_response(raw_response)
                    unique_entities = list(dict.fromkeys(extracted_entities))
                    break
                except Exception as parse_error:
                    if attempt + 1 < len(token_budgets):
                        logger.warning(
                            "NER parse failed for %s with max_new_tokens=%s (%s); retrying with %s",
                            chunk_key,
                            max_new_tokens,
                            parse_error,
                            token_budgets[attempt + 1],
                        )
                        continue
                    raise

        except Exception as e:
            # For any other unexpected exceptions, log them and return with the error message
            logger.warning(e)
            metadata.update({
                'error': f'{type(e).__name__}: {e}',
                'openie_attempt_count': attempt_count,
                'length_retry_count': length_retry_count,
                'length_observed_count': length_observed_count,
                'length_retry_penalties_attempted': list(length_retry_penalties_attempted),
                'openie_attempt_settings': list(openie_attempt_settings),
            })
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,  # Store the error message in metadata
                unique_entities=[],
                metadata=metadata  # Store the error message in metadata
            )

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata
        )

    def _triple_messages(self, passage, named_entities, error=None, previous_response=None,
                         retained_triples=None, repair_context=''):
        messages = self.prompt_template_manager.render(
            name='triple_extraction', passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities}, ensure_ascii=False),
        )
        if repair_context:
            # Demonstrate semantic relations for every coordinated subject; a
            # list connector is not a relation and must not become a graph edge.
            example_source = (
                "Lena Vale and Tomas Reed are twins and are both children of Mira Vale and Owen Reed. "
                "Lena is their daughter and Tomas their son. Both characters are portrayed by Alex Rowan "
                "and appear in Film Two."
            )
            example_triples = [
                ['Lena Vale', 'portrayed by', 'Alex Rowan'],
                ['Tomas Reed', 'portrayed by', 'Alex Rowan'],
                ['Lena Vale', 'daughter of', 'Mira Vale'],
                ['Lena Vale', 'daughter of', 'Owen Reed'],
                ['Tomas Reed', 'son of', 'Mira Vale'],
                ['Tomas Reed', 'son of', 'Owen Reed'],
                ['Lena Vale', 'twin of', 'Tomas Reed'],
                ['Tomas Reed', 'twin of', 'Lena Vale'],
                ['Lena Vale', 'appears in', 'Film Two'],
                ['Tomas Reed', 'appears in', 'Film Two'],
            ]
            messages = [
                {'role': 'system', 'content': (
                    'Extract source-supported semantic relations for index repair. '
                    'Return one JSON object with triples (paired records) and status. '
                    'Every record has triple (exactly three non-empty strings: subject, predicate, object) '
                    'and support_quote (an exact minimal source cell/span from the original passage). '
                    'The whole original passage determines whether a relation is supported. The quote '
                    'only identifies copied source text; it need not contain the subject and predicate. '
                    'A predicate must express a real relationship, never a connector such as and/or. '
                    'Expand coordinated subjects into separate complete relations. '
                    'For explicitly shared parentage, every child relates to every stated joint parent; '
                    'do not pair children and parents by list position. Expand shared arguments only when '
                    'the source states that they are shared, never for separately assigned relations. '
                    'Defective triples locate facts but are not immutable: rewrite predicate and object '
                    'together when needed. Never fill a missing argument with an unrelated nearby name '
                    'or a following but/which clause. Empty quotes or omitted names are missing evidence, '
                    'not permission to invent an unspecified/unknown argument. '
                    'Preserve entity names as arguments, put relation words in predicates, resolve pronouns only when unambiguous, '
                    'and preserve important qualifiers. Do not invent facts or copy the demonstration entities.'
                )},
                {'role': 'user', 'content': 'Example passage:\n' + example_source},
                {'role': 'assistant', 'content': json.dumps({
                    'triples': [{'triple': triple, 'support_quote': triple[2]} for triple in example_triples],
                    'status': 'success',
                }, ensure_ascii=False)},
                {'role': 'user', 'content': (
                    'Small property-repair example: Dara had classical theater training before acting in '
                    'Feature Z. Repair ["Dara", "had classical theater training", ""].'
                )},
                {'role': 'assistant', 'content': json.dumps({'triples': [{
                    'triple': ['Dara', 'trained in', 'classical theater'],
                    'support_quote': 'Dara had classical theater training',
                }], 'status': 'success'})},
                {'role': 'user', 'content': (
                    'Small missing-name example: Tracks were reissued on CD as part of "" but dated '
                    'incorrectly. Repair only the missing release name after "as part of".'
                )},
                {'role': 'assistant', 'content': '{"triples":[],"status":"no_supported_relations"}'},
                {'role': 'user', 'content': (
                    'Actual original passage:\n' + passage + '\n'
                    + json.dumps({'named_entities': named_entities}, ensure_ascii=False)
                )},
            ]
        if error:
            # Change the actual task on recovery, rather than merely changing
            # a cache seed at temperature zero. The source remains in context.
            feedback = (
                "The previous extraction failed validation: " + str(error)[:3000] + "\n"
                + ('Return paired triple/support_quote records with status. ' if repair_context
                 else 'Return only {"triples": [["subject", "relation", "object"]]}. ')
                + "Use exactly three non-empty strings per triple. Re-extract from the "
                "original passage; never truncate four/five-field records or invent a missing field. "
                "Keep time/place qualifiers inside a relation or object string. "
                "Complete property relations using a meaningful object. "
                "Remove repetition and placeholders. Do not use outside knowledge.\n"
                "Previous output (diagnostic only, not factual evidence):\n"
                + (previous_response or "")[:1500]
            )
            if retained_triples:
                feedback += ("\nAlready accepted relations are preserved. Return the corrected "
                             "relations plus any supported missing relations; do not delete correct facts:\n"
                             + json.dumps(retained_triples[:10], ensure_ascii=False)[:800])
            messages.append({"role": "user", "content": feedback})
        if repair_context:
            messages.append({'role': 'user', 'content': (
                "Targeted index repair. The original passage above is the only factual source.\n"
                + repair_context[:6000] + "\n"
                "Follow the repair scope in the context: for an originally empty/failed result, extract all "
                "explicit relations from the original source, including each relation for each coordinated subject. "
                "For listed defective records, repair those; existing valid relations are retained externally. "
                'Return exactly {"triples": [{"triple": ["subject", "relation", "object"], '
                '"support_quote": "verbatim original-passage evidence"}], "status": "success"}. '
                'Each relation and its evidence must be one paired record; do not output parallel arrays. '
                'Use the smallest exact source cell/span as support_quote, such as a specific year or amount. '
                'The quote does not need to contain all triple arguments or the predicate; assess the actual '
                'relation against the whole original passage. For a table, copy only the relevant cell, '
                'never the entire year/amount column for every relation. Prefer compact exact cells over '
                'repeating rows, columns, headers, or full sentences. '
                'Never split a four/five-field relation into arbitrary groups of three. '
                'For example ["A", "and", "B", "live in", "C"] means two complete relations '
                '["A", "lives in", "C"] and ["B", "lives in", "C"], each with source evidence. '
                'Likewise "A and B are both portrayed by C" requires ["A", "portrayed by", "C"] '
                'and ["B", "portrayed by", "C"]. Do not omit the shared portrayed-by relation. '
                'For explicit joint parentage (such as both children of C and D, or C and D\'s children), '
                'each child has each joint parent: A child-of C, A child-of D, B child-of C, B child-of D. '
                'Preserve any explicitly stated son/daughter roles in these relations. '
                'This is not a general Cartesian expansion: if the passage assigns individual relations '
                'with respectively or other separate assignments, follow those assignments. '
                'For entity-to-entity facts keep the entity name as the object and put relation words '
                'in the predicate: ["X", "is", "child of", "Y"] becomes ["X", "child of", "Y"], '
                'not ["X", "is", "child of Y"]. '
                'For a property fact, rewrite predicate and object together using the source concept: '
                '"a person had classical theater training" can become ["person", "trained in", '
                '"classical theater"], never ["person", "had classical theater training", "a nearby film"]. '
                'When an essential argument is absent (empty quotes or a missing name), omit that unsupported '
                'relation; do not use "unknown", "unspecified", or a neighboring but/which clause as its object. '
                'If the requested repair only concerns such a missing argument, return no_supported_relations. '
                "Preserve qualifiers, and never infer a missing argument from outside knowledge. "
                "If none of the requested defective records corresponds to a supported relation in the source, "
                'return {"triples": [], "status": "no_supported_relations"}. '
                "Do not repeat already valid unrelated facts to hide failure to recover the requested records."
            )})
        return messages

    @staticmethod
    def _recovery_windows(passage, max_windows=8, window_chars=900, overlap_chars=120):
        """Bounded source windows; offsets refer to the unchanged original chunk.

        Titles are repeated as context. Windows overlap at whitespace boundaries
        so facts near a cut are not silently discarded. Huge passages that cannot
        fit the bound are left failed rather than silently dropping their tail.
        """
        newline = passage.find("\n")
        title = passage[:newline] if 0 < newline <= 200 else ""
        offset = newline + 1 if title else 0
        windows = []
        while offset < len(passage):
            end = min(len(passage), offset + window_chars)
            if end < len(passage):
                boundary = passage.rfind(" ", offset + window_chars // 2, end)
                if boundary > offset:
                    end = boundary
            text = passage[offset:end]
            windows.append((offset, end, (title + "\n" if title else "") + text))
            if end == len(passage):
                return windows
            if len(windows) >= max_windows:
                return []
            offset = max(offset + 1, end - overlap_chars)
            while offset > 0 and not passage[offset - 1].isspace():
                offset += 1
                if offset >= end:
                    break
        return windows

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str],
                          repair_context: str = '', _allow_window_recovery: bool = True,
                          _support_source: str | None = None) -> TripleRawOutput:
        raw_response, metadata, retained = "", {}, []
        triple_max_tokens = getattr(self, 'triple_max_tokens', _safe_env_int('HIPPO_OPENIE_TRIPLE_MAX_TOKENS', 2048))
        attempt_settings, recovery_history, penalties = [], [], []
        attempt_count = length_count = length_retry_count = 0
        last_error = None
        invalid_count = raw_count = 0
        total_rounds = 1 + self.quality_max_retries
        quality_idx = 0

        def diagnostics():
            metadata.update({
                'openie_attempt_count': attempt_count,
                'length_retry_count': length_retry_count,
                'length_observed_count': length_count,
                'length_retry_penalties_attempted': list(penalties),
                'openie_attempt_settings': list(attempt_settings),
                'quality_retry_count': quality_idx,
                'quality_attempt_index': quality_idx,
                'quality_max_retries': self.quality_max_retries,
                'raw_triple_count': raw_count,
                'valid_triple_count': len(retained),
                'invalid_triple_count': invalid_count,
                'recovery_history': list(recovery_history),
            })

        for quality_idx in range(total_rounds):
            try:
                for attempt in range(len(_LENGTH_RETRY_FREQUENCY_PENALTIES) + 2):
                    messages = self._triple_messages(
                        passage, named_entities, last_error,
                        raw_response if last_error else None, retained, repair_context,
                    )
                    kwargs = {'messages': messages, 'max_completion_tokens': triple_max_tokens}
                    if quality_idx or attempt:
                        kwargs['seed'] = _base_generate_seed(self.llm_model) + 100 + quality_idx * 10 + attempt
                    if attempt > 1:
                        penalty = _LENGTH_RETRY_FREQUENCY_PENALTIES[attempt - 2]
                        kwargs['frequency_penalty'] = penalty
                        penalties.append(penalty)
                    if attempt:
                        length_retry_count += 1
                    if (last_error or repair_context) and getattr(self, 'guided_recovery', False):
                        configured = getattr(getattr(self.llm_model, 'llm_config', None), 'generate_params', {}) or {}
                        kwargs['extra_body'] = dict(configured.get('extra_body') or {})
                        kwargs['extra_body']['guided_json'] = REPAIR_JSON_SCHEMA if repair_context else TRIPLE_JSON_SCHEMA
                        if getattr(self, 'bounded_structured_output', False):
                            from ..utils.openie_structured_output import guided_json_parameters
                            kwargs['extra_body'] = guided_json_parameters(
                                kwargs['extra_body']['guided_json'], kwargs['extra_body'])
                    attempt_count += 1
                    attempt_settings.append({key: value for key, value in kwargs.items() if key != 'messages'})
                    raw_response, response_metadata, cache_hit = self.llm_model.infer(**kwargs)
                    metadata = dict(response_metadata)
                    metadata['cache_hit'] = cache_hit
                    if 'seed' in kwargs:
                        metadata['length_retry_seed'] = kwargs['seed']
                    if 'frequency_penalty' in kwargs:
                        metadata['length_retry_frequency_penalty'] = kwargs['frequency_penalty']
                    if metadata.get('finish_reason') != 'length':
                        break
                    length_count += 1
                    last_error = RuntimeError('finish_reason=length: output was truncated; use complete, concise triples without duplicates')
                    recovery_history.append({'kind': 'length', 'attempt': attempt_count,
                                             'error': str(last_error), 'response': raw_response})
                    if (repair_context and _allow_window_recovery and len(passage) > 1000
                            and len(self._recovery_windows(passage)) > 1):
                        # A recoverable long table already exceeded the fixed
                        # output cap. Partition its full source immediately,
                        # avoiding three more whole-table decoding attempts.
                        metadata['window_early_fallback'] = True
                        raise last_error
                    if attempt == len(_LENGTH_RETRY_FREQUENCY_PENALTIES) + 1:
                        raise last_error
                if metadata.get('finish_reason') != 'stop':
                    raise RuntimeError(f"Triple extraction did not complete: {metadata.get('finish_reason')!r}")
                payload = extract_triple_payload(raw_response)
                repair_triples, quotes, status = (normalize_repair_payload(payload) if repair_context
                                                   else (payload['triples'], None, None))
                report = validate_triples(repair_triples)
                if repair_context:
                    argument_errors = entity_argument_issues(report.valid_triples, named_entities)
                    if argument_errors:
                        raise ValueError('; '.join(argument_errors[:8]))
                explicitly_empty = False
                if repair_context:
                    if status not in ('success', 'no_supported_relations'):
                        raise ValueError('Repair response needs an explicit success/no_supported_relations status')
                    if not isinstance(quotes, list) or len(quotes) != report.raw_count:
                        raise ValueError('Each paired repair record must have one support quote')
                    if status == 'no_supported_relations' and report.raw_count:
                        raise ValueError('no_supported_relations must have empty triples and support_quotes')
                    if status == 'success' and not report.raw_count:
                        raise ValueError('Empty repair must explicitly confirm no_supported_relations')
                    source_normalized = ' '.join((_support_source if _support_source is not None else passage).split())
                    quote_matches = [isinstance(quote, str) and bool(quote.strip())
                                     and ' '.join(quote.split()) in source_normalized for quote in quotes]
                    metadata['support_quotes'] = quotes
                    metadata['support_quote_matches'] = quote_matches
                    metadata['repair_status'] = status
                    metadata['support_check'] = 'source_quote_match_only_not_semantic_entailment'
                    if not all(quote_matches):
                        raise ValueError(support_quote_error_feedback(
                            quotes, repair_triples, quote_matches,
                            _support_source if _support_source is not None else passage,
                        ))
                    explicitly_empty = status == 'no_supported_relations'
                raw_count = report.raw_count
                if report.raw_count:
                    invalid_count = len(report.invalid_triples)
                retained = merge_triples(retained, report.valid_triples)
                if report.invalid_triples:
                    raise ValueError('; '.join(report.issues[:8]))
                # An empty initial result is legitimate. Empty recovery of known
                # invalid records does not prove that the source lacks relations.
                if raw_count == 0 and recovery_history and not explicitly_empty:
                    raise ValueError('Recovery returned no relations after a known extraction failure')
                diagnostics()
                metadata.update({'quality_status': 'success' if retained else 'empty_valid',
                                 'quality_recovered': bool(recovery_history or repair_context)})
                metadata.pop('error', None)
                metadata.pop('openie_skipped', None)
                return TripleRawOutput(chunk_id=chunk_key, response=raw_response,
                                       metadata=metadata, triples=retained)
            except Exception as error:
                last_error = error
                recovery_history.append({'kind': 'quality', 'attempt': attempt_count,
                                         'error': f'{type(error).__name__}: {error}',
                                         'response': raw_response})
                logger.warning('OpenIE quality failure for %s, round %s/%s: %s',
                               chunk_key, quality_idx, self.quality_max_retries, error)
                # Repeating whole-table generation after four truncations is
                # unproductive. Move directly to smaller source windows.
                if metadata.get('finish_reason') == 'length' and len(passage) > 1000:
                    break

        diagnostics()
        metadata['quality_status'] = 'partial' if retained else 'failed'
        metadata['openie_skip_reason'] = f'{type(last_error).__name__}: {last_error}'
        if _allow_window_recovery and len(passage) > 1000:
            windows = self._recovery_windows(passage)
            if len(windows) > 1:
                window_reports, responses = [], []
                for start, end, window_text in windows:
                    window_entities = [entity for entity in named_entities
                                       if isinstance(entity, str) and entity.casefold() in window_text.casefold()]
                    output = self.triple_extraction(chunk_key, window_text, window_entities,
                                                    repair_context=repair_context, _allow_window_recovery=False,
                                                    _support_source=passage)
                    retained = merge_triples(retained, output.triples)
                    window_reports.append({'source_start': start, 'source_end': end,
                                           'metadata': output.metadata})
                    responses.append(output.response)
                metadata['window_recovery'] = window_reports
                metadata['window_recovery_attempt_count'] = sum(
                    report['metadata'].get('openie_attempt_count', 0) for report in window_reports)
                metadata['valid_triple_count'] = len(retained)
                window_success = all(report['metadata'].get('quality_status') in ('success', 'empty_valid')
                                     for report in window_reports)
                metadata['quality_status'] = ('success' if retained and window_success
                                              else 'empty_valid' if window_success and repair_context
                                              else 'partial' if retained else 'failed')
                metadata['window_recovery_complete'] = bool(window_success and (retained or repair_context))
                if metadata['window_recovery_complete']:
                    metadata['quality_recovered'] = True
                    metadata['invalid_triple_count'] = 0
                    metadata.pop('openie_skip_reason', None)
                # Persist all raw window outputs without pretending they are a
                # single model answer. Accepted relations are stored separately.
                raw_response = json.dumps({'window_responses': responses}, ensure_ascii=False)
        if metadata['quality_status'] not in ('success', 'empty_valid'):
            metadata['openie_skipped'] = True
        metadata.pop('error', None)
        return TripleRawOutput(chunk_id=chunk_key, response=raw_response,
                               metadata=metadata, triples=retained)

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(chunk_key=chunk_key, passage=passage, named_entities=ner_output.unique_entities)
        return {"ner": ner_output, "triplets": triple_output}

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk 
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        ner_results_by_id: Dict[str, NerRawOutput] = {}
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        ner_max_workers, triple_max_workers = self.worker_limits()
        logger.info(
            "Online OpenIE concurrency: NER workers=%d, triple workers=%d, "
            "process HTTP ceiling=%d (PATHCONDRAG_LLM_MAX_IN_FLIGHT), legacy worker env=%s",
            ner_max_workers, triple_max_workers, LLM_MAX_IN_FLIGHT,
            "enabled" if getattr(self, "respect_env_workers", True) else "overridden by config",
        )

        with ThreadPoolExecutor(max_workers=ner_max_workers) as executor:
            # Create NER futures for each chunk
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                chunk_key = ner_futures[future]
                if result.chunk_id != chunk_key:
                    raise RuntimeError(f"NER returned chunk {result.chunk_id!r} for {chunk_key!r}")
                ner_results_by_id[chunk_key] = result
                # Update metrics based on the metadata from the result
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1

                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        failed_ner_chunk_ids = [
            chunk_key for chunk_key in chunk_passages
            if ner_results_by_id[chunk_key].metadata.get("error")
        ]
        if failed_ner_chunk_ids:
            raise RuntimeError(f"NER failed for {len(failed_ner_chunk_ids)} chunk(s): {failed_ner_chunk_ids}")

        triple_results_by_id: Dict[str, TripleRawOutput] = {}
        total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0
        with ThreadPoolExecutor(max_workers=triple_max_workers) as executor:
            # Create triple extraction futures for each chunk
            re_futures = {
                executor.submit(self.triple_extraction, chunk_key,
                                passage, ner_results_by_id[chunk_key].unique_entities): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }
            # Collect triple extraction results with progress bar
            pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples")
            for future in pbar:
                result = future.result()
                chunk_key = re_futures[future]
                if result.chunk_id != chunk_key:
                    raise RuntimeError(f"Triple extraction returned chunk {result.chunk_id!r} for {chunk_key!r}")
                triple_results_by_id[chunk_key] = result
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1
                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        failed_triple_chunk_ids = [
            chunk_key for chunk_key in chunk_passages
            if triple_results_by_id[chunk_key].metadata.get("error")
        ]
        if failed_triple_chunk_ids:
            raise RuntimeError(f"Triple extraction failed for {len(failed_triple_chunk_ids)} chunk(s): {failed_triple_chunk_ids}")
        skipped_triple_chunk_ids = [
            chunk_key for chunk_key in chunk_passages
            if triple_results_by_id[chunk_key].metadata.get("openie_skipped")
        ]
        if skipped_triple_chunk_ids:
            logger.warning(
                "Skipped triple extraction for %s chunk(s) after quality retries: %s",
                len(skipped_triple_chunk_ids),
                skipped_triple_chunk_ids,
            )

        ner_results_dict = {chunk_key: ner_results_by_id[chunk_key] for chunk_key in chunk_passages}
        triple_results_dict = {chunk_key: triple_results_by_id[chunk_key] for chunk_key in chunk_passages}

        return ner_results_dict, triple_results_dict
