import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import filter_invalid_triples
from ..utils.misc_utils import TripleRawOutput, NerRawOutput
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)
_LENGTH_RETRY_FREQUENCY_PENALTIES = (0.2, 0.5)


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


def _length_retry_seed(llm_model: CacheOpenAI) -> int:
    """Change the cache key once without changing the decoding settings."""
    config = getattr(llm_model, "llm_config", None)
    params = getattr(config, "generate_params", {}) or {}
    base_seed = params.get("seed")
    if base_seed is None:
        base_seed = 0
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise ValueError(f"OpenIE retry requires an integer or null seed, got {base_seed!r}")
    return base_seed + 1


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI, max_workers: int = 8):
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1.")
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        self.max_workers = max_workers

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        raw_response = ""
        metadata = {}
        # Align with HippoRAG: default 512, escalate to 1024 on parse failure or truncation.
        ner_max_tokens = _safe_env_int('HIPPO_OPENIE_NER_MAX_TOKENS', 512)
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

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str]) -> TripleRawOutput:
        def _extract_triples_from_response(real_response):
            return _extract_json_list_field(real_response, "triples")

        # PREPROCESSING
        messages = self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )

        raw_response = ""
        metadata = {}
        # Align with HippoRAG openie_triple_max_tokens default=2048
        triple_max_tokens = _safe_env_int('HIPPO_OPENIE_TRIPLE_MAX_TOKENS', 2048)
        length_observed_count = 0
        length_retry_count = 0
        length_retry_penalties_attempted = []
        openie_attempt_settings = []
        attempt_count = 0
        try:
            # LLM INFERENCE
            for attempt in range(len(_LENGTH_RETRY_FREQUENCY_PENALTIES) + 2):
                kwargs = {"messages": messages, "max_completion_tokens": triple_max_tokens}
                if attempt == 1:
                    kwargs['seed'] = _length_retry_seed(self.llm_model)
                elif attempt > 1:
                    penalty = _LENGTH_RETRY_FREQUENCY_PENALTIES[attempt - 2]
                    kwargs['frequency_penalty'] = penalty
                    length_retry_penalties_attempted.append(penalty)
                if attempt:
                    length_retry_count += 1
                attempt_count += 1
                openie_attempt_settings.append({key: value for key, value in kwargs.items()
                                                if key != "messages"})
                raw_response, response_metadata, cache_hit = self.llm_model.infer(**kwargs)
                metadata = dict(response_metadata)
                metadata.update({
                    'cache_hit': cache_hit,
                    'openie_attempt_count': attempt_count,
                    'length_retry_count': length_retry_count,
                    'length_observed_count': length_observed_count,
                    'length_retry_penalties_attempted': list(length_retry_penalties_attempted),
                    'openie_attempt_settings': list(openie_attempt_settings),
                })
                if attempt == 1:
                    metadata['length_retry_seed'] = kwargs['seed']
                elif attempt > 1:
                    metadata['length_retry_frequency_penalty'] = kwargs['frequency_penalty']
                if metadata.get('finish_reason') != 'length':
                    break
                length_observed_count += 1
                metadata['length_observed_count'] = length_observed_count
                if attempt == len(_LENGTH_RETRY_FREQUENCY_PENALTIES) + 1:
                    raise RuntimeError(
                        f"Triple extraction chunk {chunk_key} remains truncated "
                        f"(finish_reason=length) after {attempt + 1} attempts"
                    )
                logger.warning("Triple response truncated for %s; retrying at max_new_tokens=%s",
                               chunk_key, triple_max_tokens)
            extracted_triples = _extract_triples_from_response(raw_response)
            triplets = filter_invalid_triples(triples=extracted_triples)

        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key}: {e}")
            metadata.update({
                'error': f'{type(e).__name__}: {e}',
                'openie_attempt_count': attempt_count,
                'length_retry_count': length_retry_count,
                'length_observed_count': length_observed_count,
                'length_retry_penalties_attempted': list(length_retry_penalties_attempted),
                'openie_attempt_settings': list(openie_attempt_settings),
            })
            return TripleRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                triples=[]
            )

        # Success
        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triplets
        )

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

        openie_max_workers = _safe_env_int('HIPPO_OPENIE_MAX_WORKERS', self.max_workers)
        ner_max_workers = _safe_env_int('HIPPO_OPENIE_NER_WORKERS', openie_max_workers)
        triple_max_workers = _safe_env_int('HIPPO_OPENIE_TRIPLE_WORKERS', openie_max_workers)

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

        ner_results_dict = {chunk_key: ner_results_by_id[chunk_key] for chunk_key in chunk_passages}
        triple_results_dict = {chunk_key: triple_results_by_id[chunk_key] for chunk_key in chunk_passages}

        return ner_results_dict, triple_results_dict
