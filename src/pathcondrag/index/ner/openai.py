"""Named-entity extraction used by the OpenAI-compatible index extractor."""

from ...utils.logging_utils import get_logger
from ...utils.misc_utils import NerRawOutput
from ..extraction_utils import (
    _LENGTH_RETRY_FREQUENCY_PENALTIES, _extract_ner_from_response,
    _length_retry_seed, _safe_env_int,
)

logger = get_logger(__name__)


class OpenAINERMixin:
    """Keep ordinary NER request and recovery settings together."""

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
