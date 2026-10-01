import json
import difflib
from pydantic import BaseModel, Field, TypeAdapter
from openai import OpenAI
from copy import deepcopy
from typing import Union, Optional, List, Dict, Any, Tuple, Literal
import re
import ast
import logging
from .prompts.filter_default_prompt import best_dspy_prompt
from .utils.llm_utils import fix_broken_generated_json, robust_json_extract, filter_invalid_triples


logger = logging.getLogger(__name__)


class Fact(BaseModel):
    fact: list[list[str]] = Field(description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]")


class DSPyFilter:
    def __init__(self, rag_runtime):
        """
        Initializes the object with the necessary configurations and templates for processing input and output messages.

        Parameters:
        rag_runtime : An object that provides the global configuration and the LLM model required for inference.

        Attributes:
        dspy_file_path : The file path for reranking as specified in the global configuration.
        one_input_template : A string template for formatting the input message with placeholders for specific fields.
        one_output_template : A string template for formatting the output message with specific fields.
        message_template : A template generated using the specified dspy file path.
        llm_infer_fn : A function reference for making inferences using the provided LLM model.
        model_name : The name of the language model as specified in the global configuration.
        default_gen_kwargs : A dictionary for storing the default generation keyword arguments.
        """
        dspy_file_path = rag_runtime.global_config.rerank_dspy_file_path
        self.one_input_template = """[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\nRespond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), and then ending with the marker for `[[ ## completed ## ]]`."""
        self.one_output_template = """[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]"""
        self.message_template = self.make_template(dspy_file_path)
        self.llm_infer_fn = rag_runtime.llm_model.infer
        self.model_name = rag_runtime.global_config.llm_name
        self.default_gen_kwargs = {}

    def make_template(self, dspy_file_path):
        if dspy_file_path is not None:
            dspy_saved = json.load(open(dspy_file_path, 'r'))
        else:
            dspy_saved = best_dspy_prompt

        system_prompt = dspy_saved['prog']['system']
        message_template = [
            {"role": "system", "content": system_prompt},
        ]
        demos = dspy_saved["prog"]["demos"]
        for demo in demos:
            message_template.append({"role": "user", "content": self.one_input_template.format(question=demo["question"], fact_before_filter=demo["fact_before_filter"])})
            message_template.append({"role": "assistant", "content": self.one_output_template.format(fact_after_filter=demo["fact_after_filter"])})
        return message_template

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json|JSON)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?\s*```\s*$", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def _try_load_json_like(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str):
            return value

        stripped = value.strip()
        if not stripped:
            return stripped

        try:
            return json.loads(stripped)
        except Exception:
            pass

        try:
            return ast.literal_eval(stripped)
        except Exception:
            return value

    def _parse_fact_payload(self, raw_value: str) -> Tuple[Dict[str, Any], str]:
        """Robustly parse fact payload from LLM output section."""
        cleaned = self._strip_code_fence(raw_value)
        candidates = [cleaned]

        # Add repaired JSON candidate for truncated outputs.
        repaired = fix_broken_generated_json(cleaned)
        if repaired != cleaned:
            candidates.append(repaired)

        for idx, candidate in enumerate(candidates):
            parsed_value = self._try_load_json_like(candidate)

            # Some models output JSON as a quoted string; unwrap recursively.
            for _ in range(3):
                if isinstance(parsed_value, str):
                    unwrapped = self._try_load_json_like(parsed_value)
                    if unwrapped is parsed_value:
                        break
                    parsed_value = unwrapped
                else:
                    break

            if isinstance(parsed_value, dict):
                fact_value = parsed_value.get("fact", [])
                # Handle nested string-encoded fact field.
                if isinstance(fact_value, str):
                    nested = self._try_load_json_like(fact_value)
                    if isinstance(nested, dict) and "fact" in nested:
                        parsed_value = nested
                    elif isinstance(nested, list):
                        parsed_value = {"fact": nested}
                if "fact" in parsed_value:
                    reason = "ok_direct" if idx == 0 else "ok_repaired"
                    return parsed_value, reason

            if isinstance(parsed_value, list):
                reason = "ok_list" if idx == 0 else "ok_list_repaired"
                return {"fact": parsed_value}, reason

        # Final fallback: extract fact list from noisy response text.
        extracted = robust_json_extract(cleaned, "fact")
        if isinstance(extracted, list):
            return {"fact": extracted}, "ok_robust_extract"

        return {"fact": []}, "unparseable"

    def _extract_sections(self, response: str) -> Dict[str, str]:
        sections = [(None, [])]
        field_header_pattern = re.compile('\\[\\[ ## (\\w+) ## \\]\\]')
        for line in response.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)
        return {k: "\n".join(v).strip() for k, v in sections if k is not None}

    def parse_filter(self, response, return_diagnostics: bool = False):
        sections = [(None, [])]
        field_header_pattern = re.compile('\\[\\[ ## (\\w+) ## \\]\\]')
        for line in response.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]
        parsed = []
        diagnostics = {
            "has_fact_after_filter_field": False,
            "payload_parse_reason": None,
            "fact_items_raw_count": 0,
            "fact_items_valid_count": 0,
            "parsed_count": 0,
            "no_facts_reason": None,
            "error": None,
        }
        for k, value in sections:
            if k == "fact_after_filter":
                diagnostics["has_fact_after_filter_field"] = True
                try:
                    parsed_payload, payload_reason = self._parse_fact_payload(value)
                    diagnostics["payload_parse_reason"] = payload_reason
                    fact_items = parsed_payload.get("fact", [])
                    diagnostics["fact_items_raw_count"] = len(fact_items) if isinstance(fact_items, list) else 0
                    fact_items = filter_invalid_triples(fact_items if isinstance(fact_items, list) else [])
                    diagnostics["fact_items_valid_count"] = len(fact_items)
                    parsed = TypeAdapter(Fact).validate_python({"fact": fact_items}).fact
                    diagnostics["parsed_count"] = len(parsed)
                except Exception as e:
                    diagnostics["error"] = str(e)
                    logger.error(
                        "Error parsing field %s: %s. Raw value: %s",
                        k,
                        e,
                        value[:512],
                    )

        if diagnostics["has_fact_after_filter_field"] is False:
            diagnostics["no_facts_reason"] = "missing_fact_after_filter_field"
        elif diagnostics["parsed_count"] == 0:
            if diagnostics["payload_parse_reason"] == "unparseable":
                diagnostics["no_facts_reason"] = "parse_unparseable"
            elif diagnostics["fact_items_raw_count"] == 0:
                diagnostics["no_facts_reason"] = "explicit_empty_fact_list"
            elif diagnostics["fact_items_valid_count"] == 0:
                diagnostics["no_facts_reason"] = "all_invalid_triples_filtered"
            else:
                diagnostics["no_facts_reason"] = "parse_exception_or_validation_failed"

        if return_diagnostics:
            return parsed, diagnostics
        return parsed

    def llm_call(self, question, fact_before_filter):
        # make prompt
        messages = deepcopy(self.message_template)
        messages.append({"role": "user", "content": self.one_input_template.format(question=question, fact_before_filter=fact_before_filter)})
        # call openai

        # Independent query workers share this filter instance. Build request
        # kwargs locally so a call cannot mutate another call's defaults.
        generation_kwargs = deepcopy(self.default_gen_kwargs)
        generation_kwargs['max_completion_tokens'] = 512

        response = self.llm_infer_fn(
            messages=messages,
            model=self.model_name,
            **generation_kwargs
        )
        response_text = response
        metadata = {}
        cache_hit = None

        if isinstance(response, tuple):
            if len(response) >= 1:
                response_text = response[0]
            if len(response) >= 2 and isinstance(response[1], dict):
                metadata = response[1]
            if len(response) >= 3 and isinstance(response[2], bool):
                cache_hit = response[2]
        elif isinstance(response, list):
            if len(response) > 0:
                response_text = response[0]
        if response_text is None:
            response_text = ""
        if not isinstance(response_text, str):
            response_text = str(response_text)

        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        metadata["cache_hit"] = cache_hit
        return response_text, metadata

    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)

    def rerank(self,
               query: str,
               candidate_items: List[Tuple],
               candidate_indices: List[int],
               len_after_rerank: int =None) -> Tuple[List[int], List[Tuple], dict]:
        rerank_info = {
            "candidate_count": len(candidate_items),
            "generated_facts_count": 0,
            "matched_facts_count": 0,
            "selected_facts_count": 0,
            "cache_hit": None,
            "llm_finish_reason": None,
            "llm_prompt_tokens": None,
            "llm_completion_tokens": None,
            "response_chars": 0,
            "parse_diagnostics": {},
            "no_facts_reason": None,
            "exception": None,
        }
        fact_before_filter = {"fact": [list(candidate_item) for candidate_item in candidate_items]}
        # Terminal transport/provider failures must abort the run. Only semantic
        # parsing failures below may produce the original empty-facts fallback.
        response, llm_meta = self.llm_call(query, json.dumps(fact_before_filter))
        rerank_info["response"] = response
        rerank_info["llm_metadata"] = llm_meta
        rerank_info["response_chars"] = len(response)
        rerank_info["cache_hit"] = llm_meta.get("cache_hit")
        rerank_info["llm_finish_reason"] = llm_meta.get("finish_reason")
        rerank_info["llm_prompt_tokens"] = llm_meta.get("prompt_tokens")
        rerank_info["llm_completion_tokens"] = llm_meta.get("completion_tokens")
        try:
            generated_facts, parse_diag = self.parse_filter(response, return_diagnostics=True)
            rerank_info["parse_diagnostics"] = parse_diag
            rerank_info["generated_facts_count"] = len(generated_facts)
        except Exception as e:
            rerank_info["exception"] = str(e)
            logger.error("rerank exception: %s", e)
            generated_facts = []
            rerank_info["no_facts_reason"] = "parse_exception"

        rerank_info["generated_facts"] = generated_facts

        result_indices = []
        for generated_fact in generated_facts:
            matched = difflib.get_close_matches(str(generated_fact), [str(i) for i in candidate_items], n=1, cutoff=0.0)
            if not matched:
                continue
            closest_matched_fact = matched[0]
            try:
                result_indices.append(candidate_items.index(eval(closest_matched_fact)))
            except Exception as e:
                logger.warning("result_indices exception: %s", e)

        # de-dup indices while preserving order
        dedup_indices = []
        seen = set()
        for idx in result_indices:
            if idx not in seen:
                dedup_indices.append(idx)
                seen.add(idx)
        result_indices = dedup_indices
        rerank_info["matched_facts_count"] = len(result_indices)

        sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
        sorted_candidate_items = [candidate_items[i] for i in result_indices]
        final_indices = sorted_candidate_indices[:len_after_rerank]
        final_items = sorted_candidate_items[:len_after_rerank]
        rerank_info["selected_facts_count"] = len(final_items)

        if len(final_items) == 0:
            if rerank_info["no_facts_reason"] is None:
                parse_diag = rerank_info.get("parse_diagnostics", {})
                if isinstance(parse_diag, dict) and parse_diag.get("no_facts_reason"):
                    rerank_info["no_facts_reason"] = parse_diag["no_facts_reason"]
                elif rerank_info.get("generated_facts_count", 0) > 0 and rerank_info.get("matched_facts_count", 0) == 0:
                    rerank_info["no_facts_reason"] = "no_candidate_alignment_after_rerank"
                else:
                    rerank_info["no_facts_reason"] = "empty_after_rerank"
        else:
            rerank_info["no_facts_reason"] = None

        return final_indices, final_items, rerank_info
