"""Shared offline batch OpenIE orchestration and triple response parsing."""

import json
from typing import Dict, Tuple

from ..ner.offline import OfflineNERMixin
from .openie_openai import ChunkInfo, OpenIE
from ...utils.logging_utils import get_logger
from ...utils.misc_utils import NerRawOutput, TripleRawOutput

logger = get_logger(__name__)


class OfflineOpenIE(OfflineNERMixin, OpenIE):
    """Coordinate the unchanged offline NER and triple extraction requests."""

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}
        ner_output, ner_results = self._batch_ner(chunk_passages)
        messages = [self.prompt_template_manager.render(
            name='triple_extraction', passage=passage, named_entity_json=named_entities
        ) for passage, named_entities in zip(chunk_passages.values(), ner_output)]
        triple_output, _ = self.llm_model.batch_infer(
            messages, json_template='triples', max_tokens=2048)

        chunk_ids = list(chunks)
        triple_raw_outputs = []
        for index, response in enumerate(triple_output):
            chunk_id = chunk_ids[index]
            try:
                triples = json.loads(response)["triples"]
            except Exception as error:
                triples = []
                logger.warning(f"Could not parse response from OpenIE: {error}")
            if len(triples) == 0:
                logger.warning("No triples extracted for chunk_id: {}".format(chunk_id))
            triple_raw_outputs.append(TripleRawOutput(chunk_id, response, triples, {}))

        triple_results = dict(zip(chunks, triple_raw_outputs))
        return ner_results, triple_results
