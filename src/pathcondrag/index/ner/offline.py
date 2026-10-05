"""NER preparation and response parsing for offline batch backends."""

import json
from typing import Dict, List, Tuple

from ...utils.logging_utils import get_logger
from ...utils.misc_utils import NerRawOutput

logger = get_logger(__name__)


class OfflineNERMixin:
    """Run the shared offline NER stage without rewriting raw responses."""

    def _batch_ner(self, chunk_passages: Dict[str, str]) -> Tuple[List[str], Dict[str, NerRawOutput]]:
        messages = [self.prompt_template_manager.render(name='ner', passage=passage)
                    for passage in chunk_passages.values()]
        ner_output, _ = self.llm_model.batch_infer(messages, json_template='ner', max_tokens=512)

        chunk_ids = list(chunk_passages)
        ner_raw_outputs = []
        for index, response in enumerate(ner_output):
            chunk_id = chunk_ids[index]
            try:
                unique_entities = json.loads(response)["named_entities"]
            except Exception as error:
                unique_entities = []
                logger.warning(f"Could not parse response from OpenIE: {error}")
            if len(unique_entities) == 0:
                logger.warning("No entities extracted for chunk_id: {}".format(chunk_id))
            ner_raw_outputs.append(NerRawOutput(chunk_id, response, unique_entities, {}))

        ner_results = dict(zip(chunk_passages, ner_raw_outputs))
        return ner_output, ner_results
