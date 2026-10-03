from argparse import ArgumentTypeError
from dataclasses import dataclass
from hashlib import md5
from typing import Dict, Any, List, Tuple, Literal, Union, Optional
from copy import deepcopy
import numpy as np
import re
import logging

from .typing import Triple
from .openie_quality import validate_triples

logger = logging.getLogger(__name__)

@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]

@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal['node', 'dpr']

@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    doc_scores: np.ndarray = None
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None
    retrieval_trace: Optional[Dict[str, Any]] = None


    def to_dict(self, top_k: int = 10):
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:top_k],
            "doc_scores": [float(v) for v in self.doc_scores[:top_k]] if self.doc_scores is not None else None,
            "gold_docs": self.gold_docs,
            "retrieval_trace": self.retrieval_trace or {},
        }

def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()


def unicode_text_processing(text):
    """Match indexes declaring HippoRAG's unicode_alnum_casefold_v1 schema."""
    if isinstance(text, list):
        return [unicode_text_processing(item) for item in text]
    if not isinstance(text, str):
        text = str(text)
    normalized = ''.join(character if character.isalnum() or character.isspace() else ' '
                         for character in text.casefold())
    return ' '.join(normalized.split())

def reformat_openie_results(corpus_openie_results) -> (Dict[str, NerRawOutput], Dict[str, TripleRawOutput]):
    """Restore cached outputs without discarding extraction diagnostics.

    Older HippoRAG rows do not contain responses or metadata.  Their triples are
    still checked strictly; malformed cached relations must not become graph
    nodes merely because their container happens to have length three.
    """
    ner_output_dict, triple_output_dict = {}, {}
    for chunk_item in corpus_openie_results:
        chunk_id = chunk_item['idx']
        metadata = chunk_item.get('openie_metadata') or {}
        responses = chunk_item.get('openie_responses') or {}
        validation = validate_triples(chunk_item.get('extracted_triples', []))
        triple_metadata = deepcopy(metadata.get('triples') or {})
        if validation.invalid_triples:
            triple_metadata['cached_validation'] = {
                'invalid_triple_count': len(validation.invalid_triples),
                'issues': validation.issues,
            }
        ner_output_dict[chunk_id] = NerRawOutput(
            chunk_id=chunk_id,
            response=responses.get('ner'),
            metadata=deepcopy(metadata.get('ner') or {}),
            unique_entities=list(np.unique(chunk_item.get('extracted_entities', []))),
        )
        triple_output_dict[chunk_id] = TripleRawOutput(
            chunk_id=chunk_id,
            response=responses.get('triples'),
            metadata=triple_metadata,
            triples=validation.valid_triples,
        )
    return ner_output_dict, triple_output_dict


def normalize_graph_triples(triples: List[List[str]], *, normalizer=text_processing) -> List[List[str]]:
    """Use the existing graph normalization while rejecting empty results.

    The caller chooses the index manifest's normalizer. Keeping the legacy
    default preserves old entity/fact IDs. Non-ASCII text can disappear under
    that historical rule, so normalized fields are checked again.
    """
    normalized = []
    for triple in validate_triples(triples).valid_triples:
        processed = normalizer(triple)
        if all(field.strip() for field in processed):
            normalized.append(processed)
    return normalized


def openie_row_needs_retry(row: dict) -> bool:
    """Retry explicit extraction failures, not every legitimate empty result."""
    metadata = row.get('openie_metadata') or {}
    for stage in ('ner', 'triples'):
        stage_metadata = metadata.get(stage) or {}
        if (stage_metadata.get('error') or stage_metadata.get('openie_skipped')
                or stage_metadata.get('quality_status') in ('failed', 'partial')):
            return True
    return False

def extract_entity_nodes(chunk_triples: List[List[Triple]]) -> (List[str], List[List[str]]):
    chunk_triple_entities = []  # a list of lists of unique entities from each chunk's triples
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
            else:
                logger.warning(f"During graph construction, invalid triple is found: {t}")
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities

def flatten_facts(chunk_triples: List[Triple]) -> List[Triple]:
    graph_triples = []  # a list of unique relation triple (in tuple) from all chunks
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    graph_triples = list(set(graph_triples))
    return graph_triples

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()


def all_values_of_same_length(data: dict) -> bool:
    """
    Return True if all values in 'data' have the same length or data is an empty dict,
    otherwise return False.
    """
    # Get an iterator over the dictionary's values
    value_iter = iter(data.values())

    # Get the length of the first sequence (handle empty dict case safely)
    try:
        first_length = len(next(value_iter))
    except StopIteration:
        # If the dictionary is empty, treat it as all having "the same length"
        return True

    # Check that every remaining sequence has this same length
    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )
