"""Strict, deterministic OpenIE validation shared by extraction and indexing.

This module never guesses a relation from a malformed four/five-field record.
Such records need a new, source-grounded extraction rather than truncation.
"""

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, List

_COORDINATING_PREDICATES = {'and', 'or', '&', 'as well as', 'and/or'}
_ENTITY_RELATION_PREFIXES = (
    'son of', 'daughter of', 'child of', 'parent of', 'father of', 'mother of',
    'husband of', 'wife of', 'twin of', 'portrayed by', 'played by',
)


@dataclass
class TripleValidation:
    valid_triples: List[List[str]]
    invalid_triples: List[Any]
    issues: List[str]
    raw_count: int


def has_semantic_text(value: Any) -> bool:
    """Keep Unicode letters/numbers; reject empty and punctuation-only fields."""
    return isinstance(value, str) and any(character.isalnum() for character in value)


def validate_triples(triples: Any) -> TripleValidation:
    if not isinstance(triples, list):
        raise ValueError("The triples field must be a JSON array.")
    valid, invalid, issues, seen = [], [], [], set()
    for index, triple in enumerate(triples):
        issue = None
        if not isinstance(triple, (list, tuple)):
            issue = "must be an array, not a string or object"
        elif len(triple) != 3:
            issue = f"has {len(triple)} fields; exactly three are required"
        elif not all(isinstance(field, str) for field in triple):
            issue = "contains a non-string field"
        elif not all(has_semantic_text(field) for field in triple):
            issue = "contains an empty or punctuation-only field"
        elif ' '.join(triple[1].strip(' \t\n.,;:').casefold().split()) in _COORDINATING_PREDICATES:
            issue = "uses a coordinating word as predicate; express the actual source relation for each subject"
        if issue:
            invalid.append(triple)
            issues.append(f"triples[{index}] {issue}")
            continue
        cleaned = [field.strip() for field in triple]
        key = tuple(cleaned)
        if key not in seen:
            seen.add(key)
            valid.append(cleaned)
    return TripleValidation(valid, invalid, issues, len(triples))


def merge_triples(*groups: List[List[str]]) -> List[List[str]]:
    """Deduplicate only strictly valid triples while preserving source order."""
    return validate_triples([triple for group in groups for triple in group]).valid_triples


def entity_argument_issues(triples: List[List[str]], named_entities: List[str]) -> List[str]:
    """Detect a known entity hidden inside a relational object during repair.

    Only exact known-entity suffixes are checked. This does not require every
    named entity to participate in a triple or reject ordinary property values.
    """
    known = {entity.strip().casefold() for entity in named_entities if isinstance(entity, str)}
    issues = []
    for index, triple in enumerate(triples):
        if triple[1].strip().casefold() not in {'is', 'was', 'are', 'were', 'be'}:
            continue
        obj = triple[2].strip().casefold()
        for relation in _ENTITY_RELATION_PREFIXES:
            prefix = relation + ' '
            if obj.startswith(prefix) and obj[len(prefix):].strip() in known:
                issues.append(f'triples[{index}] hides a known entity inside the object; '
                              f'use {relation!r} as predicate and the entity itself as object')
                break
    return issues


def support_quote_error_feedback(quotes, triples, matches, source: str) -> str:
    """Describe exact quote mismatches without replacing or approving evidence.

    Nearby sentences are source substrings offered to the model as diagnostics.
    They are not automatically assigned as evidence for a particular relation.
    """
    spans = list(dict.fromkeys(span.strip() for span in re.split(r'(?<=[.!?])\s+|\n+', source)
                               if span.strip()))
    grouped = {}
    for index, match in enumerate(matches):
        if match:
            continue
        quote = quotes[index]
        key = json.dumps(quote, ensure_ascii=False)
        grouped.setdefault(key, {'quote': quote, 'indices': [], 'triples': []})
        grouped[key]['indices'].append(index)
        grouped[key]['triples'].append(triples[index])
    def source_excerpt(span, quote, limit=250):
        if len(span) <= limit:
            return span
        match = SequenceMatcher(None, quote, span, autojunk=False).find_longest_match(
            0, len(quote), 0, len(span))
        center = match.b + min(match.size, limit) // 2
        start = max(0, min(len(span) - limit, center - limit // 2))
        return span[start:start + limit]

    details = []
    for group in list(grouped.values())[:3]:
        quote = group['quote'] if isinstance(group['quote'], str) else ''
        closest = sorted(spans, key=lambda span: SequenceMatcher(
            None, ' '.join(quote.split()), ' '.join(span.split())).ratio(), reverse=True)[:2]
        first_triple = group['triples'][0]
        display = {
            'indices': group['indices'][:8],
            'triple': [field[:48] if isinstance(field, str) else field for field in first_triple],
            'quote': quote[:120],
            'source_spans': [],
        }
        if len(group['indices']) > 8:
            display['additional_indices'] = len(group['indices']) - 8
        available = max(0, 500 - len(json.dumps(display, ensure_ascii=False)) - 12)
        per_span = min(250, max(0, available // max(1, len(closest)) - 4))
        display['source_spans'] = [source_excerpt(span, quote, per_span)
                                   for span in closest if per_span > 0]
        # JSON escapes can enlarge a span. Trim only diagnostic substrings;
        # source matching and extraction acceptance are completely unchanged.
        while len(json.dumps(display, ensure_ascii=False)) > 500 and display['source_spans']:
            display['source_spans'][-1] = display['source_spans'][-1][:-10]
            if not display['source_spans'][-1]:
                display['source_spans'].pop()
        # Escaped quotes/control characters can enlarge even the diagnostic
        # fields themselves. Keep the per-group bound for those inputs too.
        while len(json.dumps(display, ensure_ascii=False)) > 500:
            if len(display['quote']) > 8:
                display['quote'] = display['quote'][:-8]
                continue
            field_index = max(range(len(display['triple'])),
                              key=lambda index: len(str(display['triple'][index])))
            field = display['triple'][field_index]
            if isinstance(field, str) and len(field) > 8:
                display['triple'][field_index] = field[:-8]
            else:
                break
        details.append(json.dumps(display, ensure_ascii=False))
    prefix = ('Support quotes do not occur verbatim in the supplied source passage. '
              'Each diagnostic gives bad record indices, triple, supplied quote excerpt, '
              'and nearby real source substrings (candidates, not verified entailment):\n')
    suffix = (
        '\nCopy the shortest supporting original text exactly, including quotes and punctuation. '
        'Do not paraphrase, join separated sentences, drop quote characters, use ellipses, '
        'or copy the demonstration passage. Recheck every evidence quote against the actual source.'
    )
    while details:
        omitted = len(grouped) - len(details)
        remainder = f'\n{omitted} additional quote groups also need correction.' if omitted else ''
        result = prefix + '\n'.join(details) + remainder + suffix
        if len(result) <= 3000 and len(result.encode('utf-8')) <= 3000:
            return result
        details.pop()
    return prefix + f'{len(grouped)} mismatching quote groups need correction.' + suffix


def extract_triple_payload(response: str) -> dict:
    """Read a JSON object or a bare JSON array, including fenced responses.

    A successfully parsed containing object is checked before scanning arrays,
    so an unrelated named_entities array cannot masquerade as the triple list.
    Python literals and truncated JSON are deliberately not repaired here.
    """
    if not isinstance(response, str):
        raise ValueError("OpenIE response must be text.")
    decoder = json.JSONDecoder()
    objects_found = False
    for start, character in enumerate(response):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(response[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            objects_found = True
            if "triples" in payload:
                if not isinstance(payload["triples"], list):
                    raise ValueError("OpenIE response field 'triples' must be a list.")
                return payload
    # Bare arrays are accepted only when the *entire* unfenced response is JSON.
    stripped = response.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        first_newline = stripped.find("\n")
        stripped = stripped[first_newline + 1:-3].strip() if first_newline >= 0 else stripped
    if not objects_found:
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            return {"triples": payload}
    raise ValueError("OpenIE response does not contain a valid JSON object with 'triples'.")


def extract_triple_list(response: str) -> List:
    return extract_triple_payload(response)["triples"]


def normalize_repair_payload(payload: dict):
    """Unpack paired repair records; retain strict legacy-array compatibility.

    Pairing makes evidence correspondence structural rather than asking a model
    to keep two independently generated arrays at the same length.
    """
    records = payload.get('triples')
    if not isinstance(records, list):
        raise ValueError('Repair triples must be an array')
    if any(isinstance(record, dict) for record in records):
        triples, quotes = [], []
        for index, record in enumerate(records):
            if not isinstance(record, dict) or set(record) != {'triple', 'support_quote'}:
                raise ValueError(f'Repair triples[{index}] must contain exactly triple and support_quote')
            triples.append(record['triple'])
            quotes.append(record['support_quote'])
        return triples, quotes, payload.get('status')
    # Empty paired results have no legacy support_quotes member.
    if not records and 'support_quotes' not in payload:
        return [], [], payload.get('status')
    quotes = payload.get('support_quotes')
    if not isinstance(quotes, list) or len(quotes) != len(records):
        count = len(quotes) if isinstance(quotes, list) else 'missing'
        raise ValueError(f'Repair has {len(records)} triples but {count} support quotes; '
                         'return paired records, each containing its own triple and support_quote')
    return records, quotes, payload.get('status')


TRIPLE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 3,
                "maxItems": 3,
            },
        },
    },
    "required": ["triples"],
    "additionalProperties": False,
}

REPAIR_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "triple": TRIPLE_JSON_SCHEMA["properties"]["triples"]["items"],
                    "support_quote": {"type": "string", "minLength": 1},
                },
                "required": ["triple", "support_quote"],
                "additionalProperties": False,
            },
        },
        "status": {"type": "string", "enum": ["success", "no_supported_relations"]},
    },
    "required": ["triples", "status"],
    "additionalProperties": False,
}
