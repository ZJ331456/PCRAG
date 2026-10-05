"""Compile compact JSON schemas to EBNF before submitting vLLM requests.

vLLM V1 can ignore per-request JSON whitespace options. An explicit grammar
encodes the format in the request itself, preserving schema-defined strings
while preventing structural whitespace loops. Compilation uses CPU only.
"""

from functools import lru_cache
import copy
import hashlib
import json


STRUCTURED_OUTPUT_VERSION = 'pathcondrag_compact_json_portable_ebnf_v3_exact_arrays'


class StructuredOutputDependencyError(RuntimeError):
    """The active runtime cannot compile the required grammar contract."""


class StructuredOutputSchemaError(ValueError):
    """A schema cannot be represented by the required structured output."""


def _positive_lookahead_end(grammar, start):
    """Find the complete group without counting quoted/class parentheses."""
    depth, quote, character_class, escaped = 0, False, False, False
    for index in range(start, len(grammar)):
        character = grammar[index]
        if escaped:
            escaped = False
            continue
        if character == '\\':
            escaped = True
            continue
        if quote:
            if character == '"':
                quote = False
            continue
        if character_class:
            if character == ']':
                character_class = False
            continue
        if character == '"':
            quote = True
        elif character == '[':
            character_class = True
        elif character == '(':
            depth += 1
        elif character == ')':
            depth -= 1
            if depth == 0:
                return index + 1
    raise StructuredOutputSchemaError('Unbalanced Xgrammar positive-lookahead optimization')


def _strip_positive_lookaheads(grammar):
    """Remove generated lookahead hints, preserving quoted/schema characters.

    The Xgrammar JSON converter exports positive lookahead optimizations in
    normalized EBNF. Guidance's GBNF converter cannot parse those extensions.
    This transformer is intended only for generated JSON-schema grammars.
    """
    output, index, quote, character_class, escaped = [], 0, False, False, False
    while index < len(grammar):
        character = grammar[index]
        if escaped:
            output.append(character)
            escaped = False
            index += 1
            continue
        if character == '\\':
            output.append(character)
            escaped = True
            index += 1
            continue
        if quote:
            output.append(character)
            if character == '"':
                quote = False
            index += 1
            continue
        if character_class:
            output.append(character)
            if character == ']':
                character_class = False
            index += 1
            continue
        if character == '"':
            quote = True
        elif character == '[':
            character_class = True
        elif grammar.startswith('(=', index) or grammar.startswith('(?=', index):
            index = _positive_lookahead_end(grammar, index)
            continue
        output.append(character)
        index += 1
    return ''.join(output)


@lru_cache(maxsize=256)
def _compile_grammar(serialized_schema):
    try:
        import xgrammar
    except ImportError as error:
        raise StructuredOutputDependencyError(
            'Strict OpenIE structured output requires xgrammar and its import '
            'dependencies in the active runtime (validated with xgrammar 0.1.18). '
            'Install them in the configured isolated runtime_deps environment. '
            'No guided_json or unrestricted-whitespace fallback is permitted.'
        ) from error
    try:
        schema = _fixed_array_schema(json.loads(serialized_schema))
        grammar = xgrammar.Grammar.from_json_schema(
            json.dumps(schema, ensure_ascii=False, separators=(',', ':'), allow_nan=False),
            any_whitespace=False, indent=None,
            separators=(',', ':'), strict_mode=True,
        )
        result = _strip_positive_lookaheads(str(grammar))
    except AttributeError as error:
        raise StructuredOutputDependencyError(
            'The active xgrammar runtime must provide Grammar.from_json_schema '
            'with any_whitespace, indent and separators support; xgrammar '
            '0.1.18 was validated. No structured-output fallback is permitted.'
        ) from error
    except Exception as error:
        digest = hashlib.sha256(serialized_schema.encode('utf-8')).hexdigest()
        raise StructuredOutputSchemaError(
            f'Xgrammar could not compile the strict OpenIE schema '
            f'(sha256={digest}): {type(error).__name__}: {error}'
        ) from error
    if not result.strip():
        raise StructuredOutputSchemaError('Xgrammar returned an empty strict OpenIE grammar')
    return result


def _fixed_array_schema(value):
    """Express homogeneous fixed-length arrays as exact positional items.

    Xgrammar 0.1.18 silently ignores minItems/maxItems on homogeneous arrays.
    The equivalent prefixItems form really constrains three-field triples;
    this only changes grammar compilation, not the public validation schema.
    """
    if isinstance(value, list):
        return [_fixed_array_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _fixed_array_schema(item) for key, item in value.items()}
    count = result.get('minItems')
    if (result.get('type') == 'array' and isinstance(count, int)
            and not isinstance(count, bool) and count >= 0
            and result.get('maxItems') == count
            and isinstance(result.get('items'), dict) and 'prefixItems' not in result):
        result['prefixItems'] = [copy.deepcopy(result['items']) for _ in range(count)]
        result['items'] = False
    return result


def guided_json_parameters(schema, extra_body=None):
    """Return exactly one grammar constraint and disable model thinking.

    Field order is preserved, matching the JSON schema supplied by the caller.
    The cache key contains the complete schema rather than its object identity,
    so mutating a schema cannot silently reuse a grammar for older constraints.
    This helper does not change the caller's completion-token budget.
    """
    if not isinstance(schema, dict):
        raise StructuredOutputSchemaError('The strict OpenIE JSON schema must be an object')
    try:
        serialized = json.dumps(schema, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise StructuredOutputSchemaError('The strict OpenIE schema must contain valid JSON values') from error
    if extra_body is not None and not isinstance(extra_body, dict):
        raise StructuredOutputSchemaError('The structured-output extra_body must be an object')
    parameters = copy.deepcopy(extra_body or {})
    for key in ('guided_json', 'guided_regex', 'guided_choice', 'guided_grammar',
                'guided_json_object', 'guided_whitespace_pattern',
                'guided_decoding_backend', 'guided_structural_tag', 'structured_outputs'):
        parameters.pop(key, None)
    template = parameters.get('chat_template_kwargs')
    if template is not None and not isinstance(template, dict):
        raise StructuredOutputSchemaError('chat_template_kwargs must be an object')
    template = copy.deepcopy(template or {})
    template['enable_thinking'] = False
    parameters['chat_template_kwargs'] = template
    parameters['guided_grammar'] = _compile_grammar(serialized)
    return parameters
