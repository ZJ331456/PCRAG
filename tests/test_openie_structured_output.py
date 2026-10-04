"""CPU-only checks using installed Xgrammar, without any model or API client."""

import builtins
import ast
import copy
import importlib.util
import json
from pathlib import Path
import unittest
import re
import unicodedata
from unittest.mock import patch


# Load this standard-library-only helper without importing the retriever package
# and its unrelated embedding/graph dependencies into the vLLM test environment.
MODULE_PATH = Path(__file__).resolve().parents[1] / 'src/pathcondrag/utils/openie_structured_output.py'
SPEC = importlib.util.spec_from_file_location('openie_structured_output_cpu_test', MODULE_PATH)
structured = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(structured)


class StructuredOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import xgrammar
        from xgrammar.testing import _is_grammar_accept_string
        cls.xgrammar = xgrammar
        cls.accepts = staticmethod(_is_grammar_accept_string)

    def setUp(self):
        structured._compile_grammar.cache_clear()
        self.schema = {'type': 'object', 'properties': {
            'subject': {'type': 'string', 'minLength': 1},
            'predicate': {'type': 'string', 'minLength': 1},
            'object': {'type': 'string', 'minLength': 1},
        }, 'required': ['subject', 'predicate', 'object'], 'additionalProperties': False}

    def grammar(self):
        return self.xgrammar.Grammar.from_ebnf(
            structured.guided_json_parameters(self.schema)['guided_grammar'])

    def test_single_constraint_and_thinking_disabled_without_token_budget_changes(self):
        parameters = structured.guided_json_parameters(self.schema)
        self.assertEqual(set(parameters), {'guided_grammar', 'chat_template_kwargs'})
        self.assertEqual(parameters['chat_template_kwargs'], {'enable_thinking': False})
        self.assertIsInstance(parameters['guided_grammar'], str)
        self.assertIn('root ::=', parameters['guided_grammar'])
        self.assertNotIn('guided_json', parameters)
        self.assertNotIn('guided_whitespace_pattern', parameters)
        self.assertNotIn('max_completion_tokens', parameters)

    def test_long_structural_whitespace_is_rejected(self):
        grammar = self.grammar()
        valid = json.dumps({'subject': 'Alpha', 'predicate': 'is', 'object': 'Beta'}, separators=(',', ':'))
        self.assertTrue(self.accepts(grammar, valid))
        for whitespace in (' ' * 2048, '\n' * 2048, '\t' * 2048, '\r\n' * 1024):
            self.assertFalse(self.accepts(grammar, valid[:1] + whitespace + valid[1:]))
            self.assertFalse(self.accepts(grammar, valid + whitespace))

    def test_strings_preserve_spaces_and_unicode(self):
        grammar = self.grammar()
        for value in ('classical theater', 'Alpha' + ' ' * 2048 + 'Beta',
                      'théâtre 中文'):
            output = json.dumps({'subject': 'Alpha', 'predicate': 'trained in', 'object': value},
                                separators=(',', ':'), ensure_ascii=False)
            self.assertTrue(self.accepts(grammar, output), repr(value))

    def test_unconstrained_strings_preserve_legal_json_escapes(self):
        # Xgrammar 0.1.18 itself excludes escapes when minLength is supplied.
        # The helper preserves schemas rather than silently weakening them.
        for properties in self.schema['properties'].values():
            properties.pop('minLength')
        grammar = self.grammar()
        for value in ('A\nB\tC', 'A "quoted" title', 'A\\B'):
            output = json.dumps({'subject': 'Alpha', 'predicate': 'trained in', 'object': value},
                                separators=(',', ':'), ensure_ascii=False)
            self.assertTrue(self.accepts(grammar, output), repr(value))

    def test_schema_constraints_and_field_order_are_preserved(self):
        self.schema['properties']['predicate'] = {'type': 'string', 'pattern': '^is$'}
        grammar = self.grammar()
        valid = {'subject': 'Alpha', 'predicate': 'is', 'object': 'Beta'}
        self.assertTrue(self.accepts(grammar, json.dumps(valid, separators=(',', ':'))))
        invalid = dict(valid, predicate='and')
        self.assertFalse(self.accepts(grammar, json.dumps(invalid, separators=(',', ':'))))
        self.assertFalse(self.accepts(grammar, json.dumps(dict(valid, extra=True), separators=(',', ':'))))

    def test_compilation_is_cached_with_explicit_finite_format(self):
        native = self.xgrammar.Grammar.from_json_schema
        with patch.object(self.xgrammar.Grammar, 'from_json_schema', wraps=native) as compile_schema:
            first = structured.guided_json_parameters(self.schema)
            second = structured.guided_json_parameters(copy.deepcopy(self.schema))
        self.assertEqual(first, second)
        self.assertEqual(compile_schema.call_count, 1)
        self.assertEqual(compile_schema.call_args.kwargs, {
            'any_whitespace': False, 'indent': None,
            'separators': (',', ':'), 'strict_mode': True,
        })

    def test_schema_mutation_changes_cache_key_and_input_is_not_modified(self):
        original = copy.deepcopy(self.schema)
        first = structured.guided_json_parameters(self.schema)['guided_grammar']
        self.assertEqual(self.schema, original)
        self.schema['properties']['predicate']['const'] = 'is'
        second = structured.guided_json_parameters(self.schema)['guided_grammar']
        self.assertNotEqual(first, second)
        self.assertEqual(structured._compile_grammar.cache_info().misses, 2)

    def test_missing_dependency_raises_instead_of_silent_fallback(self):
        native_import = builtins.__import__

        def missing(name, *args, **kwargs):
            if name == 'xgrammar':
                raise ModuleNotFoundError('No module named xgrammar')
            return native_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=missing):
            with self.assertRaisesRegex(structured.StructuredOutputDependencyError,
                                        'isolated runtime_deps.*No guided_json'):
                structured.guided_json_parameters(self.schema)

    def test_existing_body_preserves_options_but_replaces_mutually_exclusive_constraints(self):
        body = {'guided_json': {'old': True}, 'guided_regex': 'old',
                'guided_choice': ['old'], 'guided_grammar': 'old',
                'guided_json_object': True, 'guided_whitespace_pattern': '.*',
                'guided_decoding_backend': 'outlines', 'guided_structural_tag': 'old',
                'structured_outputs': {'json': 'old'}, 'seed': 42,
                'chat_template_kwargs': {'enable_thinking': True, 'keep': 'yes'}}
        original = copy.deepcopy(body)
        parameters = structured.guided_json_parameters(self.schema, extra_body=body)
        self.assertEqual(set(parameters), {'guided_grammar', 'chat_template_kwargs', 'seed'})
        self.assertEqual(parameters['seed'], 42)
        self.assertEqual(parameters['chat_template_kwargs'], {'enable_thinking': False, 'keep': 'yes'})
        self.assertEqual(body, original)
        self.assertEqual(structured._compile_grammar.cache_info().maxsize, 256)

    def test_invalid_schema_or_json_values_raise(self):
        for value in ([], 'schema', {'type': float('nan')}):
            with self.subTest(value=value):
                with self.assertRaises(structured.StructuredOutputSchemaError):
                    structured.guided_json_parameters(value)
        self.schema['properties']['predicate'] = {'type': 'string', 'pattern': '('}
        with self.assertRaisesRegex(structured.StructuredOutputSchemaError, 'Xgrammar could not compile'):
            structured.guided_json_parameters(self.schema)

    def test_positive_lookahead_stripping_preserves_literals_and_character_classes(self):
        prefix = 'root ::= "literal (?=keep)" [()?=] "escaped \\" (?=also literal)" '
        optimization = '(=(foo ("(" ")") [()]))'
        self.assertEqual(structured._strip_positive_lookaheads(prefix + optimization + '\n'), prefix + '\n')
        self.assertEqual(structured._strip_positive_lookaheads('root ::= "(?=value)" (?=("x"))'),
                         'root ::= "(?=value)" ')
        with self.assertRaisesRegex(structured.StructuredOutputSchemaError, 'Unbalanced'):
            structured._strip_positive_lookaheads('root ::= "x" (=("y")')

    def test_literals_containing_lookahead_text_are_still_exact(self):
        self.schema['properties']['object'] = {'enum': ['literal (?=inside)', 'literal (=inside)']}
        grammar = self.grammar()
        for value in ('literal (?=inside)', 'literal (=inside)'):
            output = json.dumps({'subject': 'Alpha', 'predicate': 'is', 'object': value}, separators=(',', ':'))
            self.assertTrue(self.accepts(grammar, output))

    def test_actual_role_schema_acceptance_and_rejection_on_both_backends(self):
        try:
            import llguidance
        except ImportError:
            self.skipTest('Optional Guidance CPU proof uses the installed vLLM environment')
        # Extract the pure schema functions without loading graph/embedding code.
        source_file = MODULE_PATH.with_name('openie_source_evidence.py')
        context = {'copy': copy, 're': re, 'unicodedata': unicodedata}
        for node in ast.parse(source_file.read_text()).body:
            if (isinstance(node, ast.FunctionDef) and node.name in ('_schema', '_source_quotes', '_normalized')
                    or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'FLAGS'
                                                            for target in node.targets)):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_file), 'exec'), context)
        source = 'Alpha is Beta.\nGamma is Delta.'
        schema = context['_schema'](1, source=source, triples=[['Alpha', 'is', 'Beta']])
        native = structured.guided_json_parameters(schema)['guided_grammar']
        self.assertNotIn('(=(', native)
        serialized = llguidance.grammar_from('gbnf', native)
        self.assertEqual(llguidance.LLMatcher.validate_grammar(serialized), '')
        xgrammar_grammar = self.xgrammar.Grammar.from_ebnf(native)

        class ByteTokenizer:
            eos_token_id = 256
            bos_token_id = None
            tokens = [bytes([value]) for value in range(256)] + [b'<eos>']
            special_token_ids = [256]

            def __call__(self, value):
                return list(value if isinstance(value, bytes) else value.encode())

        tokenizer = llguidance.LLTokenizer(llguidance.TokenizerWrapper(ByteTokenizer()),
                                          n_vocab=257, eos_token=256, slices=[])

        def guidance_accepts(text):
            matcher = llguidance.LLMatcher(tokenizer, serialized, log_level=0)
            return all(matcher.consume_token(value) for value in text.encode()) and matcher.is_accepting()

        verdict = {'supported': True, 'quote_id': 0,
                   'subject_roles': [{'source_subject': 'Alpha', 'mention': 'Alpha',
                                      'relation_quote_id': 0, 'relation_supported': True}],
                   'reason': 'The source directly states the relation.',
                   **{flag: True for flag in context['FLAGS']}}
        valid = {'source_has_supported_relations': True, 'source_evidence_quote_id': 0,
                 'source_reason': 'The original source contains the relation.', 'verdicts': [verdict]}
        examples = [(valid, True)]
        for field in ('source_evidence_quote_id', 'source_reason'):
            invalid = copy.deepcopy(valid)
            invalid[field] = 999 if field == 'source_evidence_quote_id' else 'x' * 241
            examples.append((invalid, False))
        for field in ('quote_id', 'supported'):
            invalid = copy.deepcopy(valid)
            invalid['verdicts'][0][field] = 1 if field == 'quote_id' else 'true'
            examples.append((invalid, False))
        invalid = copy.deepcopy(valid)
        invalid['verdicts'][0]['subject_roles'][0]['relation_quote_id'] = 1
        examples.append((invalid, False))
        for number, (example, expected) in enumerate(examples):
            output = json.dumps(example, ensure_ascii=False, separators=(',', ':'))
            self.assertEqual(self.accepts(xgrammar_grammar, output), expected,
                             f'Xgrammar example {number}: {output[:350]}')
            self.assertEqual(guidance_accepts(output), expected,
                             f'Guidance example {number}: {output[:350]}')
        output = json.dumps(valid, separators=(',', ':'))
        self.assertFalse(guidance_accepts('{' + '\n' * 2048 + output[1:]))
        # Installed Xgrammar 0.1.18 ignores minItems/maxItems. Exact candidate
        # counts must still be checked by the source verifier's application
        # gate. Stripping must preserve, not silently change, native behavior.
        original = self.xgrammar.Grammar.from_json_schema(
            schema, any_whitespace=False, indent=None, separators=(',', ':'), strict_mode=True)
        for count in (0, 1, 2):
            candidate = copy.deepcopy(valid)
            candidate['verdicts'] = [copy.deepcopy(verdict) for _ in range(count)]
            output = json.dumps(candidate, separators=(',', ':'))
            native_accepts = self.accepts(original, output)
            self.assertEqual(self.accepts(xgrammar_grammar, output), native_accepts)
            self.assertEqual(guidance_accepts(output), native_accepts)


if __name__ == '__main__':
    unittest.main()
