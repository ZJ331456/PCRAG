"""CPU-only regression checks for complete, bounded sentence recovery."""

import copy
import json
import re
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pathcondrag.index.openie import openie_atomic_recovery as module


def reply(triple=None, quote=None, status=None, complete=True):
    records = [] if triple is None else [{'triple': triple, 'support_quote': quote}]
    return json.dumps({'triples': records, 'status': status or ('success' if records else 'no_supported_relations'),
                       'coverage_complete': complete})


class FakeLLM:
    def __init__(self, responses=(), token_count=100, responder=None):
        self.responses, self.calls = iter(responses), []
        self.token_count, self.responder = token_count, responder

    def count_prompt_tokens(self, messages):
        return self.token_count(messages) if callable(self.token_count) else self.token_count

    def infer(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        value = self.responder(kwargs) if self.responder else next(self.responses)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, tuple):
            return value[0], {'finish_reason': value[1], 'prompt_tokens': 10, 'completion_tokens': 20}, value[2] if len(value) > 2 else False
        return value, {'finish_reason': 'stop', 'prompt_tokens': 10, 'completion_tokens': 20}, False


SOURCE = 'Ada lives in Rome.'
TRIPLE = ['Ada', 'lives in', 'Rome']


class AtomicRecoveryTests(unittest.TestCase):
    def test_exact_source_quotes_fixed_decoding_and_full_source(self):
        llm = FakeLLM([reply(TRIPLE, SOURCE)])
        output = module.atomic_recovery(llm, 'id', SOURCE, ['Ada', 'Rome'])
        self.assertEqual(output.triples, [TRIPLE])
        self.assertTrue(output.metadata['complete'])
        self.assertFalse(output.metadata['semantic_verified'])
        self.assertTrue(output.metadata['requires_semantic_verification'])
        self.assertEqual(output.metadata['covered_source_spans'], [[0, len(SOURCE)]])
        self.assertEqual(output.metadata['atomic_support_records'][0]['source_end'], len(SOURCE))
        call = llm.calls[0]
        self.assertEqual(call['max_completion_tokens'], 2048)
        self.assertEqual(call['temperature'], 0.0)
        self.assertFalse(call['extra_body']['chat_template_kwargs']['enable_thinking'])
        self.assertTrue(call['extra_body']['guided_grammar'].strip())
        self.assertNotIn('guided_json', call['extra_body'])
        self.assertNotIn('guided_whitespace_pattern', call['extra_body'])
        self.assertEqual(module.SCHEMA['properties']['triples']['maxItems'], 8)
        self.assertEqual(json.loads(call['messages'][1]['content'])['SOURCE'], SOURCE)

    def test_every_sentence_and_heading_is_processed_without_eight_window_limit(self):
        passage = 'A heading\n' + ' '.join(f'Person {index} lives in Rome.' for index in range(12))
        def responder(call):
            focus = json.loads(call['messages'][1]['content'])['FOCUS']
            return reply(['Person', 'lives in', 'Rome'], focus) if 'lives in' in focus else reply()
        llm = FakeLLM(responder=responder)
        output = module.atomic_recovery(llm, 'id', passage, [])
        self.assertTrue(output.metadata['complete'])
        self.assertEqual(len(llm.calls), 13)
        spans = output.metadata['covered_source_spans']
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(passage))
        self.assertTrue(all(left[1] == right[0] for left, right in zip(spans, spans[1:])))
        self.assertEqual(output.metadata['unverified_empty_focuses'], [[0, len('A heading\n')]])

    def test_wrong_quote_receives_changed_feedback_and_is_not_approved(self):
        llm = FakeLLM([reply(TRIPLE, 'Ada was living in Rome.'), reply(TRIPLE, SOURCE)])
        output = module.atomic_recovery(llm, 'id', SOURCE, [])
        self.assertTrue(output.metadata['complete'])
        attempts = output.metadata['atomic_recovery_attempts']
        self.assertEqual(attempts[0]['failure_class'], 'source_quote')
        self.assertNotEqual(llm.calls[0]['messages'], llm.calls[1]['messages'])
        self.assertIn('Choose an offered source quote ID', llm.calls[1]['messages'][-1]['content'])

    def test_coordinating_predicate_retries_meaningful_relation(self):
        llm = FakeLLM([reply(['Ada', 'and', 'Rome'], SOURCE), reply(TRIPLE, SOURCE)])
        output = module.atomic_recovery(llm, 'id', SOURCE, [])
        self.assertTrue(output.metadata['complete'])
        self.assertEqual(output.metadata['atomic_recovery_attempts'][0]['failure_class'], 'coordination')

    def test_wrong_shape_and_missing_coverage_boolean_cannot_succeed(self):
        invalid = json.dumps({'triples': [{'triple': TRIPLE, 'support_quote': SOURCE}], 'status': 'success'})
        llm = FakeLLM([invalid] * 3)
        output = module.atomic_recovery(llm, 'id', SOURCE, [])
        self.assertFalse(output.metadata['complete'])
        self.assertTrue(output.metadata['openie_skipped'])
        self.assertEqual(output.triples, [])
        self.assertEqual(len(llm.calls), 3)

    def test_parent_length_failure_splits_at_original_clause_offsets(self):
        passage = 'Ada lived in Rome for many years; Ada worked at the local public library for many years'
        def responder(call):
            focus = json.loads(call['messages'][1]['content'])['FOCUS']
            if focus == passage:
                return 'truncated', 'length'
            return reply(['Ada', 'described in', 'source'], focus)
        llm = FakeLLM(responder=responder)
        output = module.atomic_recovery(llm, 'id', passage, [])
        self.assertTrue(output.metadata['complete'])
        self.assertEqual(len(llm.calls), 5)
        parent, left, right = output.metadata['atomic_recovery_units']
        self.assertEqual(parent['status'], 'split')
        self.assertEqual(parent['split_reason'], 'output_length')
        self.assertEqual(left['source_end'], right['source_start'])
        self.assertEqual(passage[left['source_start']:left['source_end']], left['focus'])
        self.assertEqual(output.metadata['failed_source_spans'], [])
        for call in llm.calls:
            self.assertEqual(json.loads(call['messages'][1]['content'])['SOURCE'], passage)

    def test_focus_context_preflight_can_split_without_sending_oversize_request(self):
        passage = 'Ada lived in Rome for many years; Ada worked at the local public library for many years'
        def tokens(messages):
            focus = json.loads(messages[1]['content'])['FOCUS']
            return 7000 if len(focus) > 80 else 100
        def responder(call):
            focus = json.loads(call['messages'][1]['content'])['FOCUS']
            return reply(['Ada', 'described in', 'source'], focus)
        llm = FakeLLM(token_count=tokens, responder=responder)
        output = module.atomic_recovery(llm, 'id', passage, [])
        self.assertTrue(output.metadata['complete'])
        self.assertEqual(len(llm.calls), 2)
        self.assertFalse(output.metadata['atomic_recovery_attempts'][0]['request_sent'])

    def test_whole_source_context_overflow_never_truncates_or_calls_llm(self):
        llm = FakeLLM(token_count=7000)
        output = module.atomic_recovery(llm, 'id', SOURCE, [])
        self.assertEqual(llm.calls, [])
        self.assertFalse(output.metadata['complete'])
        self.assertEqual(output.metadata['failure_class'], 'source_context_limit')
        self.assertNotEqual(output.metadata['repair_status'], 'no_supported_relations')

    def test_eight_record_cap_cannot_be_ignored_by_fake_or_unguided_response(self):
        payload = {'triples': [{'triple': TRIPLE, 'support_quote': SOURCE}] * 9,
                   'status': 'success', 'coverage_complete': True}
        llm = FakeLLM([json.dumps(payload)] * 3)
        output = module.atomic_recovery(llm, 'id', SOURCE, [])
        self.assertFalse(output.metadata['complete'])
        self.assertEqual(output.triples, [])

    def test_explicit_empty_requires_external_source_validation(self):
        llm = FakeLLM([reply()])
        output = module.atomic_recovery(llm, 'id', 'A heading', [])
        self.assertTrue(output.metadata['complete'])
        self.assertEqual(output.metadata['repair_status'], 'no_supported_relations')
        self.assertEqual(output.metadata['quality_status'], 'empty_unverified')
        self.assertTrue(output.metadata['requires_source_empty_verification'])
        self.assertFalse(output.metadata['source_no_supported_relations'])

    def test_unmarked_empty_is_a_shape_failure_not_legitimate_empty(self):
        llm = FakeLLM([reply(status='success')] * 3)
        output = module.atomic_recovery(llm, 'id', SOURCE, [])
        self.assertFalse(output.metadata['complete'])
        self.assertTrue(output.metadata['openie_skipped'])

    def test_one_leaf_failure_blocks_partial_candidates(self):
        passage = SOURCE + ' Bo lives in Paris.'
        llm = FakeLLM([reply(TRIPLE, SOURCE)] + ['not json'] * 3)
        output = module.atomic_recovery(llm, 'id', passage, [])
        self.assertFalse(output.metadata['complete'])
        self.assertEqual(output.triples, [])
        self.assertEqual(output.metadata['partial_candidate_count'], 1)
        self.assertEqual(len(output.metadata['atomic_support_records']), 1)

    def test_transport_error_does_not_split_into_many_api_requests(self):
        llm = FakeLLM([RuntimeError('server unavailable')])
        output = module.atomic_recovery(llm, 'id', SOURCE, [])
        self.assertFalse(output.metadata['complete'])
        self.assertEqual(output.metadata['failure_class'], 'transport_error')
        self.assertEqual(len(llm.calls), 1)

    def test_hard_call_budget_marks_tail_pending_and_does_not_drop_it(self):
        passage = ' '.join(f'Person {index} lives in Rome.' for index in range(70))
        llm = FakeLLM(responder=lambda call: reply(TRIPLE, json.loads(call['messages'][1]['content'])['FOCUS']))
        output = module.atomic_recovery(llm, 'id', passage, [])
        self.assertEqual(len(llm.calls), 64)
        self.assertFalse(output.metadata['complete'])
        self.assertEqual(output.metadata['failure_class'], 'call_budget_exhausted')
        self.assertTrue(output.metadata['pending_source_spans'])
        self.assertEqual(output.metadata['pending_source_spans'][-1][1], len(passage))

    def test_coverage_false_causes_split_not_silent_eight_fact_truncation(self):
        passage = 'Ada lived in Rome for many years; Ada worked at the local public library for many years'
        def responder(call):
            focus = json.loads(call['messages'][1]['content'])['FOCUS']
            return reply(['Ada', 'described in', 'source'], focus, complete=focus != passage)
        output = module.atomic_recovery(FakeLLM(responder=responder), 'id', passage, [])
        self.assertTrue(output.metadata['complete'])
        self.assertEqual(output.metadata['atomic_recovery_units'][0]['split_reason'], 'focus_incomplete')

    def test_repeated_quote_is_matched_to_current_focus_not_first_occurrence(self):
        passage = SOURCE + ' ' + SOURCE
        llm = FakeLLM([reply(TRIPLE, SOURCE)] * 2)
        output = module.atomic_recovery(llm, 'id', passage, [])
        self.assertTrue(output.metadata['complete'])
        offsets = [record['source_start'] for record in output.metadata['atomic_support_records']]
        self.assertEqual(offsets, [0, len(SOURCE) + 1])

    def test_cache_and_token_counters_preserve_attempt_timing(self):
        llm = FakeLLM([(reply(TRIPLE, SOURCE), 'stop', True)])
        output = module.atomic_recovery(llm, 'id', SOURCE, [])
        self.assertEqual(output.metadata['cache_hit_count'], 1)
        self.assertEqual(output.metadata['prompt_tokens'], 10)
        self.assertEqual(output.metadata['completion_tokens'], 20)
        self.assertGreaterEqual(output.metadata['atomic_recovery_attempts'][0]['seconds'], 0)
        self.assertIn('started_at', output.metadata['atomic_recovery_attempts'][0])

    def test_invalid_sources_are_explicit_failed(self):
        for source in ['', '  ', None, 12]:
            output = module.atomic_recovery(FakeLLM(), 'id', source, [])
            self.assertFalse(output.metadata['complete'])
            self.assertEqual(output.metadata['failure_class'], 'invalid_source')


    def test_source_units_retains_every_original_character_for_ner(self):
        passage = 'Title\n\n  Ada lives in Rome.\nBo lives in Paris!  '
        units = module.source_units(passage)
        self.assertEqual(''.join(text for _, _, text in units), passage)
        self.assertEqual(units[0][0], 0)
        self.assertEqual(units[-1][1], len(passage))
        for start, end, text in units:
            self.assertEqual(passage[start:end], text)
        self.assertEqual(module.source_units(''), [])

    def test_seventeen_facts_split_repeatedly_instead_of_stopping_at_eight(self):
        passage = '; '.join(f'Ada has property {index}' for index in range(17))
        def responder(call):
            focus = json.loads(call['messages'][1]['content'])['FOCUS']
            found = list(re.finditer(r'Ada has property (\d+)', focus))
            records = [{'triple': ['Ada', 'has property', match.group(1)],
                        'support_quote': match.group(0)} for match in found[:8]]
            return json.dumps({'triples': records, 'status': 'success',
                               'coverage_complete': len(found) <= 8})
        llm = FakeLLM(responder=responder)
        output = module.atomic_recovery(llm, 'id', passage, [])
        self.assertTrue(output.metadata['complete'])
        self.assertEqual({triple[2] for triple in output.triples}, {str(index) for index in range(17)})
        self.assertEqual(len(output.triples), 17)
        self.assertGreater(len(llm.calls), 3)
        self.assertTrue(all(len(unit['support_records']) <= 8 for unit in output.metadata['atomic_recovery_units']))


    def test_named_fields_normalize_to_public_array_triples(self):
        response = json.dumps({'triples': [{'triple': {'subject': 'Ada', 'predicate': 'lives in', 'object': 'Rome'},
                                            'support_quote': SOURCE}],
                               'status': 'success', 'coverage_complete': True})
        llm = FakeLLM([response])
        with patch.object(module, 'guided_json_parameters', wraps=module.guided_json_parameters) as constrain:
            output = module.atomic_recovery(llm, 'id', SOURCE, [])
        self.assertEqual(output.triples, [TRIPLE])
        triple_schema = constrain.call_args.args[0]['properties']['triples']['items']['properties']['triple']
        self.assertEqual(triple_schema['type'], 'object')
        self.assertEqual(set(triple_schema['required']), {'subject', 'predicate', 'object'})
        self.assertIn('pattern', triple_schema['properties']['predicate'])
        self.assertTrue(llm.calls[0]['extra_body']['guided_grammar'].strip())
        self.assertIn('predicate', llm.calls[0]['extra_body']['guided_grammar'])

    def test_named_fields_missing_argument_is_not_repaired_by_truncation(self):
        response = json.dumps({'triples': [{'triple': {'subject': 'Ada', 'predicate': 'lives in'},
                                            'support_quote': SOURCE}],
                               'status': 'success', 'coverage_complete': True})
        output = module.atomic_recovery(FakeLLM([response] * 3), 'id', SOURCE, [])
        self.assertFalse(output.metadata['complete'])
        self.assertEqual(output.triples, [])

    def test_finite_predicate_regex_blocks_coordination_without_lookaround(self):
        self.assertNotIn('(?=', module.PREDICATE_PATTERN)
        self.assertNotIn('(?!', module.PREDICATE_PATTERN)
        self.assertNotIn('|)', module.PREDICATE_PATTERN)
        for invalid in ['and', 'AND', 'And', 'or', 'OR', '&', 'as well as', 'and/or']:
            self.assertIsNone(re.fullmatch(module.PREDICATE_PATTERN, invalid))
        for valid in ['is', 'are', 'has', 'ordered by', 'and is associated with', 'lives in', '位于']:
            self.assertIsNotNone(re.fullmatch(module.PREDICATE_PATTERN, valid))

    def test_cached_real_coordinated_failure_stays_failed_not_partially_approved(self):
        # This is the real response pattern observed on the Randy/Sharon source:
        # two useful relations followed by repeated coordinated-name records.
        passage = 'Randy Marsh and Sharon Marsh are fictional characters in South Park.'
        records = [{'triple': ['Randy Marsh', 'are', 'fictional characters'], 'support_quote': passage},
                   {'triple': ['Sharon Marsh', 'are', 'fictional characters'], 'support_quote': passage}]
        records.extend({'triple': ['Randy Marsh', 'and', 'Sharon Marsh'], 'support_quote': passage}
                       for _ in range(6))
        raw = json.dumps({'triples': records, 'status': 'success', 'coverage_complete': False})
        with self.assertRaisesRegex(ValueError, 'coordinating word'):
            module._parse(raw, {'finish_reason': 'stop'}, passage, 0, len(passage))


    def test_source_quote_ids_resolve_original_quotes_without_double_escaping(self):
        passage = 'Ada appears in "Moon Harbor".'
        response = json.dumps({'triples': [{'triple': {'subject': 'Ada', 'predicate': 'appears in',
                                                       'object': 'Moon Harbor'}, 'support_quote_id': 0}],
                               'status': 'success', 'coverage_complete': True})
        llm = FakeLLM([response])
        with patch.object(module, 'guided_json_parameters', wraps=module.guided_json_parameters) as constrain:
            output = module.atomic_recovery(llm, 'id', passage, [])
        self.assertTrue(output.metadata['complete'])
        self.assertEqual(output.metadata['atomic_support_records'][0]['support_quote'], passage)
        self.assertEqual(output.metadata['atomic_support_records'][0]['source_start'], 0)
        self.assertNotIn('guided_whitespace_pattern', llm.calls[0]['extra_body'])
        self.assertNotIn('guided_json', llm.calls[0]['extra_body'])
        self.assertTrue(llm.calls[0]['extra_body']['guided_grammar'].strip())
        properties = constrain.call_args.args[0]['properties']['triples']['items']['properties']
        self.assertNotIn('support_quote', properties)
        self.assertEqual(properties['support_quote_id']['enum'], [0])
        evidence = json.loads(llm.calls[0]['messages'][1]['content'])['evidence_spans']
        self.assertEqual(evidence[0]['id'], 0)
        self.assertEqual(evidence[0]['text'], passage)

    def test_header_quote_pool_cannot_offer_unrelated_body_quotes(self):
        passage = 'Ada and Bo\nAda lives in Rome.'
        start, end, text = module.source_units(passage)[0]
        pool = module._quote_pool(passage, start, end)
        self.assertEqual(pool, [(0, len('Ada and Bo'), 'Ada and Bo')])
        payload = json.loads(module._messages(passage, start, end, [], '')[1]['content'])
        self.assertEqual(payload['focus_kind'], 'heading_context_only')
        self.assertEqual(payload['FOCUS'], text)
        self.assertEqual(payload['SOURCE'], passage)

    def test_invalid_source_quote_id_is_not_converted_to_some_other_quote(self):
        passage = SOURCE
        for identifier in [True, -1, 9, '0']:
            raw = json.dumps({'triples': [{'triple': {'subject': 'Ada', 'predicate': 'lives in', 'object': 'Rome'},
                                           'support_quote_id': identifier}],
                              'status': 'success', 'coverage_complete': True})
            with self.assertRaisesRegex(ValueError, 'support_quote_id'):
                module._parse(raw, {'finish_reason': 'stop'}, passage, 0, len(passage), [(0, len(passage), passage)])


if __name__ == '__main__':
    unittest.main()
