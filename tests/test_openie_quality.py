"""CPU-only regressions for real OpenIE format failures and safe recovery."""

import json
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathcondrag.information_extraction.openie_openai import OpenIE
from pathcondrag.utils.openie_quality import entity_argument_issues, extract_triple_list, validate_triples


class FakeLLM:
    llm_config = SimpleNamespace(generate_params={"seed": None})

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, tuple):
            text, finish = response
        else:
            text, finish = response, "stop"
        return text, {"finish_reason": finish}, False


class OpenIEQualityTests(unittest.TestCase):
    def test_rejects_empty_nested_dict_and_four_field_relations_without_guessing(self):
        triples = [
            ["Marty (Jr.)", "is", "son of", "Marty McFly"],
            ["A.C. Milan", "won", "Serie A", "1901"],
            ["station", "is below grade", ""],
            {"subject": "station", "relation": "is", "object": "below grade"},
            "abc", ["A", "has", None], ["A", "has", ["B"]],
            ["A", "has", "..."], ["静岡県", "has", "静岡市"],
        ]
        result = validate_triples(triples)
        self.assertEqual(result.valid_triples, [["静岡県", "has", "静岡市"]])
        self.assertEqual(len(result.invalid_triples), 8)
        self.assertEqual(result.raw_count, 9)

    def test_parser_accepts_json_fences_but_not_truncated_or_python_literals(self):
        self.assertEqual(extract_triple_list('```json\n{"triples":[["A","r","B"]]}\n```'),
                         [["A", "r", "B"]])
        self.assertEqual(extract_triple_list('[["A","r","B"]]'), [["A", "r", "B"]])
        for bad in ('{"triples":[["A","r","B"]]', "{'triples': [['A','r','B']]}",
                    '{"named_entities":["A","r","B"]}'):
            with self.assertRaises(ValueError):
                extract_triple_list(bad)

    def test_mixed_output_preserves_valid_relations_and_feedback_corrects_bad_record(self):
        llm = FakeLLM([
            '{"triples":[["station","located in","Newton"],["station","is below grade",""]]}',
            '{"triples":[["station","is","below grade"]]}',
        ])
        result = OpenIE(llm, quality_max_retries=1).triple_extraction(
            "id", "station is below grade in Newton", ["station", "Newton"])
        self.assertEqual(result.triples, [["station", "located in", "Newton"],
                                         ["station", "is", "below grade"]])
        self.assertEqual(result.metadata["quality_status"], "success")
        self.assertTrue(result.metadata["quality_recovered"])
        self.assertIn("empty or punctuation-only", llm.calls[1]["messages"][-1]["content"])
        self.assertNotEqual(llm.calls[0]["messages"], llm.calls[1]["messages"])
        self.assertEqual(result.metadata["recovery_history"][0]["response"],
                         '{"triples":[["station","located in","Newton"],["station","is below grade",""]]}')

    def test_exhausted_partial_keeps_valid_facts_and_marks_unresolved(self):
        llm = FakeLLM(['{"triples":[["A","r","B"],["C","r",""]]}'])
        result = OpenIE(llm, quality_max_retries=0).triple_extraction("id", "passage", [])
        self.assertEqual(result.triples, [["A", "r", "B"]])
        self.assertEqual(result.metadata["quality_status"], "partial")
        self.assertEqual(result.metadata["invalid_triple_count"], 1)
        self.assertTrue(result.metadata["openie_skipped"])

    def test_valid_empty_is_not_forced_to_hallucinate_a_fact(self):
        llm = FakeLLM(['{"triples":[]}'])
        result = OpenIE(llm).triple_extraction("id", "An index heading.", [])
        self.assertEqual(result.triples, [])
        self.assertEqual(result.metadata["quality_status"], "empty_valid")
        self.assertEqual(len(llm.calls), 1)

    def test_recovery_empty_does_not_hide_unresolved_bad_record(self):
        llm = FakeLLM(['{"triples":[["A","r","B"],["C","r",""]]}', '{"triples":[]}'])
        result = OpenIE(llm, quality_max_retries=1).triple_extraction("id", "passage", [])
        self.assertEqual(result.metadata["quality_status"], "partial")
        self.assertTrue(result.metadata["openie_skipped"])
        self.assertEqual(result.metadata["invalid_triple_count"], 1)

    def test_window_recovery_preserves_chunk_identity_and_output_cap(self):
        passage = "A long table\n" + " ".join(f"row{i} names team{i}" for i in range(100))
        windows = OpenIE._recovery_windows(passage)
        self.assertGreater(len(windows), 1)
        llm = FakeLLM([("bad", "length")] * 4 + [
            json.dumps({"triples": [[f"row{i}", "names", f"team{i}"]]})
            for i in range(len(windows))
        ])
        result = OpenIE(llm, quality_max_retries=0).triple_extraction("original-id", passage, [])
        self.assertEqual(result.chunk_id, "original-id")
        self.assertEqual(len(result.triples), len(windows))
        self.assertEqual(result.metadata["quality_status"], "success")
        self.assertEqual(result.metadata["window_recovery"][-1]["source_end"], len(passage))
        self.assertTrue(all(call["max_completion_tokens"] == 2048 for call in llm.calls))
        self.assertTrue(all(window[2].startswith("A long table\n") for window in windows))

    def test_window_bound_cannot_silently_drop_source_tail(self):
        self.assertEqual(OpenIE._recovery_windows("word " * 3000, max_windows=2), [])

    def test_guided_recovery_only_changes_failed_call_and_keeps_client_options(self):
        llm = FakeLLM(['not json', '{"triples":[["A","r","B"]]}'])
        llm.llm_config = SimpleNamespace(generate_params={
            "seed": None, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        })
        result = OpenIE(llm, quality_max_retries=1, guided_recovery=True).triple_extraction(
            "id", "A is related to B", [])
        self.assertEqual(result.metadata["quality_status"], "success")
        self.assertNotIn("extra_body", llm.calls[0])
        self.assertIn("guided_json", llm.calls[1]["extra_body"])
        self.assertFalse(llm.calls[1]["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_targeted_repair_first_call_has_schema_and_source_evidence(self):
        llm = FakeLLM([json.dumps({
            'triples': [['Waban station', 'is', 'below grade']],
            'support_quotes': ['Waban station is below grade'], 'status': 'success',
        })])
        result = OpenIE(llm, guided_recovery=True).triple_extraction(
            'id', 'Waban station is below grade in Newton.', [],
            repair_context='Fix ["Waban station", "is below grade", ""].')
        self.assertEqual(result.metadata['quality_status'], 'success')
        self.assertEqual(result.metadata['support_quote_matches'], [True])
        record_schema = llm.calls[0]['extra_body']['guided_json']['properties']['triples']['items']
        self.assertIn('support_quote', record_schema['properties'])
        self.assertIn('Targeted index repair', llm.calls[0]['messages'][-1]['content'])

    def test_targeted_repair_can_explicitly_withdraw_unsupported_bad_record(self):
        llm = FakeLLM(['{"triples":[],"support_quotes":[],"status":"no_supported_relations"}'])
        result = OpenIE(llm).triple_extraction('id', 'An index heading.', [],
                                              repair_context='Check defective unsupported record.')
        self.assertEqual(result.metadata['quality_status'], 'empty_valid')
        self.assertEqual(result.metadata['repair_status'], 'no_supported_relations')
        self.assertNotIn('openie_skipped', result.metadata)

    def test_targeted_repair_rejects_quote_not_in_source(self):
        llm = FakeLLM([json.dumps({
            'triples': [['station', 'is', 'below grade']],
            'support_quotes': ['station is below grade'], 'status': 'success',
        })])
        result = OpenIE(llm, quality_max_retries=0).triple_extraction(
            'id', 'Station appears in an index.', [], repair_context='Fix a defective record.')
        self.assertEqual(result.metadata['quality_status'], 'failed')
        self.assertEqual(result.triples, [])
        self.assertEqual(result.metadata['support_quote_matches'], [False])

    def test_window_quote_cannot_use_artificial_title_body_adjacency(self):
        llm = FakeLLM([json.dumps({
            'triples': [['List title', 'lists', 'later row']],
            'support_quotes': ['List title\nlater row'], 'status': 'success',
        })])
        result = OpenIE(llm, quality_max_retries=0).triple_extraction(
            'id', 'List title\nlater row', [], repair_context='Repair a table record.',
            _support_source='List title\nfirst row. later row')
        self.assertEqual(result.metadata['quality_status'], 'failed')
        self.assertEqual(result.metadata['support_quote_matches'], [False])

    def test_paired_repair_records_map_to_public_arrays_and_quotes(self):
        llm = FakeLLM([json.dumps({
            'triples': [
                {'triple': ['A', 'lives in', 'C'], 'support_quote': 'A and B live in C'},
                {'triple': ['B', 'lives in', 'C'], 'support_quote': 'A and B live in C'},
            ], 'status': 'success',
        })])
        result = OpenIE(llm, guided_recovery=True).triple_extraction(
            'id', 'A and B live in C.', [], repair_context='Repair the parallel-subject relation.')
        self.assertEqual(result.triples, [['A', 'lives in', 'C'], ['B', 'lives in', 'C']])
        self.assertEqual(result.metadata['support_quotes'], ['A and B live in C'] * 2)
        self.assertEqual(result.metadata['support_quote_matches'], [True, True])
        self.assertEqual(result.metadata['quality_status'], 'success')

    def test_paired_repair_rejects_missing_quote_key(self):
        llm = FakeLLM(['{"triples":[{"triple":["A","r","B"]}],"status":"success"}'])
        result = OpenIE(llm, quality_max_retries=0).triple_extraction(
            'id', 'A relates to B', [], repair_context='Repair record.')
        self.assertEqual(result.metadata['quality_status'], 'failed')
        self.assertEqual(result.triples, [])

    def test_empty_paired_repair_explicitly_confirms_withdrawal(self):
        llm = FakeLLM(['{"triples":[],"status":"no_supported_relations"}'])
        result = OpenIE(llm).triple_extraction('id', 'An index heading.', [], repair_context='Check bad record.')
        self.assertEqual(result.metadata['quality_status'], 'empty_valid')
        self.assertEqual(result.metadata['repair_status'], 'no_supported_relations')

    def test_coordinators_cannot_be_predicates(self):
        result = validate_triples([['A', predicate, 'B']
                                  for predicate in ('and', 'or', '&', 'As Well As', 'and/or')]
                                 + [['A', 'portrayed by', 'C']])
        self.assertEqual(result.valid_triples, [['A', 'portrayed by', 'C']])
        self.assertEqual(len(result.invalid_triples), 5)

    def test_repair_demonstration_expands_shared_predicates_with_entity_arguments(self):
        extractor = OpenIE(FakeLLM([]))
        messages = extractor._triple_messages('Actual source.', [], repair_context='Original extraction empty.')
        demonstration = json.loads(next(message['content'] for message in messages
                                        if message['role'] == 'assistant'))
        triples = [record['triple'] for record in demonstration['triples']]
        self.assertEqual(len([triple for triple in triples if triple[1] == 'portrayed by']), 2)
        family_triples = [triple for triple in triples if triple[1] in ('daughter of', 'son of')]
        self.assertEqual(len(family_triples), 4)
        self.assertEqual(len({tuple(triple) for triple in family_triples}), 4)
        self.assertEqual(len({triple[0] for triple in family_triples}), 2)
        self.assertEqual(len({triple[2] for triple in family_triples}), 2)
        self.assertFalse(any(triple[1] in ('and', 'or') for triple in triples))
        self.assertIn('originally empty/failed', messages[-1]['content'])

    def test_connector_output_enters_recovery_instead_of_success(self):
        quote = 'A and B are portrayed by C'
        llm = FakeLLM([
            json.dumps({'triples': [{'triple': ['A', 'and', 'B'], 'support_quote': quote}],
                        'status': 'success'}),
            json.dumps({'triples': [
                {'triple': ['A', 'portrayed by', 'C'], 'support_quote': quote},
                {'triple': ['B', 'portrayed by', 'C'], 'support_quote': quote},
            ], 'status': 'success'}),
        ])
        result = OpenIE(llm, quality_max_retries=1).triple_extraction(
            'id', quote, [], repair_context='Original extraction empty; recover all relations.')
        self.assertEqual(result.triples, [['A', 'portrayed by', 'C'], ['B', 'portrayed by', 'C']])
        self.assertEqual(result.metadata['quality_status'], 'success')
        self.assertIn('coordinating word', llm.calls[1]['messages'][-2]['content'])

    def test_entity_argument_detection_leaves_real_properties_and_ner_absences_alone(self):
        triples = [['A', 'is', 'son of B'], ['station', 'is', 'below grade'],
                   ['A', 'son of', 'B'], ['A', 'is', 'child of an unknown person']]
        self.assertEqual(len(entity_argument_issues(triples, ['A', 'B', 'unused entity'])), 1)

    def test_repair_puts_known_relative_entity_in_object_after_feedback(self):
        source = 'A is a son of B.'
        llm = FakeLLM([
            json.dumps({'triples': [{'triple': ['A', 'is', 'son of B'], 'support_quote': source}],
                        'status': 'success'}),
            json.dumps({'triples': [{'triple': ['A', 'son of', 'B'], 'support_quote': source}],
                        'status': 'success'}),
        ])
        result = OpenIE(llm, quality_max_retries=1).triple_extraction(
            'id', source, ['A', 'B'], repair_context='Empty result; extract source relations.')
        self.assertEqual(result.triples, [['A', 'son of', 'B']])
        self.assertEqual(result.metadata['quality_status'], 'success')

    def test_bad_quote_feedback_identifies_record_and_preserves_literal_source_punctuation(self):
        source = 'Album tracks were reissued as part of "" but dated incorrectly.'
        bad_quote = 'Album tracks were reissued as part of " but dated incorrectly.'
        triple = ['Album', 'tracks were reissued as part of', 'an unspecified collection']
        llm = FakeLLM([
            json.dumps({'triples': [{'triple': triple, 'support_quote': bad_quote}], 'status': 'success'}),
            json.dumps({'triples': [{'triple': triple, 'support_quote': source}], 'status': 'success'}),
        ])
        result = OpenIE(llm, quality_max_retries=1).triple_extraction(
            'id', source, [], repair_context='Check defective record.')
        self.assertEqual(result.metadata['quality_status'], 'success')
        feedback = llm.calls[1]['messages'][-2]['content']
        self.assertIn('record indices [0]', feedback)
        self.assertIn(json.dumps(triple, ensure_ascii=False), feedback)
        self.assertIn(json.dumps(bad_quote, ensure_ascii=False), feedback)
        self.assertIn(json.dumps(source, ensure_ascii=False), feedback)
        self.assertIn('including quotes and punctuation', feedback)

    def test_joined_sentence_quote_is_not_accepted_even_with_overlapping_words(self):
        source = 'A controls two thirds of a region. Assessments include Valley, Jammu and Ladakh.'
        bad_quote = 'A controls two thirds of a region including Valley, Jammu and Ladakh.'
        llm = FakeLLM([json.dumps({'triples': [
            {'triple': ['A', 'controls', 'Valley'], 'support_quote': bad_quote},
        ], 'status': 'success'})])
        result = OpenIE(llm, quality_max_retries=0).triple_extraction(
            'id', source, [], repair_context='Check record.')
        self.assertEqual(result.metadata['quality_status'], 'failed')
        self.assertEqual(result.metadata['support_quote_matches'], [False])
        self.assertIn('join separated sentences', result.metadata['openie_skip_reason'])

    def test_long_targeted_repair_uses_windows_after_first_whole_chunk_length(self):
        passage = 'A long table\n' + ' '.join(f'row{i} names team{i}' for i in range(100))
        windows = OpenIE._recovery_windows(passage)
        responses = [('truncated', 'length')]
        for _, _, text in windows:
            match = re.search(r'row(\d+) names team\d+', text)
            number = match.group(1)
            responses.append(json.dumps({'triples': [{
                'triple': [f'row{number}', 'names', f'team{number}'],
                'support_quote': match.group(0),
            }], 'status': 'success'}))
        llm = FakeLLM(responses)
        result = OpenIE(llm, quality_max_retries=0).triple_extraction(
            'original-id', passage, [], repair_context='Empty table extraction; recover source relations.')
        self.assertEqual(result.metadata['quality_status'], 'success')
        self.assertTrue(result.metadata['window_early_fallback'])
        self.assertEqual(result.metadata['openie_attempt_count'], 1)
        self.assertEqual(len(llm.calls), 1 + len(windows))
        self.assertEqual(result.metadata['window_recovery'][-1]['source_end'], len(passage))
        self.assertTrue(all(call['max_completion_tokens'] == 2048 for call in llm.calls))

    def test_targeted_repair_cannot_use_partial_windows_for_over_bound_source(self):
        passage = 'word ' * 3000
        self.assertEqual(OpenIE._recovery_windows(passage), [])
        llm = FakeLLM([('truncated', 'length')] * 4)
        result = OpenIE(llm, quality_max_retries=0).triple_extraction(
            'id', passage, [], repair_context='Recover original source.')
        self.assertEqual(result.metadata['quality_status'], 'failed')
        self.assertEqual(result.triples, [])
        self.assertNotIn('window_recovery', result.metadata)
        self.assertNotIn('window_early_fallback', result.metadata)
        self.assertEqual(len(llm.calls), 4)


if __name__ == "__main__":
    unittest.main()
