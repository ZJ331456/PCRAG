"""CPU tests of evidence contracts, group scope and incomplete audit handling."""

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pathcondrag.index import openie_source_evidence as module


SOURCE = ('Randy and Sharon Marsh\nRandy Marsh and Sharon Marsh are fictional characters. '
          'Parker describes Randy as "the biggest dingbat in the entire show".')
QUOTE = 'Parker describes Randy as "the biggest dingbat in the entire show".'
SINGLE = ['Randy Marsh', 'described as', 'the biggest dingbat in the entire show']
GROUP = ['Randy and Sharon Marsh', 'described as', 'the biggest dingbat in the entire show']


def role(subject='Randy Marsh', mention='Randy', quote=QUOTE, supported=True):
    return {'source_subject': subject, 'mention': mention, 'relation_quote': quote,
            'relation_supported': supported}


def verdict(*, supported=True, quote=QUOTE, roles=None, **flags):
    result = {'supported': supported, 'quote': quote,
              'subject_roles': [role()] if roles is None else roles,
              'reason': 'The quoted clause attributes this description to the stated subject.'}
    result.update({flag: flags.get(flag, True) for flag in module.FLAGS})
    return result


def response(verdicts=(), *, has_relations=True, source_quote=QUOTE):
    return json.dumps({'source_has_supported_relations': has_relations,
                       'source_evidence_quote': source_quote,
                       'source_reason': 'Inspect the explicit original source statements.',
                       'verdicts': list(verdicts)})


class FakeLLM:
    def __init__(self, replies=(), *, token_count=1000, reply_factory=None):
        self.replies = iter(replies)
        self.calls = []
        self.token_count = token_count
        self.reply_factory = reply_factory

    def count_prompt_tokens(self, messages):
        return self.token_count

    def infer(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        reply = self.reply_factory(kwargs) if self.reply_factory else next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, tuple):
            return reply
        return reply, {'finish_reason': 'stop'}, False


class SourceEvidenceTests(unittest.TestCase):
    def setUp(self):
        # Inspect the exact schema handed to the CPU grammar compiler. The
        # compiler itself has independent real xgrammar contract tests.
        self.compiler_patch = patch.object(
            module, 'guided_json_parameters', side_effect=lambda schema: {
                'guided_json': copy.deepcopy(schema),
                'chat_template_kwargs': {'enable_thinking': False}})
        self.compiler = self.compiler_patch.start()
        self.addCleanup(self.compiler_patch.stop)

    def test_group_false_positive_is_blocked_but_original_single_subject_is_accepted(self):
        # Even an all-true LLM verdict cannot extend Randy's property to Sharon.
        llm = FakeLLM([response([verdict(), verdict()])])
        accepted, audit = module.verify_source_relations(llm, SOURCE, [GROUP, SINGLE])
        self.assertEqual(accepted, [SINGLE])
        self.assertEqual(audit['supported'], [False, True])
        self.assertEqual(audit['checks'][0]['rejection_kind'], 'subject_scope')
        self.assertEqual(audit['checks'][0]['subject_member_checks'][1]['candidate_member'], 'Sharon Marsh')
        self.assertFalse(audit['checks'][0]['subject_member_checks'][1]['supported'])
        self.assertTrue(audit['complete'])
        self.assertEqual(audit['n_unverified'], 0)
        self.assertEqual(audit['contract_version'], module.EVIDENCE_VERSION)

    def test_group_supported_for_both_actual_members_is_accepted(self):
        source = 'Randy Marsh and Sharon Marsh raise their child Stan.'
        candidate = ['Randy and Sharon Marsh', 'raise', 'Stan']
        roles = [role('Randy Marsh', 'Randy Marsh', source), role('Sharon Marsh', 'Sharon Marsh', source)]
        llm = FakeLLM([response([verdict(quote=source, roles=roles)], source_quote=source)])
        accepted, audit = module.verify_source_relations(llm, source, [candidate])
        self.assertEqual(accepted, [candidate])
        self.assertTrue(audit['checks'][0]['subject_scope_complete'])

    def test_group_resolved_name_cannot_hide_a_single_actual_mention(self):
        bad_role = role('Randy and Sharon Marsh', 'Randy')
        llm = FakeLLM([response([verdict(roles=[bad_role])])])
        accepted, audit = module.verify_source_relations(llm, SOURCE, [GROUP])
        self.assertEqual(accepted, [])
        self.assertFalse(audit['checks'][0]['whole_named_argument_supported'])

    def test_whole_passage_quote_cannot_turn_title_group_into_actual_clause_subject(self):
        invalid = response([verdict(quote=SOURCE, roles=[role(GROUP[0], GROUP[0], SOURCE)])])
        llm = FakeLLM([invalid] * 3)
        with self.assertRaisesRegex(module.SourceEvidenceError, 'only in the title'):
            module.verify_source_relations(llm, SOURCE, [GROUP])

    def test_named_organization_with_and_is_kept_as_one_exact_subject(self):
        source = 'Research and Development Department manages the laboratory.'
        candidate = ['Research and Development Department', 'manages', 'laboratory']
        roles = [role(candidate[0], candidate[0], source)]
        llm = FakeLLM([response([verdict(quote=source, roles=roles)], source_quote=source)])
        accepted, audit = module.verify_source_relations(llm, source, [candidate])
        self.assertEqual(accepted, [candidate])
        self.assertTrue(audit['checks'][0]['whole_named_argument_supported'])

    def test_original_plural_pronoun_resolves_a_whole_source_group_antecedent(self):
        source = 'Randy and Sharon Marsh\nRandy Marsh and Sharon Marsh are characters. They raise their child Stan.'
        quote = 'They raise their child Stan.'
        candidate = ['Randy and Sharon Marsh', 'raise', 'Stan']
        llm = FakeLLM([response([verdict(quote=quote, roles=[role(candidate[0], 'They', quote)])],
                               source_quote=quote)])
        accepted, audit = module.verify_source_relations(llm, source, [candidate])
        self.assertEqual(accepted, [candidate])
        self.assertTrue(all(item['supported'] for item in audit['checks'][0]['subject_member_checks']))
        self.assertTrue(audit['checks'][0]['subject_roles'][0]['coreference_resolved_from_whole_source'])

    def test_article_topic_with_a_real_original_it_argument_is_grounded(self):
        source = "Day's journey\nIn the Bible, it is not as precisely defined; the distance has been estimated from 32 to 40 kilometers."
        quote = source.split('\n', 1)[1]
        candidate = ["Day's journey", 'estimated distance', '32 to 40 kilometers']
        llm = FakeLLM([response([verdict(quote=quote, roles=[role(candidate[0], 'it', quote)])],
                               source_quote=quote)])
        accepted, audit = module.verify_source_relations(llm, source, [candidate])
        self.assertEqual(accepted, [candidate])
        self.assertTrue(audit['checks'][0]['subject_roles'][0]['coreference_resolved_from_whole_source'])

    def test_table_event_title_and_specific_body_row_establish_topic_context(self):
        title = '1999 Major League Baseball draft'
        body = 'Pick Player Team Position School Josh Hamilton Tampa Bay Devil Rays OF Athens Drive HS.'
        source = title + '\n' + body
        candidate = [title, 'includes player', 'Josh Hamilton']
        llm = FakeLLM([response([verdict(quote=body, roles=[role(title, title, body)])],
                               source_quote=body)])
        accepted, audit = module.verify_source_relations(llm, source, [candidate])
        self.assertEqual(accepted, [candidate])
        saved = audit['checks'][0]['subject_roles'][0]
        self.assertTrue(saved['title_topic_with_body_evidence'])
        self.assertTrue(saved['evidence_grounded'])
        self.assertEqual(source[saved['source_subject_offsets'][0]['start']:
                                saved['source_subject_offsets'][0]['end']], title)

    def test_title_topic_does_not_supply_a_missing_object_from_a_different_body_fact(self):
        source = '1999 Major League Baseball draft\nJosh Hamilton joined Team A.'
        quote = 'Josh Hamilton joined Team A.'
        candidate = ['1999 Major League Baseball draft', 'includes player', 'Absent Player']
        invalid = response([verdict(quote=quote, roles=[role(candidate[0], candidate[0], quote)])],
                           source_quote=quote)
        llm = FakeLLM([invalid] * 3)
        with self.assertRaises(module.SourceEvidenceError):
            module.verify_source_relations(llm, source, [candidate])

    def _id_response(self, original, source, focus=None):
        data = json.loads(original)
        pool = module._source_quotes(source, focus)
        data['source_evidence_quote_id'] = pool.index(data.pop('source_evidence_quote'))
        for result in data['verdicts']:
            result['quote_id'] = pool.index(result.pop('quote'))
            for subject in result['subject_roles']:
                subject['relation_quote_id'] = pool.index(subject.pop('relation_quote'))
        return json.dumps(data)

    def test_production_quote_ids_expand_to_original_evidence_and_preserve_group_scope(self):
        original = response([verdict(supported=False, roles=[role(GROUP[0], GROUP[0], supported=False)],
                                    subject_scope_supported=False), verdict()])
        llm = FakeLLM([self._id_response(original, SOURCE)])
        accepted, audit = module.verify_source_relations(llm, SOURCE, [GROUP, SINGLE])
        self.assertEqual(accepted, [SINGLE])
        self.assertEqual(audit['checks'][1]['quote'], QUOTE)
        self.assertIsInstance(audit['checks'][1]['quote_id'], int)
        self.assertIsInstance(audit['checks'][1]['subject_roles'][0]['relation_quote_id'], int)
        guided = llm.calls[0]['extra_body']['guided_json']
        self.assertIn('source_evidence_quote_id', guided['properties'])
        self.assertNotIn('source_evidence_quote', guided['properties'])
        payload = json.loads(llm.calls[0]['messages'][1]['content'])
        for span in payload['evidence_spans']:
            for offset in span['positions']:
                self.assertEqual(SOURCE[offset['start']:offset['end']], span['text'])

    def test_production_quote_ids_never_accept_boolean_or_unknown_ids(self):
        original = self._id_response(response([verdict()]), SOURCE)
        for wrong_id in (True, -1, 9999):
            data = json.loads(original)
            data['verdicts'][0]['quote_id'] = wrong_id
            llm = FakeLLM([json.dumps(data)] * 3)
            with self.assertRaisesRegex(module.SourceEvidenceError, 'actual original source span ID'):
                module.verify_source_relations(llm, SOURCE, [SINGLE])

    def test_wrong_row_id_feedback_identifies_real_literal_mention_ids_without_approving_a_relation(self):
        source = ('Draft\nJosh Hamilton joined Team A. '
                  'B.J. Garbe attended Moses Lake HS. Josh Girdley joined Montreal Expos.')
        candidate = ['B.J. Garbe', 'attended', 'Moses Lake HS']
        pool = module._source_quotes(source)
        wrong_quote = 'Josh Hamilton joined Team A.'
        correct_quote = next(quote for quote in pool if 'B.J. Garbe' in quote and 'Moses Lake HS' in quote)
        wrong = verdict(quote=wrong_quote, roles=[role(candidate[0], candidate[0], wrong_quote)])
        corrected = verdict(quote=correct_quote, roles=[role(candidate[0], candidate[0], correct_quote)])
        llm = FakeLLM([self._id_response(response([wrong], source_quote=wrong_quote), source),
                       self._id_response(response([corrected], source_quote=correct_quote), source)])
        accepted, audit = module.verify_source_relations(llm, source, [candidate])
        self.assertEqual(accepted, [candidate])
        self.assertEqual(audit['shape_repair_count'], 1)
        self.assertIn('literal_span_ids', llm.calls[1]['messages'][-1]['content'])
        self.assertIn('B.J. Garbe', llm.calls[1]['messages'][-1]['content'])
        hint = json.loads(llm.calls[0]['messages'][1]['content'])['candidate_span_hints'][0]
        self.assertIn(pool.index(correct_quote), hint['joint_subject_object_ids'])
        self.assertNotIn(pool.index(wrong_quote), hint['joint_subject_object_ids'])

    def test_production_reason_strings_are_bounded_without_reducing_the_output_budget(self):
        llm = FakeLLM([response([verdict()])])
        module.verify_source_relations(llm, SOURCE, [SINGLE])
        settings = llm.calls[0]
        schema = settings['extra_body']['guided_json']
        self.assertEqual(schema['properties']['source_reason']['maxLength'], 240)
        self.assertEqual(schema['properties']['verdicts']['items']['properties']['reason']['maxLength'], 240)
        self.assertEqual(settings['max_completion_tokens'], 2048)

    def test_single_candidate_schema_forces_object_bearing_quote_ids_and_keeps_negative_verdicts(self):
        source = ('1999 Major League Baseball draft\nJosh Hamilton joined Tampa Bay. '
                  'Brett Myers joined Philadelphia Phillies.')
        candidate = ['1999 Major League Baseball draft', 'includes', 'Brett Myers']
        schema = module._schema(1, source, triples=[candidate])
        pool = module._source_quotes(source)
        early = pool.index('Josh Hamilton joined Tampa Bay.')
        verdict_schema = schema['properties']['verdicts']['items']
        quote_ids = verdict_schema['properties']['quote_id']['enum']
        role_ids = verdict_schema['properties']['subject_roles']['items']['properties']['relation_quote_id']['enum']
        self.assertNotIn(early, quote_ids)
        self.assertNotIn(early, role_ids)
        self.assertIn(pool.index(source), quote_ids)
        self.assertIn(pool.index(source), role_ids)
        self.assertTrue(all('Brett Myers' in pool[identifier] for identifier in quote_ids))
        self.assertEqual(verdict_schema['properties']['supported'], {'type': 'boolean'})

    def test_single_candidate_without_literal_source_object_does_not_force_fake_evidence(self):
        source = 'Josh Hamilton joined Tampa Bay.'
        schema = module._schema(1, source, triples=[['Absent Person', 'joined', 'Absent Team']])
        ids = schema['properties']['verdicts']['items']['properties']['quote_id']['enum']
        self.assertEqual(ids, list(range(len(module._source_quotes(source)))))

    def test_long_source_whole_span_remains_available_without_duplicating_its_text_in_prompt(self):
        source = 'Draft\n' + 'Several players joined several teams. ' * 40
        messages = module._messages(source, [])
        payload = json.loads(messages[1]['content'])
        whole_id = module._source_quotes(source).index(source.strip())
        whole = payload['evidence_spans'][whole_id]
        self.assertEqual(whole['scope'], 'whole_SOURCE')
        self.assertNotIn('text', whole)
        self.assertEqual(payload['SOURCE'], source)

    def test_invalid_multicandidate_evidence_uses_bounded_shape_repairs_then_splits_to_singletons(self):
        triples, source, factory = self._batch_data(4)

        def replies(kwargs):
            data = json.loads(kwargs['messages'][1]['content'])
            if len(data['candidate_triples']) > 1:
                first_quote = 'Actor0 trained in classical theater.'
                output = [verdict(quote=first_quote, roles=[role(triple[0], triple[0], first_quote)])
                          for triple in data['candidate_triples']]
                return response(output, source_quote=first_quote)
            return factory(kwargs)

        llm = FakeLLM(reply_factory=replies)
        accepted, audit = module.verify_source_relations(llm, source, triples)
        self.assertEqual(accepted, triples)
        self.assertEqual([len(json.loads(call['messages'][1]['content'])['candidate_triples'])
                          for call in llm.calls], [4, 4, 4, 2, 2, 2, 1, 1, 1, 1])
        self.assertEqual(audit['adaptive_split_count'], 2)
        self.assertTrue(all(batch['failure_kind'] == 'invalid_evidence'
                            for batch in audit['adaptive_incomplete_batches']))
        self.assertTrue(all(batch['complete'] for batch in audit['batches']))

    def test_multicandidate_transport_failure_does_not_split_into_more_requests(self):
        triples, source, _ = self._batch_data(4)
        llm = FakeLLM([RuntimeError('server refused request')])
        with self.assertRaises(module.SourceEvidenceError) as caught:
            module.verify_source_relations(llm, source, triples)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(caught.exception.audit_metadata['adaptive_split_count'], 0)

    def test_wrong_directed_subject_is_rejected(self):
        llm = FakeLLM([response([verdict(roles=[role('Parker', 'Parker')])])])
        accepted, audit = module.verify_source_relations(llm, SOURCE, [SINGLE])
        self.assertEqual(accepted, [])
        self.assertEqual(audit['checks'][0]['rejection_kind'], 'subject_scope')

    def test_surname_alone_is_not_alias_for_specific_person(self):
        llm = FakeLLM([response([verdict(roles=[role('Marsh', 'Randy')])])])
        accepted, _ = module.verify_source_relations(llm, SOURCE, [SINGLE])
        self.assertEqual(accepted, [])

    def test_exact_quote_offsets_are_saved_and_whole_original_source_is_preserved(self):
        llm = FakeLLM([response([verdict()])])
        accepted, audit = module.verify_source_relations(llm, SOURCE, [SINGLE])
        self.assertEqual(accepted, [SINGLE])
        offset = audit['checks'][0]['quote_offsets'][0]
        self.assertEqual(SOURCE[offset['start']:offset['end']], QUOTE)
        source_payload = json.loads(llm.calls[0]['messages'][1]['content'])
        self.assertEqual(source_payload['SOURCE'], SOURCE)
        settings = llm.calls[0]
        self.assertEqual(settings['max_completion_tokens'], 2048)
        self.assertEqual(settings['temperature'], 0.0)
        self.assertFalse(settings['extra_body']['chat_template_kwargs']['enable_thinking'])
        self.compiler.assert_called_once()
        self.assertEqual(settings['extra_body']['guided_json']['properties']['verdicts']['maxItems'], 1)

    def test_missing_or_paraphrased_quote_is_incomplete_after_two_shape_repairs(self):
        invalid = response([verdict(quote='Randy is described as the biggest dingbat.')])
        llm = FakeLLM([invalid] * 3)
        with self.assertRaises(module.SourceEvidenceError) as caught:
            module.verify_source_relations(llm, SOURCE, [SINGLE])
        audit = caught.exception.audit_metadata
        self.assertFalse(audit['complete'])
        self.assertEqual(audit['n_unverified'], 1)
        self.assertIsNone(audit['supported'])
        self.assertEqual(len(llm.calls), 3)
        self.assertIn('not an exact substring', str(caught.exception))

    def test_title_only_cooccurrence_quote_does_not_validate_a_relation(self):
        invalid = response([verdict(quote='Randy and Sharon Marsh',
                                    roles=[role('Randy Marsh', 'Randy', 'Randy and Sharon Marsh')])])
        llm = FakeLLM([invalid] * 3)
        with self.assertRaisesRegex(module.SourceEvidenceError, 'title-only'):
            module.verify_source_relations(llm, SOURCE, [SINGLE])

    def test_role_actual_mention_must_be_in_relation_bearing_quote(self):
        invalid = response([verdict(roles=[role('Sharon Marsh', 'Sharon')])])
        llm = FakeLLM([invalid] * 3)
        with self.assertRaisesRegex(module.SourceEvidenceError, 'actual subject mention'):
            module.verify_source_relations(llm, SOURCE, [GROUP])

    def test_shape_feedback_retry_can_complete_and_changes_the_request(self):
        llm = FakeLLM(['{}', response([verdict()])])
        accepted, audit = module.verify_source_relations(llm, SOURCE, [SINGLE])
        self.assertEqual(accepted, [SINGLE])
        self.assertEqual(audit['shape_repair_count'], 1)
        self.assertNotEqual(llm.calls[0]['messages'], llm.calls[1]['messages'])
        self.assertEqual(json.loads(llm.calls[1]['messages'][1]['content'])['SOURCE'], SOURCE)

    def test_transport_exception_is_unverified_not_a_negative_verdict(self):
        llm = FakeLLM([RuntimeError('server unavailable')])
        with self.assertRaises(module.SourceEvidenceError) as caught:
            module.verify_source_relations(llm, SOURCE, [SINGLE])
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(caught.exception.audit_metadata['n_unverified'], 1)
        self.assertFalse(caught.exception.audit_metadata['complete'])

    def test_explicit_grounded_negative_is_complete(self):
        llm = FakeLLM([response([verdict(supported=False, subject_scope_supported=False)])])
        accepted, audit = module.verify_source_relations(llm, SOURCE, [GROUP])
        self.assertEqual(accepted, [])
        self.assertTrue(audit['complete'])
        self.assertEqual(audit['n_rejected'], 1)
        self.assertFalse(audit['source_no_supported_relations'])

    def test_false_role_flags_override_a_true_model_supported_flag(self):
        llm = FakeLLM([response([verdict(attribution_supported=False)])])
        accepted, audit = module.verify_source_relations(llm, SOURCE, [SINGLE])
        self.assertEqual(accepted, [])
        self.assertEqual(audit['checks'][0]['rejection_kind'], 'attribution')

    def test_empty_candidates_trigger_an_independent_actual_source_request(self):
        llm = FakeLLM([response([], has_relations=True)])
        accepted, audit = module.verify_source_relations(llm, SOURCE, [])
        self.assertEqual(accepted, [])
        self.assertEqual(len(llm.calls), 1)
        self.assertTrue(audit['source_has_supported_relations'])
        self.assertFalse(audit['source_no_supported_relations'])

    def test_legitimate_no_relation_source_has_explicit_nonempty_evidence(self):
        source = 'Contents'
        llm = FakeLLM([response([], has_relations=False, source_quote=source)])
        accepted, audit = module.verify_source_relations(llm, source, [])
        self.assertEqual(accepted, [])
        self.assertTrue(audit['source_no_supported_relations'])
        self.assertTrue(audit['complete'])
        self.assertEqual(len(llm.calls), 1)

    def test_context_limit_is_checked_before_the_request(self):
        llm = FakeLLM(token_count=7000)
        with self.assertRaises(module.SourceEvidenceError) as caught:
            module.verify_source_relations(llm, SOURCE, [])
        self.assertEqual(llm.calls, [])
        self.assertIn('input+2048 exceeds 8192', str(caught.exception))

    def test_invalid_candidate_is_not_silently_normalized(self):
        llm = FakeLLM()
        with self.assertRaises(ValueError):
            module.verify_source_relations(llm, SOURCE, [['Randy', 'and', 'Sharon']])
        self.assertEqual(llm.calls, [])

    def test_duplicate_candidates_are_rejected_to_preserve_order_identity(self):
        with self.assertRaises(ValueError):
            module.verify_source_relations(FakeLLM(), SOURCE, [SINGLE, SINGLE])

    def test_non_boolean_flags_are_not_accepted(self):
        invalid = response([verdict(relation_supported='true')])
        with self.assertRaises(module.SourceEvidenceError):
            module.verify_source_relations(FakeLLM([invalid] * 3), SOURCE, [SINGLE])

    def _batch_data(self, size=9):
        triples = [[f'Actor{i}', 'trained in', 'classical theater'] for i in range(size)]
        source = ' '.join(f'{triple[0]} trained in classical theater.' for triple in triples)

        def factory(kwargs):
            data = json.loads(kwargs['messages'][1]['content'])
            verdicts = []
            for triple in data['candidate_triples']:
                quote = f'{triple[0]} trained in classical theater.'
                verdicts.append(verdict(quote=quote, roles=[role(triple[0], triple[0], quote)]))
            return response(verdicts, source_quote=source)
        return triples, source, factory

    def test_candidate_batches_never_exceed_eight(self):
        triples, source, factory = self._batch_data()
        llm = FakeLLM(reply_factory=factory)
        accepted, audit = module.verify_source_relations(llm, source, triples)
        self.assertEqual(accepted, triples)
        self.assertEqual(audit['batch_count'], 3)
        self.assertEqual([len(json.loads(call['messages'][1]['content'])['candidate_triples'])
                          for call in llm.calls], [4, 4, 1])
        self.assertEqual([check['index'] for check in audit['checks']], list(range(9)))

    def test_retry_context_changes_cache_key_and_reduces_batches_to_four(self):
        triples, source, factory = self._batch_data()
        llm = FakeLLM(reply_factory=factory)
        accepted, audit = module.verify_source_relations(llm, source, triples,
                                                        retry_context='Previous role evidence was missing.')
        self.assertEqual(accepted, triples)
        self.assertEqual([len(json.loads(call['messages'][1]['content'])['candidate_triples'])
                          for call in llm.calls], [2, 2, 2, 2, 1])
        self.assertTrue(audit['retry_context_supplied'])
        self.assertIn('diagnostics are NOT factual evidence', llm.calls[0]['messages'][2]['content'])

    def test_length_response_immediately_halves_candidates_and_retains_the_whole_source(self):
        triples, source, factory = self._batch_data(8)

        def replies(kwargs):
            selected = json.loads(kwargs['messages'][1]['content'])['candidate_triples']
            if len(selected) > 2:
                return '{"verdicts": [', {'finish_reason': 'length'}, False
            return factory(kwargs)

        llm = FakeLLM(reply_factory=replies)
        accepted, audit = module.verify_source_relations(llm, source, triples)
        self.assertEqual(accepted, triples)
        sizes = [len(json.loads(call['messages'][1]['content'])['candidate_triples']) for call in llm.calls]
        self.assertEqual(sizes, [4, 2, 2, 2, 2])
        self.assertEqual(audit['adaptive_split_count'], 1)
        self.assertTrue(audit['complete'])
        self.assertTrue(all(batch['complete'] for batch in audit['batches']))
        self.assertTrue(all(not batch['complete'] for batch in audit['adaptive_incomplete_batches']))
        for call in llm.calls:
            self.assertEqual(json.loads(call['messages'][1]['content'])['SOURCE'], source)
            self.assertEqual(call['max_completion_tokens'], 2048)
            self.assertFalse(call['extra_body']['chat_template_kwargs']['enable_thinking'])

    def test_length_even_for_one_candidate_remains_incomplete_without_expanding_token_budget(self):
        llm = FakeLLM([('{}', {'finish_reason': 'length'}, False)])
        with self.assertRaises(module.SourceEvidenceError) as caught:
            module.verify_source_relations(llm, SOURCE, [SINGLE])
        self.assertEqual(len(llm.calls), 1)
        self.assertFalse(caught.exception.audit_metadata['complete'])
        self.assertEqual(caught.exception.audit_metadata['batches'][-1]['failure_kind'], 'output_length')

    def test_adaptive_length_splits_reach_one_if_quotes_need_large_outputs(self):
        triples, source, factory = self._batch_data(4)

        def replies(kwargs):
            selected = json.loads(kwargs['messages'][1]['content'])['candidate_triples']
            if len(selected) > 1:
                return '{', {'finish_reason': 'length'}, False
            return factory(kwargs)

        llm = FakeLLM(reply_factory=replies)
        accepted, audit = module.verify_source_relations(llm, source, triples)
        self.assertEqual(accepted, triples)
        sizes = [len(json.loads(call['messages'][1]['content'])['candidate_triples']) for call in llm.calls]
        self.assertEqual(sizes, [4, 2, 1, 1, 1, 1])
        self.assertEqual(audit['max_batch_triples'], 1)

    def test_later_batch_failure_never_publishes_partial_acceptance(self):
        triples, source, factory = self._batch_data()
        count = 0

        def replies(kwargs):
            nonlocal count
            count += 1
            return factory(kwargs) if count == 1 else RuntimeError('later backend failure')

        llm = FakeLLM(reply_factory=replies)
        with self.assertRaises(module.SourceEvidenceError) as caught:
            module.verify_source_relations(llm, source, triples)
        audit = caught.exception.audit_metadata
        self.assertFalse(audit['complete'])
        self.assertEqual(audit['n_unverified'], 9)
        self.assertEqual(audit['completed_batches'], 1)
        self.assertEqual(len(audit['partial_accepted']), 4)
        self.assertIsNone(audit['supported'])

    def test_empty_heading_focus_is_audited_independently_of_body_relations(self):
        end = SOURCE.index('\n')
        llm = FakeLLM([response([], has_relations=False, source_quote=SOURCE[:end])])
        audit = module.verify_source_empty_focus(llm, SOURCE, 0, end)
        self.assertTrue(audit['source_no_supported_relations'])
        self.assertEqual(audit['source_scope'], 'whole_source_with_original_focus')
        self.assertEqual(audit['source_relation_verdict_scope'], 'original_focus_only')
        data = json.loads(llm.calls[0]['messages'][1]['content'])
        self.assertEqual(data['SOURCE'], SOURCE)
        self.assertEqual(data['FOCUS'], {'start': 0, 'end': end, 'text': SOURCE[:end]})
        self.assertIn('Do not count relations only', llm.calls[0]['messages'][0]['content'])

    def test_empty_focus_with_an_actual_relation_is_not_approved_as_no_relation(self):
        start = SOURCE.index(QUOTE)
        llm = FakeLLM([response([], has_relations=True)])
        audit = module.verify_source_empty_focus(llm, SOURCE, start, len(SOURCE))
        self.assertFalse(audit['source_no_supported_relations'])
        self.assertTrue(audit['source_has_supported_relations'])

    def test_empty_focus_evidence_cannot_be_borrowed_from_another_source_clause(self):
        llm = FakeLLM([response([], has_relations=False)] * 3)
        with self.assertRaisesRegex(module.SourceEvidenceError, 'inside the exact original FOCUS'):
            module.verify_source_empty_focus(llm, SOURCE, 0, SOURCE.index('\n'))
        self.assertEqual(len(llm.calls), 3)

    def test_explicit_false_group_role_uses_real_conflict_quote_without_fabricating_a_mention(self):
        negative_role = role(GROUP[0], GROUP[0], supported=False)
        llm = FakeLLM([response([verdict(supported=False, roles=[negative_role],
                                       subject_scope_supported=False), verdict()])])
        accepted, audit = module.verify_source_relations(llm, SOURCE, [GROUP, SINGLE])
        self.assertEqual(accepted, [SINGLE])
        self.assertEqual(len(llm.calls), 1)
        self.assertTrue(audit['complete'])
        self.assertEqual(audit['supported'], [False, True])
        saved_role = audit['checks'][0]['subject_roles'][0]
        self.assertFalse(saved_role['evidence_grounded'])
        self.assertTrue(saved_role['diagnostic_only'])
        self.assertEqual(saved_role['relation_quote'], QUOTE)

    def test_false_role_for_a_missing_source_name_is_complete_negative_not_fictitious_evidence(self):
        candidate = ['Absent Person', 'described as', SINGLE[2]]
        negative_role = role('Absent Person', 'Absent Person', supported=False)
        llm = FakeLLM([response([verdict(supported=False, roles=[negative_role],
                                       subject_scope_supported=False)])])
        accepted, audit = module.verify_source_relations(llm, SOURCE, [candidate])
        self.assertEqual(accepted, [])
        self.assertTrue(audit['complete'])
        self.assertEqual(audit['checks'][0]['subject_roles'][0]['source_subject_offsets'], [])
        self.assertFalse(audit['checks'][0]['subject_roles'][0]['evidence_grounded'])

    def test_true_candidate_flag_cannot_be_approved_using_only_explicit_false_roles(self):
        negative_role = role(GROUP[0], GROUP[0], supported=False)
        llm = FakeLLM([response([verdict(supported=True, roles=[negative_role])])])
        accepted, audit = module.verify_source_relations(llm, SOURCE, [GROUP])
        self.assertEqual(accepted, [])
        self.assertFalse(audit['checks'][0]['subject_scope_complete'])
        self.assertEqual(audit['checks'][0]['rejection_kind'], 'subject_scope')

    def test_false_role_still_needs_a_real_relation_bearing_conflict_quote(self):
        negative_role = role(GROUP[0], GROUP[0], quote='Invented conflicting evidence.', supported=False)
        invalid = response([verdict(supported=False, roles=[negative_role])])
        llm = FakeLLM([invalid] * 3)
        with self.assertRaisesRegex(module.SourceEvidenceError, 'not an exact substring'):
            module.verify_source_relations(llm, SOURCE, [GROUP])

    def test_empty_focus_bounds_are_original_character_offsets_and_are_validated(self):
        llm = FakeLLM()
        for bounds in ((-1, 3), (0, len(SOURCE) + 1), (3, 3), (True, 3)):
            with self.assertRaises(ValueError):
                module.verify_source_empty_focus(llm, SOURCE, *bounds)
        self.assertEqual(llm.calls, [])

    def test_whitespace_focus_is_deterministically_empty_without_dropping_its_span(self):
        source = 'Heading\n  \nActual sentence has a relation.'
        llm = FakeLLM()
        audit = module.verify_source_empty_focus(llm, source, 8, 10)
        self.assertTrue(audit['source_no_supported_relations'])
        self.assertTrue(audit['deterministic_empty_focus'])
        self.assertEqual(audit['source_focus']['text'], '  ')
        self.assertEqual(llm.calls, [])


if __name__ == '__main__':
    unittest.main()
