"""CPU checks for actual source offsets, local ownership and verifier reuse."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE=ModuleType("source_witness_test_package");PACKAGE.__path__=[str(ROOT/"src/pathcondrag")];sys.modules[PACKAGE.__name__]=PACKAGE
SPEC=importlib.util.spec_from_file_location(PACKAGE.__name__+".evidence_source_witness",ROOT/"src/pathcondrag/evidence_source_witness.py")
m=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=m;SPEC.loader.exec_module(m)


def check(q,a,quote,doc=None,kind="entity",node=None,bindings=None):
    doc=doc or "Article\n"+quote
    return m.source_witness_check(node or {"answer_type":kind,"depends_on":[]},q,{"answer":a,"evidence":quote,"confidence":.99},doc,bindings or {})


class Parent:
    def _verify(self,q,kind,docs):
        self.parent_calls+=1;self._verification_diagnostics[q+"::"+",".join(map(str,docs))]={"attempts":2,"outputs":self.payloads}
        return self.result


class Engine(m.SourceWitnessMixin,Parent):
    def __init__(self,accepted,payloads,enabled=True):
        self.improvements={"source_witness"} if enabled else set();self.parent_calls=0;self.payloads=payloads;self._verification_diagnostics={};self.result=(accepted,[{"reason":"original_rejection"}],None)


class SourceWitnessTests(unittest.TestCase):
    def test_exact_span_roundtrips(self):
        doc='Alice Example\nAlice Example designed Sample Bridge.';q=doc.splitlines()[1]
        span,mode=m.literal_source_span(q,doc);self.assertEqual(doc[slice(*span)],q);self.assertEqual(mode,"exact")

    def test_reversible_whitespace_and_curly_quotes(self):
        doc='Work\nAlice  Example wrote “Sample Work”.'
        span,mode=m.literal_source_span('Alice Example wrote "Sample Work".',doc)
        self.assertEqual(doc[slice(*span)],'Alice  Example wrote “Sample Work”.');self.assertIn('reversible',mode)

    def test_ellipsis_concat_and_paraphrase_not_repaired(self):
        doc='Work\nAlice Example wrote Sample Work and painted Another Work.'
        for q in ['Alice Example wrote ... Another Work.','Alice Example authored Sample Work.']:
            self.assertIsNone(m.literal_source_span(q,doc)[0])

    def test_ambiguous_normalized_occurrences_not_repaired(self):
        doc='Work\nAlice  Example wrote Sample Work.\nAlice   Example wrote Sample Work.'
        self.assertIsNone(m.literal_source_span('Alice Example wrote Sample Work.',doc)[0])

    def test_active_and_passive_predicate_direction(self):
        for quote in ['Alice Example designed Sample Bridge.','Sample Bridge was designed by Alice Example.']:
            self.assertIsNotNone(check('Who designed Sample Bridge?','Alice Example',quote)[0])
        bad=check('Who designed Sample Bridge?','Alice Example','Sample Bridge designed Alice Example.')
        self.assertTrue(bad[1].startswith('contradicted_direction'))

    def test_typed_agent_and_role_agent_questions_do_not_reverse_direction(self):
        quote='Alice Example designed Sample Bridge.'
        self.assertIsNotNone(check('Which architect designed Sample Bridge?','Alice Example',quote)[0])
        self.assertIsNotNone(check('Who is the captain of Sample Team?','Alice Example','Alice Example is captain of Sample Team.')[0])

    def test_year_cannot_come_from_an_unrelated_second_clause(self):
        quote='Sample Company was founded by Alice Example in 2005. Sample Company hired Alice Example in 1999.'
        self.assertIsNone(check('Who founded Sample Company in 1999?','Alice Example',quote)[0])

    def test_negative_question_is_not_supported_with_positive_source(self):
        self.assertIsNone(check('What did Alice Example not design?','Sample Bridge','Alice Example designed Sample Bridge.')[0])

    def test_object_question_active_and_passive_direction(self):
        for quote in ['Alice Example designed Sample Bridge.','Sample Bridge was designed by Alice Example.']:
            self.assertIsNotNone(check('What did Alice Example design?','Sample Bridge',quote)[0])
        self.assertTrue(check('What did Alice Example design?','Sample Bridge','Sample Bridge designed Alice Example.')[1].startswith('contradicted_direction'))

    def test_known_director_alias_and_role_question(self):
        quote='Sample Film was directed by Alice Example.'
        self.assertIsNotNone(check('Who directed Sample Film?','Alice Example',quote)[0])
        self.assertIsNotNone(check('Who is the director of Sample Film?','Alice Example',quote)[0])

    def test_quantity_unit_is_adjacent_and_owned(self):
        quote='La jolie fille de Perth is an opera in four acts by Georges Bizet.'
        self.assertIsNotNone(check('How many acts does La jolie fille de Perth have?','four',quote)[0])
        self.assertIsNone(check('How many acts does La jolie fille de Perth have?','four','La jolie fille de Perth has a cast of four singers and three acts.')[0])

    def test_shared_capitalized_words_do_not_establish_relation(self):
        self.assertIsNone(check('Who designed Sample Bridge?','Alice Example','Alice Example met Bob Other at Sample Bridge.')[0])

    def test_relative_owner_is_not_parent_attribute(self):
        quote="Alice Example's father designed Sample Bridge."
        self.assertIsNone(check('What did Alice Example design?','Sample Bridge',quote)[0])
        self.assertTrue(check('What did Alice Example design?','Sample Bridge',quote)[1].startswith('contradicted_owner'))

    def test_cross_clause_entity_cannot_take_predicate(self):
        quote='Alice Example met Bob Other, who designed Sample Bridge.'
        self.assertIsNone(check('What did Alice Example design?','Sample Bridge',quote)[0])

    def test_negated_fact_does_not_answer_positive_question(self):
        result=check('What did Alice Example design?','Sample Bridge','Alice Example never designed Sample Bridge.')
        self.assertTrue(result[1].startswith('contradicted_negation'))

    def test_dropped_or_wrong_year_not_supported(self):
        q='What did Alice Example design in 1999?'
        self.assertIsNone(check(q,'Sample Bridge','Alice Example designed Sample Bridge.')[0])
        self.assertTrue(check(q,'Sample Bridge','Alice Example designed Sample Bridge in 2005.')[1].startswith('unknown_'))
        self.assertIsNotNone(check(q,'Sample Bridge','Alice Example designed Sample Bridge in 1999.')[0])

    def test_named_work_constraint_must_be_in_predicate_clause(self):
        quote='Alice Example designed Another Bridge. Sample Bridge was nearby.'
        self.assertIsNone(check('Who designed Sample Bridge?','Alice Example',quote)[0])

    def test_multiple_people_cannot_bind_single_person(self):
        result=check('Who founded Sample Company?','Alice and Bob','Alice and Bob founded Sample Company.',kind='person')
        self.assertTrue(result[1].startswith('contradicted_type'))

    def test_unknown_predicate_not_upgraded_by_confidence(self):
        self.assertIsNone(check('What did Alice Example admire?','Sample Bridge','Alice Example admired Sample Bridge.')[0])

    def test_adjacent_topic_pronoun_with_actual_offset(self):
        doc='Alice Example\nAlice Example is an architect. She designed Sample Bridge.'
        self.assertIsNotNone(check('What did Alice Example design?','Sample Bridge','She designed Sample Bridge.',doc)[0])
        bad='Alice Example\nAlice Example is an architect. Bob Other moved away. She designed Sample Bridge.'
        self.assertIsNone(check('What did Alice Example design?','Sample Bridge','She designed Sample Bridge.',bad)[0])

    def test_ambiguous_previous_sentence_multiple_names_not_resolved(self):
        doc='Alice Example\nAlice Example met Bob Other. She designed Sample Bridge.'
        self.assertIsNone(check('What did Alice Example design?','Sample Bridge','She designed Sample Bridge.',doc)[0])

    def test_surname_proof_recovery_preserves_true_contiguous_text(self):
        doc='Alice Example\nAlice Example is an actress. Example voiced Sample Character.'
        item={'answer':'Alice Example','doc_id':'D0','evidence':'Example voiced Sample Character.','confidence':.9}
        recovered,reason=m._recover_candidate(item,doc,'Who voiced Sample Character?','person')
        self.assertIsNotNone(recovered,reason);self.assertIn('contiguous_identity',reason)
        self.assertEqual(doc[slice(*recovered['source_witness']['source_span'])],recovered['evidence'])
        self.assertTrue(m.verify_hypotheses({'hypotheses':[recovered]},{0:doc})[0])

    def test_explicit_alias_recovery(self):
        doc='Robert Smith\nRobert Smith, also known as Bob Example, is an actor. Bob Example voiced Sample Character.'
        item={'answer':'Robert Smith','doc_id':0,'evidence':'Bob Example voiced Sample Character.','confidence':.9}
        recovered,reason=m._recover_candidate(item,doc,'Who voiced Sample Character?','person')
        self.assertIsNotNone(recovered,reason)

    def test_same_surname_wrong_full_name_not_recovered(self):
        doc='Alice Smith\nAlice Smith is an actress. Bob Smith voiced Sample Character.'
        item={'answer':'Alice Smith','doc_id':0,'evidence':'Bob Smith voiced Sample Character.','confidence':.9}
        self.assertIsNone(m._recover_candidate(item,doc,'Who voiced Sample Character?','person')[0])

    def test_recovered_pronoun_checks_relation_in_original_predicate(self):
        doc='Alice Example\nAlice Example is an actress. She visited Sample Character.'
        item={'answer':'Alice Example','doc_id':0,'evidence':'She visited Sample Character.','confidence':.9}
        self.assertIsNone(m._recover_candidate(item,doc,'Who voiced Sample Character?','person')[0])

    def test_wrapper_disabled_returns_exact_parent_tuple(self):
        engine=Engine([],[],False)
        self.assertIs(engine._verify('Question','entity',{}),engine.result);self.assertEqual(engine.parent_calls,1)

    def test_original_unknown_is_retained_with_unknown_metadata(self):
        quote='Alice Example admired Sample Bridge.';item={'answer':'Sample Bridge','doc_id':0,'evidence':quote,'confidence':.9};engine=Engine([item],[])
        actual=engine._verify('What did Alice Example admire?','entity',{0:'Alice Example\n'+quote})
        self.assertEqual(actual[0][0]['answer'],item['answer']);self.assertEqual(actual[0][0]['source_witness']['status'],'unknown');self.assertEqual(engine.parent_calls,1)

    def test_contradicted_existing_is_explicitly_rejected(self):
        quote='Alice Example never designed Sample Bridge.';item={'answer':'Sample Bridge','doc_id':0,'evidence':quote,'confidence':.9};engine=Engine([item],[])
        actual=engine._verify('What did Alice Example design?','entity',{0:'Alice Example\n'+quote})
        self.assertEqual(actual[0],[]);self.assertTrue(actual[1][-1]['reason'].startswith('contradicted_negation'))

    def test_wrapper_reuses_payload_preserves_attempts_outputs_and_zero_direct_calls(self):
        doc='Sample Bridge\nAlice  Example designed Sample Bridge.'
        payload={'hypotheses':[{'answer':'Alice Example','doc_id':0,'evidence':'Alice Example designed Sample Bridge.','confidence':.9}]};engine=Engine([],[payload])
        actual=engine._verify('Who designed Sample Bridge?','person',{0:doc})
        self.assertEqual(len(actual[0]),1);self.assertEqual(actual[0][0]['evidence'],'Alice  Example designed Sample Bridge.')
        diag=engine._verification_diagnostics['Who designed Sample Bridge?::0'];self.assertEqual(diag['attempts'],2);self.assertEqual(diag['outputs'],[payload]);self.assertEqual(diag['source_witness']['direct_extra_requests'],0);self.assertTrue(diag['source_witness']['may_enable_downstream_requests']);self.assertEqual(engine.parent_calls,1)

    def test_existing_accepted_stays_prioritized_over_repairs(self):
        quote='Alice Example designed Sample Bridge.';item={'answer':'Alice Example','doc_id':0,'evidence':quote,'confidence':.9};payload={'hypotheses':[dict(item,answer='Bob Other')]};engine=Engine([item],[payload])
        actual=engine._verify('Who designed Sample Bridge?','person',{0:'Sample Bridge\n'+quote});self.assertEqual([p['answer'] for p in actual[0]],['Alice Example']);self.assertEqual(engine._verification_diagnostics['Who designed Sample Bridge?::0']['source_witness']['repair_attempts'],0)

    def test_callback_does_not_consult_annotations(self):
        class Guard(dict):
            def get(self,key,default=None):
                if key in {'dataset','gold_docs','gold_answers','type','question_decomposition'}:raise AssertionError(key)
                return super().get(key,default)
        node=Guard(answer_type='entity',depends_on=['a']);bindings=Guard(a='Alice Example')
        quote='Alice Example designed Sample Bridge.'
        self.assertIsNotNone(check('What did Alice Example design?','Sample Bridge',quote,node=node,bindings=bindings)[0])

    def test_life_attribute_subject_does_not_include_born_suffix(self):
        quote = 'Alpha was born in Rome in 1960.'
        node = {'question': 'Where was ${r1.answer} born?', 'answer_type': 'place', 'depends_on': ['r1']}
        self.assertIsNotNone(check('Where was Alpha born?', 'Rome', quote,
                                   'Alpha\n' + quote, node=node, bindings={'r1': 'Alpha'})[0])
        self.assertIsNone(check('Where was Beta born?', 'Rome', quote, 'Alpha\n' + quote)[0])

    def test_unrelated_negation_is_unknown_rather_than_rejected_original(self):
        quote = 'Alice Example was born in Rome and never designed Sample Bridge.'
        proof = {'answer': 'Rome', 'doc_id': 0, 'evidence': quote, 'confidence': .9}
        result = check('Where was Alice Example born?', 'Rome', quote, kind='place')
        self.assertTrue(result[1].startswith('unknown_'))
        engine = Engine([proof], [])
        accepted, _, _ = engine._verify('Where was Alice Example born?', 'place', {0: 'Alice Example\n' + quote})
        self.assertEqual(accepted[0]['answer'], 'Rome')

    def test_title_with_lowercase_and_is_not_a_person_list(self):
        quote = 'John Smith, 3rd Baron of North and South, designed Sample Bridge.'
        result = check('Who designed Sample Bridge?', 'John Smith, 3rd Baron of North and South', quote, kind='person')
        self.assertFalse(str(result[1]).startswith('contradicted_type'))


if __name__=='__main__':unittest.main()
