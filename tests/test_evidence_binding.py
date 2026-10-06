"""CPU tests for same-document alias support and relation ownership."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_evidence_binding_test_package"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/pathcondrag")]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(PACKAGE + "." + name,
                                                  ROOT / "src/pathcondrag" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load("evidence_retrieval")
binding = load("evidence_binding")


def hypothesis(answer, evidence, entity, relation, subject=None, **extra):
    item = {"answer": answer, "evidence": evidence, "doc_id": "D0", "confidence": .9,
            "answer_entity": entity, "answer_relation": relation,
            "subject_evidence": subject or evidence}
    item.update(extra)
    return {"hypotheses": [item]}


def verify(payload, document, question, kind="person"):
    return binding.verify_typed_hypotheses(payload, {0: document}, question, kind)


class BindingTests(unittest.TestCase):
    def test_title_and_middle_name_support_canonical_person(self):
        quote = "Antonio Lucio Vivaldi was an Italian composer."
        document = "Antonio Vivaldi\n" + quote
        accepted, rejected = verify(hypothesis("Antonio Vivaldi", quote, "Antonio Lucio Vivaldi", "other"),
                                     document, "Who was an Italian composer?")
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["answer"], "Antonio Vivaldi")
        self.assertTrue(accepted[0]["relation_supported"])
        self.assertEqual(accepted[0]["binding_support"]["answer_surface"], "introductory_full_name")

    def test_title_anchors_surname_and_separate_identity(self):
        intro = "Bob Iger is an American businessman."
        quote = "Iger was born on February 10, 1951."
        document = "Bob Iger\n" + intro + " " + quote
        accepted, rejected = verify(hypothesis("1951", quote, "Bob Iger", "birth_date", intro),
                                     document, "In what year was Bob Iger born?", "year")
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["answer"], "1951")

    def test_surname_answer_needs_document_identity(self):
        intro = "Bob Iger is an American businessman."
        quote = "Iger became the chief executive in 2005."
        document = "Bob Iger\n" + intro + " " + quote
        accepted, rejected = verify(hypothesis("Bob Iger", quote, "Bob Iger", "other", intro),
                                     document, "Who became the chief executive in 2005?")
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["binding_support"]["answer_surface"], "document_anchored_surname")
        accepted, rejected = verify(hypothesis("Bob Iger", quote, "Bob Iger", "other", intro),
                                     "Business\n" + quote, "Who became the chief executive in 2005?")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "answer_not_supported_in_quote")

    def test_different_person_sharing_surname_is_not_alias(self):
        intro = "Bob Iger is an American businessman."
        quote = "Arthur Iger was born in 1926."
        accepted, rejected = verify(hypothesis("Bob Iger", quote, "Bob Iger", "other", intro),
                                     "Bob Iger\n" + intro + " " + quote, "Who was born in 1926?")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "answer_not_supported_in_quote")

    def test_film_nationality_is_not_director_nationality(self):
        quote = "Il saprofita is an Italian film directed by Sergio Nasca."
        accepted, rejected = verify(hypothesis("Italian", quote, "Sergio Nasca", "nationality"),
                                     "Il saprofita\n" + quote, "What nationality was Sergio Nasca?", "nationality")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "answer_attribute_belongs_to_other_entity")

    def test_person_nationality_can_include_film_director(self):
        quote = "Sergio Nasca was an Italian film director."
        accepted, rejected = verify(hypothesis("Italian", quote, "Sergio Nasca", "nationality"),
                                     "Sergio Nasca\n" + quote, "What nationality was Sergio Nasca?", "nationality")
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["answer"], "Italian")

    def test_birth_country_is_not_automatically_citizenship(self):
        quote = "Sergio Nasca was born in Italy."
        accepted, rejected = verify(hypothesis("Italy", quote, "Sergio Nasca", "nationality"),
                                     "Sergio Nasca\n" + quote, "What nationality was Sergio Nasca?", "nationality")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "birthplace_does_not_establish_nationality")

    def test_question_subject_mismatch_not_fixed_by_cooccurrence(self):
        quote = "Bob Iger was born in 1951. Arthur Iger was born in 1926."
        accepted, rejected = verify(hypothesis("1926", quote, "Arthur Iger", "birth_date"),
                                     "Bob Iger\n" + quote, "When was Bob Iger born?", "year")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "relationship_subject_mismatch")

    def test_wrong_person_in_same_multi_sentence_quote_is_rejected(self):
        quote = "Bob Iger was born in 1951. Arthur Iger was born in 1926."
        accepted, rejected = verify(hypothesis("1926", quote, "Bob Iger", "birth_date"),
                                     "Bob Iger\n" + quote, "When was Bob Iger born?", "year")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "answer_not_supported_for_requested_relationship")

    def test_birth_and_death_years_not_interchangeable(self):
        quote = "Sergio Nasca (born 1937) died in 1989."
        accepted, rejected = verify(hypothesis("1989", quote, "Sergio Nasca", "birth_date"),
                                     "Sergio Nasca\n" + quote, "When was Sergio Nasca born?", "year")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "answer_not_supported_for_requested_relationship")

    def test_explicit_wikipedia_life_dates_match_left_and_right(self):
        quote = "John Smith (13 November 1948 – 2 July 2000) was an English writer."
        document = "John Smith\n" + quote
        accepted, rejected = verify(hypothesis("1948", quote, "John Smith", "birth_date"),
                                     document, "When was John Smith born?", "year")
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["answer"], "1948")
        accepted, rejected = verify(hypothesis("2 July 2000", quote, "John Smith", "death_date"),
                                     document, "When did John Smith die?", "date")
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["answer"], "2 July 2000")

    def test_life_range_does_not_swap_birth_and_death(self):
        quote = "John Smith (13 November 1948 – 2 July 2000) was an English writer."
        for answer, relation, question in [("2000", "birth_date", "When was John Smith born?"),
                                            ("1948", "death_date", "When did John Smith die?")]:
            with self.subTest(relation=relation):
                accepted, rejected = verify(hypothesis(answer, quote, "John Smith", relation),
                                             "John Smith\n" + quote, question, "year")
                self.assertFalse(accepted)
                self.assertEqual(rejected[0]["reason"], "requested_relationship_not_in_quote")

    def test_life_range_of_other_person_or_relative_cannot_supply_date(self):
        for quote in ["John Smith was an English writer. Arthur Smith (13 November 1948 – 2 July 2000) was a carpenter.",
                      "John Smith's father Arthur Smith (13 November 1948 – 2 July 2000) was a carpenter."]:
            with self.subTest(quote=quote):
                accepted, rejected = verify(hypothesis("1948", quote, "John Smith", "birth_date"),
                                             "John Smith\n" + quote, "When was John Smith born?", "year")
                self.assertFalse(accepted)
                self.assertEqual(rejected[0]["reason"], "requested_relationship_not_in_quote")

    def test_non_date_parenthetical_interval_is_not_life_range(self):
        quote = "John Smith (office 1948 – term 2000) was an English writer."
        accepted, rejected = verify(hypothesis("1948", quote, "John Smith", "birth_date"),
                                     "John Smith\n" + quote, "When was John Smith born?", "year")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "requested_relationship_not_in_quote")

    def test_exact_quotes_and_identity_support_required(self):
        quote = "Sergio Nasca was an Italian director."
        payload = hypothesis("Italian", quote, "Sergio Nasca", "nationality", "Sergio Nasca is a director.")
        accepted, rejected = verify(payload, quote, "What nationality was Sergio Nasca?", "nationality")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "subject_evidence_not_exact_substring")

    def test_requested_relation_must_match(self):
        quote = "Sergio Nasca was an Italian director."
        accepted, rejected = verify(hypothesis("Italian", quote, "Sergio Nasca", "birth_place"),
                                     quote, "What nationality was Sergio Nasca?", "nationality")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "requested_relationship_mismatch")

    def test_year_requires_numeric_value(self):
        quote = "Sergio Nasca was born in Italy."
        accepted, rejected = verify(hypothesis("Italy", quote, "Sergio Nasca", "birth_date"),
                                     quote, "When was Sergio Nasca born?", "year")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "answer_type_mismatch")

    def test_unsupported_hypotheses_empty_is_valid(self):
        self.assertEqual(verify({"hypotheses": []}, "No supporting fact.", "Who directed X?"), ([], []))

    def test_mixin_disabled_delegates_and_http_failure_propagates(self):
        class Parent:
            def _verify(self, *args):
                return "original"

        class Engine(binding.BindingImprovementMixin, Parent):
            improvements = frozenset()

        engine = Engine()
        self.assertEqual(engine._verify("Question?", "person", {}), "original")
        engine.improvements = frozenset({"binding"})
        engine._verification_diagnostics = {}
        engine._infer_object = lambda messages: (_ for _ in ()).throw(ConnectionError("HTTP failed"))
        with self.assertRaisesRegex(ConnectionError, "HTTP failed"):
            engine._verify("Question?", "person", {})

    def test_one_batched_request_and_at_most_one_repair(self):
        class Engine(binding.BindingImprovementMixin):
            improvements = frozenset({"binding"})

            def __init__(self, responses):
                self.responses = iter(responses)
                self.calls = []
                self._verification_diagnostics = {}

            def _infer_object(self, messages):
                self.calls.append(messages)
                return next(self.responses)

        quote = "Sergio Nasca was an Italian film director."
        docs = {0: quote, 1: "Another passage with no answer."}
        valid = hypothesis("Italian", quote, "Sergio Nasca", "nationality")
        engine = Engine([(valid, None)])
        accepted, _, _ = engine._verify("What nationality was Sergio Nasca?", "nationality", docs)
        self.assertTrue(accepted)
        self.assertEqual(len(engine.calls), 1)
        self.assertIn("[D1]", engine.calls[0][1]["content"])
        engine = Engine([(None, "invalid_json_object"), ({"hypotheses": []}, None)])
        self.assertFalse(engine._verify("What nationality was Sergio Nasca?", "nationality", docs)[0])
        self.assertEqual(len(engine.calls), 2)


if __name__ == "__main__":
    unittest.main()
