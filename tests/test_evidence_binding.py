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


def verify(payload, document, question, kind="person", validation="legacy"):
    return binding.verify_typed_hypotheses(payload, {0: document}, question, kind, validation)


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

    def test_strict_country_location_rejects_port_location(self):
        intro = "Israel is a country in the Middle East."
        quote = "On the Mediterranean coast, Haifa Port is the country's oldest and largest port."
        payload = hypothesis("Mediterranean coast", quote, "Israel", "location", intro)
        accepted, rejected = verify(payload, "Israel\n" + intro + " " + quote,
                                     "Where is Israel located?", "place", "strict_relation")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "requested_location_not_supported_for_subject")
        accepted, rejected = verify(hypothesis("Middle East", intro, "Israel", "location"),
                                     "Israel\n" + intro, "Where is Israel located?", "place", "strict_relation")
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["binding_support"]["validation"], "strict_relation")
        self.assertTrue(accepted[0]["binding_support"]["mechanical_relation_checked"])
        self.assertFalse(accepted[0]["binding_support"]["entailment_guaranteed"])

    def test_strict_country_mention_cannot_own_its_port_location(self):
        intro = "Israel is a country."
        for quote in ("Israel's Haifa Port is located on the Mediterranean coast.",
                      "The port in Israel is located on the Mediterranean coast."):
            with self.subTest(quote=quote):
                accepted, rejected = verify(hypothesis("Mediterranean coast", quote, "Israel", "location", intro),
                                             "Israel\n" + intro + " " + quote,
                                             "Where is Israel located?", "place", "strict_relation")
                self.assertFalse(accepted)
                self.assertEqual(rejected[0]["reason"], "requested_location_not_supported_for_subject")

    def test_strict_location_uses_document_anchored_pronoun(self):
        intro = "Israel is a country."
        quote = "It is located in Western Asia."
        accepted, rejected = verify(hypothesis("Western Asia", quote, "Israel", "location", intro),
                                     "Israel\n" + intro + " " + quote,
                                     "Where is Israel located?", "place", "strict_relation")
        self.assertFalse(rejected)
        self.assertTrue(accepted)

    def test_strict_neighbor_cannot_change_requested_subject(self):
        quote = "Spain is bordered to the north and northeast by France, Andorra, and the Bay of Biscay."
        accepted, rejected = verify(hypothesis("Andorra", quote, "Spain", "north_of"),
                                     "Spain\n" + quote,
                                     "What is the region immediately north of Mediterranean coast?", "place", "strict_relation")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "relationship_subject_mismatch")
        accepted, rejected = verify(hypothesis("Andorra", quote, "Spain", "north_of"),
                                     "Spain\n" + quote, "What is north of Spain?", "place", "strict_relation")
        self.assertFalse(rejected)
        self.assertTrue(accepted)

    def test_strict_neighbor_checks_direction_orientation(self):
        quote = "France is north of Spain."
        accepted, rejected = verify(hypothesis("Spain", quote, "France", "north_of"),
                                     "France\n" + quote, "What is north of France?", "place", "strict_relation")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "requested_direction_not_supported_for_subject")
        accepted, rejected = verify(hypothesis("France", quote, "Spain", "north_of"),
                                     "France\n" + quote, "What is north of Spain?", "place", "strict_relation")
        self.assertFalse(rejected)
        self.assertTrue(accepted)

    def test_strict_establishment_cannot_transfer_polity_date(self):
        quote = "The Crown of Aragon was established in 1162. Andorra is a neighboring country."
        for subject in ("Crown of Aragon", "Andorra"):
            accepted, rejected = verify(hypothesis("1162", quote, subject, "established_date"),
                                         "Crown of Aragon\n" + quote,
                                         "When was Andorra established?", "year", "strict_relation")
            self.assertFalse(accepted)
            self.assertIn(rejected[0]["reason"], {"relationship_subject_mismatch", "requested_establishment_not_supported_for_subject"})
        quote = "Andorra was established in 1278."
        accepted, rejected = verify(hypothesis("1278", quote, "Andorra", "established_date"),
                                     "Andorra\n" + quote, "When was Andorra established?", "year", "strict_relation")
        self.assertFalse(rejected)
        self.assertTrue(accepted)

    def test_strict_establishment_mentions_are_not_the_owning_subject(self):
        for quote in ("Andorra's ally, the Crown of Aragon, was founded in 1162.",
                      "The Crown of Aragon, an ally of Andorra, was founded in 1162."):
            with self.subTest(quote=quote):
                accepted, rejected = verify(hypothesis("1162", quote, "Andorra", "established_date"),
                                             "Andorra\n" + quote, "When was Andorra established?", "year", "strict_relation")
                self.assertFalse(accepted)
                self.assertEqual(rejected[0]["reason"], "requested_establishment_not_supported_for_subject")

    def test_strict_creator_birth_date_requires_the_work_creator_link(self):
        quote = "Joshua Jacob Marston (born August 13, 1968) is an American screenwriter and film director."
        accepted, rejected = verify(hypothesis("August 13, 1968", quote, "Joshua Jacob Marston", "birth_date"),
                                     "Joshua Jacob Marston\n" + quote,
                                     "When was The Black Marble director born?", "date", "strict_relation")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "linked_subject_relationship_not_supported")
        identity = "The Black Marble is a film directed by Harold Becker."
        quote = "Harold Becker was born on September 25, 1928."
        accepted, rejected = verify(hypothesis("September 25, 1928", quote, "Harold Becker", "birth_date"),
                                     "The Black Marble\n" + identity + " " + quote,
                                     "When was the director of The Black Marble born?", "date", "strict_relation")
        self.assertFalse(rejected)
        self.assertTrue(accepted)

    def test_strict_creators_cannot_swap_names_in_same_document(self):
        quote = "Film Alpha was directed by Dana Jones. Film Beta was directed by Morgan Lee."
        accepted, rejected = verify(hypothesis("Morgan Lee", quote, "Film Alpha", "director"),
                                     "Film Alpha\n" + quote, "Who directed Film Alpha?", "person", "strict_relation")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "requested_creator_not_supported_for_subject")
        accepted, rejected = verify(hypothesis("Dana Jones", quote, "Film Alpha", "director"),
                                     "Film Alpha\n" + quote, "Who directed Film Alpha?", "person", "strict_relation")
        self.assertFalse(rejected)
        self.assertTrue(accepted)

    def test_strict_comparison_does_not_accept_death_as_birth_order(self):
        quote = "Dave Brockie died on March 23, 2014."
        accepted, rejected = verify(hypothesis("Dave Brockie", quote, "Dave Brockie", "comparison"),
                                     "Dave Brockie\n" + quote,
                                     "Who was born first, Ronnie Radke or Dave Brockie?", "order", "strict_relation")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "comparison_requires_atomic_attribute_evidence")

    def test_strict_unknown_relation_rejects_binding_but_keeps_input_documents(self):
        quote = "Aster is an example."
        docs = {0: "Aster\n" + quote}
        before = dict(docs)
        accepted, rejected = binding.verify_typed_hypotheses(
            hypothesis("Aster", quote, "Aster", "other"), docs, "Which example is best?", "unknown", "strict_relation")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "unrecognized_requested_relationship")
        self.assertEqual(docs, before)

    def test_strict_unknown_relation_needs_no_llm_request(self):
        class Engine(binding.BindingImprovementMixin):
            improvements = frozenset({"binding"})
            cfg = SimpleNamespace(evidence_binding_validation="strict_relation")

            def __init__(self):
                self._verification_diagnostics = {}

            def _infer_object(self, messages):
                self.fail("An unrecognized relation must be handled locally")

        engine = Engine()
        accepted, rejected, reason = engine._verify("Which example is best?", "unknown", {0: "Aster is an example."})
        self.assertFalse(accepted)
        self.assertIsNone(reason)
        self.assertEqual(rejected[0]["reason"], "unrecognized_requested_relationship")
        self.assertEqual(engine._verification_diagnostics["Which example is best?::0"]["attempts"], 0)

    def test_strict_binding_has_the_same_two_call_budget_and_legacy_prompt_unchanged(self):
        class Engine(binding.BindingImprovementMixin):
            improvements = frozenset({"binding"})

            def __init__(self, validation=None):
                self._verification_diagnostics = {}
                self.calls = []
                if validation:
                    self.cfg = SimpleNamespace(evidence_binding_validation=validation)

            def _infer_object(self, messages):
                self.calls.append(json.loads(json.dumps(messages)))
                if len(self.calls) == 1:
                    return hypothesis("Mediterranean coast", quote, "Israel", "location", intro), None
                return {"hypotheses": []}, None

        intro = "Israel is a country in the Middle East."
        quote = "On the Mediterranean coast, Haifa Port is the country's oldest and largest port."
        docs = {0: "Israel\n" + intro + " " + quote}
        implicit, explicit, strict = Engine(), Engine("legacy"), Engine("strict_relation")
        implicit._verify("Where is Israel located?", "place", docs)
        explicit._verify("Where is Israel located?", "place", docs)
        accepted, _, _ = strict._verify("Where is Israel located?", "place", docs)
        self.assertEqual(implicit.calls, explicit.calls)
        self.assertFalse(accepted)
        self.assertEqual(len(strict.calls), 2)
        self.assertIn("Strict relation mode", strict.calls[0][0]["content"])
        self.assertEqual(strict._verification_diagnostics["Where is Israel located?::0"]["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
