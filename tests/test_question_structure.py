"""CPU checks for question-only, conservative planning-capacity decisions."""
import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / "src/pathcondrag/question_structure.py"
SPEC = importlib.util.spec_from_file_location("_question_structure_test", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
route = module.route_question_structure
route_depth = module.route_dependency_depth


class QuestionStructureTests(unittest.TestCase):
    def test_comparison_and_bridge_comparison(self):
        self.assertEqual(route("Which film was released first, Film A or Film B?")["kind"], "parallel_comparison")
        self.assertEqual(route("Which film has the director born first, Film A or Film B?")["kind"], "bridge_comparison")
        self.assertEqual(route("Are the directors of both films Film A and Film B from the same country?")["kind"], "bridge_comparison")

    def test_yes_no_and_shared_attribute_comparisons(self):
        for question in ("Are Alpha and Beta from the same country?",
                         "Are Alpha and Beta from different countries?",
                         "What profession do Alpha and Beta share?",
                         "Who is older, Alpha or Beta?"):
            with self.subTest(question=question):
                self.assertTrue(route(question)["expand"])

    def test_explicit_multi_object_time_and_attribute(self):
        for question in ("When did Alpha and Beta open assembly plants?",
                         "Where were Alpha and Beta born?",
                         "What nationalities do Alpha and Beta have?",
                         "When did the maker of Work A, the largest company, and Beta open factories?"):
            with self.subTest(question=question):
                self.assertEqual(route(question)["kind"], "parallel_attribute")

    def test_conjunction_inside_unknown_entity_description_stays_original(self):
        for question in ("When did the country where Britain and France fought become independent?",
                         "When did the country of Britain and France become independent?",
                         "Who owns the record label of the Another Page performer?",
                         "What bluegrass singer provides vocals for a song on the album released through Label A?",
                         "Which British driver raced for different teams and won the European Grand Prix?",
                         "When did the birthplace of the Live and Beyond performer become the capital of State A?",
                         "What is the salary of a person of the same nationality as the creator of Work A and B?",
                         "When did Alpha and the country controlling Beta become allies?"):
            with self.subTest(question=question):
                self.assertFalse(route(question)["expand"])

    def test_title_words_and_single_alternative_predicate_do_not_imply_branches(self):
        self.assertEqual(route("Which film was released earlier, Jazz Boat or Her Husband's Trademark?")["kind"],
                         "parallel_comparison")
        self.assertFalse(route("Who scored or orchestrated more films for Studio A?")["expand"])
        self.assertFalse(route("")["expand"])
        self.assertFalse(route(None)["expand"])

    def test_router_accepts_only_text_and_never_annotations(self):
        class TextOnly(str):
            def __getattr__(self, key):
                raise AssertionError("Question routing attempted to read annotations")
        self.assertTrue(route(TextOnly("Which film was released first, Film A or Film B?"))["expand"])
        self.assertTrue(route("When did the country where Britain and France fought become independent?")["qualifier_risk"])

    def test_authorized_depth_can_route_a_chain_without_forcing_a_node_count(self):
        question = "What year did the university attended by the founder close?"
        decision = route_depth(question, 4)
        self.assertTrue(decision["expand"])
        self.assertEqual(decision["kind"], "dependency_chain")
        self.assertEqual(decision["depth_hint"], 4)
        self.assertNotIn("min_nodes", decision)

    def test_explicit_nested_entity_description_routes_at_two_hops(self):
        for question in ("When was the writer of Work X born?",
                         "Who owns the record label of the Another Page performer?",
                         "Where was Work X's author born?",
                         "When did the city where Alpha works become a capital?"):
            with self.subTest(question=question):
                self.assertEqual(route_depth(question, 2)["kind"], "nested_bridge")

    def test_simple_question_and_ambiguous_title_conjunction_stay_original(self):
        for question in ("When was Alpha born?", "Who composed Castor et Pollux?",
                         "Which British driver raced for different teams and won the European Grand Prix?",
                         "Who scored or orchestrated more films for Studio A?"):
            with self.subTest(question=question):
                self.assertFalse(route_depth(question, 2)["expand"])

    def test_depth_routing_keeps_existing_explicit_comparison_decision(self):
        question = "Are the directors of both films Film A and Film B from the same country?"
        self.assertEqual(route_depth(question, 2), route(question))


if __name__ == "__main__":
    unittest.main()
