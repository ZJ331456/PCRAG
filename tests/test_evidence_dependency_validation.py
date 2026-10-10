"""CPU source-ownership tests for the opt-in dependency checker."""
import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_dependency_validation_test_package"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/pathcondrag")]
sys.modules[PACKAGE] = package
checker = importlib.import_module(PACKAGE + ".evidence_dependency_validation")
legacy = importlib.import_module(PACKAGE + ".evidence_dag_package")


def verify(question, answer, quote, document=None, dependencies=None, bindings=None):
    node = {"id": "s2", "question": question, "depends_on": dependencies or [],
            "answer_type": "count" if question.startswith("How many") else "entity"}
    proof = {"answer": answer, "evidence": quote, "confidence": 0.99}
    return checker.expanded_proof_relation(node, question, proof, document or quote, bindings or {})


class ExpandedDependencyValidationTests(unittest.TestCase):
    def assert_supported(self, *args, **kwargs):
        check, reason = verify(*args, **kwargs)
        self.assertIsNotNone(check)
        self.assertIsNone(reason)
        return check

    def assert_rejected(self, *args, **kwargs):
        check, reason = verify(*args, **kwargs)
        self.assertIsNone(check)
        self.assertIsNotNone(reason)
        return reason

    def test_legacy_positive_keeps_the_same_check(self):
        question = "Who directed Sample Film?"
        quote = "Sample Film was directed by Alice Example."
        node = {"id": "s1", "question": question, "depends_on": [], "answer_type": "person"}
        proof = {"answer": "Alice Example", "evidence": quote}
        document = "Sample Film\n" + quote
        self.assertEqual(checker.expanded_proof_relation(node, question, proof, document, {}),
                         legacy._proof_relation(node, question, proof, document, {}))

    def test_adjacent_explicit_nickname_owner_preserves_numeric_bound(self):
        quote = "His production company, 40 Acres and a Mule Filmworks, has produced over 35 films since 1983."
        document = ('Spike Lee\nShelton Jackson "Spike" Lee (born March 20, 1957) is an American film '
                    'director, producer, writer, and actor. ' + quote)
        self.assert_supported("How many films has Spike Lee produced since 1983?", "over 35", quote, document)
        self.assertEqual(self.assert_rejected("How many films has Spike Lee produced since 1983?",
                                              "35", quote, document), "expanded_count_bound_mismatch")

    def test_explicit_company_question_has_company_owner(self):
        quote = "His production company has produced 35 films since 1983."
        document = "Alice Example\nAlice Example is a film producer. " + quote
        self.assert_supported("How many films has Alice Example's production company produced since 1983?",
                              "35", quote, document)

    def test_count_without_requested_since_year_is_rejected(self):
        quote = "Alice Example has produced 35 films."
        self.assertEqual(self.assert_rejected("How many films has Alice Example produced since 1983?",
                                              "35", quote, "Alice Example\n" + quote),
                         "expanded_count_time_scope_not_supported")

    def test_count_different_temporal_relation_is_rejected(self):
        quote = "Alice Example has produced 35 films before 1983."
        self.assert_rejected("How many films has Alice Example produced since 1983?", "35", quote,
                             "Alice Example\n" + quote)

    def test_count_answer_must_be_the_predicate_quantity(self):
        quote = "Alice Example has produced 12 films since 1983 and owns 35 records."
        self.assert_rejected("How many films has Alice Example produced since 1983?", "35", quote,
                             "Alice Example\n" + quote)

    def test_adjacent_other_person_blocks_pronoun_inheritance(self):
        quote = "His production company has produced 35 films since 1983."
        document = "Alice Example\nAlice Example is a producer. John Other is a director. " + quote
        self.assert_rejected("How many films has Alice Example produced since 1983?", "35", quote, document)

    def test_adjacent_family_subject_blocks_pronoun_inheritance(self):
        quote = "His production company has produced 35 films since 1983."
        for previous in ["Alice Example's father is a producer.",
                         "Alice Example and John Other are film directors.",
                         "Alice Example is a director married to John Other."]:
            with self.subTest(previous=previous):
                self.assert_rejected("How many films has Alice Example produced since 1983?", "35", quote,
                                     "Alice Example\n" + previous + " " + quote)

    def test_shared_surname_is_not_identity(self):
        quote = "His production company has produced 35 films since 1983."
        document = "Alice Example\nBob Example is a producer. " + quote
        self.assert_rejected("How many films has Alice Example produced since 1983?", "35", quote, document)

    def test_duplicate_quote_cannot_select_a_convenient_antecedent(self):
        quote = "His production company has produced 35 films since 1983."
        document = "Alice Example\nAlice Example is a producer. " + quote + " John Other is a producer. " + quote
        self.assert_rejected("How many films has Alice Example produced since 1983?", "35", quote, document)

    def test_country_based_explicit_predicate(self):
        quote = "Example Studios is based in Lithuania."
        self.assert_supported("Which country is Example Studios based in?", "Lithuania", quote,
                              "Example Studios\n" + quote)

    def test_owned_adjacent_pronoun_location(self):
        quote = "It is based in Lithuania."
        document = "Example Studios\nExample Studios is a games company. " + quote
        self.assert_supported("Which country is Example Studios based in?", "Lithuania", quote, document)

    def test_other_company_location_not_inherited(self):
        quote = "Other Studios is based in Lithuania. Example Studios operates internationally."
        self.assert_rejected("Which country is Example Studios based in?", "Lithuania", quote,
                             "Example Studios\n" + quote)

    def test_possessive_other_entity_location_not_transferred(self):
        quote = "Example Studios's subsidiary is based in Lithuania."
        self.assert_rejected("Which country is Example Studios based in?", "Lithuania", quote,
                             "Example Studios\n" + quote)

    def test_headquarters_explicit_relation(self):
        quote = "Example Broadcaster has its headquarters in Broadcasting House, London."
        self.assert_supported("Where are the headquarters of Example Broadcaster?", "Broadcasting House, London",
                              quote, "Example Broadcaster\n" + quote)

    def test_headquarters_lead_explicit_acronym_is_identity(self):
        quote = "The Example Broadcasting Corporation (EBC) is a British public service broadcaster with its headquarters at Example House in London."
        self.assert_supported("Where is the headquarters of EBC?", "Example House in London", quote,
                              "EBC\n" + quote)
        self.assert_rejected("Where is the headquarters of EBC?", "London",
                             "EBC is in London.", "EBC\nEBC is in London.")

    def test_naked_same_page_acronym_does_not_alias_another_company(self):
        quote = "Other Broadcasting Corporation (OBC) is a broadcaster with its headquarters at Example House."
        self.assert_rejected("Where is the headquarters of EBC?", "Example House", quote, "EBC\n" + quote)

    def test_passive_broadcast_pair(self):
        quote = "Example Series was first broadcast on BBC."
        self.assert_supported("On which television channel was Example Series broadcast?", "BBC", quote,
                              "Example Series\n" + quote)

    def test_series_owned_relative_broadcast_relation(self):
        quote = "Example Series is a Western television series starring Alice Example that aired on NBC."
        self.assert_supported("Which network aired Example Series?", "NBC", quote, "Example Series\n" + quote)

    def test_co_producer_does_not_prove_broadcast_network(self):
        quote = "Example Series was a television drama, co-produced by the BBC and the ABC."
        self.assert_rejected("Which network aired Example Series?", "BBC", quote, "Example Series\n" + quote)

    def test_pronoun_broadcast_pair_uses_adjacent_owned_lead(self):
        quote = "It was originally aired on BBC."
        document = "Example Series\nExample Series is a television drama. " + quote
        self.assert_supported("On which channel was Example Series aired?", "BBC", quote, document)

    def test_broadcast_cooccurrence_and_wrong_direction_rejected(self):
        for quote in ["BBC aired Other Series; Example Series was discussed separately.",
                      "Example Series was broadcast on Channel Four, and BBC produced a documentary."]:
            with self.subTest(quote=quote):
                self.assert_rejected("On which channel was Example Series broadcast?", "BBC", quote,
                                     "Example Series\n" + quote)

    def test_album_artist_attribution_does_not_prove_composer(self):
        quote = "Example Album is an album by Alice Example."
        document = "Example Album\n" + quote
        self.assert_supported("Who is the artist of the album Example Album?", "Alice Example", quote, document)
        self.assert_rejected("Who composed Example Album?", "Alice Example", quote, document)

    def test_album_named_alias_and_artist_role_labels(self):
        quote = "Example Album (also referred to as Trio) is an album by American bassist, composer and bandleader Alice Example with pianist Other Person."
        self.assert_supported("Who is the performer of the album Example Album?", "Alice Example", quote,
                              "Example Album\n" + quote)
        self.assert_rejected("Who is the performer of the album Example Album?", "Other Person", quote,
                             "Example Album\n" + quote)

    def test_explicit_creator_active_and_passive(self):
        self.assert_supported("Who developed Example Game?", "Example Studios",
                              "Example Game was developed by Example Studios.")
        self.assert_supported("Who published Example Book?", "Example Press",
                              "Example Press published Example Book.")

    def test_creator_answer_not_unrelated_name_in_tail(self):
        quote = "Example Game was developed by Other Studios and reviewed by Example Studios."
        self.assert_rejected("Who developed Example Game?", "Example Studios", quote)

    def test_argument_owner_cannot_replace_its_sister_or_subsidiary(self):
        for mark in ("'", "’"):
            with self.subTest(mark=mark):
                self.assert_rejected("Who is the mother of Bob?", "Alice",
                                     f"Bob's mother is Alice{mark}s sister Carol.")
                self.assert_rejected("Who produced Project X?", "StudioA",
                                     f"Project X was produced by StudioA{mark}s subsidiary StudioB.")

    def test_partial_name_is_not_the_full_argument(self):
        self.assert_rejected("Who produced Project X?", "StudioA",
                             "Project X was produced by StudioA Studios.")

    def test_argument_keeps_explicit_apposition_and_location(self):
        self.assert_supported("Where is the headquarters of Example Company?", "Example House",
                              "Example Company has headquarters at Example House in London.")
        self.assert_supported("Who produced Project X?", "StudioA",
                              "Project X was produced by StudioA, an independent studio.")

    def test_explicit_mother_and_capital_have_direction(self):
        self.assert_supported("Who was the mother of Alice Example?", "Mary Example",
                              "Alice Example's mother was Mary Example.")
        self.assert_supported("Who was the mother of Alice Example?", "Mary Example",
                              "Mary Example was the mother of Alice Example.")
        self.assert_rejected("Who was the mother of Alice Example?", "Mary Example",
                             "Alice Example was the mother of Mary Example.")
        self.assert_supported("What is the capital of Example Country?", "Example City",
                              "Example Country's capital is Example City.")

    def test_capital_inverse_city_rank_modifier_and_state_type(self):
        quote = "Example City is the capital and second largest city of the U.S. state of Example State."
        self.assert_supported("What is the capital of Example State?", "Example City", quote,
                              "Example City, Example State\n" + quote)
        self.assert_rejected("What is the capital of Example State?", "Other City",
                             quote + " Other City is mentioned separately.")

    def test_count_time_in_other_predicate_not_transferred(self):
        quote = "Alice Example has produced 35 films and has owned a company since 1983."
        self.assert_rejected("How many films has Alice Example produced since 1983?", "35", quote,
                             "Alice Example\n" + quote)

    def test_negated_or_unquoted_or_missing_literal_answer_rejected(self):
        for quote, document, answer in [
            ("Example Studios is not based in Lithuania.", "Example Studios is not based in Lithuania.", "Lithuania"),
            ("Example Studios is based in Lithuania.", "Example Studios is based in Canada.", "Lithuania"),
            ("Example Studios is based in Lithuania.", "Example Studios is based in Lithuania.", "Canada")]:
            with self.subTest(quote=quote, document=document, answer=answer):
                self.assert_rejected("Which country is Example Studios based in?", answer, quote, document)

    def test_multi_parent_question_cannot_inherit_absent_input(self):
        quote = "Example Studios is based in Lithuania."
        self.assert_rejected("Which country is Example Studios based in?", "Lithuania", quote,
                             dependencies=["s1", "s3"], bindings={"s1": "Example Studios", "s3": "Other Entity"})


if __name__ == "__main__":
    unittest.main()
