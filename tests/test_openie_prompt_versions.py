"""Extraction prompt contracts and semantically checked demonstrations."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pathcondrag.prompts import PromptTemplateManager
from pathcondrag.prompts.templates import (
    ner, origin_ner_prompt, origin_triple_extraction_prompt, triple_extraction,
)


class OpenIEPromptVersionTests(unittest.TestCase):
    def setUp(self):
        self.manager = PromptTemplateManager()

    def test_both_versions_preserve_source_and_names_without_resubstitution(self):
        passage = 'A and B named the building "$budget" in 2008.'
        entity_json = json.dumps({'named_entities': ['A', 'B', '$budget', '2008']})
        for name in ('origin_ner_prompt', 'ner',
                     'origin_triple_extraction_prompt', 'triple_extraction'):
            with self.subTest(template=name):
                messages = self.manager.render(name, passage=passage,
                                               named_entity_json=entity_json)
                self.assertEqual([message['role'] for message in messages],
                                 ['system', 'user', 'assistant', 'user'])
                self.assertIn(passage, messages[-1]['content'])
                if 'triple' in name:
                    self.assertIn(entity_json, messages[-1]['content'])

    def test_demonstrations_follow_single_key_json_contracts(self):
        for name in ('origin_ner_prompt', 'ner',
                     'origin_triple_extraction_prompt', 'triple_extraction'):
            with self.subTest(template=name):
                messages = self.manager.render(name, passage='Source.', named_entity_json='{}')
                payload = json.loads(messages[2]['content'])
                key = 'triples' if 'triple' in name else 'named_entities'
                self.assertEqual(set(payload), {key})
                if key == 'triples':
                    self.assertTrue(all(len(row) == 3 and all(
                        isinstance(value, str) and value.strip() for value in row)
                        for row in payload[key]))
                    self.assertEqual(len(payload[key]), len({tuple(row) for row in payload[key]}))
                else:
                    self.assertTrue(all(isinstance(value, str) and value.strip()
                                        for value in payload[key]))
                    self.assertEqual(len(payload[key]), len(set(payload[key])))

    def test_new_demonstration_preserves_roles_and_attribution(self):
        messages = self.manager.render('triple_extraction', passage='Source.',
                                       named_entity_json='{}')
        triples = json.loads(messages[2]['content'])['triples']
        self.assertIn(['Nia', 'portrayed by', 'Lio'], triples)
        self.assertIn(['Oren', 'portrayed by', 'Lio'], triples)
        self.assertIn(['Mara', 'describes Oren as', 'reckless'], triples)
        self.assertFalse(any(row[1] == 'and' or row[0] == 'Nia' and row[2] == 'reckless'
                             for row in triples))
        family = [row for row in triples if row[1] in ('daughter of', 'son of')]
        self.assertEqual({tuple(row) for row in family}, {
            ('Nia', 'daughter of', 'Mara'), ('Nia', 'daughter of', 'Vale'),
            ('Oren', 'son of', 'Mara'), ('Oren', 'son of', 'Vale')})
        self.assertIn(['Riverlight', 'release date', '3 July 2008'], triples)
        self.assertIn(['Lio', 'trained in', 'classical theater'], triples)
        self.assertIn(['Riverlight', 'was meant to look like', 'a historical tale'], triples)
        self.assertFalse(any(row[0] == 'Lio' and row[2] == 'a historical tale'
                             for row in triples))

    def test_origin_triples_keep_origin_example_independent_of_new_ner(self):
        self.assertNotEqual(ner.one_shot_ner_paragraph, origin_ner_prompt.one_shot_ner_paragraph)
        self.assertIn(origin_ner_prompt.one_shot_ner_paragraph,
                      origin_triple_extraction_prompt.ner_conditioned_re_input)
        self.assertIn(origin_ner_prompt.one_shot_ner_output,
                      origin_triple_extraction_prompt.ner_conditioned_re_input)
        self.assertIn('India\'s first private FM', origin_triple_extraction_prompt.ner_conditioned_re_input)
        self.assertNotEqual(origin_triple_extraction_prompt.ner_conditioned_re_input,
                            triple_extraction.ner_conditioned_re_input)


if __name__ == '__main__':
    unittest.main()
