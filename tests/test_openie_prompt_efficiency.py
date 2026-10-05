"""Quality-preserving preflight caching and first-request structure checks."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pathcondrag.index.openie import openie_semantic_validation as validation
from pathcondrag.index.openie.source_verified_openie import SourceVerifiedOpenIE
from pathcondrag.index.ner.source_verified import SourceVerifiedNERMixin
from pathcondrag.prompts import PromptTemplateManager


class PromptEfficiencyTests(unittest.TestCase):
    def test_failed_ner_retries_do_not_replay_the_identical_cached_bad_task(self):
        client = SimpleNamespace(infer=Mock(return_value=(
            '{"named_entities":"invalid"}', {'finish_reason': 'stop'}, False)))
        extractor = SourceVerifiedOpenIE(client)
        with patch('pathcondrag.index.ner.source_verified.guided_json_parameters',
                   return_value={'chat_template_kwargs': {'enable_thinking': False}}):
            result = SourceVerifiedNERMixin.ner(extractor, 'chunk', 'Ada trained in theater.')
        self.assertFalse(result.metadata['complete'])
        calls = client.infer.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertEqual(len({json.dumps(call.kwargs['messages']) for call in calls}), 3)
        self.assertEqual([call.kwargs['max_completion_tokens'] for call in calls], [512] * 3)

    def test_exact_prompt_counts_are_cached_and_content_changes_invalidate(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = [[1, 2], [1, 2, 3]]
        validation._qwen_prompt_length.cache_clear()
        client = SimpleNamespace(llm_name='qwen3-8b')
        messages = [{'role': 'user', 'content': 'original source'}]
        with patch.object(validation, '_qwen_tokenizer', return_value=tokenizer):
            self.assertEqual(validation._prompt_tokens(client, messages), 2)
            self.assertEqual(validation._prompt_tokens(client, json.loads(json.dumps(messages))), 2)
            messages[0]['content'] += ' changed'
            self.assertEqual(validation._prompt_tokens(client, messages), 3)
        self.assertEqual(tokenizer.apply_chat_template.call_count, 2)
        self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs['enable_thinking'])
        validation._qwen_prompt_length.cache_clear()

    def test_provider_role_mapping_does_not_mutate_original_templates(self):
        changed = PromptTemplateManager(role_mapping={'system': 'user'})
        normal = PromptTemplateManager()
        self.assertEqual(changed.render('ner', passage='Ada')[0]['role'], 'user')
        self.assertEqual(normal.render('ner', passage='Ada')[0]['role'], 'system')

    def test_initial_strict_triples_have_grammar_and_still_receive_semantic_audit(self):
        client = SimpleNamespace(infer=Mock(return_value=(
            '{"triples":[["Ada","trained in","theater"]]}',
            {'finish_reason': 'stop'}, False)))
        extractor = SourceVerifiedOpenIE(client)
        with patch('pathcondrag.index.openie.openie_structured_output.guided_json_parameters',
                   return_value={'guided_grammar': 'test',
                                 'chat_template_kwargs': {'enable_thinking': False}}), \
                patch.object(extractor, '_verify_and_recover', side_effect=lambda *args: args[-1]) as audit:
            result = extractor.triple_extraction('chunk', 'Ada trained in theater.', ['Ada'])
        audit.assert_called_once()
        self.assertEqual(result.triples, [['Ada', 'trained in', 'theater']])
        request = client.infer.call_args.kwargs
        self.assertEqual(request['max_completion_tokens'], 2048)
        self.assertIn('guided_grammar', request['extra_body'])
        self.assertFalse(request['extra_body']['chat_template_kwargs']['enable_thinking'])


if __name__ == '__main__':
    unittest.main()
