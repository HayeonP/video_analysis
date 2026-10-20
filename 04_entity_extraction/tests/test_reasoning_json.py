"""Reasoning drafts must never become final structured decisions."""
from pathlib import Path
import sys
import unittest
import tempfile
import json
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_low_freq_event'))
from low_freq_event.backend import parse_json_response
from low_freq_event.config import Settings
from pydantic import BaseModel


class ReasoningJsonTests(unittest.TestCase):
    def test_physical_model_switch_restores_settings_and_shares_budget_and_cache(self):
        from story_semantics import StoryBackend
        from identity_review import ReviewBackend
        class Answer(BaseModel):value:int
        with tempfile.TemporaryDirectory() as folder:
            settings=Settings(model_id='example/Thinking',model_revision='reasoning',json_retries=0)
            backend=StoryBackend(settings,Path(folder),limit=3)
            backend.physical_settings=settings.model_copy(update=dict(model_id='example/Instruct',model_revision='visual'))
            with patch.object(ReviewBackend,'_generate',return_value='{"value":1}') as generate:
                self.assertEqual(backend.request_physical('count',Answer).value,1)
                self.assertIs(backend.settings,settings)
                backend.request_physical('count',Answer)
                self.assertEqual(generate.call_count,1)
                self.assertEqual(backend.ledger['generations'],1)
                self.assertEqual(backend.ledger['events'][-1]['model_id'],'example/Instruct')
            with patch.object(ReviewBackend,'_generate',side_effect=RuntimeError('failed')):
                with self.assertRaises(RuntimeError):backend.request_physical('new count',Answer)
            self.assertIs(backend.settings,settings)
            self.assertEqual(backend.ledger['generations'],2)

    def test_closed_reasoning_or_prefilled_opening_uses_only_final_json(self):
        for opening in ('<think>\n',''):
            text=opening+'Draft: {"decision":"same"}\n</think>\n```json\n{"decision":"uncertain"}\n```'
            self.assertEqual(parse_json_response(text),{'decision':'uncertain'})

    def test_unclosed_explicit_reasoning_never_accepts_its_draft_json(self):
        with self.assertRaises(ValueError):parse_json_response('<think>\nDraft: {"decision":"same"}')
        with self.assertRaises(ValueError):parse_json_response('draft\n</think>\n{"decision":')
        with self.assertRaises(ValueError):parse_json_response('draft\n</think>\n')
        with self.assertRaises(ValueError):parse_json_response('Draft: {"decision":"same"}',reasoning_expected=True)
        with self.assertRaises(ValueError):parse_json_response('{"decision":"same"}',reasoning_expected=True)
        self.assertEqual(parse_json_response('{"draft":1}\n</think>\n{"final":2}',reasoning_expected=True),{'final':2})

    def test_instruct_json_and_literal_tag_strings_are_preserved(self):
        self.assertEqual(parse_json_response('Result: {"value":3}'),{'value':3})
        self.assertEqual(parse_json_response('```json\n{"value":3}\n```'),{'value':3})
        self.assertEqual(parse_json_response('{"value":"</think>"}'),{'value':'</think>'})

    def test_sampling_and_seed_are_bound_to_cache_and_trace(self):
        from low_freq_event.backend import QwenBackend
        class Answer(BaseModel):value:int
        with tempfile.TemporaryDirectory() as folder:
            backend=QwenBackend(Settings(),Path(folder)/'traces')
            with patch.object(backend,'_generate',return_value='{"value":1}') as generate:
                first={'do_sample':True,'temperature':1.,'top_p':.95,'top_k':20,'seed':0}
                backend.request('test',Answer,generation_options=first)
                backend.request('test',Answer,generation_options=dict(reversed(list(first.items()))))
                self.assertEqual(generate.call_count,1)
                backend.request('test',Answer,generation_options=dict(first,seed=1))
                backend.request('test',Answer)
                self.assertEqual(generate.call_count,3)
            traces=[json.loads(p.read_text()) for p in (Path(folder)/'traces').glob('*.json')]
            self.assertTrue(any(t.get('generation_options',{}).get('seed')==0 for t in traces))

    def test_stricter_validator_rechecks_cached_response_without_losing_old_evidence(self):
        from low_freq_event.backend import QwenBackend
        class Answer(BaseModel):value:int
        def strict(answer):
            if answer.value!=2:raise ValueError('Current evidence requires value two')
            return answer
        with tempfile.TemporaryDirectory() as folder:
            trace=Path(folder)/'traces';backend=QwenBackend(Settings(json_retries=0),trace)
            with patch.object(backend,'_generate',side_effect=['{"value":1}','{"value":2}']) as generate:
                backend.request('same input',Answer)
                self.assertEqual(backend.request('same input',Answer,validate=strict).value,2)
                self.assertEqual(generate.call_count,2)
                self.assertIn('Current evidence requires',generate.call_args.args[0])
                backend.request('same input',Answer,validate=strict)
                self.assertEqual(generate.call_count,2)
            invalid=list(trace.glob('cache_invalid_*.json'))
            self.assertEqual(len(invalid),1)
            self.assertEqual(json.loads(json.loads(invalid[0].read_text())['cached_response'])['value'],1)


if __name__=='__main__':unittest.main()
