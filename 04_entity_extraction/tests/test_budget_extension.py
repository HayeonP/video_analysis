"""Explicit extensions retain the development run's full cost history."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'01_low_freq_event')]
from story_pipeline import extend_generation_budget


class BudgetExtensionTests(unittest.TestCase):
    def test_explicit_extension_preserves_usage_and_records_both_limits(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);path=output/'qwen_budget.json'
            events=[{'number':1,'status':'failed'},{'number':2,'status':'generated'}]
            path.write_text(json.dumps({'limit':8,'generations':2,'events':events}))
            extend_generation_budget(output,16,'Additional regression validation')
            actual=json.loads(path.read_text())
            self.assertEqual(actual['events'],events)
            self.assertEqual(actual['generations'],2)
            self.assertEqual(actual['limit'],16)
            self.assertEqual(actual['budget_changes'][0]['previous_limit'],8)
            self.assertEqual(actual['budget_changes'][0]['generations_so_far'],2)
            before=path.read_text()
            for limit,reason in [(16,'same'),(4,'lower'),(32,'')]:
                with self.assertRaises(ValueError):extend_generation_budget(output,limit,reason)
                self.assertEqual(path.read_text(),before)

    def test_cannot_extend_a_missing_ledger(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):extend_generation_budget(Path(temp),16,'reason')


if __name__=='__main__':unittest.main()
