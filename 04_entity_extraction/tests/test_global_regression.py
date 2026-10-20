from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_global_regression import evaluate, evaluate_run, source_observations


class GlobalRegressionTests(unittest.TestCase):
    def setUp(self):
        self.old = [dict(entity_id=eid, detection_id=f'{eid}_{frame}', shot_id=eid, source_frame=frame,
                         bbox_pixels=[0, 0, 20, 30], kind='person')
                    for eid in ('A', 'B', 'C') for frame in range(3)]
        self.labels = dict(groups=[dict(name='main', ids=['A', 'B'], kind='person'),
                                   dict(name='other', ids=['C'], kind='person')])
        self.good = [dict(row, entity_id='G1' if row['entity_id'] in ('A', 'B') else 'G2') for row in self.old]

    def test_consistent_ids_pass_and_missing_identity_fails(self):
        self.assertTrue(evaluate(self.good, self.old, self.labels)['gate']['passed'])
        missing = [row for row in self.good if row['shot_id'] != 'B']
        self.assertFalse(evaluate(missing, self.old, self.labels)['gate']['passed'])

    def test_split_hidden_by_dominant_vote_fails(self):
        rows = deepcopy(self.good)
        rows[0]['entity_id'] = 'G3'
        result = evaluate(rows, self.old, self.labels)
        self.assertEqual(result['evaluation']['current']['same_id_pair_recall'], 1)
        self.assertFalse(result['gate']['passed'])

    def test_minority_false_merge_fails(self):
        rows = deepcopy(self.good)
        rows[0]['entity_id'] = 'G2'
        result = evaluate(rows, self.old, self.labels)
        self.assertEqual(result['evaluation']['current']['false_merge_pairs'], 0)
        self.assertFalse(result['gate']['passed'])
        self.assertEqual(result['gate']['cross_group_supported_id_conflicts'][0]['global_ids'], ['G2'])

    def test_pre_export_mapping_uses_boxes_and_rejects_missing_mapping(self):
        bundle = dict(shots=[dict(shot_id='s', width=100, height=100, frame_count=1,
            tracks=[dict(track_id='t', observations=[dict(frame=0, bbox=[1, 2, 20, 30])])])])
        local = dict(shots=[dict(shot_id='s', entities=[dict(local_id='L1', track_ids=['t'], kind='person')])])
        catalog = dict(entities=[dict(local_ids=['L1'], kind='person')])
        rows = source_observations(bundle, local, catalog)
        self.assertEqual(rows[0]['entity_id'], 'P0001')
        self.assertEqual(rows[0]['bbox_pixels'], [1., 2., 20., 30.])
        with self.assertRaisesRegex(ValueError, 'mapping'):
            source_observations(bundle, local, dict(entities=[]))

    def test_report_hashes_exact_catalog_labels_bundle_and_optional_export(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);baseline=run/'baseline.json';labels=run/'labels.json';export=run/'observations.json'
            baseline.write_text(json.dumps(self.old));labels.write_text(json.dumps(self.labels));export.write_text(json.dumps(self.good))
            bundle=dict(shots=[dict(shot_id=eid,width=100,height=100,frame_count=3,tracks=[dict(track_id=eid,
                observations=[dict(frame=i,bbox=[0,0,20,30]) for i in range(3)])]) for eid in ('A','B','C')])
            local=dict(shots=[dict(shot_id=eid,entities=[dict(local_id=eid,track_ids=[eid],kind='person')]) for eid in ('A','B','C')])
            catalog=dict(entities=[dict(entity_id='G1',local_ids=['A','B'],kind='person'),dict(entity_id='G2',local_ids=['C'],kind='person')])
            (run/'reviewed_tracking.json').write_text(json.dumps(bundle));(run/'local_entities.json').write_text(json.dumps(local))
            chosen=run/'chosen_catalog.json';chosen.write_text(json.dumps(catalog))
            report=evaluate_run(run,baseline,labels,final_catalog=chosen,observations=export)
            self.assertTrue(report['passed'])
            self.assertEqual(set(report['artifacts']),{'final_catalog','local_entities','bundle','labels','baseline','exported_observations'})
            for item in report['artifacts'].values():
                self.assertEqual(item['sha256'],hashlib.sha256(Path(item['path']).read_bytes()).hexdigest())
            before=report['artifacts']['final_catalog']['sha256'];chosen.write_text(chosen.read_text()+'\n')
            after=evaluate_run(run,baseline,labels,final_catalog=chosen)
            self.assertTrue(after['passed'])
            self.assertNotEqual(before,after['artifacts']['final_catalog']['sha256'])
            self.assertNotIn('exported_observations',after['artifacts'])
            # A previous SAM directory must not shadow the active no-SAM run.
            stale=run/'mask_refinement';stale.mkdir()
            (stale/'bundle.json').write_text('{"shots":[]}')
            (stale/'local.json').write_text('{"shots":[]}')
            (run/'run_config.json').write_text('{"mask_refinement":false}')
            no_mask=evaluate_run(run,baseline,labels,final_catalog=chosen)
            self.assertTrue(no_mask['passed'])
            self.assertEqual(no_mask['artifacts']['bundle']['path'],str((run/'reviewed_tracking.json').resolve()))
            self.assertIn('run_config',no_mask['artifacts'])
            (stale/'bundle.json').write_text(json.dumps(bundle));(stale/'local.json').write_text(json.dumps(local))
            (run/'run_config.json').write_text('{"mask_refinement":true}')
            mask=evaluate_run(run,baseline,labels,final_catalog=chosen)
            self.assertTrue(mask['passed'])
            self.assertEqual(mask['artifacts']['local_entities']['path'],str((stale/'local.json').resolve()))


if __name__ == '__main__':
    unittest.main()
