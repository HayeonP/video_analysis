"""Model-free export guards for the anchored pipeline."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'01_low_freq_event'),str(ROOT.parent/'03_adaptive_sampling')]
from bounded_matching import export_mapping
from efficient_pipeline import defer_conflicting_tracks


class ExportGuards(unittest.TestCase):
    def setUp(self):
        self.groups={key:dict(kind='person',observations=[dict(detection_id=key,frame_key='f1',kind='person')])
                     for key in ('T1','T2')}
        self.memory={'A1':dict(label='인물',appearance='visible',kind='person')}
        self.mapping={'T1':'A1','T2':'A1'}

    def test_unresolved_conflicts_keep_both_observations_separate(self):
        audit=dict(remaining_conflicts=[dict(track_ids=['T1','T2'])])
        original=deepcopy(self.groups)
        defer_conflicting_tracks(self.groups,self.memory,self.mapping,audit)
        registry,visual,observed=export_mapping(self.groups,self.memory,self.mapping,identity_method='anchored')
        self.assertEqual(len(observed),2)
        self.assertEqual(len(registry),2)
        self.assertEqual({r['identity_status'] for r in registry.values()},{'unresolved'})
        self.assertEqual({p['identity_method'] for p in visual['assignments'].values()},{'anchored'})
        self.assertEqual(self.groups,original)
        self.assertEqual(len(audit['conflict_resolutions']),2)

    def test_export_does_not_include_reference_observations(self):
        self.memory['A1']['observations']=[dict(detection_id='reference_only')]
        registry,visual,observed=export_mapping(self.groups,self.memory,{'T1':'A1'})
        self.assertEqual([o['detection_id'] for o in observed],['T1'])
        self.assertEqual(registry['E0001']['reference_key'],'A1')
        self.assertEqual(visual['assignments']['T1']['identity_status'],'estimated')

    def test_duplicate_inside_one_track_is_invariant_error(self):
        with self.assertRaises(RuntimeError):
            defer_conflicting_tracks(self.groups,self.memory,self.mapping,
                                     dict(remaining_conflicts=[dict(track_ids=['T1','T1'])]))

    def test_off_export_method_is_explicit(self):
        _,visual,_=export_mapping(self.groups,self.memory,{'T1':'A1'},identity_method='no Qwen')
        self.assertEqual(visual['assignments']['T1']['identity_method'],'no Qwen')


if __name__=='__main__':unittest.main()
