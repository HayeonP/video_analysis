"""Synthetic contract, timeline and partial-label regression checks (no models)."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from story_export import export_story, suppress_duplicate_boxes, validate_story_contract, write_json, _rate
from story_evaluation import evaluate_partial_identities


def fixture(folder):
    from PIL import Image
    Image.new('RGB', (30, 40), 'red').save(folder/'crop.jpg')
    bundle = dict(asset_root=str(folder), shots=[dict(shot_id='s1', source=str(folder/'source.mp4'),
        fps='24', frame_count=5, full_start_sec=0., original_start_frame=72, width=160, height=96,
        tracks=[dict(track_id='s1:T1', observations=[dict(frame=i, time_sec=i/24,
            original_time_sec=3+i/24, bbox=[20, 20, 50, 60], confidence=.9,
            category='person', crop='crop.jpg') for i in range(5)])])])
    local = dict(shots=[dict(shot_id='s1', entities=[dict(local_id='s1:L1', track_ids=['s1:T1'],
        kind='person', label='사람', appearance='붉은 옷')], rejected_tracks=[], summary='사람이 서 있다.',
        events=[dict(subject='s1:L1', action='서 있다', object=None, start_sec=0., end_sec=5/24)])])
    global_result = dict(entities=[dict(entity_id='G1', label='사람', kind='person', appearance='붉은 옷')],
        mapping={'s1:L1': 'G1'}, shots=[dict(shot_id='s1', summary='G1이 서 있다.', time_basis='full',
        events=[dict(subject='G1', action='서 있다', object=None, start_sec=0., end_sec=5/24)])])
    return bundle, local, global_result


class ExportTests(unittest.TestCase):
    def test_fractional_fps_float_restores_original_grid(self):
        from fractions import Fraction
        self.assertEqual(_rate(30000/1001), Fraction(30000, 1001))

    def test_complete_contract_and_partition_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, local, global_result = fixture(Path(temp))
            self.assertTrue(validate_story_contract(bundle, local, global_result)['complete_track_review'])
            local['shots'][0]['rejected_tracks'] = ['s1:T1']
            with self.assertRaises(ValueError):
                validate_story_contract(bundle, local, global_result)

    def test_unreviewed_track_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, local, global_result = fixture(Path(temp))
            bundle['shots'][0]['tracks'].append(dict(track_id='s1:T2', observations=[]))
            quality = validate_story_contract(bundle, local, global_result)
            self.assertEqual(quality['unreviewed_tracks'], 1)
            self.assertFalse(quality['complete_track_review'])

    def test_duplicates_conflicts_and_carried_subject(self):
        def row(eid, tid, box):
            return dict(shot_id='s', source_frame=0, entity_id=eid, tracklet_id=tid,
                        detection_id=tid, detection_confidence=.8, bbox_pixels=box)
        rows = [row('G1', 't1', [0, 0, 40, 40]), row('G1', 't2', [1, 1, 41, 41]),
                row('G1', 't3', [90, 90, 120, 120]), row('G2', 't4', [5, 5, 20, 20])]
        kept, audit = suppress_duplicate_boxes(rows)
        self.assertEqual(len(kept), 3)
        self.assertEqual(audit['suppressed_count'], 1)
        self.assertEqual(audit['conflict_count'], 1)
        self.assertIn('G2', {r['entity_id'] for r in kept})

    def test_nested_face_keeps_outer_body_and_provenance(self):
        outer = dict(shot_id='s', source_frame=0, entity_id='G1', tracklet_id='body',
            detection_id='body', confidence=.5, detection_source='sam2_mask', evidence='sam2_mask',
            bbox_pixels=[0, 0, 100, 200])
        face = dict(shot_id='s', source_frame=0, entity_id='G1', tracklet_id='face',
            detection_id='face', confidence=.99, detection_source='yolo_full', evidence='observed_detection',
            bbox_pixels=[20, 10, 50, 40])
        kept, audit = suppress_duplicate_boxes([face, outer])
        self.assertEqual([row['detection_id'] for row in kept], ['body'])
        self.assertEqual(audit['suppressed'][0]['method'], 'same_id_containment')
        self.assertEqual(audit['suppressed'][0]['removed_observation']['evidence'], 'observed_detection')
        self.assertEqual(kept[0]['merged_source_observations'][0]['detection_source'], 'yolo_full')
        self.assertEqual(kept[0]['bbox_pixels'], [0, 0, 100, 200])

    def test_nested_carrier_and_passenger_with_distinct_ids_survive(self):
        rows = [dict(shot_id='s', source_frame=0, entity_id=eid, tracklet_id=eid,
            detection_id=eid, confidence=.9, bbox_pixels=box) for eid,box in
            [('G1', [0, 0, 100, 200]), ('G2', [20, 10, 50, 40])]]
        kept, audit = suppress_duplicate_boxes(rows)
        self.assertEqual(len(kept), 2)
        self.assertEqual(audit['suppressed_count'], 0)

    def test_same_id_disjoint_boxes_remain_conflict(self):
        rows = [dict(shot_id='s', source_frame=0, entity_id='G1', tracklet_id=str(i),
            detection_id=str(i), confidence=.9, bbox_pixels=box) for i,box in
            enumerate(([0, 0, 20, 20], [100, 100, 120, 120]))]
        kept, audit = suppress_duplicate_boxes(rows)
        self.assertEqual(len(kept), 2)
        self.assertEqual(audit['suppressed_count'], 0)
        self.assertEqual(audit['conflict_count'], 1)

    def test_model_free_export_offsets_and_status_ownership(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); bundle, local, global_result = fixture(folder)
            output = folder/'out'; output.mkdir(); write_json(output/'status.json', {'owner': 'root'})
            result = export_story(bundle, local, global_result, output, render_videos=False)
            self.assertEqual(json.loads((output/'status.json').read_text()), {'owner': 'root'})
            story = json.loads((output/'story.json').read_text())
            event = story['shots'][0]['events'][0]
            self.assertEqual(event['original_start_sec'], 3.)
            self.assertEqual(story['entities'][0]['first_seen_sec'], 3.)
            self.assertTrue(result['validation']['structurally_complete'])

    def test_sam_and_detector_fallback_provenance_survives_export(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); bundle, local, global_result = fixture(folder)
            rows = bundle['shots'][0]['tracks'][0]['observations']
            rows[0].update(evidence='sam2_mask', detection_source='sam2_mask', confidence=.61,
                confidence_semantics='uncalibrated mask stability times directional agreement',
                mask='masks/f0.png', masked_crop='masked_crops/f0.png',
                source_track_ids=['detector:T1', 'detector:T2'], detector_supported=False,
                nearest_detector_frame=1, mask_quality=dict(stability=.8, forward_backward_iou=.7625, calibrated=False))
            rows[1].update(evidence='observed_detection', detection_source='yolo_small_target_tile',
                           mask_refinement_status='detector_fallback', tracker_confirmed=True,
                           source_track_ids=['detector:T1'], original_box_candidates=2)
            original = deepcopy(bundle)
            export_story(bundle, local, global_result, folder/'out', render_videos=False)
            exported = json.loads((folder/'out/observations.json').read_text())['observations']
            mask, detection = exported[:2]
            self.assertEqual(mask['observation_type'], 'sam2_mask')
            self.assertEqual(mask['evidence'], 'sam2_mask')
            self.assertEqual(mask['detection_source'], 'sam2_mask')
            self.assertEqual(mask['confidence'], .61)
            self.assertIsNone(mask['detection_confidence'])
            self.assertFalse(mask['detector_supported'])
            self.assertEqual(mask['source_track_ids'], ['detector:T1', 'detector:T2'])
            self.assertEqual(mask['mask_quality'], rows[0]['mask_quality'])
            self.assertEqual(mask['mask'], 'masks/f0.png')
            self.assertEqual(detection['observation_type'], 'observed_detection')
            self.assertEqual(detection['detection_source'], 'yolo_small_target_tile')
            self.assertEqual(detection['detection_confidence'], .9)
            self.assertEqual(detection['mask_refinement_status'], 'detector_fallback')
            self.assertEqual(bundle, original)

    def test_explicit_observation_type_takes_priority(self):
        from story_export import _observation_provenance
        provenance = _observation_provenance(dict(observation_type='manual_verified_mask',
            evidence='sam2_mask', detection_source='sam2_mask', confidence=.4))
        self.assertEqual(provenance['observation_type'], 'manual_verified_mask')
        self.assertIsNone(provenance['detection_confidence'])

    def test_original_and_adaptive_exact_timing(self):
        import numpy as np
        from fractions import Fraction
        import av
        sys.path.insert(0, str(ROOT.parent/'03_adaptive_sampling'))
        from media import VideoWriter
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); bundle, local, global_result = fixture(folder)
            writer = VideoWriter(folder/'source.mp4', 160, 96, Fraction(24))
            for index in range(5):
                writer.write(index, np.full((96, 160, 3), index*40, np.uint8))
            writer.close()
            adaptive = folder/'adaptive'; adaptive.mkdir()
            write_json(adaptive/'summary.json', {'videos': [{'id': 's1', 'samples': 'samples.json'}]})
            write_json(adaptive/'samples.json', dict(source_frames=5, source_fps='24', original_start_frame=72,
                samples=[dict(source_frame=0, hold_until_frame=3), dict(source_frame=3, hold_until_frame=5)]))
            result = export_story(bundle, local, global_result, folder/'out', adaptive)
            self.assertTrue(result['validation']['video']['passed'])
            self.assertEqual(result['validation']['video']['sampled_frames'], 2)
            with av.open(str(folder/'out/adaptive_entities_full.mp4')) as container:
                stream = container.streams.video[0]
                self.assertEqual(stream.duration*stream.time_base, Fraction(5, 24))
                self.assertEqual([f.pts*f.time_base for f in container.decode(stream)], [Fraction(0), Fraction(3, 24)])


class EvaluationTests(unittest.TestCase):
    @staticmethod
    def obs(eid, frame, x=0, kind='person'):
        return dict(entity_id=eid, detection_id=f'{eid}:{frame}', shot_id='s1', source_frame=frame,
                    bbox=[x, 0, x+20, 20], kind=kind)

    def test_pair_recall_unknowns_and_false_merges(self):
        old = [self.obs('A', 0), self.obs('B', 1), self.obs('C', 2), self.obs('unlabeled', 3)]
        new = [self.obs('G1', 0), self.obs('G1', 1), self.obs('G2', 2), self.obs('G1', 3)]
        labels = dict(groups=[dict(name='one', kind='person', ids=['A', 'B', 'A']), dict(name='two', ids=['C'])])
        result = evaluate_partial_identities(old, new, labels)
        self.assertEqual(result['current']['same_id_pair_recall'], 1.)
        self.assertEqual(result['baseline']['same_id_pair_recall'], 0.)
        self.assertEqual(result['current']['false_merge_pairs'], 0)
        self.assertEqual(result['annotation_ids'], 3)
        new[2]['entity_id'] = 'G1'
        result = evaluate_partial_identities(old, new, labels)
        self.assertEqual(result['current']['false_merge_pairs'], 2)

    def test_missing_detection_cannot_improve_unconditional_recall(self):
        old = [self.obs('A', 0), self.obs('B', 1), self.obs('C', 2)]
        new = [self.obs('G1', 0), self.obs('G1', 1)]
        result = evaluate_partial_identities(old, new, dict(groups=[dict(name='one', ids=['A', 'B', 'C'])]))
        self.assertEqual(result['current']['same_id_pair_recall'], 1/3)
        self.assertEqual(result['current']['covered_same_id_pair_recall'], 1.)

    def test_contradiction_is_not_transitively_merged(self):
        old = [self.obs('A', 0), self.obs('B', 1), self.obs('C', 2)]
        new = [self.obs('G1', 0), self.obs('G2', 1), self.obs('G3', 2)]
        labels = dict(groups=[dict(name='one', ids=['A', 'B']), dict(name='two', ids=['B', 'C', 'typo'])])
        result = evaluate_partial_identities(old, new, labels)
        self.assertEqual(result['exclusions']['contradictory_ids'], ['B'])
        self.assertEqual(result['exclusions']['unavailable_ids'], ['typo'])
        self.assertEqual(result['current']['known_different_pairs_evaluated'], 0)

    def test_normalized_old_coordinates(self):
        old = self.obs('A', 0)
        old.update(bbox=[0, 0, 100, 200], source_size=[200, 100], bbox_format='xyxy normalized 0..1000')
        result = evaluate_partial_identities([old], [self.obs('G1', 0)], dict(groups=[dict(name='one', ids=['A'])]))
        self.assertEqual(result['annotation_id_coverage'], 1.)

    def test_fused_source_box_preserves_geometry_transfer_coverage(self):
        old = [self.obs('A', 0), self.obs('B', 1)]
        new = [dict(self.obs('G1', frame), bbox=[0, 0, 400, 400],
            merged_source_observations=[dict(detection_id=f'face:{frame}', bbox_pixels=[0, 0, 20, 20])])
            for frame in range(2)]
        result = evaluate_partial_identities(old, new, dict(groups=[dict(name='one', ids=['A', 'B'])]))
        self.assertEqual(result['annotation_id_coverage'], 1.)
        self.assertEqual(result['current']['same_id_pair_recall'], 1.)
        self.assertTrue(all(row['matched_geometry'] == 'merged_source_box' for row in result['transfer']['matches']))


if __name__ == '__main__':
    unittest.main()
