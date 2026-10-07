"""Synthetic CPU-only checks; no model imports or inference."""
from copy import deepcopy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from duplicate_tracks import duplicate_evidence, resolve_assigned_duplicates


def feature(cosine=1.):
    return [cosine, math.sqrt(1-cosine*cosine)]


def observation(key, *, box=(100, 100, 400, 800), kind='person', confidence=.8,
                body=1., visual=1., face=None, face_quality=0., frame='s1_f0'):
    result = dict(detection_id=key, frame_key=frame, shot_id='s1', bbox=list(box),
                  kind='person' if kind == 'person' else 'object', entity_kind=kind,
                  detection_confidence=confidence, body_feature=feature(body),
                  visual_feature=feature(visual), object_feature=feature(visual),
                  face_quality=face_quality)
    if face is not None:
        result['face_feature'] = feature(face)
    return result


def groups(*observations):
    return {f'T{i}': dict(kind=o['entity_kind'], observations=[o]) for i, o in enumerate(observations)}


class DuplicateTracksTests(unittest.TestCase):
    def test_person_body_and_visual_with_moderate_overlap(self):
        a = observation('a')
        b = observation('b', box=(248, 100, 548, 800), body=.843, visual=.888)
        proof = duplicate_evidence(a, b)
        self.assertIsNotNone(proof)
        self.assertAlmostEqual(proof['iou'], 152/448)

    def test_body_only_does_not_pass(self):
        self.assertIsNone(duplicate_evidence(observation('a'), observation('b', visual=.84)))

    def test_quality_face_can_supply_evidence(self):
        a = observation('a', face=1., face_quality=100.)
        b = observation('b', body=.3, visual=.3, face=.70, face_quality=90.)
        self.assertEqual(duplicate_evidence(a, b)['method'], 'reliable face')
        b['face_quality'] = 20.
        self.assertIsNone(duplicate_evidence(a, b))

    def test_reliable_face_conflict_blocks_clothing_match(self):
        self.assertIsNone(duplicate_evidence(observation('a', face=1., face_quality=100.),
                                           observation('b', face=.1, face_quality=100.)))

    def test_settings_can_raise_face_quality_bar(self):
        a = observation('a', face=1., face_quality=90.)
        b = observation('b', body=.1, visual=.1, face=.9, face_quality=90.)
        self.assertIsNone(duplicate_evidence(a, b, SimpleNamespace(face_conflict_quality=100.)))

    def test_carrier_and_small_passenger_are_not_duplicates(self):
        a = observation('a', box=(0, 0, 1000, 1000), kind='animal')
        b = observation('b', box=(400, 400, 550, 600), kind='animal')
        self.assertIsNone(duplicate_evidence(a, b))

    def test_nonperson_threshold_is_stricter(self):
        a = observation('a', kind='animal')
        self.assertIsNotNone(duplicate_evidence(a, observation('b', kind='animal', visual=.95)))
        self.assertIsNone(duplicate_evidence(a, observation('b', kind='animal', visual=.93)))
        self.assertIsNone(duplicate_evidence(a, observation('b', kind='animal', box=(248, 100, 548, 800))))

    def test_corrected_person_to_animal_reuses_person_dino(self):
        a = observation('a', kind='animal')
        b = observation('b', kind='animal', visual=.96)
        del a['object_feature']
        del b['object_feature']
        proof = duplicate_evidence(a, b)
        self.assertIsNotNone(proof)
        self.assertAlmostEqual(proof['object_similarity'], .96)

    def test_person_visual_comparison_can_reuse_object_dino(self):
        a = observation('a')
        b = observation('b', body=.84, visual=.89)
        del b['visual_feature']
        proof = duplicate_evidence(a, b)
        self.assertIsNotNone(proof)
        self.assertAlmostEqual(proof['visual_similarity'], .89)

    def test_different_frame_family_or_invalid_features_fail(self):
        a = observation('a')
        self.assertIsNone(duplicate_evidence(a, observation('b', frame='s1_f1')))
        self.assertIsNone(duplicate_evidence(a, observation('b', kind='animal')))
        b = observation('b'); b['visual_feature'] = [float('nan'), 0.]
        self.assertIsNone(duplicate_evidence(a, b))

    def test_highest_confidence_wins_and_inputs_remain_unchanged(self):
        source = groups(observation('a', confidence=.5), observation('b', confidence=.9))
        mapping = {'T0': 'G1', 'T1': 'G1'}
        before = deepcopy((source, mapping))
        out, assigned, audit = resolve_assigned_duplicates(source, mapping)
        self.assertEqual(set(out), {'T1'})
        self.assertEqual(assigned, {'T1': 'G1'})
        self.assertEqual(audit['suppressions'][0]['suppressed_detection_id'], 'a')
        self.assertEqual((source, mapping), before)

    def test_larger_box_breaks_equal_confidence_tie(self):
        source = groups(observation('a', box=(100, 100, 380, 780)), observation('b'))
        out, _, _ = resolve_assigned_duplicates(source, {'T0': 'G1', 'T1': 'G1'})
        self.assertEqual(set(out), {'T1'})

    def test_different_assigned_identities_remain_separate(self):
        source = groups(observation('a'), observation('b'))
        out, assigned, audit = resolve_assigned_duplicates(source, {'T0': 'G1', 'T1': 'G2'})
        self.assertEqual(len(out), 2)
        self.assertEqual(len(assigned), 2)
        self.assertEqual(audit['suppressed_count'], 0)

    def test_unsupported_covisibility_is_reported_not_removed(self):
        source = groups(observation('a'), observation('b', box=(700, 100, 900, 800)))
        out, _, audit = resolve_assigned_duplicates(source, {'T0': 'G1', 'T1': 'G1'})
        self.assertEqual(len(out), 2)
        self.assertEqual(audit['suppressed_count'], 0)
        self.assertEqual(len(audit['remaining_conflicts']), 1)

    def test_only_duplicate_observation_removed_from_multiframe_track(self):
        a = observation('a', confidence=.9)
        b = observation('b', confidence=.5)
        c = observation('c', frame='s1_f1')
        source = groups(a, b); source['T1']['observations'].append(c)
        out, assigned, audit = resolve_assigned_duplicates(source, {'T0': 'G1', 'T1': 'G1'})
        self.assertEqual([o['detection_id'] for o in out['T1']['observations']], ['c'])
        self.assertEqual(len(assigned), 2)
        self.assertEqual(audit['suppressed_count'], 1)


if __name__ == '__main__':
    unittest.main()
