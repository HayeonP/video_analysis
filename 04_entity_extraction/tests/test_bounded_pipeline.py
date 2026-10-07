"""CPU-only regressions: fake Qwen generations, temporary outputs, no downloads."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent/'01_low_freq_event'),
               str(ROOT.parent/'03_adaptive_sampling')]

import bounded_matching as bm
from efficient_tracks import build_tracklets, rank_candidates
from low_freq_event.backend import QwenBackend
from low_freq_event.config import Settings


VALID = json.dumps({'scene_summary': 'observed', 'groups': [], 'rejected_ids': []})


def observation(shot='s1', index=0, *, time=None, kind='person',
                vector=None, bbox=None, suffix=''):
    vector = [1., 0., 0., 0.] if vector is None else vector
    value = dict(frame_key=f'{shot}_f{index}', shot_id=shot,
        detection_id=f'{shot}_f{index}_d{suffix}',
        full_time_sec=float(index if time is None else time),
        original_time_sec=float(index if time is None else time),
        bbox=bbox or [100., 100., 300., 700.], source_size=[1920, 1080],
        kind='person' if kind == 'person' else 'object', entity_kind=kind,
        label='인물' if kind == 'person' else '동물', appearance='observed appearance',
        detector_class='person' if kind == 'person' else 'bird',
        detection_confidence=.9, face_quality=0., face_feature=None)
    value['body_feature' if kind == 'person' else 'object_feature'] = vector
    return value


def entry(obs):
    return dict(label=obs['label'], kind=obs['entity_kind'],
                appearance=obs['appearance'], shot_id=obs['shot_id'], observations=[obs])


def frames_for(observations):
    values = {}
    for obs in observations:
        values[obs['frame_key']] = {k: obs[k] for k in ('frame_key', 'shot_id', 'full_time_sec')}
    return list(values.values())


def group(ids, *, kind='person', match_id=None):
    return bm.EntityGroup(ids=ids, kind=kind, match_id=match_id,
                          label='검토한 인물', appearance='visible traits', reason='visible evidence')


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='bounded-budget-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = Settings(json_retries=0)
        self.backend = bm.BudgetBackend(self.settings, self.root/'review_traces', 8)

    def request(self, prompt):
        return self.backend.request(prompt, bm.BatchDecision)

    def test_absolute_cap_counts_eight_actual_generations(self):
        with patch.object(QwenBackend, '_generate', return_value=VALID) as generate:
            for index in range(8):
                self.request(f'unique request {index}')
            with self.assertRaises(bm.BudgetExhausted):
                self.request('ninth uncached request')
        self.assertEqual(generate.call_count, 8)
        self.assertEqual(self.backend.ledger['generations'], 8)
        self.assertEqual(self.backend.remaining, 0)

    def test_cache_hit_at_zero_budget_and_reserved_slot_does_not_generate(self):
        with patch.object(QwenBackend, '_generate', return_value=VALID) as generate:
            for index in range(8):
                self.request(f'cached request {index}')
            self.backend.reserve = 1
            result = self.request('cached request 0')
        self.assertEqual(result.scene_summary, 'observed')
        self.assertEqual(generate.call_count, 8)
        self.assertEqual(self.backend.ledger['generations'], 8)

    def test_final_one_generation_is_reserved(self):
        self.backend.reserve = 1
        with patch.object(QwenBackend, '_generate', return_value=VALID) as generate:
            for index in range(7):
                self.request(f'batch {index}')
            with self.assertRaises(bm.BudgetExhausted):
                self.request('eighth batch is not allowed')
            self.assertEqual(self.backend.remaining, 1)
            self.backend.reserve = 0
            self.request('final catalog')
        self.assertEqual(generate.call_count, 8)
        self.assertEqual(self.backend.remaining, 0)

    def test_structured_output_retry_consumes_actual_generation(self):
        self.backend.settings = Settings(json_retries=1)
        with patch.object(QwenBackend, '_generate', side_effect=['invalid JSON', VALID]) as generate:
            self.request('retry once')
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(self.backend.ledger['generations'], 2)

    def test_failed_generation_consumes_slot_and_persists_for_resume(self):
        with patch.object(QwenBackend, '_generate', side_effect=RuntimeError('mock OOM')):
            with self.assertRaisesRegex(RuntimeError, 'mock OOM'):
                self.request('failed generation')
        self.assertEqual(self.backend.ledger['generations'], 1)
        self.assertEqual(self.backend.ledger['events'][0]['status'], 'failed')
        resumed = bm.BudgetBackend(self.settings, self.root/'review_traces', 8)
        self.assertEqual(resumed.remaining, 7)

    def test_manual_ask_retry_is_cached_and_counts_two_generations(self):
        dest = self.root/'ask'
        dest.mkdir()
        with patch.object(bm, 'gallery', return_value=([], [])), \
             patch.object(bm, 'scene_clip', return_value=None), \
             patch.object(QwenBackend, '_generate', side_effect=['invalid JSON', VALID]) as generate:
            result = bm.ask(self.backend, 'retry batch', [], {}, self.root, dest, 24.)
        self.assertIsNotNone(result)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(self.backend.ledger['generations'], 2)

    def test_successful_retry_cache_replays_when_budget_is_exhausted(self):
        dest = self.root/'retry-cache'
        dest.mkdir()
        with patch.object(bm, 'gallery', return_value=([], [])), \
             patch.object(bm, 'scene_clip', return_value=None), \
             patch.object(QwenBackend, '_generate', side_effect=['invalid JSON', VALID]):
            first = bm.ask(self.backend, 'retry and resume', [], {}, self.root, dest, 24.)
        self.assertIsNotNone(first)
        self.backend.reserve = 0
        with patch.object(QwenBackend, '_generate', return_value=VALID):
            for index in range(6):
                self.request(f'consume remaining {index}')
        self.assertEqual(self.backend.remaining, 0)
        with patch.object(bm, 'gallery', return_value=([], [])), \
             patch.object(bm, 'scene_clip', return_value=None), \
             patch.object(QwenBackend, '_generate', side_effect=AssertionError('must use cache')):
            replay = bm.ask(self.backend, 'retry and resume', [], {}, self.root, dest, 24.)
        self.assertIsNotNone(replay, 'Successful corrected response must survive an exhausted-budget rerun')
        self.assertEqual(replay.scene_summary, first.scene_summary)
        self.assertEqual(self.backend.ledger['generations'], 8)


class GroupConstraintTests(unittest.TestCase):
    def apply(self, current, decisions, memory=None):
        memory = {} if memory is None else memory
        assignments = {}
        joins = []
        rejected = bm.apply_groups(bm.BatchDecision(groups=decisions), current,
                                   memory, assignments, joins, 'unit-test')
        return memory, assignments, joins, rejected

    def test_person_cannot_merge_with_animal(self):
        current = {'T1': entry(observation('s1', 0)),
                   'T2': entry(observation('s2', 1, kind='animal'))}
        _, assignments, joins, rejected = self.apply(current, [group(['T1', 'T2'])])
        self.assertEqual(len(set(assignments.values())), 2)
        self.assertEqual(joins, [])
        self.assertEqual(rejected, [])

    def test_person_cannot_match_animal_memory(self):
        memory = {'G0001': dict(entry(observation('s1', 0, kind='animal')), members=['old'], history=[])}
        _, assignments, joins, _ = self.apply({'T1': entry(observation('s2', 1))},
                                             [group(['T1'], match_id='G0001')], memory)
        self.assertNotEqual(assignments['T1'], 'G0001')
        self.assertEqual(joins, [])

    def test_simultaneous_people_cannot_merge(self):
        current = {'T1': entry(observation('s1', 0, suffix='a')),
                   'T2': entry(observation('s1', 0, suffix='b', bbox=[600., 100., 800., 700.]))}
        _, assignments, _, _ = self.apply(current, [group(['T1', 'T2'])])
        self.assertEqual(len(set(assignments.values())), 2)

    def test_separate_groups_cannot_reuse_memory_for_simultaneous_people(self):
        memory = {'G0001': dict(entry(observation('s0', 0)), members=['old'], history=[])}
        current = {'T1': entry(observation('s1', 1, suffix='a')),
                   'T2': entry(observation('s1', 1, suffix='b'))}
        _, assignments, _, _ = self.apply(current,
            [group(['T1'], match_id='G0001'), group(['T2'], match_id='G0001')], memory)
        self.assertEqual(assignments['T1'], 'G0001')
        self.assertNotEqual(assignments['T2'], 'G0001')

    def test_valid_group_preserves_all_observations_and_inputs(self):
        current = {'T1': entry(observation('s1', 0)), 'T2': entry(observation('s2', 1))}
        original = deepcopy(current)
        memory, assignments, _, _ = self.apply(current, [group(['T1', 'T2'])])
        self.assertEqual(len(set(assignments.values())), 1)
        self.assertEqual(len(memory[assignments['T1']]['observations']), 2)
        self.assertEqual(current, original)

    def test_invalid_duplicate_group_does_not_drop_an_observation(self):
        current = {'T1': entry(observation('s1', 0)), 'T2': entry(observation('s2', 1))}
        _, assignments, _, _ = self.apply(current, [group(['T1', 'T1', 'T2'])])
        self.assertEqual(set(assignments), set(current))
        self.assertEqual(len(set(assignments.values())), 2)


class TrackletTests(unittest.TestCase):
    def test_adjacent_samples_track_without_mutating_inputs(self):
        raw = [observation(index=0, time=0.),
               observation(index=1, time=.25, bbox=[110., 100., 310., 700.])]
        saved = deepcopy(raw)
        frames = frames_for(raw)
        saved_frames = deepcopy(frames)
        tracks = build_tracklets(raw, frames, Settings())
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(next(iter(tracks.values()))['observations']), 2)
        self.assertEqual(raw, saved)
        self.assertEqual(frames, saved_frames)
        json.dumps(tracks, allow_nan=False)

    def test_same_appearance_never_connects_across_shots(self):
        raw = [observation('s1', 0, time=0.), observation('s2', 0, time=.1)]
        tracks = build_tracklets(raw, frames_for(raw), Settings())
        self.assertEqual(len(tracks), 2)
        for track in tracks.values():
            self.assertEqual(len({o['shot_id'] for o in track['observations']}), 1)

    def test_each_frame_assigns_each_track_at_most_once(self):
        raw = []
        for index in (0, 1):
            raw.append(observation(index=index, time=index*.25, suffix='a'))
            raw.append(observation(index=index, time=index*.25, suffix='b',
                vector=[0., 1., 0., 0.], bbox=[650., 100., 850., 700.]))
        tracks = build_tracklets(raw, frames_for(raw), Settings())
        self.assertEqual(len(tracks), 2)
        self.assertEqual(sorted(len(t['observations']) for t in tracks.values()), [2, 2])
        for track in tracks.values():
            keys = [o['frame_key'] for o in track['observations']]
            self.assertEqual(len(keys), len(set(keys)))

    def test_long_time_gap_starts_new_track(self):
        raw = [observation(index=0, time=0.), observation(index=1, time=3.)]
        self.assertEqual(len(build_tracklets(raw, frames_for(raw), Settings())), 2)

    def test_nested_carrier_passenger_are_not_deduplicated(self):
        raw = [observation(kind='animal', suffix='carrier', bbox=[100., 100., 800., 800.]),
               observation(kind='animal', suffix='passenger', bbox=[350., 350., 450., 450.])]
        self.assertEqual(len(build_tracklets(raw, frames_for(raw), Settings())), 2)

    def test_bird_airplane_class_flip_can_keep_same_track(self):
        raw = [observation(index=0, time=0., kind='animal'),
               observation(index=1, time=.2, kind='object', bbox=[105., 100., 305., 700.])]
        raw[1]['detector_class'] = 'airplane'
        self.assertEqual(len(build_tracklets(raw, frames_for(raw), Settings())), 1)


class CandidateTests(unittest.TestCase):
    def test_top_three_are_deterministic_and_inputs_immutable(self):
        query = entry(observation('current', 20))
        registry = {f'G{i:04d}': entry(observation(f's{i}', i)) for i in range(6)}
        original = deepcopy((query, registry))
        result = rank_candidates(query, registry, limit=3)
        self.assertEqual([r['entity_id'] for r in result], ['G0000', 'G0001', 'G0002'])
        self.assertEqual((query, registry), original)
        json.dumps(result, allow_nan=False)

    def test_candidate_guards_exclude_person_animal_and_covisible(self):
        query = entry(observation('current', 20))
        registry = {'animal': entry(observation('old', 1, kind='animal')),
                    'covisible': entry(observation('current', 20, suffix='other')),
                    'compatible': entry(observation('old', 2))}
        self.assertEqual([r['entity_id'] for r in rank_candidates(query, registry)], ['compatible'])

    def test_visual_feature_works_without_face_or_body_embedding(self):
        query_obs = observation('current', 20)
        old_obs = observation('old', 1)
        for obs in (query_obs, old_obs):
            obs.pop('body_feature')
            obs['visual_feature'] = [0., 1., 0., 0.]
        result = rank_candidates(entry(query_obs), {'G1': entry(old_obs)})
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['score'], .95)
        self.assertIsNotNone(result[0]['evidence']['visual'])

    def test_zero_limit_is_empty(self):
        query = entry(observation())
        self.assertEqual(rank_candidates(query, {'G1': entry(observation('s2', 0))}, limit=0), [])


if __name__ == '__main__':
    unittest.main()
