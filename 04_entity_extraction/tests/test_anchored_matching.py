"""CPU-only anchor invariants; no model generation, GPU use or project outputs."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_bounded_pipeline import ROOT, observation, entry, frames_for
import anchored_matching as am
from efficient_tracks import rank_candidates
from low_freq_event.backend import QwenBackend
from low_freq_event.config import Settings


def anchor_spec(key='A001', refs=None, kind='person'):
    return am.AnchorSpec(anchor_id=key, label='기준 인물', kind=kind,
        appearance='visible stable traits', reference_track_ids=refs or ['T1'], evidence='actual visible reference')


def verdict(key, anchor='A001', *, decision='match', correction=None):
    return am.TrackDecision(track_id=key, decision=decision, anchor_id=anchor,
        correction=correction, reason='matching visible traits', evidence=['specific observed traits'])


class AnchorInvariantTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(json_retries=0)

    def make(self, groups, specs=None, corrections=None):
        return am.make_anchors(am.OverviewDecision(anchors=specs or [anchor_spec()],
            corrections=corrections or []), groups, self.settings)

    def test_anchor_references_never_export_as_unassigned_observations(self):
        groups = {'T1': entry(observation('reference', 0)), 'T2': entry(observation('current', 1))}
        anchors, corrections, invalid = self.make(groups)
        self.assertEqual(invalid, [])
        memory, mapping, _ = am.finalize_memory(groups, anchors, {'T2': 'A001'},
            {'T2': {'status': 'matched', 'reason': 'observed'}}, corrections)
        self.assertEqual(set(mapping), {'T2'})
        self.assertEqual([o['detection_id'] for o in memory['A001']['observations']],
                         [groups['T2']['observations'][0]['detection_id']])
        self.assertEqual(memory['A001']['members'], ['T2'])

    def test_reference_current_track_does_not_collide_with_itself(self):
        groups = {'T1': entry(observation())}
        anchors, corrections, _ = self.make(groups)
        mapping, states = {}, {}
        am.apply_assignments(am.AssignmentDecision(assignments=[verdict('T1')]), groups,
            anchors, groups, mapping, states, corrections, self.settings)
        self.assertEqual(mapping, {'T1': 'A001'})
        self.assertEqual(states['T1']['status'], 'matched')

    def test_anchor_gallery_is_fixed_after_assignment(self):
        groups = {'T1': entry(observation('reference', 0)), 'T2': entry(observation('current', 1))}
        anchors, corrections, _ = self.make(groups)
        original = deepcopy((anchors, groups))
        am.apply_assignments(am.AssignmentDecision(assignments=[verdict('T2')]),
            {'T2': groups['T2']}, anchors, groups, {}, {}, corrections, self.settings)
        self.assertEqual((anchors, groups), original)

    def test_same_reference_cannot_define_two_anchors(self):
        groups = {'T1': entry(observation())}
        anchors, _, invalid = self.make(groups, [anchor_spec('A001'), anchor_spec('A002')])
        self.assertEqual(list(anchors), ['A001'])
        self.assertEqual(len(invalid), 1)
        self.assertIn('already owns', invalid[0]['reason'])

    def test_distinct_simultaneous_references_cannot_define_one_anchor(self):
        groups = {'T1': entry(observation(suffix='a')),
                  'T2': entry(observation(suffix='b', bbox=[700., 100., 900., 700.]))}
        anchors, _, invalid = self.make(groups, [anchor_spec(refs=['T1', 'T2'])])
        self.assertEqual(anchors, {})
        self.assertIn('simultaneous', invalid[0]['reason'])

    def test_duplicate_reference_boxes_can_define_one_anchor(self):
        left, right = observation(suffix='a'), observation(suffix='b')
        left['visual_feature'] = right['visual_feature'] = [1., 0., 0., 0.]
        anchors, _, invalid = self.make({'T1': entry(left), 'T2': entry(right)},
                                      [anchor_spec(refs=['T1', 'T2'])])
        self.assertEqual(list(anchors), ['A001'])
        self.assertEqual(invalid, [])

    def test_clear_face_contradiction_rejects_reference_group(self):
        a, b = observation('s1', 0), observation('s2', 1)
        a.update(face_quality=100., face_feature=[1., 0., 0., 0.])
        b.update(face_quality=100., face_feature=[0., 1., 0., 0.])
        anchors, _, invalid = self.make({'T1': entry(a), 'T2': entry(b)},
                                      [anchor_spec(refs=['T1', 'T2'])])
        self.assertEqual(anchors, {})
        self.assertIn('face contradiction', invalid[0]['reason'])

    def test_person_animal_reference_requires_explicit_kind_correction(self):
        groups = {'T1': entry(observation(kind='person'))}
        anchors, _, invalid = self.make(groups, [anchor_spec(kind='animal')])
        self.assertEqual(anchors, {})
        self.assertTrue(invalid)
        correction = am.ReferenceCorrection(track_id='T1', kind='animal', reason='visible wings and nonhuman anatomy')
        anchors, corrections, _ = self.make(groups, [anchor_spec(kind='animal')], [correction])
        self.assertEqual(anchors['A001']['kind'], 'animal')
        self.assertEqual(groups['T1']['kind'], 'person')
        self.assertEqual(corrections['T1']['kind'], 'animal')

    def test_distinct_simultaneous_assignments_are_not_merged(self):
        groups = {'T1': entry(observation('reference', 0)),
                  'T2': entry(observation('current', 1, suffix='a')),
                  'T3': entry(observation('current', 1, suffix='b', bbox=[700., 100., 900., 700.]))}
        anchors, corrections, _ = self.make(groups)
        mapping, states = {}, {}
        current = {key: groups[key] for key in ('T2', 'T3')}
        am.apply_assignments(am.AssignmentDecision(assignments=[verdict('T2'), verdict('T3')]),
            current, anchors, groups, mapping, states, corrections, self.settings)
        self.assertEqual(mapping['T2'], 'A001')
        self.assertEqual(mapping['T3'], 'U_T3')
        self.assertEqual(states['T3']['status'], 'co_visible_conflict')

    def test_unsupported_model_match_stays_unknown(self):
        groups = {'T1': entry(observation('reference', 0)),
                  'T2': entry(observation('current', 1, vector=[0., 1., 0., 0.]))}
        anchors, corrections, _ = self.make(groups)
        mapping, states = {}, {}
        am.apply_assignments(am.AssignmentDecision(assignments=[verdict('T2')]),
            {'T2': groups['T2']}, anchors, groups, mapping, states, corrections, self.settings)
        self.assertEqual(mapping, {'T2': 'U_T2'})
        self.assertEqual(states['T2']['status'], 'low_support')

    def test_unknowns_remain_distinct_and_expose_unresolved_status(self):
        groups = {'T1': entry(observation('s1', 0)), 'T2': entry(observation('s2', 1))}
        mapping, states = {}, {}
        am.apply_assignments(None, groups, {}, groups, mapping, states, {}, self.settings)
        memory, mapping, _ = am.finalize_memory(groups, {}, mapping, states, {})
        self.assertEqual(set(mapping.values()), {'U_T1', 'U_T2'})
        self.assertTrue(all(m['identity_status'] == 'unresolved' for m in memory.values()))

    def test_duplicate_response_rows_do_not_silently_override(self):
        groups = {'T1': entry(observation())}
        anchors, corrections, _ = self.make(groups)
        mapping, states = {}, {}
        am.apply_assignments(am.AssignmentDecision(assignments=[verdict('T1'), verdict('T1')]),
            groups, anchors, groups, mapping, states, corrections, self.settings)
        self.assertEqual(mapping['T1'], 'U_T1')

    def test_retrieval_includes_reference_without_covisibility_penalty(self):
        groups = {'T1': entry(observation())}
        anchors, _, _ = self.make(groups)
        self.assertEqual([c['entity_id'] for c in am.retrieve(groups['T1'], anchors)], ['A001'])


class FeatureCorrectionTests(unittest.TestCase):
    def test_corrected_animal_uses_existing_person_dino(self):
        q = observation('q', 0)
        q['visual_feature'] = [1., 0., 0., 0.]
        query = entry(q)
        query['kind'] = 'animal'
        candidate = entry(observation('old', 1, kind='animal'))
        result = rank_candidates(query, {'A1': candidate})
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['evidence']['object']['score'], 1.)
        self.assertEqual(q['entity_kind'], 'person')

    def test_corrected_person_uses_existing_object_dino(self):
        q = observation('q', 0, kind='animal')
        query = entry(q)
        query['kind'] = 'person'
        old = observation('old', 1)
        old['visual_feature'] = [1., 0., 0., 0.]
        result = rank_candidates(query, {'A1': entry(old)})
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['evidence']['visual']['score'], 1.)


class CoordinatorTests(unittest.TestCase):
    def test_overview_covers_late_shots_before_extra_kind_buckets(self):
        groups = {}
        for index in range(12):
            for kind in ('person', 'animal', 'object'):
                key = f'T{index:02d}_{kind}'
                groups[key] = entry(observation(f's{index:02d}', index, kind=kind, suffix=kind))
        frames = frames_for([g['observations'][0] for g in groups.values()])
        with patch.object(am, 'representative_score', return_value=1.):
            selected = am.select_overview(groups, frames, Path('/unused'), limit=24)
        self.assertEqual(len(selected), 24)
        self.assertEqual({g['shot_id'] for g in selected.values()}, {f's{i:02d}' for i in range(12)})
        # The first pass supplies one track from every shot before any extras.
        self.assertEqual(len({g['shot_id'] for g in list(selected.values())[:12]}), 12)

    def test_overview_evenly_covers_more_shots_than_the_track_budget(self):
        groups = {f'T{i:02d}': entry(observation(f's{i:02d}', i)) for i in range(40)}
        frames = frames_for([g['observations'][0] for g in groups.values()])
        with patch.object(am, 'representative_score', return_value=1.):
            selected = am.select_overview(groups, frames, Path('/unused'), limit=24)
        expected = {f's{round(i*39/23):02d}' for i in range(24)}
        self.assertEqual({g['shot_id'] for g in selected.values()}, expected)
        self.assertIn('T00', selected)
        self.assertIn('T39', selected)

    def test_catalog_five_batches_and_one_final_are_fixed_request_sites(self):
        from catalog_review import CatalogDecision
        groups = {f'T{i:02d}': entry(observation(f's{i:02d}', i)) for i in range(17)}
        frames = frames_for([g['observations'][0] for g in groups.values()])
        timeline = dict(fps='24', shots=[dict(id=f's{i:02d}') for i in range(17)])
        calls = []
        def fake_request(backend, prompt, schema, selected, current, anchors, output, dest, fps, *,
                         final=False, current_are_identities=False):
            calls.append((schema, list(current), final, current_are_identities))
            if schema is CatalogDecision:
                return CatalogDecision(annotations=[dict(id=key, label='검토한 인물', kind=value['kind'],
                    appearance='visible stable traits', decision='keep', reason='observed appearance',
                    evidence=['actual source observation']) for key,value in current.items()])
            return am.AssignmentDecision(assignments=[verdict(key, None, decision='unknown') for key in current])
        def fake_select(current, frames, output, *, limit):
            return dict(list(current.items())[:limit])
        with tempfile.TemporaryDirectory(prefix='anchored-coordinator-test-') as tmp, \
             patch.object(am, 'select_overview', side_effect=fake_select), \
             patch.object(am, 'request_stage', side_effect=fake_request):
            memory, mapping, _ = am.match_anchored(groups, frames, timeline, Path(tmp),
                                                  Settings(json_retries=0), limit=8, batch_shots=1)
        self.assertEqual(len(calls), 7)
        self.assertEqual(sum(schema is CatalogDecision for schema, _, _, _ in calls), 2)
        self.assertEqual(sum(final for _, _, final, _ in calls), 1)
        self.assertTrue(calls[0][3])
        self.assertTrue(calls[-1][3])
        batches = [keys for schema, keys, _, identities in calls if schema is am.AssignmentDecision and not identities]
        self.assertEqual(len(batches), 5)
        self.assertEqual(sorted(key for keys in batches for key in keys), sorted(groups))
        self.assertLessEqual(len(calls[-1][1]), 24)
        self.assertEqual(len(mapping), 17)
        self.assertEqual(len(memory), 17)


class RequestStageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='anchored-request-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.backend = am.BudgetBackend(Settings(json_retries=0), self.root/'review_traces', 8)
        self.valid = '{"scene_summary":"cached response","assignments":[]}'

    def ask(self, prompt, *, final=False):
        return am.request_stage(self.backend, prompt, am.AssignmentDecision,
            [], {}, {}, self.root, self.root/'stage', 24., final=final)

    def test_production_request_replays_corrected_prompt_cache_at_zero_budget(self):
        with patch.object(am, 'render_gallery', return_value=([], [])), \
             patch.object(am, 'scene_clip', return_value=None), \
             patch.object(QwenBackend, '_generate', side_effect=['invalid JSON', self.valid]) as generate:
            first = self.ask('retry then replay')
        self.assertIsNotNone(first)
        self.assertEqual(generate.call_count, 2)
        self.backend.reserve = 0
        with patch.object(QwenBackend, '_generate', return_value=self.valid):
            for index in range(6):
                self.backend.request(f'consume slot {index}', am.AssignmentDecision)
        self.assertEqual(self.backend.remaining, 0)
        with patch.object(am, 'render_gallery', return_value=([], [])), \
             patch.object(am, 'scene_clip', return_value=None), \
             patch.object(QwenBackend, '_generate', side_effect=AssertionError('cache must be used')) as generate:
            replay = self.ask('retry then replay')
        self.assertIsNotNone(replay)
        self.assertEqual(replay.scene_summary, first.scene_summary)
        self.assertEqual(generate.call_count, 0)
        self.assertEqual(self.backend.ledger['generations'], 8)

    def test_production_oom_retry_counts_and_reserves_final_generation(self):
        answers = [am.torch.OutOfMemoryError('mock OOM'), self.valid] + [self.valid]*6
        with patch.object(am, 'render_gallery', return_value=([], [])), \
             patch.object(am, 'scene_clip', return_value=None), \
             patch.object(am.torch.cuda, 'empty_cache') as release, \
             patch.object(QwenBackend, '_generate', side_effect=answers) as generate:
            retried = self.ask('OOM then retry')
            self.assertIsNotNone(retried)
            self.assertEqual(self.backend.ledger['generations'], 2)
            self.assertEqual(self.backend.ledger['events'][0]['status'], 'failed')
            release.assert_called_once()
            for index in range(5):
                self.backend.request(f'other batch {index}', am.AssignmentDecision)
            self.assertEqual(self.backend.remaining, 1)
            self.assertIsNone(self.ask('batch cannot consume reserved final slot'))
            self.assertEqual(self.backend.remaining, 1)
            self.assertIsNotNone(self.ask('final unresolved review', final=True))
        self.assertEqual(generate.call_count, 8)
        self.assertEqual(self.backend.ledger['generations'], 8)
        self.assertEqual(self.backend.remaining, 0)


if __name__ == '__main__':
    unittest.main()
