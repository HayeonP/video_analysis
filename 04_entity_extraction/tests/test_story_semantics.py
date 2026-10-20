"""CPU-only semantic contracts; fake responses exercise production validators."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent/'01_low_freq_event')]
from PIL import Image
from pydantic import BaseModel
from low_freq_event.backend import ModelOutputError
from low_freq_event.config import Settings
from identity_review import ReviewBackend
from story_export import validate_story_contract
from story_semantics import (StoryBackend, BudgetExhausted, ShotDecision, GlobalDecision,
    validate_local, validate_global, review_shots, resolve_global, revise_narrative,
    cannot_links, scene_sheet, crop_sheet, gallery_pages, geometry_context,
    apply_global_rejections, pending_local_reviews, annotate_identity_scopes, guard_fragment_narrative)


class IdentityScopeTests(unittest.TestCase):
    def test_unowned_fragment_cannot_survive_as_a_separate_narrative_actor(self):
        entities=[dict(entity_id='G0001',label='traveler'),
                  dict(entity_id='G0002',label='unowned hand',identity_scope='part_owner_unresolved')]
        original='G0001 meets another person G0002.'
        shots=[dict(shot_id='scene',summary=original,entities=['G0001','G0002'],
                    events=[dict(subject='G0002',action='fights',object='G0001')]),
               dict(shot_id='other',summary='G0001 walks.',entities=['G0001'],events=[])]
        untouched=deepcopy(shots[1])
        self.assertEqual(guard_fragment_narrative(shots,entities),['scene'])
        self.assertEqual(shots[0]['unverified_summary'],original)
        self.assertEqual(shots[0]['events'],[])
        self.assertNotIn('another person',shots[0]['summary'])
        self.assertEqual(shots[0]['summary_scope'],'observed_ids_only')
        self.assertEqual(shots[1],untouched)

    def test_only_unowned_parts_are_relabelled_without_changing_partition(self):
        local=dict(shots=[dict(track_roles={'face':'whole_subject','hand':'body_part','sleeve':'worn_item'},
            entities=[dict(local_id='face',track_ids=['face']),dict(local_id='hand',track_ids=['hand']),
                      dict(local_id='sleeve',track_ids=['sleeve'])])])
        result=dict(entities=[dict(entity_id='A',local_ids=['face'],label='traveler',uncertain=False),
                             dict(entity_id='B',local_ids=['hand'],label='traveler',uncertain=False),
                             dict(entity_id='C',local_ids=['sleeve'],label='coat',uncertain=False)],
                    mapping={'face':'A','hand':'B','sleeve':'C'})
        before=deepcopy(result)
        annotated=annotate_identity_scopes(result,local)
        self.assertEqual(annotated['mapping'],before['mapping'])
        self.assertEqual(annotated['entities'][0],before['entities'][0])
        self.assertEqual(annotated['entities'][1]['identity_status'],'unresolved')
        self.assertEqual(annotated['entities'][1]['observation_label'],'traveler')
        self.assertIn('신체 부분',annotated['entities'][1]['label'])
        self.assertIn('착용물',annotated['entities'][2]['label'])
        snapshot=deepcopy(annotated)
        self.assertEqual(annotate_identity_scopes(annotated,local),snapshot)
        owned=dict(entities=[dict(entity_id='A',local_ids=['face','hand'],label='traveler',uncertain=False)])
        self.assertEqual(annotate_identity_scopes(deepcopy(owned),local),owned)


class Probe(BaseModel):
    value: int


class FakeBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.phase = ''
        self.reserve = 0

    def request(self, prompt, schema, **kwargs):
        self.calls.append(dict(prompt=prompt, schema=schema.__name__, **kwargs))
        payload = self.responses.pop(0)
        if isinstance(payload, Exception):
            raise payload
        if schema.__name__=='GlobalPartition' and 'entities' in payload:
            payload=dict(groups=[row['local_ids'] for row in payload['entities']],
                         rejections=payload.get('rejection_reasons',{}))
        response = schema.model_validate(payload)
        try:
            return kwargs.get('validate', lambda x: x)(response)
        except ValueError as error:
            raise ModelOutputError(str(error)) from error


def raw_local(*, ids=('L01',), tracks=('T001',), segments=None):
    entities = [dict(local_id=lid, track_ids=[tid], kind='person', label='사람',
                     appearance='짧은 머리', reason='관측') for lid, tid in zip(ids, tracks)]
    if segments:
        for entity, span in zip(entities, segments):
            entity['segments'] = [dict(track_id=entity['track_ids'][0], first_frame=span[0], last_frame=span[1])]
    return dict(entities=entities, rejected_tracks=[], summary=' '.join(ids)+' 등장한다.',
                events=[], missing_objects=[])


def fixture(folder, *, two_tracks=False):
    folder = Path(folder)
    Image.new('RGB', (120, 80), 'gray').save(folder/'scene.jpg')
    Image.new('RGB', (30, 40), 'red').save(folder/'crop.jpg')
    tracks = []
    for number in range(2 if two_tracks else 1):
        tid = f's1:T{number+1}'
        observations = [dict(frame=i, source_frame=i, time_sec=i/2, original_time_sec=10+i/2,
                            bbox=[10+number*70, 10, 40+number*70, 60], category='person', confidence=.9,
                            crop='crop.jpg', image='scene.jpg') for i in range(4)]
        tracks.append(dict(track_id=tid, observations=observations, representatives=observations))
    shot = dict(shot_id='s1', fps=2, frame_count=4, width=120, height=80,
                full_start_sec=3., original_start_frame=20, source=str(folder/'source.mp4'),
                tracks=tracks, scene_frames=[dict(frame=0, time_sec=0., image='scene.jpg')])
    bundle = dict(asset_root=str(folder), shots=[shot])
    local = dict(shots=[dict(shot_id='s1', entities=[dict(local_id=f's1:L{i+1:02d}', track_ids=[t['track_id']],
                  label='사람', appearance='짧은 머리', kind='person') for i,t in enumerate(tracks)],
                  rejected_tracks=[], summary='s1:L01이 서 있다.', events=[])])
    return bundle, local


class BudgetTests(unittest.TestCase):
    def test_resume_reserves_changed_and_missing_shots_but_not_cached_future_shots(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)
            bundle={'shots':[{'shot_id':f's{i}','tracks':[{'track_id':f's{i}:T1'}]} for i in range(1,4)]}
            self.assertEqual(pending_local_reviews(bundle,output),{'s1','s2','s3'})
            (output/'local_entities.json').write_text(json.dumps({'shots':[{'shot_id':'s1'},{'shot_id':'s2'}]}))
            (output/'reviewed_tracking.json').write_text(json.dumps(bundle))
            changed=deepcopy(bundle)
            changed['shots'][0]['tracks']=[{'track_id':'s1:T1:R001'},{'track_id':'s1:T1:R002'}]
            self.assertEqual(pending_local_reviews(changed,output),{'s1','s3'})

    def test_cache_and_resume_do_not_consume_new_generations(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp); settings = Settings(json_retries=0)
            with patch.object(ReviewBackend, '_generate', return_value='{"value":7}') as generate:
                backend = StoryBackend(settings, output, limit=1)
                self.assertEqual(backend.request('same', Probe).value, 7)
                self.assertEqual(backend.request('same', Probe).value, 7)
                resumed = StoryBackend(settings, output, limit=1)
                self.assertEqual(resumed.request('same', Probe).value, 7)
                self.assertEqual(generate.call_count, 1)
                self.assertEqual(resumed.ledger['generations'], 1)
                with self.assertRaises(BudgetExhausted):
                    resumed.request('different', Probe)
                self.assertEqual(resumed.ledger['generations'], 1)

    def test_runtime_failure_consumes_budget_and_is_durable(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp); settings = Settings(json_retries=0)
            backend = StoryBackend(settings, output, limit=1)
            with patch.object(ReviewBackend, '_generate', side_effect=RuntimeError('synthetic failure')):
                with self.assertRaises(RuntimeError):
                    backend.request('failure', Probe)
            resumed = StoryBackend(settings, output, limit=1)
            self.assertEqual(resumed.ledger['generations'], 1)
            self.assertEqual(resumed.ledger['events'][0]['status'], 'failed')
            with self.assertRaises(BudgetExhausted):
                resumed.request('again', Probe)

    def test_thinking_request_has_bound_decoding_and_final_answer_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            backend=StoryBackend(Settings(model_id='Qwen/Qwen3-VL-8B-Thinking',json_retries=0),Path(temp),limit=1)
            with patch.object(ReviewBackend,'request',return_value=Probe(value=7)) as request:
                backend.request('compare',Probe,max_tokens=800)
            self.assertEqual(request.call_args.kwargs['max_tokens'],8192)
            self.assertEqual(request.call_args.kwargs['generation_options']['seed'],0)
            self.assertTrue(request.call_args.kwargs['generation_options']['do_sample'])
            self.assertIn('8192',request.call_args.args[0])

    def test_bounded_reasoning_also_caps_final_output_without_spending_on_cache_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            backend=StoryBackend(Settings(model_id='Qwen/Qwen3-VL-8B-Thinking',json_retries=0),Path(temp),limit=2)
            backend.identity_thinking_budget=1024
            backend.phase='global_focused:test'
            with patch.object(ReviewBackend,'_generate',return_value='{"value":7}') as generate:
                self.assertEqual(backend.request('bounded',Probe,max_tokens=800).value,7)
                self.assertEqual(generate.call_args.args[3],1888)
                self.assertEqual(backend.ledger['generations'],1)
                self.assertEqual(backend.request('bounded',Probe,max_tokens=800).value,7)
                self.assertEqual(generate.call_count,1)

    def test_invalid_json_retry_counts_both_generations(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = StoryBackend(Settings(json_retries=1), Path(temp), limit=2)
            with patch.object(ReviewBackend, '_generate', side_effect=['{"value":', '{"value":9}']):
                self.assertEqual(backend.request('retry', Probe).value, 9)
            self.assertEqual(backend.ledger['generations'], 2)
            self.assertEqual(len(list((Path(temp)/'review_traces').glob('*.json'))), 2)

    def test_reserved_generations_are_not_consumed(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = StoryBackend(Settings(json_retries=0), Path(temp), limit=2)
            backend.reserve = 2
            with patch.object(ReviewBackend, '_generate') as generate:
                with self.assertRaises(BudgetExhausted):
                    backend.request('reserved', Probe)
                generate.assert_not_called()
            self.assertEqual(backend.ledger['generations'], 0)

    def test_resumed_budget_cannot_silently_expand(self):
        with tempfile.TemporaryDirectory() as temp:
            StoryBackend(Settings(), Path(temp), limit=8)
            with self.assertRaises(ValueError):
                StoryBackend(Settings(), Path(temp), limit=24)


class LocalValidationTests(unittest.TestCase):
    def test_class_and_body_part_role_words_do_not_invent_an_owner(self):
        raw=raw_local()
        raw['entities'][0]['kind']='body_part'
        raw['track_roles']={'T001':'hand','T002':'animal'}
        result=ShotDecision.model_validate(raw)
        self.assertEqual(result.track_roles,{'T001':'body_part','T002':'whole_subject'})
        self.assertEqual(result.entities[0].kind,'unknown')
        self.assertTrue(result.entities[0].uncertain)
        self.assertIn('owner unresolved',result.entities[0].reason)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle, self.local = fixture(self.temp.name)
        self.shot = self.bundle['shots'][0]
        self.aliases = {'s1:T1': 'T001'}

    def check(self, data):
        return validate_local(ShotDecision.model_validate(data), self.shot, self.aliases)

    def test_missing_track_is_rejected(self):
        data = raw_local(); data['entities'] = []
        with self.assertRaises(ValueError):
            self.check(data)

    def test_assigned_and_rejected_track_is_rejected(self):
        data = raw_local(); data['rejected_tracks'] = ['T001']
        with self.assertRaises(ValueError):
            self.check(data)

    def test_split_requires_complete_nonoverlapping_observations(self):
        valid = raw_local(ids=('L01', 'L02'), tracks=('T001', 'T001'), segments=((0, 1), (2, 3)))
        self.check(valid)
        overlapping = deepcopy(valid); overlapping['entities'][1]['segments'][0]['first_frame'] = 1
        with self.assertRaises(ValueError):
            self.check(overlapping)
        omitted = deepcopy(valid); omitted['entities'][1]['segments'][0]['first_frame'] = 3
        with self.assertRaises(ValueError):
            self.check(omitted)

    def test_single_owner_segment_cannot_drop_observed_frames(self):
        with self.assertRaises(ValueError):
            self.check(raw_local(segments=((0, 1),)))

    def test_empty_split_child_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check(raw_local(ids=('L01', 'L02'), tracks=('T001', 'T001'), segments=((0, 3), (5, 6))))

    def test_unknown_segment_track_is_rejected(self):
        data = raw_local(segments=((0, 3),)); data['entities'][0]['segments'][0]['track_id'] = 'unknown'
        with self.assertRaises(ValueError):
            self.check(data)

    def test_mixed_split_and_whole_track_in_one_entity_is_valid(self):
        self.bundle, _ = fixture(self.temp.name, two_tracks=True)
        self.shot = self.bundle['shots'][0]
        self.aliases = {'s1:T1': 'T001', 's1:T2': 'T002'}
        data = raw_local(ids=('L01', 'L02'), tracks=('T001', 'T001'), segments=((0, 1), (2, 3)))
        data['entities'][0]['track_ids'].append('T002')
        self.check(data)

    def test_unknown_event_actor_is_rejected(self):
        data = raw_local(); data['events'] = [dict(subject='L99', action='걷는다', start_sec=0, end_sec=1)]
        with self.assertRaises(ValueError):
            self.check(data)

    def test_event_cannot_extend_beyond_actual_shot(self):
        data = raw_local(); data['events'] = [dict(subject='L01', action='걷는다', start_sec=0, end_sec=2.05)]
        with self.assertRaises(ValueError):
            self.check(data)

    def test_event_evidence_cannot_point_outside_shot(self):
        data = raw_local(); data['events'] = [dict(subject='L01', action='걷는다', evidence_frames=[4])]
        with self.assertRaises(ValueError):
            self.check(data)

    def test_split_materialization_replaces_parent_and_keeps_complete_partition(self):
        backend = FakeBackend([raw_local(ids=('L01', 'L02'), tracks=('T001', 'T001'), segments=((0, 1), (2, 3)))])
        reviewed = review_shots(self.bundle, Path(self.temp.name)/'out', backend)
        result = dict(entities=[dict(entity_id='G1'), dict(entity_id='G2')], mapping={'s1:L01': 'G1', 's1:L02': 'G2'})
        coverage = validate_story_contract(self.bundle, reviewed, result)
        self.assertTrue(coverage['complete_track_review'])
        tracks = self.bundle['shots'][0]['tracks']
        self.assertEqual(len(tracks), 2)
        self.assertEqual(sorted(o['frame'] for track in tracks for o in track['observations']), [0, 1, 2, 3])

    def test_split_without_old_keyframe_materializes_real_crops(self):
        import numpy as np
        from fractions import Fraction
        sys.path.insert(0, str(ROOT.parent/'03_adaptive_sampling'))
        from media import VideoWriter
        writer = VideoWriter(Path(self.temp.name)/'source.mp4', 120, 80, Fraction(2))
        for index in range(4):
            writer.write(index, np.full((80, 120, 3), index*50, np.uint8))
        writer.close()
        track = self.shot['tracks'][0]
        track['representatives'] = track['observations'][:1]
        for observation in track['observations'][1:]:
            observation['crop'] = observation['image'] = None
        backend = FakeBackend([raw_local(ids=('L01', 'L02'), tracks=('T001', 'T001'), segments=((0, 1), (2, 3)))])
        review_shots(self.bundle, Path(self.temp.name)/'out', backend)
        for child in self.shot['tracks']:
            self.assertTrue(child['representatives'])
            for observation in child['representatives']:
                self.assertTrue(observation['crop'])
                self.assertTrue((Path(self.temp.name)/observation['crop']).is_file())

    def test_id_replacement_is_token_based(self):
        self.bundle, _ = fixture(self.temp.name, two_tracks=True)
        data = raw_local(ids=('L1', 'L10'), tracks=('T001', 'T002'))
        reviewed = review_shots(self.bundle, Path(self.temp.name)/'out', FakeBackend([data]))
        self.assertEqual(reviewed['shots'][0]['summary'], 's1:L1 s1:L10 등장한다.')

    def test_relative_and_absolute_asset_paths_work(self):
        self.assertGreater(scene_sheet(self.bundle, self.shot, Path('/not-the-asset-root'), self.aliases).width, 0)
        self.assertGreater(crop_sheet(self.bundle, self.shot['tracks'], Path('/not-the-asset-root'), 'T1').height, 0)
        self.shot['tracks'][0]['representatives'][0]['crop'] = str(Path(self.temp.name)/'crop.jpg')
        self.assertGreater(crop_sheet(self.bundle, self.shot['tracks'], Path('/not-the-asset-root'), 'T1').height, 0)


class AssignmentMapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.bundle, _ = fixture(self.temp.name, two_tracks=True)
        self.shot = self.bundle['shots'][0]
        self.aliases = {'s1:T1': 'T001', 's1:T2': 'T002'}

    @staticmethod
    def entity(lid='L01', **kwargs):
        return dict(local_id=lid, kind='person', label='인물', appearance='보이는 외형', **kwargs)

    def check(self, mapping, entities, **kwargs):
        result = ShotDecision.model_validate(dict(track_assignments=mapping, entities=entities,
            summary='인물이 보인다.', **kwargs))
        return validate_local(result, self.shot, self.aliases)

    def test_map_assigns_tracks_and_explicit_reject(self):
        result = self.check({'T001': 'L01', 'T002': 'reject'}, [self.entity()],
                            rejection_reasons={'T002': '배경의 고정된 형상'})
        self.assertEqual(result.entities[0].track_ids, ['T001'])
        self.assertEqual(result.rejected_tracks, ['T002'])
        self.assertEqual(result.rejection_reasons['T002'], '배경의 고정된 형상')

    def test_all_rejected_map_has_no_synthetic_entities(self):
        result = self.check({'T001': 'reject', 'T002': 'reject'}, [])
        self.assertFalse(result.entities)
        self.assertEqual(set(result.rejected_tracks), {'T001', 'T002'})

    def test_unknown_local_entity_assignment_fails(self):
        with self.assertRaises(ValueError):
            self.check({'T001': 'L99', 'T002': 'reject'}, [self.entity()])

    def test_unknown_track_key_fails(self):
        with self.assertRaises(ValueError):
            self.check({'T001': 'L01', 'T002': 'reject', 'T999': 'L01'}, [self.entity()])

    def test_duplicate_target_within_assignment_list_fails(self):
        with self.assertRaises(ValueError):
            self.check({'T001': ['L01', 'L01'], 'T002': 'reject'}, [self.entity()])

    def test_duplicate_entity_ids_fail(self):
        with self.assertRaises(ValueError):
            self.check({'T001': 'L01', 'T002': 'reject'}, [self.entity(), self.entity()])

    def test_split_list_with_complete_segments_is_valid(self):
        entities = [self.entity('L01', segments=[dict(track_id='T001', first_frame=0, last_frame=1)]),
                    self.entity('L02', segments=[dict(track_id='T001', first_frame=2, last_frame=3)])]
        result = self.check({'T001': ['L01', 'L02'], 'T002': 'reject'}, entities)
        self.assertEqual([e.track_ids for e in result.entities], [['T001'], ['T001']])

    def test_split_list_without_segments_fails(self):
        with self.assertRaises(ValueError):
            self.check({'T001': ['L01', 'L02'], 'T002': 'reject'}, [self.entity('L01'), self.entity('L02')])

    def test_empty_assignment_list_cannot_hide_track(self):
        with self.assertRaises(ValueError):
            self.check({'T001': 'L01', 'T002': []}, [self.entity()])

    def test_missing_map_key_cannot_hide_track(self):
        with self.assertRaises(ValueError):
            self.check({'T001': 'L01'}, [self.entity()])

    def test_empty_map_cannot_hide_unreviewed_tracks(self):
        with self.assertRaises(ValueError):
            self.check({}, [])

    def test_legacy_explicit_partition_remains_supported(self):
        result = self.check({}, [self.entity(track_ids=['T001'])], rejected_tracks=['T002'])
        self.assertEqual(result.entities[0].track_ids, ['T001'])


class GeometryRelationshipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.bundle, self.local = fixture(self.temp.name, two_tracks=True)
        self.shot = self.bundle['shots'][0]
        self.aliases = {'s1:T1': 'T001', 's1:T2': 'T002'}

    def decision(self, *, together, relation=None):
        mapping = {'T001': 'L01', 'T002': 'L01' if together else 'L02'}
        entities = [AssignmentMapTests.entity('L01')]
        if not together:
            entities.append(AssignmentMapTests.entity('L02'))
        return ShotDecision.model_validate(dict(track_assignments=mapping, entities=entities, summary='관측된 장면.',
            relations=[] if relation is None else [dict(left='T001', right='T002', relation=relation, reason='영상 근거')]))

    def test_geometry_maps_ids_preserving_hard_and_conditional_distinction(self):
        self.shot['mask_graph'] = dict(cannot_link=[['s1:T1', 's1:T2'], ['unavailable', 's1:T2']],
                                       conditional_separation=[['s1:T1', 's1:T2']])
        self.shot['tracks'][0]['identity_boundary_candidates'] = [dict(frame=2, reason='appearance change')]
        context = geometry_context(self.shot, self.aliases)
        self.assertEqual(context['cannot_merge'], [['T001', 'T002']])
        self.assertEqual(context['inspect_relationship'], [['T001', 'T002']])
        self.assertEqual(context['identity_change_candidates'][0]['track'], 'T001')
        self.assertEqual(context['identity_change_candidates'][0]['boundaries'][0]['frame'], 2)

    def test_hard_cannot_link_blocks_common_identity(self):
        self.shot['mask_graph'] = dict(cannot_link=[['s1:T1', 's1:T2']])
        with self.assertRaises(ValueError):
            validate_local(self.decision(together=True), self.shot, self.aliases)
        validate_local(self.decision(together=False), self.shot, self.aliases)

    def test_conditional_body_part_separation_can_merge_after_visual_review(self):
        self.shot['mask_graph'] = dict(conditional_separation=[['s1:T1', 's1:T2']])
        validate_local(self.decision(together=True, relation='part_of'), self.shot, self.aliases)

    def test_held_and_carried_require_separate_identities(self):
        for relation in ('held_by', 'carried_by'):
            with self.subTest(relation=relation):
                with self.assertRaises(ValueError):
                    validate_local(self.decision(together=True, relation=relation), self.shot, self.aliases)
                validate_local(self.decision(together=False, relation=relation), self.shot, self.aliases)

    def test_explicit_visual_difference_repairs_shared_assignment(self):
        result=validate_local(self.decision(together=True,relation='different_individual'),self.shot,self.aliases)
        self.assertEqual(len(result.entities),2)
        self.assertNotEqual(result.track_assignments['T001'],result.track_assignments['T002'])
        self.assertEqual(len(validate_local(result,self.shot,self.aliases).entities),2)

    def test_unknown_or_self_relation_is_rejected(self):
        for target in ('unknown', 'T001'):
            result = self.decision(together=True, relation='part_of')
            result.relations[0].right = target
            with self.assertRaises(ValueError):
                validate_local(result, self.shot, self.aliases)

    def test_split_subject_swap_respects_time_specific_identity(self):
        self.shot['mask_graph'] = dict(cannot_link=[['s1:T1', 's1:T2']])
        entities = [AssignmentMapTests.entity('L01', segments=[
                        dict(track_id='T001', first_frame=0, last_frame=1),
                        dict(track_id='T002', first_frame=2, last_frame=3)]),
                    AssignmentMapTests.entity('L02', segments=[
                        dict(track_id='T001', first_frame=2, last_frame=3),
                        dict(track_id='T002', first_frame=0, last_frame=1)])]
        result = ShotDecision.model_validate(dict(track_assignments={'T001': ['L01', 'L02'], 'T002': ['L01', 'L02']},
            entities=entities, summary='서로 다른 두 대상이 교차한다.'))
        validate_local(result, self.shot, self.aliases)

    def test_relationship_ids_materialize_as_original_track_ids(self):
        data = self.decision(together=False, relation='held_by').model_dump()
        result = review_shots(self.bundle, Path(self.temp.name)/'out', FakeBackend([data]))
        relation = result['shots'][0]['relations'][0]
        self.assertEqual((relation['left'], relation['right']), ('s1:T1', 's1:T2'))

    def test_global_does_not_promote_conditional_parts_to_hard_constraint(self):
        self.shot['mask_graph'] = dict(cannot_link=[], conditional_separation=[['s1:T1', 's1:T2']])
        self.assertEqual(cannot_links(self.bundle, self.local), [])

    def test_global_preserves_explicit_carried_relationship(self):
        self.shot['mask_graph'] = dict(cannot_link=[], conditional_separation=[['s1:T1', 's1:T2']])
        for row in self.shot['tracks'][1]['observations']:
            row['bbox'] = [15, 15, 25, 25]
        self.local['shots'][0]['relations'] = [dict(left='s1:T1', right='s1:T2', relation='carried_by')]
        self.assertEqual(cannot_links(self.bundle, self.local), [['s1:L01', 's1:L02']])

    def test_region_graph_veto_does_not_assert_physical_identity(self):
        self.shot['mask_graph'] = dict(cannot_link=[['s1:T1', 's1:T2']])
        for row in self.shot['tracks'][1]['observations']:
            row['bbox'] = [10, 10, 40, 60]
        self.assertEqual(cannot_links(self.bundle, self.local), [])


class GalleryTests(unittest.TestCase):
    def test_gallery_keeps_all_candidates_across_pages(self):
        items = [(f'T{i:03d}', Image.new('RGB', (200, 100), (i, 20, 30))) for i in range(1, 14)]
        pages = gallery_pages(items)
        self.assertEqual(len(pages), 3)
        self.assertEqual([size.size for _, size in pages], [(1360, 1296), (1360, 1296), (680, 432)])
        labels = ', '.join(label for label, _ in pages)
        for label, _ in items:
            self.assertEqual(labels.count(label), 1)
        self.assertEqual(gallery_pages([]), [])

    def test_masked_crop_is_used_and_missing_mask_falls_back_to_original(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, _ = fixture(temp)
            Image.new('RGB', (30, 40), 'blue').save(Path(temp)/'masked.png')
            track = bundle['shots'][0]['tracks'][0]
            track['representatives'][0]['masked_crop'] = 'masked.png'
            sheet = crop_sheet(bundle, [track], Path(temp), 'T001', masked=True)
            self.assertEqual(sheet.getpixel((112, 100)), (0, 0, 255))
            red = sheet.getpixel((336, 100))
            self.assertGreater(red[0], 240)
            self.assertLess(red[2], 10)
            unmasked = crop_sheet(bundle, [track], Path(temp), 'T001', masked=False)
            self.assertGreater(unmasked.getpixel((112, 100))[0], 240)

    def test_local_review_uses_context_plus_complete_compact_gallery(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, _ = fixture(temp, two_tracks=True)
            backend = FakeBackend([raw_local(ids=('L01', 'L02'), tracks=('T001', 'T002'))])
            review_shots(bundle, Path(temp)/'out', backend)
            images = backend.calls[0]['images']
            self.assertEqual(len(images), 2)
            self.assertIn('Full scene', images[0][0])
            self.assertIn('T001', images[1][0])
            self.assertIn('T002', images[1][0])


class CliSmokeTests(unittest.TestCase):
    def test_actual_python_entrypoint_help_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, str(ROOT/'story_pipeline.py'), '--help'],
                cwd=temp, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--qwen-max-calls', result.stdout)
            self.assertIn('--stage', result.stdout)

    def test_run_sh_help_resolves_workspace_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ, ENTITY_PYTHON=sys.executable)
            result = subprocess.run(['bash', str(ROOT/'run.sh'), '--help'], cwd=temp, env=env,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--input', result.stdout)

    def test_cli_default_input_discovers_only_dataset_without_model_calls(self):
        import story_pipeline
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp); dataset = project/'inputs'/'a_generic_video'; dataset.mkdir(parents=True)
            (dataset/'summary.json').write_text('{"videos":[]}', encoding='utf-8')
            output = project/'output'
            bundle = dict(shots=[], stats={'frames': 0})
            with patch.object(story_pipeline, 'ROOT', project), \
                 patch.object(sys, 'argv', ['story_pipeline.py', '--stage', 'tracking', '--output', str(output)]), \
                 patch.object(story_pipeline, 'build_dense_tracks', return_value=bundle) as tracking, \
                 patch.object(story_pipeline, 'StoryBackend') as semantic_backend:
                story_pipeline.main()
            self.assertEqual(tracking.call_args.args[0], dataset.resolve())
            self.assertEqual(tracking.call_args.args[1], output.resolve())
            semantic_backend.assert_not_called()
            self.assertTrue((output/'tracking_summary.json').is_file())

    def test_cli_ambiguous_default_input_requires_explicit_path(self):
        import story_pipeline
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            for name in ('video_one', 'video_two'):
                dataset = project/'inputs'/name; dataset.mkdir(parents=True)
                (dataset/'summary.json').write_text('{"videos":[]}', encoding='utf-8')
            with patch.object(story_pipeline, 'ROOT', project), patch.object(sys, 'argv', ['story_pipeline.py']), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                story_pipeline.main()
            self.assertEqual(stopped.exception.code, 2)


class GlobalValidationTests(unittest.TestCase):
    def test_local_seed_resolution_uses_prebound_pair_views_without_model_id_choices(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle,local=fixture(temp)
            second=deepcopy(bundle['shots'][0]);second['shot_id']='s2';second['full_start_sec']=6.
            second['tracks'][0]['track_id']='s2:T1'
            for row in second['tracks'][0]['observations']+second['tracks'][0]['representatives']:
                row['original_time_sec']+=6
            bundle['shots'].append(second)
            row=deepcopy(local['shots'][0]);row['shot_id']='s2'
            row['entities'][0].update(local_id='s2:L01',track_ids=['s2:T1'])
            local['shots'].append(row)
            response=dict(decision='same',left_view=1,right_view=1,
                left_observation='Visible matching face and distinctive hair',
                right_observation='Visible matching face and distinctive hair',
                basis='distinctive_visual_traits',reason='Matching visible facial structure and hair.')
            backend=FakeBackend([response])
            result=resolve_global(bundle,local,Path(temp),backend,initialization='local',review_max_calls=1)
            self.assertEqual(len(backend.calls),1)
            self.assertEqual(backend.calls[0]['schema'],'PairedViewDecision')
            self.assertNotIn('s1:L01',backend.calls[0]['prompt'])
            self.assertEqual(result['mapping']['s1:L01'],result['mapping']['s2:L01'])

    def test_partition_must_include_every_local_entity_once(self):
        one = dict(local_ids=['a'], kind='animal', label='생물', appearance='날개', reason='관측')
        with self.assertRaises(ValueError):
            validate_global(GlobalDecision(entities=[one]), {'a', 'b'}, [])
        with self.assertRaises(ValueError):
            validate_global(GlobalDecision(entities=[one, one]), {'a'}, [])

    def test_kind_correction_does_not_override_cannot_link(self):
        data = dict(entities=[dict(local_ids=['a', 'b'], kind='animal', label='생물', appearance='날개', reason='외형')])
        validate_global(GlobalDecision.model_validate(data), {'a', 'b'}, [])
        with self.assertRaises(ValueError):
            validate_global(GlobalDecision.model_validate(data), {'a', 'b'}, [['a', 'b']])

    def test_covisible_separate_boxes_require_semantic_identity_review(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, local = fixture(temp, two_tracks=True)
            self.assertEqual(cannot_links(bundle, local), [])
            # Same species/class contributes no cannot-link without co-visibility.
            for row in bundle['shots'][0]['tracks'][1]['observations']:
                row['frame'] += 4
            self.assertEqual(cannot_links(bundle, local), [])

    def test_global_forward_and_reverse_are_fixed_two_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, local = fixture(temp)
            response = dict(entities=[dict(local_ids=['s1:L01'], kind='animal', label='생물', appearance='날개', reason='영상 근거')])
            backend = FakeBackend([response, response])
            output = Path(temp)/'out'; output.mkdir()
            result = resolve_global(bundle, local, output, backend, initialization="whole")
            self.assertEqual(len(backend.calls), 2)
            self.assertEqual(result['entities'][0]['kind'], 'person')
            self.assertEqual(result['mapping'], {'s1:L01': 'G0001'})


class GlobalRejectionTests(unittest.TestCase):
    def test_explicit_rejection_counts_as_partition_coverage(self):
        data = dict(entities=[dict(local_ids=['a'], kind='person', label='사람',
                    appearance='짧은 머리', reason='관측')],
                    rejected_local_ids=['b'], rejection_reasons={'b':'눈 덮인 바위 표면'})
        result = validate_global(GlobalDecision.model_validate(data), {'a', 'b'}, [['a', 'b']])
        self.assertEqual(result.rejected_local_ids, ['b'])

    def test_rejection_requires_nonempty_reason(self):
        for reasons in ({}, {'a':''}, {'a':'   '}):
            with self.subTest(reasons=reasons), self.assertRaises(ValueError):
                validate_global(GlobalDecision(entities=[], rejected_local_ids=['a'],
                    rejection_reasons=reasons), {'a'}, [])

    def test_rejection_cannot_duplicate_assignment_or_include_unknown_id(self):
        one = dict(local_ids=['a'], kind='person', label='사람', appearance='얼굴', reason='관측')
        for entities, rejected in (([one], ['a']), ([], ['a', 'a']), ([], ['a', 'b'])):
            with self.subTest(rejected=rejected), self.assertRaises(ValueError):
                validate_global(GlobalDecision(entities=entities, rejected_local_ids=rejected,
                    rejection_reasons={'a':'바위', 'b':'배경'}), {'a'}, [])

    def test_applied_rejections_preserve_audit_partition_and_remove_related_events(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, local = fixture(temp, two_tracks=True)
            shot = local['shots'][0]
            shot['summary'] = 's1:L01이 s1:L02를 바라본다.'
            shot['events'] = [dict(subject='s1:L01', action='걷는다'),
                              dict(subject='s1:L02', action='움직인다'),
                              dict(subject='s1:L01', object='s1:L02', action='본다')]
            before = deepcopy(local)
            global_result = dict(entities=[dict(entity_id='G0001', local_ids=['s1:L01'], kind='person')],
                mapping={'s1:L01':'G0001'}, rejected_local_ids=['s1:L02'],
                rejection_reasons={'s1:L02':'배경 암벽의 일부'})
            updated = apply_global_rejections(local, global_result, Path(temp))
            row = updated['shots'][0]
            self.assertEqual(local, before)
            self.assertEqual([entity['local_id'] for entity in row['entities']], ['s1:L01'])
            self.assertEqual(row['rejected_tracks'], ['s1:T2'])
            self.assertEqual(row['rejection_reasons']['s1:T2'], '배경 암벽의 일부')
            self.assertEqual(row['global_rejected_entities'], [before['shots'][0]['entities'][1]])
            self.assertEqual(row['events'], [before['shots'][0]['events'][0]])
            self.assertEqual(row['initial_summary_before_global_rejection'], shot['summary'])
            self.assertNotIn('s1:L02', row['summary'])
            self.assertIn('s1:L01', row['summary'])
            self.assertTrue(validate_story_contract(bundle, updated, global_result)['complete_track_review'])
            self.assertEqual(json.loads((Path(temp)/'final_local_entities.json').read_text()), updated)

    def test_no_rejections_preserve_original_local_result(self):
        with tempfile.TemporaryDirectory() as temp:
            _, local = fixture(temp)
            self.assertIs(apply_global_rejections(local, {}, Path(temp)), local)

    def test_global_resolution_can_reject_all_false_candidates_without_empty_id(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, local = fixture(temp)
            response = dict(entities=[], rejected_local_ids=['s1:L01'],
                            rejection_reasons={'s1:L01':'정적인 암벽과 동일한 윤곽'})
            backend = FakeBackend([response, response])
            result = resolve_global(bundle, local, Path(temp), backend, initialization="whole")
            self.assertEqual(result['entities'], [])
            self.assertEqual(result['mapping'], {})
            self.assertEqual(result['rejected_local_ids'], ['s1:L01'])
            updated = apply_global_rejections(local, result, Path(temp))
            self.assertTrue(validate_story_contract(bundle, updated, result)['complete_track_review'])


class NarrativeValidationTests(unittest.TestCase):
    def revise(self, shot_row):
        with tempfile.TemporaryDirectory() as temp:
            bundle, local = fixture(temp)
            global_result = dict(entities=[dict(entity_id='G0001', label='사람', kind='person')], mapping={'s1:L01': 'G0001'})
            backend = FakeBackend([dict(shots=[dict(shot_id='s1', **shot_row)])])
            output = Path(temp)/'out'; output.mkdir()
            return revise_narrative(bundle, local, global_result, output, backend)

    def test_success_converts_local_event_time_to_full_once(self):
        result = self.revise(dict(summary='G0001이 걷는다.', events=[dict(subject='G0001', action='걷는다', start_sec=.5, end_sec=1.5, evidence_frames=[1, 2])]))
        event = result['shots'][0]['events'][0]
        self.assertEqual(event['start_sec'], 3.5)
        self.assertEqual(event['end_sec'], 4.5)
        self.assertEqual(event['time_basis'], 'full')
        self.assertEqual(event['local_start_sec'], .5)
        self.assertFalse(result['narrative_failures'])

    def test_unknown_global_entity_cannot_be_added_in_events(self):
        result = self.revise(dict(summary='서 있다.', events=[dict(subject='G0099', action='걷는다')]))
        self.assertTrue(result['narrative_failures'])
        self.assertFalse(result['shots'][0]['reviewed'])

    def test_unknown_global_entity_cannot_be_added_in_summary(self):
        result = self.revise(dict(summary='G0099가 나타나 G0001을 바라본다.', events=[]))
        self.assertTrue(result['narrative_failures'])
        self.assertFalse(result['shots'][0]['reviewed'])

    def test_narrative_evidence_frames_must_be_in_shot(self):
        result = self.revise(dict(summary='G0001이 걷는다.', events=[dict(subject='G0001', action='걷는다', evidence_frames=[500])]))
        self.assertTrue(result['narrative_failures'])

    def test_narrative_time_cannot_extend_beyond_actual_shot(self):
        result = self.revise(dict(summary='G0001이 걷는다.', events=[dict(subject='G0001', action='걷는다', start_sec=0, end_sec=2.05)]))
        self.assertTrue(result['narrative_failures'])

    def test_final_narrative_receives_source_scene_images(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, local = fixture(temp)
            global_result = dict(entities=[dict(entity_id='G0001', label='사람', kind='person')], mapping={'s1:L01': 'G0001'})
            backend = FakeBackend([dict(shots=[dict(shot_id='s1', summary='G0001이 서 있다.', events=[])])])
            output = Path(temp)/'out'; output.mkdir()
            revise_narrative(bundle, local, global_result, output, backend)
            self.assertTrue(backend.calls[0].get('images'))


class GlobalAssignmentMapTests(unittest.TestCase):
    def group(self, key='K01'):
        return dict(group_id=key, kind='person', label='traveler', appearance='blue scarf', reason='visible face')

    def test_exact_mapping_materializes_each_group_and_explicit_rejection(self):
        response = GlobalDecision.model_validate(dict(local_assignments={'a':'K01','b':'K01','c':'reject'},
            entities=[self.group()], rejection_reasons={'c':'rock surface'}))
        validate_global(response, {'a','b','c'}, [])
        self.assertEqual(response.entities[0].local_ids, ['a','b'])
        self.assertEqual(response.rejected_local_ids, ['c'])
        self.assertEqual(GlobalDecision.model_validate(response.model_dump()), response)

    def test_legacy_partition_roundtrips_new_optional_assignment_fields(self):
        old = GlobalDecision(entities=[dict(local_ids=['a'], kind='person', label='person', appearance='face', reason='visible')])
        self.assertEqual(GlobalDecision.model_validate(old.model_dump()), old)
        rejected = GlobalDecision(entities=[], rejected_local_ids=['a'], rejection_reasons={'a':'background'})
        self.assertEqual(GlobalDecision.model_validate(rejected.model_dump()), rejected)

    def test_reconciled_canonical_partition_roundtrips_inherited_group_metadata(self):
        canonical = GlobalDecision(entities=[dict(self.group(), local_ids=['a']), dict(self.group(), local_ids=['b'])])
        loaded = GlobalDecision.model_validate(canonical.model_dump())
        validate_global(loaded, {'a','b'}, [])
        self.assertEqual([row.local_ids for row in loaded.entities], [['a'],['b']])

    def test_unknown_empty_or_duplicate_group_is_not_silently_repaired(self):
        payloads = [dict(local_assignments={'a':'missing'}, entities=[self.group()]),
                    dict(local_assignments={}, entities=[self.group()]),
                    dict(local_assignments={'a':'K01'}, entities=[self.group(),self.group()]),
                    dict(local_assignments={'a':'K01'}, entities=[dict(self.group(), local_ids=['b'])]),
                    dict(local_assignments={'a':'K01'}, entities=[self.group()], rejected_local_ids=['a'])]
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                GlobalDecision.model_validate(payload)

    def test_missing_input_id_and_invented_kind_remain_invalid(self):
        response = GlobalDecision.model_validate(dict(local_assignments={'a':'K01'}, entities=[self.group()]))
        with self.assertRaises(ValueError):validate_global(response, {'a','b'}, [])
        with self.assertRaises(ValueError):
            GlobalDecision.model_validate(dict(local_assignments={'a':'K01'}, entities=[dict(self.group(),kind='environment')]))


class EmptyShotRegressionTests(unittest.TestCase):
    def test_scene_only_review_has_no_synthetic_ids_and_retains_missed_subject_warning(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, _ = fixture(temp)
            bundle['shots'][0]['tracks'] = []
            backend = FakeBackend([dict(summary='먼 배경에 작은 사람이 보인다.', missing_objects=['멀리 있는 작은 사람'],
                # A model may still emit extra fields. They cannot become detector
                # observations, fake rejected aliases, IDs, or unsupported events.
                rejected_tracks=['T999'], entities=[dict(local_id='L99', track_ids=['T999'])],
                events=[dict(subject='L99', action='walks')])])
            output = Path(temp)/'out'
            output.mkdir()
            result = review_shots(bundle, output, backend)
            self.assertFalse(result['failures'])
            self.assertEqual(backend.calls[0]['schema'], 'SceneOnlyDecision')
            self.assertTrue(backend.calls[0]['images'])
            self.assertEqual(len(result['shots']), 1)
            row = result['shots'][0]
            self.assertEqual(row['missing_objects'], ['멀리 있는 작은 사람'])
            for field in ('entities', 'rejected_tracks', 'events', 'relations'):
                self.assertEqual(row[field], [])
            self.assertEqual(row['summary'], '먼 배경에 작은 사람이 보인다.')


class RawGeometryIdentityTests(unittest.TestCase):
    def test_disjoint_body_part_is_not_a_different_individual(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, local = fixture(temp, two_tracks=True)
            local['shots'][0]['track_roles'] = {'s1:T1': 'whole_subject', 's1:T2': 'body_part'}
            self.assertEqual(cannot_links(bundle, local), [])
            local['shots'][0]['track_roles']['s1:T2'] = 'whole_subject'
            self.assertEqual(cannot_links(bundle, local), [])

    def test_incomplete_roles_do_not_turn_unknown_region_into_hard_constraint(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, local = fixture(temp, two_tracks=True)
            local['shots'][0]['track_roles'] = {'s1:T1': 'whole_subject'}
            self.assertEqual(cannot_links(bundle, local), [])

    def test_explicit_held_carried_or_distinct_relation_survives_without_masks(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, local = fixture(temp, two_tracks=True)
            local['shots'][0]['track_roles'] = {'s1:T1': 'whole_subject', 's1:T2': 'body_part'}
            for relation in ('held_by', 'carried_by', 'different_individual'):
                with self.subTest(relation=relation):
                    local['shots'][0]['relations'] = [dict(left='s1:T1', right='s1:T2', relation=relation)]
                    self.assertEqual(cannot_links(bundle, local), [['s1:L01', 's1:L02']])

    def test_single_frame_geometry_does_not_create_permanent_identity_veto(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, local = fixture(temp, two_tracks=True)
            for track in bundle['shots'][0]['tracks']:
                track['observations'] = track['observations'][:1]
            self.assertEqual(cannot_links(bundle, local), [])


if __name__ == '__main__':
    unittest.main()
