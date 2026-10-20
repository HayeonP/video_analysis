"""Generic evidence/graph regressions for the active global identity pipeline."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent/'01_low_freq_event')]

from PIL import Image
from low_freq_event.config import Settings
from low_freq_event.backend import ModelOutputError
from identity_review import ReviewBackend
from story_semantics import GlobalDecision, GlobalPartition, StoryBackend, resolve_global
from global_identity import (FocusedDecision, FlatPairDecision, PairedViewDecision, paired_views, apply_review, candidate_pairs,
                             focused_images, normalize_review, reconcile_global)


def cluster(lid, source, **values):
    return dict(local_ids=[lid], source_clusters=[source], kind='person',
                label='person', appearance='visible person', reason='initial visual review', **values)


def evidence(a='a', b='b', frame=0, contradiction=False):
    row = dict(left=dict(local_id=a, frame=frame, observation='visible face'),
               right=dict(local_id=b, frame=frame, observation='visible face'),
               explanation='observed facial shape comparison')
    row['trait' if contradiction else 'basis'] = 'incompatible_face' if contradiction else 'distinctive_visual_traits'
    return row


def verdict(left='A', right='B', a='a', b='b', decision='same'):
    return dict(left=left, right=right, decision=decision, reason='frame comparison',
                evidence=[evidence(a, b)] if decision == 'same' else [],
                contradictions=[evidence(a, b, contradiction=True)] if decision == 'different' else [])


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.clusters = {'A': cluster('a', 'A'), 'B': cluster('b', 'B')}
        self.pairs = [dict(left='A', right='B')]
        self.shown = {'a': [0], 'b': [0]}

    def normalize(self, row):
        return normalize_review(FocusedDecision(decisions=[row]), self.pairs, self.clusters, self.shown).decisions[0]

    def test_positive_evidence_must_name_actual_shown_frames_of_both_targets(self):
        good = verdict()
        self.assertEqual(self.normalize(good).decision, 'same')
        for field, value in [('frame', 99), ('local_id', 'unshown_person'), ('observation', '  ')]:
            bad = deepcopy(good)
            bad['evidence'][0]['right'][field] = value
            with self.subTest(field=field):
                self.assertEqual(self.normalize(bad).decision, 'uncertain')

    def test_flat_fallback_preserves_typed_evidence_and_checks_provenance(self):
        values=dict(left_local_id='a',left_frame=0,left_observation='face A',
                    right_local_id='b',right_frame=0,right_observation='face B',reason='visible facial comparison')
        for decision,basis in [('same','distinctive_visual_traits'),('different','incompatible_face')]:
            flat=FlatPairDecision(decision=decision,basis=basis,**values)
            parsed=normalize_review(flat.focused(self.pairs[0]),self.pairs,self.clusters,self.shown)
            self.assertEqual(parsed.decisions[0].decision,decision)
            restored=FlatPairDecision.model_validate(flat.model_dump())
            self.assertEqual(restored.focused(self.pairs[0]),flat.focused(self.pairs[0]))
            restored.right_frame=99
            parsed=normalize_review(restored.focused(self.pairs[0]),self.pairs,self.clusters,self.shown)
            self.assertEqual(parsed.decisions[0].decision,'uncertain')
        contradictory=FlatPairDecision(decision='same',basis='incompatible_face',**values)
        parsed=normalize_review(contradictory.focused(self.pairs[0]),self.pairs,self.clusters,self.shown)
        self.assertEqual(parsed.decisions[0].decision,'uncertain')

    def test_unrelated_hand_cannot_bridge_whole_people_from_cached_same_verdicts(self):
        clusters={'A':cluster('hand','A'),'B':cluster('traveler','B'),'C':cluster('stranger','C')}
        pairs=[dict(left='A',right='B'),dict(left='A',right='C')]
        roles={key:dict(shot_id=key,part_or_worn=key=='hand') for key in ('hand','traveler','stranger')}
        response=FocusedDecision(decisions=[verdict('A','B','hand','traveler'),verdict('A','C','hand','stranger')])
        # Revalidating a deserialized cached SAME must enforce the new gate, too.
        response=FocusedDecision.model_validate(response.model_dump())
        result=normalize_review(response,pairs,clusters,{key:[0] for key in roles},identity_roles=roles)
        self.assertEqual([row.decision for row in result.decisions],['uncertain','uncertain'])
        revised,forbidden,joins,_=apply_review(clusters,result,[])
        self.assertEqual(len(revised),3)
        self.assertFalse(forbidden)
        self.assertFalse(joins)

    def test_body_part_needs_explicit_same_frame_connection_in_an_actual_scene(self):
        roles={'a':dict(shot_id='scene',part_or_worn=True),'b':dict(shot_id='scene',part_or_worn=False)}
        row=verdict()
        for basis,scenes,expected in [('distinctive_visual_traits',{'a':[0]},'uncertain'),
                                      ('visible_physical_connection',{},'uncertain'),
                                      ('visible_physical_connection',{'a':[0]},'same')]:
            row['evidence'][0]['basis']=basis
            parsed=normalize_review(FocusedDecision(decisions=[row]),self.pairs,self.clusters,self.shown,
                                    identity_roles=roles,scene_frames=scenes)
            self.assertEqual(parsed.decisions[0].decision,expected)
        roles['b']['shot_id']='another-scene'
        parsed=normalize_review(FocusedDecision(decisions=[row]),self.pairs,self.clusters,self.shown,
                                identity_roles=roles,scene_frames={'a':[0]})
        self.assertEqual(parsed.decisions[0].decision,'uncertain')
        parsed=normalize_review(FocusedDecision(decisions=[verdict(decision='different')]),
            self.pairs,self.clusters,self.shown,identity_roles=roles,scene_frames={'a':[0]})
        self.assertEqual(parsed.decisions[0].decision,'uncertain')

    def test_synopsis_or_label_without_visual_evidence_cannot_merge(self):
        row = verdict()
        row.update(evidence=[], reason='The synopsis says the same protagonist travels here.')
        self.assertEqual(self.normalize(row).decision, 'uncertain')

    def test_anatomical_connection_cannot_reach_outside_observed_owner_extent(self):
        row=verdict();row['evidence'][0]['basis']='visible_physical_connection'
        roles={'a':dict(shot_id='scene',part_or_worn=True,body_part=True,kind='person',boxes_by_frame={0:[[0,0,10,10]]}),
               'b':dict(shot_id='scene',part_or_worn=False,kind='person',boxes_by_frame={0:[[50,0,100,100]]})}
        for box,expected in [([50,0,100,100],'uncertain'),([0,0,100,100],'same')]:
            roles['b']['boxes_by_frame']={0:[box]}
            result=normalize_review(FocusedDecision(decisions=[row]),self.pairs,self.clusters,self.shown,
                identity_roles=roles,scene_frames={'a':[0],'b':[0]})
            self.assertEqual(result.decisions[0].decision,expected)

    def test_anatomical_owner_kind_must_agree_but_worn_objects_can_attach(self):
        row=verdict();row['evidence'][0]['basis']='visible_physical_connection'
        for kind,role,owner,expected in [('person','body_part','animal','uncertain'),
                ('animal','body_part','person','uncertain'),('unknown','body_part','person','uncertain'),
                ('person','body_part','person','same'),('object','worn_item','person','same')]:
            roles={'a':dict(shot_id='scene',part_or_worn=True,kind=kind,body_part=role=='body_part'),
                   'b':dict(shot_id='scene',part_or_worn=False,kind=owner)}
            with self.subTest(kind=kind,role=role,owner=owner):
                parsed=normalize_review(FocusedDecision.model_validate(dict(decisions=[row])),
                    self.pairs,self.clusters,self.shown,identity_roles=roles,scene_frames={'a':[0]})
                self.assertEqual(parsed.decisions[0].decision,expected)

    def test_missing_or_conflicting_negative_evidence_is_uncertain(self):
        row = verdict(decision='different')
        self.assertEqual(self.normalize(row).decision, 'different')
        row['contradictions'][0]['right']['frame'] = 9
        self.assertEqual(self.normalize(row).decision, 'uncertain')

    def test_absent_details_or_context_differences_cannot_make_a_hard_negative(self):
        for observation,reason in [
                ('face is hidden','incompatible facial geometry'),
                ('a distant silhouette','the body is different'),
                ('얼굴이 가려진 인물','얼굴 구조가 다름'),
                ('visible target','두 관측은 배경과 행동이 다르고 얼굴이 확인되지 않음'),
                ('visible target','different clothing and lighting'),
                ('visible target','Different hair color and mask presence indicate distinct physical characteristics'),
                ('visible target','머리색과 마스크 유무가 다름')]:
            row=verdict(decision='different')
            row['contradictions'][0]['left']['observation']=observation
            row['reason']=reason
            with self.subTest(observation=observation,reason=reason):
                self.assertEqual(self.normalize(row).decision,'uncertain')
        row = verdict(decision='different')
        row['evidence'] = [evidence()]
        self.assertEqual(self.normalize(row).decision, 'uncertain')

    def test_same_with_contradiction_is_not_accepted(self):
        row = verdict()
        row['contradictions'] = [evidence(contradiction=True)]
        self.assertEqual(self.normalize(row).decision, 'uncertain')

    def test_invalid_counterevidence_cannot_be_erased_to_strengthen_a_decision(self):
        row = verdict()
        row['contradictions'] = [evidence(frame=99, contradiction=True)]
        self.assertEqual(self.normalize(row).decision, 'uncertain')
        row = verdict(decision='different')
        row['evidence'] = [evidence(frame=99)]
        self.assertEqual(self.normalize(row).decision, 'uncertain')

    def test_invalid_internal_conflict_rejects_entire_response(self):
        for cid, item in [('not-a-cluster', evidence('a', 'b', contradiction=True)),
                          ('A', evidence('a', 'b', frame=99, contradiction=True))]:
            response = FocusedDecision(decisions=[verdict()], internal_conflicts=[dict(cluster_id=cid, evidence=item)])
            with self.subTest(cid=cid), self.assertRaises(ValueError):
                normalize_review(response, self.pairs, self.clusters, self.shown)

    def test_duplicate_missing_and_invented_pair_outputs_are_rejected(self):
        for decisions in ([], [verdict(), verdict()], [verdict(right='X')]):
            with self.subTest(decisions=decisions), self.assertRaises(ValueError):
                normalize_review(FocusedDecision(decisions=decisions), self.pairs, self.clusters, self.shown)

    def test_reversed_pair_is_supported_with_correct_reference_sides(self):
        row = verdict(left='B', right='A', a='b', b='a')
        self.assertEqual(self.normalize(row).decision, 'same')

    def test_internal_consistency_needs_grounded_connections_covering_every_member(self):
        clusters = {'A': dict(self.clusters['A'], local_ids=['a', 'b', 'c'])}
        shown = {key: [0] for key in 'abc'}
        response = FocusedDecision(decisions=[], cluster_checks=[dict(cluster_id='A',
            decision='consistent', evidence=[evidence('a', 'b')])])
        checked = normalize_review(response, [], clusters, shown, {'A': ['a', 'b', 'c']})
        self.assertEqual(checked.cluster_checks[0].decision, 'uncertain')
        response.cluster_checks[0].evidence.append(type(response.cluster_checks[0].evidence[0]).model_validate(evidence('a', 'c')))
        checked = normalize_review(response, [], clusters, shown, {'A': ['a', 'b', 'c']})
        self.assertEqual(checked.cluster_checks[0].decision, 'consistent')


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.clusters = {key: cluster(key.lower(), key) for key in 'ABC'}

    def test_transitive_bridge_cannot_join_distinct_subjects(self):
        response = FocusedDecision(decisions=[verdict(), verdict('B', 'C', 'b', 'c')])
        result, _, joins, unresolved = apply_review(self.clusters, response, [('a', 'c')])
        self.assertEqual(len(joins), 1)
        self.assertEqual(len(result), 2)
        self.assertEqual(unresolved[0]['blocking_pair'], ['a', 'c'])
        self.assertEqual(self.clusters['A']['local_ids'], ['a'])

    def test_negatives_apply_before_same_decisions_regardless_of_output_order(self):
        response = FocusedDecision(decisions=[verdict(), verdict('B', 'C', 'b', 'c'),
                                               verdict('A', 'C', 'a', 'c', 'different')])
        response.decisions[-1].contradictions[0].trait='distinct_subjects'
        entries={key:dict(shot_id='same-scene') for key in 'abc'}
        result, forbidden, joins, unresolved = apply_review(self.clusters, response, [],entries)
        self.assertIn(('a', 'c'), forbidden)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(joins), 1)
        self.assertEqual(unresolved[0]['kind'], 'blocked_merge')

    def test_internal_cluster_conflict_splits_all_local_members_and_stops_propagation(self):
        self.clusters['A']['local_ids'].append('d')
        response = FocusedDecision(decisions=[verdict()], internal_conflicts=[dict(cluster_id='A',
            evidence=evidence('a', 'd', contradiction=True))])
        response.internal_conflicts[0].evidence.trait='distinct_subjects'
        entries={key:dict(shot_id='same-scene',kind='person',label='person',uncertain=True) for key in 'abcd'}
        result, forbidden, joins, unresolved = apply_review(self.clusters, response, [],entries)
        self.assertFalse(joins)
        self.assertNotIn('A', result)
        self.assertEqual(result['A.001']['local_ids'], ['a'])
        self.assertEqual(result['A.002']['local_ids'], ['d'])
        self.assertTrue(result['A.001']['uncertain'])  # Direct call has no independent local metadata.
        self.assertEqual(sorted(lid for value in result.values() for lid in value['local_ids']), ['a', 'b', 'c', 'd'])
        self.assertEqual({row['kind'] for row in unresolved}, {'blocked_merge', 'internal_conflict_split'})
        self.assertIn(('a', 'd'), forbidden)
        # A later high-similarity third subject must not reunite the disproved pair.
        bridge = FocusedDecision(decisions=[verdict('A.001', 'B', 'a', 'b'), verdict('B', 'A.002', 'b', 'd')])
        merged, _, _, remaining = apply_review(result, bridge, forbidden)
        self.assertTrue(all(not {'a', 'd'} <= set(row['local_ids']) for row in merged.values()))
        self.assertEqual(remaining[0]['kind'], 'blocked_merge')

        entries = {key: dict(label='person', appearance='visible face') for key in 'abcd'}
        candidates = candidate_pairs(result, entries, {}, forbidden, reviewed=[('A', 'B')])
        pairs = {tuple(sorted((row['left'], row['right']))) for row in candidates}
        self.assertIn(('A.001', 'B'), pairs)
        self.assertNotIn(('A.001', 'A.002'), pairs)

    def test_cross_shot_negative_is_soft_and_cannot_be_silently_overridden_by_a_bridge(self):
        entries={key:dict(shot_id='scene-'+key) for key in 'abc'}
        soft={}
        response=FocusedDecision(decisions=[verdict('A','C','a','c','different'),
                                            verdict(),verdict('B','C','b','c')])
        result,forbidden,joins,unresolved=apply_review(self.clusters,response,[],entries,soft_objections=soft)
        self.assertFalse(forbidden)
        self.assertEqual(len(result),2)
        self.assertEqual(len(joins),1)
        self.assertEqual(unresolved[0]['kind'],'soft_objection_requires_review')
        self.assertEqual(soft[('a','c')]['status'],'active')
        # A separate focused request must directly compare the disputed members.
        direct=FocusedDecision(decisions=[verdict('A','C','a','c')])
        result,forbidden,joins,_=apply_review(result,direct,[],entries,soft_objections=soft)
        self.assertEqual(len(result),1)
        self.assertFalse(forbidden)
        self.assertEqual(soft[('a','c')]['status'],'superseded_by_direct_visual_review')

    def test_missing_embeddings_and_changed_descriptions_still_propose_same_kind_review(self):
        entries = {'a': dict(label='silhouette', appearance='dark distant figure'),
                   'b': dict(label='traveler', appearance='bright close face')}
        rows = candidate_pairs({k: self.clusters[k] for k in 'AB'}, entries, {}, [])
        self.assertEqual([(row['left'], row['right']) for row in rows], [('A', 'B')])
        self.assertEqual(rows[0]['visual_similarity'], 0.)
        self.assertEqual(candidate_pairs({k: self.clusters[k] for k in 'AB'}, entries, {}, [('a', 'b')]), [])

    def test_one_uncertain_local_pair_does_not_exhaust_other_members_of_its_clusters(self):
        groups={'A':dict(cluster('a','A'),local_ids=['a','c']),'B':cluster('b','B')}
        entries={key:dict(label='person',appearance='face') for key in 'abc'}
        retrieval={'a':[dict(local_id='b',multiview_similarity=.9)],
                   'c':[dict(local_id='b',multiview_similarity=.8)]}
        pending=candidate_pairs(groups,entries,retrieval,[],reviewed_locals=[('a','b')])
        self.assertEqual(len(pending),1)
        self.assertEqual(pending[0]['visual_similarity'],.8)
        self.assertFalse(candidate_pairs(groups,entries,retrieval,[],reviewed_locals=[('a','b'),('b','c')]))

    def test_impossible_anatomical_owner_is_not_spent_a_generation(self):
        groups={k:self.clusters[k] for k in 'AB'}
        entries={'a':dict(shot_id='scene',kind='person',_part_or_worn=True,_body_part=True),
                 'b':dict(shot_id='scene',kind='animal')}
        self.assertFalse(candidate_pairs(groups,entries,{},[]))
        entries['b']['kind']='person'
        self.assertTrue(candidate_pairs(groups,entries,{},[]))
        entries['a'].update(kind='object',_body_part=False)
        entries['b']['kind']='animal'
        self.assertTrue(candidate_pairs(groups,entries,{},[]))
        entries['a']['_observed_frames']=[0,1]
        entries['b']['_observed_frames']=[2,3]
        self.assertFalse(candidate_pairs(groups,entries,{},[]))
        entries['b']['_observed_frames']=[1,2]
        self.assertTrue(candidate_pairs(groups,entries,{},[]))


def synthetic_video(folder):
    folder = Path(folder)
    Image.new('RGB', (160, 100), '#565966').save(folder/'scene.jpg')
    Image.new('RGB', (40, 60), '#76918e').save(folder/'crop.jpg')
    shots, local_shots, provisional = [], [], []
    # Intentionally unrelated to the Sintel fixture IDs and original shot count.
    for index, sid in enumerate(('arrival', 'return')):
        lid, tid = sid+':traveler', sid+':track'
        observations = [dict(frame=f, image='scene.jpg', crop='crop.jpg', bbox=[10, 10, 50, 70],
                              time_sec=f/2, original_time_sec=index*3+f/2) for f in range(3)]
        shots.append(dict(shot_id=sid, full_start_sec=index*3, frame_count=3, fps=2,
                          tracks=[dict(track_id=tid, observations=observations, representatives=observations)],
                          scene_frames=[dict(frame=0, image='scene.jpg')]))
        local_shots.append(dict(shot_id=sid, entities=[dict(local_id=lid, track_ids=[tid], kind='person',
            label='traveler' if index else 'silhouette', appearance='blue scarf', reason='visible')],
            summary='A traveler is present.', events=[], relations=[]))
        provisional.append(dict(local_ids=[lid], kind='person', label='traveler', appearance='blue scarf', reason='visible'))
    return dict(asset_root=str(folder), shots=shots), dict(shots=local_shots), dict(entities=provisional)


class Backend:
    def __init__(self, responses):
        self.responses, self.calls, self.reserve, self.phase = list(responses), [], 0, ''

    def request(self, prompt, schema, **kwargs):
        self.calls.append(dict(prompt=prompt, schema=schema.__name__, **kwargs))
        payload = self.responses.pop(0)
        if isinstance(payload,Exception):raise payload
        if schema is GlobalPartition and 'entities' in payload:
            payload = dict(groups=[row['local_ids'] for row in payload['entities']],
                           rejections=payload.get('rejection_reasons', {}))
        if schema is PairedViewDecision and 'decisions' in payload:
            row=payload['decisions'][0]
            payload=dict(decision=row['decision'],left_view=1,right_view=1,left_observation='visible face',
                         right_observation='visible face',basis='distinctive_visual_traits' if row['decision']=='same' else 'insufficient',
                         reason=row.get('reason','matching facial geometry'))
        response = schema.model_validate(payload)
        return kwargs.get('validate', lambda value: value)(response)


class ActivePipelineTests(unittest.TestCase):
    def test_zero_new_generation_allowance_replays_cached_visual_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);bundle,local,_=synthetic_video(output)
            payload=dict(decision='same',left_view=1,right_view=1,left_observation='face',right_observation='face',
                         basis='distinctive_visual_traits',reason='matching facial geometry')
            with patch.object(ReviewBackend,'_generate',return_value=json.dumps(payload)) as generate:
                first=StoryBackend(Settings(json_retries=0),output,limit=3)
                resolve_global(bundle,local,output,first,review_max_calls=1)
                resumed=StoryBackend(Settings(json_retries=0),output,limit=3)
                result=resolve_global(bundle,local,output,resumed,review_max_calls=0)
                self.assertEqual(generate.call_count,1)
                self.assertEqual(len(result['entities']),1)
                self.assertEqual(result['identity_reconciliation']['generations'],0)
                self.assertEqual(result['identity_reconciliation']['completed_requests'],1)

    def test_covisible_whole_bodies_require_a_source_frame_verdict_not_embedding_merge(self):
        for verdict in ('uncertain','different'):
            with self.subTest(verdict=verdict),tempfile.TemporaryDirectory() as directory:
                output=Path(directory);bundle,local,_=synthetic_video(output)
                first,second=bundle['shots'];first['tracks']+=second['tracks'];bundle['shots']=[first]
                local['shots'][0]['entities']+=local['shots'][1]['entities'];local['shots']=local['shots'][:1]
                local['shots'][0]['track_roles']={t['track_id']:'whole_subject' for t in first['tracks']}
                for o in first['tracks'][1]['observations']:o['bbox']=[75,10,115,70]
                payload=dict(decision=verdict,left_view=1,right_view=1,visible_bodies=2,
                    basis='distinct_subjects' if verdict=='different' else 'insufficient',
                    left_observation='one body visible',right_observation='a second body visible',
                    reason='Two separate bodies visible together despite similar color and clothing')
                backend=Backend([payload]);result=resolve_global(bundle,local,output,backend,review_max_calls=5)
                self.assertEqual(len(result['entities']),2)
                self.assertEqual(len(backend.calls),1)
                inventory=json.loads((output/'identity_reconciliation/001/input.json').read_text())
                views=inventory['paired_views']
                self.assertEqual(len(views['left']),1);self.assertEqual(len(views['right']),1)
                self.assertEqual(views['left'][0]['frame'],views['right'][0]['frame'])
                self.assertEqual(bool(result['cannot_link']),verdict=='different')
                if verdict=='different':
                    wrong=PairedViewDecision(**dict(payload,visible_bodies=1))
                    with self.assertRaises(ValueError):wrong.focused(inventory['pairs'][0],views)

    def test_auxiliary_view_citation_does_not_repeat_the_ignored_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);bundle,local,_=synthetic_video(output)
            shot,review=deepcopy(bundle['shots'][1]),deepcopy(local['shots'][1])
            shot.update(shot_id='departure',full_start_sec=6);shot['tracks'][0]['track_id']='departure:track'
            review['shot_id']='departure';review['entities'][0].update(local_id='departure:traveler',track_ids=['departure:track'])
            bundle['shots'].append(shot);local['shots'].append(review)
            class AuxiliaryBackend(Backend):
                def request(self,prompt,schema,**kwargs):
                    if len(self.calls)==1:
                        views=json.loads((output/'identity_reconciliation/002/input.json').read_text())['paired_views']
                        for side in ('left','right'):
                            self.responses[0][side+'_view']=2 if len({r['local_id'] for r in views[side]})>1 else 1
                    return super().request(prompt,schema,**kwargs)
            backend=AuxiliaryBackend([
                dict(decision='same',left_view=1,right_view=1,left_observation='face',right_observation='face',
                     basis='distinctive_visual_traits',reason='matching facial geometry'),
                dict(decision='uncertain',left_view=2,right_view=1,basis='insufficient')])
            result=resolve_global(bundle,local,output,backend,review_max_calls=5)
            self.assertEqual(len(backend.calls),2)
            self.assertEqual(len(result['entities']),2)
            audit=result['identity_reconciliation']
            self.assertTrue(audit['deferred_local_pairs'])
            self.assertTrue(audit['unreviewed_candidates'])
            self.assertTrue(set(map(tuple,audit['deferred_local_pairs'])).isdisjoint(
                            set(map(tuple,audit['reviewed_local_pairs']))))

    def test_high_face_similarity_only_nominates_a_review_and_cannot_join_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);bundle,local,_=synthetic_video(output)
            scores={'arrival:traveler':[dict(local_id='return:traveler',face_similarity=.99)]}
            with patch('view_quality.face_retrieval',return_value=scores):
                result=resolve_global(bundle,local,output,Backend([dict(decision='uncertain')]),review_max_calls=1)
            self.assertEqual(len(result['entities']),2)
            self.assertEqual(json.loads((output/'face_retrieval.json').read_text())['candidates'],scores)

    def test_singleton_board_has_fixed_sources_and_no_prior_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);bundle,local,raw=synthetic_video(output)
            from story_semantics import local_lookup
            clusters={f'C{i+1}':dict(group,source_clusters=[f'C{i+1}']) for i,group in enumerate(raw['entities'])}
            pair=dict(left='C1',right='C2')
            images,shown,selected,scenes,views=paired_views(bundle,local_lookup(local),clusters,pair,{},output)
            self.assertEqual(len(images),1)
            self.assertEqual(selected,{'C1':['arrival:traveler'],'C2':['return:traveler']})
            self.assertEqual({r['local_id'] for r in views['left']},{'arrival:traveler'})
            self.assertEqual({r['local_id'] for r in views['right']},{'return:traveler'})
            self.assertEqual(shown,scenes)
            backend=Backend([dict(decision='uncertain')])
            resolve_global(bundle,local,output,backend,review_max_calls=1)
            payload=json.loads(backend.calls[0]['prompt'].split('\nEVIDENCE: ',1)[1])
            self.assertEqual(set(payload),{'views','prior_distinction_needs_fresh_review','video_context'})
            self.assertEqual(payload['video_context'],{})
            self.assertNotIn('blue scarf',backend.calls[0]['prompt'])

    def test_cross_shot_prototypes_keep_each_view_bound_to_its_actual_source(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);bundle,local,raw=synthetic_video(output)
            shot,review=deepcopy(bundle['shots'][1]),deepcopy(local['shots'][1])
            shot.update(shot_id='departure',full_start_sec=6);shot['tracks'][0]['track_id']='departure:track'
            review['shot_id']='departure';review['entities'][0].update(local_id='departure:traveler',track_ids=['departure:track'])
            bundle['shots'].append(shot);local['shots'].append(review)
            from story_semantics import local_lookup
            groups={'A':dict(raw['entities'][0],local_ids=['arrival:traveler','departure:traveler']),
                    'B':raw['entities'][1]}
            _,shown,selected,_,views=paired_views(bundle,local_lookup(local),groups,
                dict(left='A',right='B'),{},output)
            self.assertEqual(set(selected['A']),{'arrival:traveler','departure:traveler'})
            self.assertLessEqual(len(views['left']),3)
            for i,ref in enumerate(views['left'],1):
                decision=PairedViewDecision(decision='same',left_view=i,right_view=1,
                    left_observation='face',right_observation='face',basis='distinctive_visual_traits',reason='facial geometry')
                evidence=decision.focused(dict(left='A',right='B'),views).decisions[0].evidence[0]
                self.assertEqual(evidence.left.local_id,ref['local_id'])
                self.assertIn(evidence.left.frame,shown[ref['local_id']])
                self.assertEqual(evidence.right.local_id,'return:traveler')

    def test_paired_ordinals_cannot_cite_the_other_side_or_unshown_frames(self):
        refs={'left':[dict(local_id='left-only',frame=12)],'right':[dict(local_id='right-only',frame=98)]}
        row=PairedViewDecision(decision='same',left_view=1,right_view=1,left_observation='visible face',
             right_observation='visible face',basis='distinctive_visual_traits',reason='matching facial structure')
        parsed=row.focused(dict(left='A',right='B'),refs).decisions[0]
        self.assertEqual(parsed.evidence[0].left.local_id,'left-only')
        self.assertEqual(parsed.evidence[0].right.local_id,'right-only')
        self.assertEqual(parsed.evidence[0].right.frame,98)
        row.right_view=2
        with self.assertRaises(ValueError):row.focused(dict(left='A',right='B'),refs)
        row.right_view=None
        with self.assertRaises(ValueError):row.focused(dict(left='A',right='B'),refs)
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);bundle,local,_=synthetic_video(output)
            backend=Backend([dict(decision='uncertain',basis='insufficient')])
            resolve_global(bundle,local,output,backend,review_max_calls=5)
            self.assertEqual(len(backend.calls),1)

    def test_boundary_views_use_actual_frames_and_only_model_verdict_can_join(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);bundle,local,_=synthetic_video(output)
            edge=dict(left_local_id='arrival:traveler',right_local_id='return:traveler',
                      left_frame=2,right_frame=0,left_time_sec=1,right_time_sec=3)
            backend=Backend([dict(decision='uncertain')])
            with patch('boundary_identity.boundary_candidates',return_value=[edge]):
                result=resolve_global(bundle,local,output,backend,review_max_calls=1)
            self.assertEqual(len(result['entities']),2)
            self.assertEqual(backend.calls[0]['schema'],'FlatPairDecision')
            data=json.loads(backend.calls[0]['prompt'].split('\nEVIDENCE: ',1)[1])
            self.assertEqual(data['pairs'][0]['boundary_continuity'],[edge])
            self.assertIn(2,data['shown_frames']['arrival:traveler'])
            self.assertTrue(any('Adjacent-shot boundary' in name for name,_ in backend.calls[0]['images']))

    def test_new_audit_archives_all_previous_batch_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);bundle,local,_=synthetic_video(output)
            old=output/'identity_reconciliation'/'099';old.mkdir(parents=True)
            for name in ('decision.json','failure.json','evidence_05.jpg'):(old/name).write_text('previous-run')
            resolve_global(bundle,local,output,Backend([]),review_max_calls=0)
            self.assertFalse((output/'identity_reconciliation'/'099').exists())
            history=list((output/'identity_reconciliation_history').iterdir())
            self.assertEqual(len(history),1)
            self.assertEqual((history[0]/'099'/'decision.json').read_text(),'previous-run')

    def test_cached_requests_do_not_spend_the_actual_generation_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);bundle,local,provisional=synthetic_video(output)
            shot,review=deepcopy(bundle['shots'][1]),deepcopy(local['shots'][1])
            shot.update(shot_id='departure',full_start_sec=6);shot['tracks'][0]['track_id']='departure:track'
            review['shot_id']='departure';review['entities'][0].update(local_id='departure:traveler',track_ids=['departure:track'])
            bundle['shots'].append(shot);local['shots'].append(review)
            payload=dict(decision='same',left_view=1,right_view=1,left_observation='face',right_observation='face',
                         basis='distinctive_visual_traits',reason='matching facial geometry')
            with patch.object(ReviewBackend,'_generate',return_value=json.dumps(payload)) as generate:
                first=StoryBackend(Settings(json_retries=0),output,limit=5)
                result=resolve_global(bundle,local,output,first,review_max_calls=1)
                self.assertEqual(first.ledger['generations'],1)
                self.assertEqual(len(result['entities']),2)
                resumed=StoryBackend(Settings(json_retries=0),output,limit=5)
                result=resolve_global(bundle,local,output,resumed,review_max_calls=1)
                self.assertEqual(generate.call_count,2)
                self.assertEqual(resumed.ledger['generations'],2)
                self.assertEqual(len(result['entities']),1)
                self.assertEqual(result['identity_reconciliation']['generations'],1)
                self.assertEqual(result['identity_reconciliation']['completed_requests'],2)

    def test_linked_resume_keeps_cumulative_generation_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);bundle,local,provisional=synthetic_video(output)
            shot,review=deepcopy(bundle['shots'][1]),deepcopy(local['shots'][1])
            shot.update(shot_id='departure',full_start_sec=6);shot['tracks'][0]['track_id']='departure:track'
            review['shot_id']='departure';review['entities'][0].update(local_id='departure:traveler',track_ids=['departure:track'])
            bundle['shots'].append(shot);local['shots'].append(review)
            payload=dict(decision='same',left_view=1,right_view=1,left_observation='face',right_observation='face',
                         basis='distinctive_visual_traits',reason='matching facial geometry')
            with patch.object(ReviewBackend,'_generate',return_value=json.dumps(payload)) as generate:
                first=StoryBackend(Settings(json_retries=0),output,limit=5)
                result=resolve_global(bundle,local,output,first,review_max_calls=1,strategy="linked")
                self.assertEqual(first.ledger['generations'],1)
                self.assertEqual(len(result['entities']),2)
                resumed=StoryBackend(Settings(json_retries=0),output,limit=5)
                result=resolve_global(bundle,local,output,resumed,review_max_calls=1,strategy="linked")
                self.assertEqual(generate.call_count,1)
                self.assertEqual(resumed.ledger['generations'],1)
                self.assertEqual(len(result['entities']),2)
                self.assertEqual(result['identity_reconciliation']['generations'],0)
                self.assertEqual(result['identity_reconciliation']['completed_requests'],1)

    def test_competing_owners_get_separate_flat_requests_with_their_own_shared_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);bundle,local,_=synthetic_video(output)
            source=deepcopy(bundle['shots'][0]['tracks'][0])
            bundle['shots']=bundle['shots'][:1];local['shots']=local['shots'][:1]
            shot,review=bundle['shots'][0],local['shots'][0]
            shot['tracks']=[];review['entities']=[];review['track_roles']={}
            for index,(name,frames) in enumerate((('hand',[0,1,3,4]),('first',[0,1]),('second',[3,4]))):
                track=deepcopy(source);track['track_id']=name
                track['observations']=[dict(source['observations'][0],frame=f) for f in frames]
                track['representatives']=track['observations'];shot['tracks'].append(track)
                review['track_roles'][name]='body_part' if index==0 else 'whole_subject'
                review['entities'].append(dict(local_id=name,track_ids=[name],kind='person',label='person',appearance='visible'))
            backend=Backend([dict(decision='uncertain'),dict(decision='uncertain')])
            resolve_global(bundle,local,output,backend,review_max_calls=2)
            self.assertEqual([row['schema'] for row in backend.calls],['FlatPairDecision','FlatPairDecision'])
            inventories=[json.loads(row['prompt'].split('\nEVIDENCE: ',1)[1]) for row in backend.calls]
            self.assertTrue(all(len(row['pairs'])==1 for row in inventories))
            first,second=inventories
            self.assertEqual(first['shown_scene_frames']['hand'],first['shown_scene_frames']['first'])
            self.assertEqual(second['shown_scene_frames']['hand'],second['shown_scene_frames']['second'])
            self.assertNotEqual(first['shown_scene_frames']['hand'],second['shown_scene_frames']['hand'])

    def test_isolated_parts_only_propose_same_shot_ownership_and_show_common_frame(self):
        clusters={'A':cluster('hand','A'),'B':cluster('owner','B'),'C':cluster('elsewhere','C')}
        entries={'hand':dict(shot_id='scene',_part_or_worn=True,label='hand'),
                 'owner':dict(shot_id='scene',label='person'),
                 'elsewhere':dict(shot_id='another',label='person')}
        pairs=candidate_pairs(clusters,entries,{},[])
        self.assertIn({'A','B'},[{row['left'],row['right']} for row in pairs])
        self.assertNotIn({'A','C'},[{row['left'],row['right']} for row in pairs])
        clusters['A']['kind']='object'  # A worn item can be attached to a person.
        pairs=candidate_pairs(clusters,entries,{},[])
        self.assertIn({'A','B'},[{row['left'],row['right']} for row in pairs])
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);bundle,local,_=synthetic_video(output)
            from story_semantics import local_lookup
            scene_frames={}
            group={'A':cluster('arrival:traveler','A')}
            _,shown,_=focused_images(bundle,local_lookup(local),group,output,
                required_frames={'arrival:traveler':2},scene_frames=scene_frames)
            self.assertIn(2,shown['arrival:traveler'])
            self.assertEqual(scene_frames,{'arrival:traveler':[2]})

    def test_failed_pair_is_deferred_while_other_pairs_can_form_supported_path(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);bundle,local,_=synthetic_video(output)
            shot,review=deepcopy(bundle['shots'][1]),deepcopy(local['shots'][1])
            shot.update(shot_id='departure',full_start_sec=6);shot['tracks'][0]['track_id']='departure:track'
            review['shot_id']='departure';review['entities'][0].update(local_id='departure:traveler',track_ids=['departure:track'])
            bundle['shots'].append(shot);local['shots'].append(review)
            good=dict(decision='same',left_view=1,right_view=1,left_observation='face',right_observation='face',
                      basis='distinctive_visual_traits',reason='matching facial geometry')
            backend=Backend([ModelOutputError('malformed individual comparison'),good,good])
            result=resolve_global(bundle,local,output,backend,review_max_calls=5)
            self.assertEqual(len(backend.calls),3)
            self.assertTrue(all(row['schema']=='PairedViewDecision' for row in backend.calls))
            self.assertEqual(len(result['entities']),1)
            self.assertEqual(len(result['identity_reconciliation']['errors']),1)
            self.assertFalse(result['identity_reconciliation']['unreviewed_candidates'])

    def test_default_starts_from_local_hypotheses_and_uses_only_focused_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);bundle,local,_=synthetic_video(output)
            response=dict(decisions=[verdict(left='C0001',right='C0002',
                a='arrival:traveler',b='return:traveler')])
            backend=Backend([response])
            result=resolve_global(bundle,local,output,backend,review_max_calls=1)
            self.assertEqual(result['initialization'],'local')
            self.assertEqual([row['schema'] for row in backend.calls],['PairedViewDecision'])
            self.assertEqual(len(result['entities']),1)
            self.assertEqual(len(json.loads((output/'global_provisional.json').read_text())['entities']),2)
            self.assertFalse((output/'global_forward.json').exists())
            inventory=json.loads((output/'identity_reconciliation/001/input.json').read_text())
            self.assertTrue(all('track_ids' not in row for row in inventory['local_entities']))
            self.assertTrue(all(set(row)=={'shot_id','summary'} for row in inventory['shot_evidence']))

    def test_zero_focused_budget_preserves_every_local_hypothesis_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);bundle,local,_=synthetic_video(output)
            backend=Backend([])
            result=resolve_global(bundle,local,output,backend,review_max_calls=0)
            self.assertEqual(len(result['entities']),2)
            self.assertEqual(set(result['mapping']),{'arrival:traveler','return:traveler'})
            self.assertEqual(result['identity_reconciliation']['status'],'needs_review')
            self.assertFalse(backend.calls)

    def test_compact_visual_aliases_decode_to_immutable_local_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            bundle, local, _ = synthetic_video(output)
            provisional = dict(groups=[['I001','I002']], rejections={})
            backend = Backend([provisional, provisional])
            result = resolve_global(bundle, local, output, backend, initialization="whole", review_max_calls=0)
            self.assertEqual(result['entities'][0]['local_ids'], ['arrival:traveler','return:traveler'])
            self.assertEqual(result['visual_aliases'], {'arrival:traveler':'I001','return:traveler':'I002'})
            self.assertIn('I001', backend.calls[0]['images'][0][0])
            self.assertNotIn('arrival:traveler', backend.calls[0]['prompt'])
            self.assertEqual(json.loads((output/'global_forward.json').read_text())['entities'][0]['local_ids'],
                             ['arrival:traveler','return:traveler'])
            for groups in ([['I001','arrival:traveler','I002']], [['I001','I999']],
                           [['I001']], [['I001'], ['I002'], []]):
                invalid = GlobalPartition(groups=groups)
                with self.subTest(groups=groups), self.assertRaises(ValueError):
                    backend.calls[0]['validate'](invalid)

    def test_group_partition_cache_roundtrip_preserves_inference_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);bundle,local,_=synthetic_video(output)
            with patch.object(ReviewBackend,'_generate',return_value='{"groups":[["I001","I002"]],"rejections":{}}') as generate:
                backend=StoryBackend(Settings(json_retries=0),output,limit=4)
                first=resolve_global(bundle,local,output,backend,initialization="whole",review_max_calls=0)
                resumed=StoryBackend(Settings(json_retries=0),output,limit=4)
                again=resolve_global(bundle,local,output,resumed,initialization="whole",review_max_calls=0)
                self.assertEqual(generate.call_count,2)
                self.assertEqual(first['mapping'],again['mapping'])
                self.assertEqual(resumed.ledger['generations'],2)
                cached=list(resumed.cache_dir.glob('*.json'))
                self.assertEqual(len(cached),2)
                for path in cached:
                    self.assertEqual(set(json.loads(path.read_text())),{'groups','rejections'})

    def test_compact_partition_rejections_cannot_overlap_or_alias_collide(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);bundle,local,_=synthetic_video(output)
            valid=dict(groups=[['I001']],rejections={'I002':'background rock'})
            backend=Backend([valid,valid])
            result=resolve_global(bundle,local,output,backend,initialization="whole",review_max_calls=0)
            self.assertEqual(result['rejected_local_ids'],['return:traveler'])
            for raw in (dict(groups=[['I001','I002']],rejections={'I002':'rock'}),
                        dict(groups=[['I001']],rejections={'I002':'rock','return:traveler':'rock'}),
                        dict(groups=[['I001']],rejections={'I002':' '})):
                with self.subTest(raw=raw),self.assertRaises(ValueError):
                    backend.calls[0]['validate'](GlobalPartition.model_validate(raw))

    def test_real_resolver_splits_disproved_inherited_identity_without_dropping_members(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            bundle, local, provisional = synthetic_video(output)
            inherited = dict(entities=[dict(provisional['entities'][0],
                local_ids=['arrival:traveler', 'return:traveler'])])
            review = dict(decisions=[], cluster_checks=[dict(cluster_id='C0001', decision='conflicted')],
                internal_conflicts=[dict(cluster_id='C0001', evidence=evidence(
                    'arrival:traveler', 'return:traveler', contradiction=True))])
            backend = Backend([inherited, inherited, review])
            result = resolve_global(bundle, local, output, backend, initialization="whole", review_max_calls=1)
            self.assertEqual(len(result['entities']), 2)
            self.assertEqual(set(result['mapping']), {'arrival:traveler', 'return:traveler'})
            self.assertNotEqual(result['mapping']['arrival:traveler'], result['mapping']['return:traveler'])
            self.assertFalse(result['cannot_link'])
            self.assertEqual(result['identity_reconciliation']['soft_objections'][0]['local_ids'],
                             ['arrival:traveler','return:traveler'])
            self.assertFalse(result['identity_reconciliation']['unreviewed_internal_groups'])
            self.assertFalse(result['identity_reconciliation']['unresolved'])
            self.assertEqual(result['identity_reconciliation']['status'], 'needs_review')
            self.assertEqual(result['identity_reconciliation']['resolved_uncertainties'][0]['resolution'],
                             'repaired_by_grounded_split')

    def test_uncertain_pair_resolved_by_later_supported_path_is_not_stale_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            bundle, local, provisional = synthetic_video(output)
            third_shot, third_local = deepcopy(bundle['shots'][1]), deepcopy(local['shots'][1])
            third_shot.update(shot_id='departure', full_start_sec=6)
            third_shot['tracks'][0]['track_id'] = 'departure:track'
            third_local.update(shot_id='departure')
            third_local['entities'][0].update(local_id='departure:traveler', track_ids=['departure:track'])
            bundle['shots'].append(third_shot)
            local['shots'].append(third_local)
            for shot in local['shots']:
                shot['entities'][0]['label'] = 'traveler'
            provisional['entities'].append(dict(provisional['entities'][1], local_ids=['departure:traveler']))
            first = dict(decisions=[verdict('C0001', 'C0002', decision='uncertain'),
                verdict('C0001', 'C0003', 'arrival:traveler', 'departure:traveler')])
            second = dict(decisions=[verdict('C0001', 'C0002', 'arrival:traveler', 'return:traveler')])
            backend = Backend([provisional, provisional, first, second, second])
            result = resolve_global(bundle, local, output, backend, initialization="whole", review_max_calls=3)
            audit = result['identity_reconciliation']
            self.assertEqual(len(result['entities']), 1)
            self.assertEqual(audit['status'], 'complete')
            self.assertFalse(audit['unresolved'])
            self.assertEqual(len(audit['resolved_uncertainties']), 1)
            self.assertEqual(audit['resolved_uncertainties'][0]['resolution'], 'resolved_by_supported_path')

    def test_inherited_multi_local_group_gets_independent_internal_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            bundle, local, provisional = synthetic_video(output)
            inherited = dict(entities=[dict(provisional['entities'][0],
                local_ids=['arrival:traveler', 'return:traveler'])])
            review = dict(decisions=[], cluster_checks=[dict(cluster_id='C0001', decision='consistent',
                evidence=[evidence('arrival:traveler', 'return:traveler')], reason='matching face and scarf')])
            backend = Backend([inherited, inherited, review])
            result = resolve_global(bundle, local, output, backend, initialization="whole", review_max_calls=1)
            self.assertEqual(len(result['entities']), 1)
            self.assertEqual(len(backend.calls), 3)
            self.assertTrue(result['identity_reconciliation']['reviews'][0]['internal_checks'])
            self.assertFalse(result['identity_reconciliation']['unreviewed_internal_groups'])
            self.assertEqual(result['identity_reconciliation']['status'], 'complete')

    def test_full_active_resolver_merges_generic_fragments_and_keeps_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            bundle, local, provisional = synthetic_video(output)
            focused = dict(decisions=[verdict('C0001', 'C0002', 'arrival:traveler', 'return:traveler')])
            backend = Backend([provisional, provisional, focused])
            result = resolve_global(bundle, local, output, backend, initialization="whole", review_max_calls=2,
                                    context={'title': 'An unrelated video'})
            self.assertEqual(len(result['entities']), 1)
            self.assertEqual(set(result['mapping'].values()), {'G0001'})
            self.assertEqual(len(json.loads((output/'global_provisional.json').read_text())['entities']), 2)
            self.assertEqual(len(backend.calls), 3)
            self.assertIn('An unrelated video', backend.calls[-1]['prompt'])
            self.assertIn('untrusted background',backend.calls[-1]['prompt'])
            self.assertIn('alone cannot justify a match',backend.calls[-1]['prompt'])
            self.assertEqual(len(backend.calls[-1]['images']), 1)
            self.assertIn('LEFT versus RIGHT', backend.calls[-1]['images'][0][0])
            self.assertTrue((output/'identity_reconciliation/001/evidence_00.jpg').is_file())
            audit = result['identity_reconciliation']
            self.assertEqual((audit['initial_entities'], audit['final_entities']), (2, 1))
            self.assertEqual(audit['status'], 'complete')

    def test_actual_generation_cap_includes_structured_output_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            bundle, local, provisional = synthetic_video(output)
            backend = StoryBackend(Settings(json_retries=1), output, limit=10)
            with patch.object(ReviewBackend, '_generate', return_value='not JSON') as generate:
                result, audit = reconcile_global(bundle, local, GlobalDecision.model_validate(provisional),
                                                output, backend, max_calls=1)
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(backend.ledger['generations'], 1)
            self.assertEqual(len(result.entities), 2)
            self.assertEqual(audit['status'], 'needs_review')
            self.assertEqual(len(audit['errors']), 1)

    def test_zero_review_budget_preserves_partition_and_reports_unreviewed_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            bundle, local, provisional = synthetic_video(output)
            backend = Backend([])
            result, audit = reconcile_global(bundle, local, GlobalDecision.model_validate(provisional),
                                            output, backend, max_calls=0)
            self.assertEqual(len(result.entities), 2)
            self.assertFalse(backend.calls)
            self.assertEqual(len(audit['unreviewed_candidates']), 1)

    def test_evidence_frames_match_rendered_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            bundle, local, provisional = synthetic_video(output)
            entries = {entity['local_id']: dict(entity, shot_id=shot['shot_id'])
                       for shot in local['shots'] for entity in shot['entities']}
            clusters = {'C0001': dict(provisional['entities'][0], source_clusters=['C0001'])}
            images, shown, selected = focused_images(bundle, entries, clusters, output)
            self.assertEqual(shown, {'arrival:traveler': [0, 1, 2]})
            self.assertEqual(selected, {'C0001': ['arrival:traveler']})
            self.assertEqual(len(images), 2)
            self.assertLess(images[0][1].height, 250)  # No empty second crop-row slot.


if __name__ == '__main__':
    unittest.main()


def test_adaptive_depth_only_rechecks_inconclusive_cross_shot_pairs():
    from types import SimpleNamespace
    from global_identity import review_pair_depth
    from story_semantics import BudgetExhausted
    from low_freq_event.backend import ModelOutputError
    for decision,eligible,expected in [('same',True,[1024]),('uncertain',False,[1024]),('uncertain',True,[1024,None])]:
        backend=SimpleNamespace(adaptive_identity_depth=True,identity_thinking_budget=1024)
        seen=[]
        def request(prompt,schema,**kwargs):
            seen.append(backend.identity_thinking_budget)
            return len(seen)
        normalize=lambda r:SimpleNamespace(decisions=[SimpleNamespace(decision=decision)])
        result=review_pair_depth(backend,request,'same pixels',object,{},normalize,eligible=eligible)
        assert seen==expected and result==len(expected) and backend.identity_thinking_budget==1024
    for error in [BudgetExhausted('cap'),ModelOutputError('invalid JSON')]:
        backend=SimpleNamespace(adaptive_identity_depth=True,identity_thinking_budget=1024)
        def request(prompt,schema,**kwargs):
            if backend.identity_thinking_budget is None:raise error
            return 'validated uncertain'
        result=review_pair_depth(backend,request,'same pixels',object,{},
            lambda r:SimpleNamespace(decisions=[SimpleNamespace(decision='uncertain')]))
        assert result=='validated uncertain' and backend.identity_thinking_budget==1024
