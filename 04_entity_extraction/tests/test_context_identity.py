import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'01_low_freq_event'),str(ROOT.parent/'03_adaptive_sampling')]
import clear_cache
import fcntl
import context_identity as ci
import sequential_identity as si


def verdict(value, contradictions=None, evidence=None):
    return ci.ContextDecision(decision=value, visual_evidence='observed', temporal_evidence='continuity',
        positive_identity_evidence=evidence if evidence is not None else ['matching face and action'],
        contradictions=contradictions or [])


def obs(shot, index=0):
    return dict(frame_key=f'{shot}_f{index}', shot_id=shot, detection_id=f'{shot}_d{index}',
                bbox=[100,100,400,900],full_time_sec=float(index),original_time_sec=float(index),
                label='인물',appearance='appearance',entity_kind='person')


def entry(shot,index=0):
    return dict(label='인물',kind='person',appearance='appearance',shot_id=shot,
                scene_summary=f'action in {shot}',observations=[obs(shot,index)])


class DecisionTests(unittest.TestCase):
    def test_dark_crop_without_contradiction_is_uncertain(self):
        self.assertEqual(ci.normalize_decision(verdict('different'),{'a'},{'b'}).decision,'uncertain')

    def test_invented_frame_cannot_block_merge(self):
        c=ci.Contradiction(frame_a='invented',frame_b='b',trait='incompatible_face',evidence_a='x',evidence_b='y')
        self.assertEqual(ci.normalize_decision(verdict('different',[c]),{'a'},{'b'}).decision,'uncertain')

    def test_grounded_different_is_retained(self):
        c=ci.Contradiction(frame_a='a',frame_b='b',trait='incompatible_face',evidence_a='x',evidence_b='y')
        self.assertEqual(ci.normalize_decision(verdict('different',[c]),{'a'},{'b'}).decision,'different')

    def test_same_requires_positive_evidence(self):
        self.assertEqual(ci.normalize_decision(verdict('same',evidence=[]),{'a'},{'b'}).decision,'uncertain')

    def test_same_with_contradiction_is_uncertain(self):
        c=ci.Contradiction(frame_a='a',frame_b='b',trait='incompatible_face',evidence_a='x',evidence_b='y')
        self.assertEqual(ci.normalize_decision(verdict('same',[c]),{'a'},{'b'}).decision,'uncertain')

    def test_supported_same_is_retained(self):
        self.assertEqual(ci.normalize_decision(verdict('same'),{'a'},{'b'}).decision,'same')

    def test_no_mutation_of_model_response(self):
        result=verdict('different')
        ci.normalize_decision(result,{'a'},{'b'})
        self.assertEqual(result.decision,'different')

    def test_frame_selection_has_neighbors_and_targets(self):
        frames=[dict(frame_key=f's{s}_f{i}',shot_id=f's{s}') for s in range(7) for i in range(10)]
        entries={'a':entry('s2',1),'b':entry('s4',8)}
        chosen=ci.select_context_frames(frames,entries)
        self.assertEqual({f['shot_id'] for f in chosen},{'s1','s2','s3','s4','s5'})
        self.assertIn('s2_f1',{f['frame_key'] for f in chosen})
        self.assertIn('s4_f8',{f['frame_key'] for f in chosen})
        self.assertGreater(len(ci.select_context_frames(frames,entries,expanded=True)),len(chosen))


class MemoryTests(unittest.TestCase):
    def test_history_accumulates(self):
        memory={}
        a,_=si.update_memory(memory,'a',entry('s1'),'new',None,'F')
        b,blocked=si.update_memory(memory,'b',entry('s2'),'same',a,'F')
        self.assertEqual(a,b);self.assertFalse(blocked)
        self.assertEqual([h['shot_id'] for h in memory[a]['history']],['s1','s2'])
        self.assertEqual(len(si.snapshot(memory)[a]['history']),2)

    def test_covisible_entities_not_merged(self):
        memory={}
        a,_=si.update_memory(memory,'a',entry('s1'),'new',None,'F')
        b,blocked=si.update_memory(memory,'b',entry('s1'),'same',a,'F')
        self.assertTrue(blocked);self.assertNotEqual(a,b)


class IntegrationTests(unittest.TestCase):
    def test_forward_reverse_history_and_final_merge(self):
        calls=[]
        class Backend:
            def request(self,prompt,schema,**kwargs):
                if schema is si.ShotReview:
                    n=sum(c[0]=='forward' for c in calls)+1
                    # Inspect the explicit inventory passed to the model, not a separate unused variable.
                    self_outer.assertIn('"shot_history":',prompt)
                    if n==2:self_outer.assertIn('summary s1',prompt)
                    calls.append(('forward',n))
                    result=si.ShotReview(scene_summary=f'summary s{n}',entities=[si.ShotEntity(
                        ids=['local'],label='인물',appearance='appearance',kind='person',
                        decision='new' if n==1 else 'same',match_id=None if n==1 else 'F0001',reason='observed action')])
                elif schema is si.MemoryReview:
                    n=sum(c[0]=='reverse' for c in calls)+1
                    self_outer.assertNotIn('F0001',prompt)
                    if n==1:self_outer.assertIn('"shot_history": []',prompt)
                    else:self_outer.assertIn('reverse summary 1',prompt)
                    calls.append(('reverse',n))
                    result=si.MemoryReview(scene_summary=f'reverse summary {n}',matches=[si.MemoryMatch(
                        local_id='S002_001' if n==1 else 'S001_001',decision='new' if n==1 else 'same',
                        match_id=None if n==1 else 'R0001',reason='action')])
                else:
                    result=si.MergeReview(groups=[])
                return kwargs.get('validate',lambda x:x)(result)
        self_outer=self
        frames=[dict(obs(f's{i}',i),image='unused',hold_frames=1) for i in [1,2]]
        timeline=dict(fps='24',shots=[dict(id='s1'),dict(id='s2')])
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)
            def seeds(observed,settings,index):return {'local':entry(f's{index}',index)}
            with patch.object(si,'local_seeds',side_effect=seeds),patch.object(si,'gallery',return_value=([],[])),patch.object(si,'scene_clip',return_value=None),patch.object(si,'adjudicate_pair',return_value=verdict('same')) as judge:
                registry,visual,observed,joins=si.run_sequential(frames,frames,timeline,output,Backend(),None,{f['frame_key']:[] for f in frames})
            self.assertEqual(len(registry),1)
            self.assertEqual(len(observed),2)
            self.assertEqual(len(joins),1)
            self.assertEqual(len(judge.call_args.args[4]),2)
            self.assertEqual(len(json.loads((output/'sequential/shot_history.json').read_text())),2)
            self.assertEqual(len(json.loads((output/'sequential/reverse_history.json').read_text())),2)

    def test_reconciliation_preserves_distinct_and_uncertain(self):
        for value in ['different','uncertain']:
            with self.subTest(value=value),tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                class Backend:
                    def request(self,*args,**kw):return si.MergeReview(groups=[])
                groups={'a':entry('s1'),'b':entry('s2')}
                with patch.object(si,'gallery',return_value=([],[])),patch.object(si,'adjudicate_pair',return_value=verdict(value)):
                    mapping,joins,_=si.reconcile(groups,{'a':'F1','b':'F1'},{'a':'R1','b':'R2'},[('a','b')],[],root,Backend(),24,root,[])
                self.assertNotEqual(mapping['a'],mapping['b']);self.assertEqual(joins,[])

    def test_adjudication_expands_uncertain_context(self):
        class Backend:
            calls=0
            def request(self,*args,**kwargs):
                self.calls+=1
                return verdict('uncertain' if self.calls==1 else 'same')
        backend=Backend();entries={'a':entry('s1'),'b':entry('s2')}
        frames=[dict(obs('s1'),hold_frames=1),dict(obs('s2'),hold_frames=1)]
        with tempfile.TemporaryDirectory() as temp,patch.object(si,'gallery',return_value=([],[])),patch.object(si,'scene_clip',return_value=None):
            result=ci.adjudicate_pair('a','b',entries,frames,[],Path(temp),backend,Path(temp),24,{})
        self.assertEqual(backend.calls,2);self.assertEqual(result.decision,'same')


class CacheTests(unittest.TestCase):
    def make_output(self,root):
        mode=root/'with_qwen';mode.mkdir()
        (mode/'run_config.json').write_text('{}')
        (mode/'yolo_proposals.json').write_text('{}')
        (mode/'entities.json').write_text('{}')
        (mode/'adaptive_entities_full.mp4').write_bytes(b'video')
        (mode/'entities').mkdir();(mode/'entities'/'representative.jpg').write_bytes(b'image')
        for name in ['frames','grounding','features','crops','request_cache','sequential']:
            (mode/name).mkdir();(mode/name/'cached').write_text('cache')
        return mode

    def test_reset_removes_inference_not_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);mode=self.make_output(root)
            removed=clear_cache.clear_caches(root,apply=True)
            self.assertIn('with_qwen/run_config.json',removed)
            for name in ['run_config.json','yolo_proposals.json','request_cache','sequential','frames','features','grounding','crops']:
                self.assertFalse((mode/name).exists())
            for name in ['entities.json','adaptive_entities_full.mp4','entities/representative.jpg']:
                self.assertTrue((mode/name).exists())
            self.assertEqual(clear_cache.clear_caches(root,apply=True),[])

    def test_dry_run_keeps_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);mode=self.make_output(root)
            clear_cache.clear_caches(root)
            self.assertTrue((mode/'yolo_proposals.json').exists())

    def test_active_run_prevents_any_deletion(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);mode=self.make_output(root)
            with (root/'.run.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):clear_cache.clear_caches(root,apply=True)
            self.assertTrue((mode/'run_config.json').exists())

    def test_symlink_target_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as temp,tempfile.TemporaryDirectory() as external:
            root=Path(temp);mode=self.make_output(root)
            outside=Path(external)/'keep';outside.write_text('keep')
            (mode/'traces').symlink_to(Path(external),target_is_directory=True)
            clear_cache.clear_caches(root,apply=True)
            self.assertTrue(outside.exists());self.assertFalse((mode/'traces').exists())


if __name__=='__main__':unittest.main()
