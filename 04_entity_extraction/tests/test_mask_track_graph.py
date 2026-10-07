"""Mask geometry/graph checks; no model loading or GPU required."""
from pathlib import Path
import sys
import tempfile
import unittest
import json

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mask_track_graph import (MaskGraphConfig,analyze_mask_track_graph,classify_pair,constrained_components,
                             identity_boundary_candidates,mask_pair_metrics,whole_region_support,
                             encode_masked_representatives,segment_temporal_candidates,remap_temporal_constraints)


def mask(rect, shape=(100,200)):
    a=np.zeros(shape,dtype=bool);x1,y1,x2,y2=rect;a[y1:y2,x1:x2]=True
    return dict(array=a,bbox=list(rect),pixels=int(a.sum()))


class MaskGraphTests(unittest.TestCase):
    def test_nested_bbox_disjoint_foreground_is_not_duplicate(self):
        big=mask((0,0,100,100));small=mask((40,40,60,60))
        big['array'][40:60,40:60]=False;big['pixels']=int(big['array'].sum())
        record=mask_pair_metrics(big,small)
        self.assertEqual(record['mask_iou'],0)
        self.assertEqual(record['intersection_over_smaller'],0)
        decision=classify_pair([dict(record,frame=i) for i in range(4)],4,4,MaskGraphConfig())
        self.assertEqual(decision['relation'],'distinct_mask_regions_require_semantic_check')
        self.assertFalse(decision['hard'])

    def test_repeated_identical_masks_support_fusion(self):
        a=mask((10,10,40,60));record=mask_pair_metrics(a,a)
        decision=classify_pair([dict(record,frame=i) for i in range(4)],4,4,MaskGraphConfig())
        self.assertEqual(decision['relation'],'same_foreground')

    def test_far_separation_veto_has_region_scope_not_physical_identity_scope(self):
        left=mask((0,0,20,40));right=mask((150,0,170,40))
        metrics=mask_pair_metrics(left,right)
        decision=classify_pair([dict(metrics,frame=i) for i in range(3)],3,3,MaskGraphConfig())
        self.assertTrue(decision['hard'])
        self.assertEqual(decision['hard_scope'],'foreground_regions')
        self.assertFalse(decision['physical_identity_constraint'])

    def test_single_frame_needs_stricter_duplicate_evidence(self):
        row=dict(mask_iou=.85,intersection_over_smaller=.90,box_gap_ratio=0,area_ratio=.8,frame=0)
        self.assertNotEqual(classify_pair([row],1,1,MaskGraphConfig())['relation'],'same_foreground')
        row['mask_iou']=.97
        self.assertEqual(classify_pair([row],1,1,MaskGraphConfig())['relation'],'same_foreground')

    def test_transitive_duplicate_merge_cannot_cross_separation(self):
        groups,rejected=constrained_components(['a','b','c'],[('a','b',.98),('b','c',.9)],[['a','c']])
        self.assertEqual(groups,[['a','b'],['c']])
        self.assertEqual(len(rejected),1)

    def test_body_parts_with_shared_whole_mask_have_support(self):
        parts={('a','whole'):[dict(frame=i,intersection_over_smaller=1,left_pixels=10,right_pixels=100) for i in range(3)],
               ('b','whole'):[dict(frame=i,intersection_over_smaller=1,left_pixels=10,right_pixels=100) for i in range(3)]}
        self.assertEqual(whole_region_support('a','b',parts)['whole_track'],'whole')

    def test_switching_foreground_affiliation_exposes_identity_boundary(self):
        pairs={('a','t'):[dict(frame=i,mask_iou=.98) for i in range(3)],
               ('b','t'):[dict(frame=i,mask_iou=.98) for i in range(3,6)],
               ('a','b'):[dict(frame=i,mask_iou=0,intersection_over_smaller=0) for i in range(6)]}
        result=identity_boundary_candidates({'track_id':'t'},pairs,MaskGraphConfig())
        self.assertEqual(len(result),1)
        self.assertEqual((result[0]['after_frame'],result[0]['before_frame']),(2,3))

    def test_end_to_end_grouping_preserves_sources_and_conditional_carried_subject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);Image.new('RGB',(200,100),'gray').save(root/'frame.jpg')
            big=mask((0,0,100,100));big['array'][40:60,40:60]=False
            small=mask((40,40,60,60));far=mask((150,0,180,60))
            tracks=[]
            for tid,data in [('s:T1',big),('s:T2',big),('s:T3',small),('s:T4',far)]:
                name=tid.replace(':','_');rows=[]
                for frame in range(3):
                    path=f'{name}_{frame}.png';Image.fromarray(data['array'].astype(np.uint8)*255).save(root/path)
                    rows.append(dict(frame=frame,bbox=data['bbox'],evidence='sam2_mask',mask=path,
                                     confidence=.9,category='unknown',crop='frame.jpg',image='frame.jpg',
                                     track_id=tid,time_sec=frame/24))
                tracks.append(dict(track_id=tid,observations=rows,representatives=rows,source_track_ids=[tid]))
            bundle=dict(fingerprint='unit-test',asset_root=str(root),shots=[dict(shot_id='s',fps=24,tracks=tracks,
                           scene_frames=[],adaptive_scene_frames=[])])
            result=analyze_mask_track_graph(bundle,root/'output')
            shot=result['shots'][0]
            self.assertEqual(len(shot['tracks']),3)
            self.assertEqual(shot['tracks'][0]['source_track_ids'],['s:T1','s:T2'])
            self.assertIn(['s:T1','s:T3'],shot['mask_graph']['conditional_separation'])
            self.assertEqual(len(shot['tracks'][0]['observations'][0]['source_observation_refs']),2)
            self.assertTrue(Path(shot['tracks'][0]['representatives'][0]['crop']).is_file())
            self.assertEqual(len(bundle['shots'][0]['tracks']),4)
            self.assertEqual(shot['mask_graph']['cannot_link_scope'],'foreground_regions')
            self.assertEqual(shot['mask_graph']['physical_cannot_link'],[])
            self.assertEqual(result['mask_track_graph']['physical_identity_constraints_from_geometry'],0)

    def test_masked_appearance_uses_exact_bytes_cache_and_preserves_raw_feature(self):
        class Encoder:
            def __init__(self):self.count=0
            def encode_images(self,images):
                self.count+=len(images)
                return [[float(np.asarray(im).mean())+1,10.,5.] for im in images]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);Image.new('RGB',(10,10),'white').save(root/'crop.png')
            Image.new('RGB',(10,10),'gray').save(root/'masked.png')
            row=dict(frame=3,bbox=[0,0,10,10],crop='crop.png',masked_crop='masked.png',
                     feature=[1.,0.,0.],feature_provenance='original detector crop')
            bundle=dict(fingerprint='raw',asset_root=str(root),shots=[dict(shot_id='s',
                        tracks=[dict(track_id='s:T1',representatives=[row])])])
            encoder=Encoder()
            result=encode_masked_representatives(bundle,root/'output',device='cpu',encoder=encoder)
            rep=result['shots'][0]['tracks'][0]['representatives'][0]
            self.assertEqual(rep['source_crop_feature'],[1.,0.,0.])
            self.assertEqual(rep['source_crop_feature_provenance'],'original detector crop')
            self.assertEqual(rep['feature_provenance']['input_kind'],'sam_foreground_masked_crop')
            self.assertEqual(encoder.count,1)
            cached=encode_masked_representatives(bundle,root/'output',device='cpu',encoder=encoder)
            self.assertEqual(encoder.count,1)
            self.assertEqual(result['fingerprint'],cached['fingerprint'])
            Image.new('RGB',(10,10),'red').save(root/'masked.png')
            changed=encode_masked_representatives(bundle,root/'output',device='cpu',encoder=encoder)
            self.assertEqual(encoder.count,2)
            self.assertNotEqual(changed['fingerprint'],cached['fingerprint'])
            self.assertEqual(row['feature'],[1.,0.,0.])

    def test_postlocal_appearance_save_path_does_not_overwrite_prelocal_bundle(self):
        class Encoder:
            def encode_images(self,images):return [[1.,2.,3.] for _ in images]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);Image.new('RGB',(10,10),'white').save(root/'crop.png')
            output=root/'output';prelocal=output/'mask_track_graph/bundle.json'
            prelocal.parent.mkdir(parents=True);prelocal.write_text('{"stage":"prelocal"}')
            bundle=dict(fingerprint='postlocal',asset_root=str(root),shots=[dict(shot_id='s',
                        tracks=[dict(track_id='s:T1',representatives=[dict(frame=0,crop='crop.png')])])])
            destination=output/'mask_refinement/bundle.json'
            result=encode_masked_representatives(bundle,output,device='cpu',encoder=Encoder(),save_path=destination)
            self.assertEqual(json.loads(prelocal.read_text()),{'stage':'prelocal'})
            self.assertEqual(json.loads(destination.read_text())['fingerprint'],result['fingerprint'])
            self.assertTrue((output/'mask_refinement/appearance.json').is_file())

    def test_temporal_candidates_partition_observations_exactly_and_allow_reunion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'frame.jpg';Image.new('RGB',(20,20),'white').save(image)
            observations=[dict(frame=frame,bbox=[0,0,10,10],confidence=.9,track_id='s:T1',
                         crop=str(image),image=str(image),source_observation_refs=[dict(track_id='raw:T9',frame=frame)])
                         for frame in [0,1,4,10,14]]
            track=dict(track_id='s:T1',observations=observations,representatives=observations,
                       source_track_ids=['raw:T9'],identity_boundary_candidates=[dict(after_frame=4,before_frame=10)])
            bundle=dict(fingerprint='before',asset_root=str(root),shots=[dict(shot_id='s',fps=24,tracks=[track],
                         mask_graph=dict(source_to_candidate={'raw:T9':'s:T1'},cannot_link=[],conditional_separation=[]))])
            result=segment_temporal_candidates(bundle,root)
            children=result['shots'][0]['tracks']
            self.assertEqual([t['track_id'] for t in children],['s:T1:C001','s:T1:C002'])
            self.assertEqual([o['frame'] for t in children for o in t['observations']],[0,1,4,10,14])
            self.assertTrue(all(t['temporal_candidate']['same_identity_allowed'] for t in children))
            self.assertEqual(children[1]['source_track_ids'],['raw:T9'])
            self.assertEqual(children[1]['observations'][0]['source_observation_refs'][0]['frame'],10)
            self.assertEqual(result['shots'][0]['mask_graph']['cannot_link'],[])
            self.assertEqual(result['shots'][0]['mask_graph']['source_to_candidates']['raw:T9'],
                             ['s:T1:C001','s:T1:C002'])
            self.assertEqual(len(bundle['shots'][0]['tracks']),1)
            repeated=segment_temporal_candidates(result,root)
            self.assertEqual(repeated['fingerprint'],result['fingerprint'])

    def test_hard_constraint_only_follows_its_supported_child_interval(self):
        graph=dict(cannot_link=[['a','b']],conditional_separation=[],
                   separation_frame_evidence=[dict(kind='hard',left='a',right='b',frames=[0,1,2])])
        children={'a':['a:C1','a:C2'],'b':['b']}
        frames={'a:C1':{0,1,2},'a:C2':{10,11,12},'b':{0,1,2,10,11,12}}
        result=remap_temporal_constraints(graph,children,frames)
        self.assertEqual(result['cannot_link'],[['a:C1','b']])
        self.assertNotIn(['a:C2','b'],result['cannot_link'])
        self.assertNotIn(['a:C1','a:C2'],result['cannot_link'])
        missing=remap_temporal_constraints(dict(cannot_link=[['a','b']]),children,frames)
        self.assertEqual(missing['cannot_link'],[])
        self.assertEqual(len(missing['conditional_separation']),2)

    def test_constraint_not_transferred_to_nonoverlapping_candidate_times(self):
        graph=dict(cannot_link=[['a','b']],conditional_separation=[],
                   separation_frame_evidence=[dict(kind='hard',left='a',right='b',frames=[0,1])])
        result=remap_temporal_constraints(graph,{'a':['a:C1','a:C2'],'b':['b']},
                    {'a:C1':{0,1},'a:C2':{10,11},'b':{0,1}})
        self.assertEqual(result['cannot_link'],[['a:C1','b']])

    def test_interval_without_prior_keyframe_gets_real_video_crop(self):
        import av
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'shot.mp4'
            with av.open(str(video),'w') as container:
                stream=container.add_stream('libx264',rate=24);stream.width=32;stream.height=24;stream.pix_fmt='yuv420p'
                for index in range(4):
                    frame=av.VideoFrame.from_ndarray(np.full((24,32,3),index*40,dtype=np.uint8),format='rgb24')
                    for packet in stream.encode(frame):container.mux(packet)
                for packet in stream.encode():container.mux(packet)
            image=root/'first.jpg';Image.new('RGB',(32,24),'black').save(image)
            observations=[dict(frame=i,bbox=[2,2,12,16],confidence=.8,image=None,crop=None) for i in range(4)]
            first=dict(observations[0],image=str(image),crop=str(image))
            track=dict(track_id='s:T1',observations=observations,representatives=[first],
                       identity_boundary_candidates=[dict(after_frame=1,before_frame=2)])
            bundle=dict(fingerprint='video',asset_root=str(root),shots=[dict(shot_id='s',fps=24,source=str(video),tracks=[track])])
            result=segment_temporal_candidates(bundle,root)
            second=result['shots'][0]['tracks'][1]
            self.assertEqual([row['frame'] for row in second['observations']],[2,3])
            self.assertEqual([row['frame'] for row in second['representatives']],[2,3])
            for row in second['representatives']:
                self.assertTrue(Path(row['crop']).is_file())
                with Image.open(row['crop']) as crop:self.assertEqual(crop.size,(10,14))


if __name__=='__main__':unittest.main()
