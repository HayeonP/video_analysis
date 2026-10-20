from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[2]/'01_low_freq_event')]
from catalog_repair import FrameDecision,validate_frame,prepare
from story_export import consolidate_reviewed_bodies


def test_physical_partition_rejects_missing_duplicate_and_repeated_identity():
    for bodies in ([dict(members=['D1'],identity='G1',reason='one body')],
                   [dict(members=['D1','D1'],identity='G1',reason='one body')],
                   [dict(members=['D1'],identity='G1',reason='left'),dict(members=['D2'],identity='G1',reason='right')]):
        with pytest.raises(ValueError):validate_frame(FrameDecision(bodies=bodies),2,['G1'])
    assert validate_frame(FrameDecision(bodies=[dict(members=['D1','D2'],identity='G1',reason='connected torso and arm')]),2,['G1'])


def fixture():
    rows=[dict(detection_id=t,shot_id='s',source_frame=4,tracklet_id=t,entity_id='G1',bbox_pixels=b,source_size=[100,100],bbox=b,confidence=.9) for t,b in [('a',[0,0,30,50]),('b',[25,10,50,40])]]
    proof=dict(shot_id='s',frame=4,entity_id='G1',members=[dict(track_id=r['tracklet_id'],bbox=r['bbox_pixels']) for r in rows],reason='visible arm connects to torso',review='catalog_repair/001')
    return rows,proof


def test_reviewed_body_union_keeps_all_original_detections_and_unrelated_body():
    rows,proof=fixture();other=dict(rows[0],detection_id='c',tracklet_id='c',entity_id='G2')
    output,audit=consolidate_reviewed_bodies(rows+[other],[proof])
    assert len(output)==2 and len(audit)==1
    body=next(r for r in output if r['entity_id']=='G1')
    assert body['bbox_pixels']==[0,0,50,50]
    assert body['consolidated_observations']==rows
    assert rows[0]['bbox_pixels']==[0,0,30,50]


@pytest.mark.parametrize('field,value',[('frame',5),('entity_id','G2')])
def test_body_proof_cannot_transfer_to_different_frame_or_identity(field,value):
    rows,proof=fixture();proof[field]=value
    with pytest.raises(ValueError):consolidate_reviewed_bodies(rows,[proof])


def test_body_proof_cannot_hide_changed_or_missing_source_box():
    rows,proof=fixture();proof['members'][0]['bbox']=[1,1,2,2]
    with pytest.raises(ValueError):consolidate_reviewed_bodies(rows,[proof])


def test_existing_cross_shot_links_survive_without_new_profile_evidence(tmp_path):
    bundle=dict(asset_root=str(tmp_path),shots=[dict(shot_id='s',tracks=[dict(track_id='t',observations=[dict(frame=0,bbox=[0,0,10,10])])])])
    local=dict(shots=[dict(shot_id='s',entities=[dict(local_id='l',track_ids=['t'])])])
    catalog=dict(mapping={'l':'G0001'},entities=[dict(entity_id='G0001',kind='object',local_ids=['l'])])
    b,l,entities,obs,assignments,prototypes,sources=prepare(bundle,local,catalog)
    assert assignments=={} and entities['G0001']['local_ids']==['l']
    assert next(iter(obs.values()))[0]=='G0001'


def test_identity_change_and_peripheral_face_union_require_stronger_review():
    from catalog_repair import needs_second_view
    rows=[(('s',0,tuple(box)),'G1',dict(track_id=tid),dict(bbox=box)) for tid,box in [('left',[0,0,30,50]),('right',[25,0,60,50])]]
    roles={'left':'whole_subject','right':'whole_subject'}
    merged=FrameDecision(bodies=[dict(members=['D1','D2'],identity='G1',reason='possibly one person')])
    assert needs_second_view(merged,rows,{},roles)
    assert not needs_second_view(merged,rows,{},dict(left='body_part',right='whole_subject'))
    changed=FrameDecision(bodies=[dict(members=['D1','D2'],identity='G2',reason='new match')])
    assert needs_second_view(changed,rows,{},dict(left='body_part',right='whole_subject'))


def test_reason_cannot_describe_two_identities_but_merge_their_boxes():
    response=FrameDecision(bodies=[dict(members=['D1','D2'],identity='G0001',reason='D1 matches G0001, D2 matches G0011, two physical bodies')])
    with pytest.raises(ValueError,match='different reference identities'):
        validate_frame(response,2,['G0001','G0011'])


def test_flat_assignments_cannot_swap_reference_ids_into_detection_keys():
    from catalog_repair import FrameAssignments
    valid=FrameAssignments(assignments={'D1':'G0001','D2':'G0011'},evidence={'D1':'foreground body','D2':'opponent behind'})
    assert len(valid.bodies(2,['G0001','G0011']).bodies)==2
    invalid=FrameAssignments(assignments={'G0001':'D1'},evidence={'G0001':'body'})
    with pytest.raises(ValueError):invalid.bodies(1,['G0001'])


def test_format_recovery_is_bounded_and_preserves_strict_validation():
    from catalog_repair import request_assignments,FrameAssignments
    from low_freq_event.backend import ModelOutputError
    class Backend:
        def __init__(self,fail=False):self.prompts=[];self.fail=fail
        def request(self,prompt,schema,**kwargs):
            self.prompts.append(prompt)
            if len(self.prompts)==1 or self.fail:raise ModelOutputError('extra reference keys')
            assert 'ONLY keys' in prompt and '"D2"' in prompt
            return kwargs['validate'](FrameAssignments(assignments={'D1':'G0001','D2':'G0001'},
                evidence={'D1':'torso','D2':'attached arm'}))
    backend=Backend()
    result=request_assignments(backend,'partition',None,2,['G0001'])
    assert len(backend.prompts)==2 and len(result.bodies(2,['G0001']).bodies)==1
    failing=Backend(True)
    with pytest.raises(ModelOutputError):request_assignments(failing,'partition',None,2,['G0001'])
    assert len(failing.prompts)==2


def test_physical_count_can_join_unknown_crop_to_known_body_and_exclude_shadow():
    from catalog_repair import PhysicalPartition,reconcile_physical_partition
    response=FrameDecision(bodies=[dict(members=['D1'],identity='new',reason='face hidden'),
        dict(members=['D2'],identity='G0001',reason='whole torso'),
        dict(members=['D3'],identity='new',reason='flat silhouette')])
    partition=PhysicalPartition(bodies=[dict(members=['D1','D2'],reason='same connected torso')],
        excluded={'D3':dict(kind='cast_shadow',reason='cast shadow on the ground')})
    result=reconcile_physical_partition(response,partition,3,['G0001'])
    assert len(result.bodies)==1 and result.bodies[0].identity=='G0001'
    assert result.bodies[0].members==['D1','D2']
    with pytest.raises(ValueError):reconcile_physical_partition(response,PhysicalPartition(bodies=partition.bodies),3,['G0001'])
    with pytest.raises(ValueError):reconcile_physical_partition(response,PhysicalPartition(bodies=partition.bodies,excluded={k:dict(kind='cast_shadow',reason='shadow') for k in ['D2','D3']}),3,['G0001'])


def test_physical_count_cannot_silently_choose_between_known_people():
    from catalog_repair import PhysicalPartition,reconcile_physical_partition
    response=FrameDecision(bodies=[dict(members=['D1'],identity='G0001',reason='left person'),dict(members=['D2'],identity='G0002',reason='right person')])
    partition=PhysicalPartition(bodies=[dict(members=['D1','D2'],reason='one body')])
    with pytest.raises(ValueError,match='conflicting known identities'):
        reconcile_physical_partition(response,partition,2,['G0001','G0002'])


def test_human_physical_recheck_does_not_reinterpret_separate_animals_as_limbs():
    from catalog_repair import needs_physical_count
    response=FrameDecision(bodies=[dict(members=['D1'],identity='new',reason='unidentified')])
    rows=[(('s',0,(0,0,10,10)),'G1',{}, {})]
    assert needs_physical_count(response,rows,{'G1':{'kind':'person'}})
    assert not needs_physical_count(response,rows,{'G1':{'kind':'animal'}})
