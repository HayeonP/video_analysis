from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'01_low_freq_event')]
from story_semantics import ShotDecision,reconcile_handled_objects
from local_consistency import separate_conflicting_subjects


def decision(roles):
    return ShotDecision.model_validate(dict(
        track_assignments={'A':'L01','B':'L01'},track_roles=roles,
        track_descriptions={'A':'인물의 몸','B':'별도 관측 대상'},
        entities=[dict(local_id='L01',kind='person',label='인물',appearance='외형')],summary='L01 등장'))


def test_held_object_separated_from_owner_using_explicit_role():
    src=decision({'A':'whole_subject','B':'handled_object'})
    out=reconcile_handled_objects(src)
    assert len(out.entities)==2
    assert [(e.kind,e.track_ids) for e in out.entities]==[('person',['A']),('object',['B'])]
    assert out.relations[0].relation=='held_by'
    assert src.entities[0].track_ids==['A','B']
    assert len(reconcile_handled_objects(out).entities)==2


def test_body_parts_are_not_split_by_region_separation():
    src=decision({'A':'whole_subject','B':'body_part'})
    shot={'mask_graph':{'separation_frame_evidence':[{'left':'x','right':'y','frames':[0,1,2]}]}}
    out=separate_conflicting_subjects(src,shot,{'x':'A','y':'B'})
    assert len(out.entities)==1


def test_repeated_separation_and_two_whole_roles_required():
    src=decision({'A':'whole_subject','B':'whole_subject'})
    shot={'mask_graph':{'separation_frame_evidence':[{'left':'x','right':'y','frames':[0]}]}}
    assert len(separate_conflicting_subjects(src,shot,{'x':'A','y':'B'}).entities)==1
    shot['mask_graph']['separation_frame_evidence'][0]['frames']=[0,1]
    out=separate_conflicting_subjects(src,shot,{'x':'A','y':'B'})
    assert len(out.entities)==2
    assert out.track_assignments['A']!=out.track_assignments['B']
    assert out.relations[0].relation=='uncertain'
    assert len(src.entities)==1
    assert len(separate_conflicting_subjects(out,shot,{'x':'A','y':'B'}).entities)==2


def test_multiple_views_without_conflicting_regions_stay_one():
    src=decision({'A':'whole_subject','B':'whole_subject'})
    assert len(separate_conflicting_subjects(src,{'mask_graph':{}},{'x':'A','y':'B'}).entities)==1


def detector_shot(frames, *, nested=False):
    return dict(tracks=[dict(track_id=tid, observations=[dict(frame=f, bbox=box,
        confidence=.8, tracker_confirmed=True, evidence='observed_detection') for f in range(frames)])
        for tid, box in [('x',[0,0,40,80]), ('y',[10,10,20,30] if nested else [90,0,130,80])]])


def test_no_mask_repeated_whole_person_boxes_separate_idempotently():
    src=decision({'A':'whole_subject','B':'whole_subject'})
    shot=detector_shot(3)
    out=separate_conflicting_subjects(src,shot,{'x':'A','y':'B'})
    assert len(out.entities)==2
    assert out.track_assignments['A']!=out.track_assignments['B']
    assert len(out.relations)==1
    assert out.relations[0].relation=='uncertain'
    assert len(separate_conflicting_subjects(out,shot,{'x':'A','y':'B'}).relations)==1


def test_no_mask_body_part_or_containment_keeps_owner():
    src=decision({'A':'whole_subject','B':'body_part'})
    assert len(separate_conflicting_subjects(src,detector_shot(4),{'x':'A','y':'B'}).entities)==1
    src=decision({'A':'whole_subject','B':'whole_subject'})
    assert len(separate_conflicting_subjects(src,detector_shot(4,nested=True),{'x':'A','y':'B'}).entities)==1


def test_no_mask_one_frame_or_weak_continuations_do_not_split():
    src=decision({'A':'whole_subject','B':'whole_subject'})
    assert len(separate_conflicting_subjects(src,detector_shot(1),{'x':'A','y':'B'}).entities)==1
    shot=detector_shot(4)
    for row in shot['tracks'][1]['observations']:
        row['confidence']=.1
    assert len(separate_conflicting_subjects(src,shot,{'x':'A','y':'B'}).entities)==1


def test_legacy_geometry_cache_veto_is_downgraded_but_visual_difference_survives():
    from story_semantics import TrackRelation
    src=decision({'A':'whole_subject','B':'whole_subject'})
    src.relations=[TrackRelation(left='A',right='B',relation='different_individual',
        reason='VLM whole-subject roles plus repeated simultaneous separation: at least two masks or three observed detector boxes')]
    out=separate_conflicting_subjects(src,detector_shot(3),{'x':'A','y':'B'})
    assert len(out.entities)==2 and all(r.relation!='different_individual' for r in out.relations)
    assert len(separate_conflicting_subjects(out,detector_shot(3),{'x':'A','y':'B'}).relations)==1
    src.relations[0].reason='Focused visual audit v3 of original frames [0,1]: two separate visible bodies'
    out=separate_conflicting_subjects(src,detector_shot(3),{'x':'A','y':'B'})
    assert any(r.relation=='different_individual' for r in out.relations)
