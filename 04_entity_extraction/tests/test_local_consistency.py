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
    assert out.relations[0].relation=='different_individual'
    assert len(src.entities)==1
    assert len(separate_conflicting_subjects(out,shot,{'x':'A','y':'B'}).entities)==2


def test_multiple_views_without_conflicting_regions_stay_one():
    src=decision({'A':'whole_subject','B':'whole_subject'})
    assert len(separate_conflicting_subjects(src,{'mask_graph':{}},{'x':'A','y':'B'}).entities)==1
