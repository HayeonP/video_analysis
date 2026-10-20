from pathlib import Path
import sys
from copy import deepcopy
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'01_low_freq_event')]
from local_identity_audit import audit_candidates,audit_local_identity,PairReview,validate_pair
from story_semantics import ShotDecision,BudgetExhausted


def fixture(tmp_path):
    from PIL import Image
    image=tmp_path/'frame.jpg';Image.new('RGB',(200,100),'gray').save(image)
    shot=dict(shot_id='arbitrary_clip',width=200,height=100,fps=2,frame_count=3,tracks=[
        dict(track_id=tid,observations=[dict(frame=f,bbox=box,confidence=.9,image=str(image),evidence='observed_detection') for f in range(3)])
        for tid,box in [('x',[0,0,70,80]),('y',[45,0,115,80])]])
    result=ShotDecision.model_validate(dict(track_assignments={'A':'L01','B':'L01'},
        track_roles={'A':'body_part','B':'whole_subject'},entities=[dict(local_id='L01',kind='person',label='person',appearance='person')],summary='person'))
    return result,shot,{'x':'A','y':'B'}


def test_candidate_is_role_independent_and_not_a_physical_split(tmp_path):
    result,shot,aliases=fixture(tmp_path)
    candidates=audit_candidates(result,shot,aliases)
    assert len(candidates)==1 and len(candidates[0]['support'])==3
    assert len(result.entities)==1
    shot['tracks'][1]['observations'][0]['confidence']=.1
    assert audit_candidates(result,shot,aliases)==[]


def test_contained_body_part_does_not_trigger(tmp_path):
    result,shot,aliases=fixture(tmp_path)
    for row in shot['tracks'][1]['observations']:row['bbox']=[10,10,20,20]
    assert audit_candidates(result,shot,aliases)==[]


class Backend:
    def __init__(self,decision):self.ledger={'generations':0};self.phase='local';self.reserve=0;self.decision=decision
    @property
    def remaining(self):return 20-self.ledger['generations']
    def request(self,prompt,schema,**kwargs):
        if self.remaining<=self.reserve:raise BudgetExhausted('capped')
        self.ledger['generations']+=1
        value=schema.model_validate(dict(decision=self.decision,left=dict(role='body_part' if self.decision=='same' else 'whole_subject',kind='person',description='first visible person'),
            right=dict(role='whole_subject',kind='person',description='second visible person'),evidence_frames=[0,1],
            count_visible_bodies=1 if self.decision=='same' else 2,
            same_basis='visibly_connected_body_parts' if self.decision=='same' else None,
            visible_connection='A visible continuous arm joins the same shoulder.' if self.decision=='same' else '',
            visible_evidence='Two visible people are separately present in the source frames.'))
        return kwargs['validate'](value)


def test_explicit_visual_difference_splits_and_saves_evidence(tmp_path):
    result,shot,aliases=fixture(tmp_path);backend=Backend('different');state={'generations':0}
    out=audit_local_identity(result,{},shot,aliases,tmp_path,backend,state)
    assert len(out.entities)==2 and out.track_assignments['A']!=out.track_assignments['B']
    assert state['generations']==1 and backend.phase=='local' and backend.reserve==0
    assert (tmp_path/'local_identity_audit/arbitrary_clip/audit.json').is_file()
    from PIL import Image
    with Image.open(tmp_path/'local_identity_audit/arbitrary_clip/pair_00_f00000000.jpg') as image:
        assert image.size==(shot['width'],shot['height'])  # One source frame, no repeated crops.
    assert len(result.entities)==1


def test_same_or_budget_limit_keeps_owner_and_pending(tmp_path):
    result,shot,aliases=fixture(tmp_path)
    out=audit_local_identity(result,{},shot,aliases,tmp_path,Backend('same'),{'generations':0})
    assert len(out.entities)==1
    shot=deepcopy(shot);backend=Backend('different')
    out=audit_local_identity(result,{},shot,aliases,tmp_path,backend,{'generations':4})
    assert len(out.entities)==2 and all(e.uncertain for e in out.entities)
    assert not any(r.relation=='different_individual' for r in out.relations)
    assert backend.ledger['generations']==0 and shot['local_identity_audit']['status']=='needs_review'


def test_different_requires_two_supplied_frames():
    r=PairReview.model_validate(dict(decision='different',left=dict(role='body_part',kind='person',description='head'),
        right=dict(role='whole_subject',kind='person',description='body'),evidence_frames=[2],visible_evidence='Visible two people.'))
    with pytest.raises(ValueError,match='two supplied'):validate_pair(r,[1,2])
    r.evidence_frames=[1,99]
    with pytest.raises(ValueError,match='supplied original'):validate_pair(r,[1,2])


def test_distinct_whole_subjects_cannot_be_claimed_from_one_visible_body():
    r=PairReview.model_validate(dict(decision='different',
        left=dict(role='whole_subject',kind='person',description='visible face'),
        right=dict(role='whole_subject',kind='person',description='hidden shoulder'),
        evidence_frames=[1,2],visible_evidence='Only one face and an obscured region are visible.',count_visible_bodies=1))
    with pytest.raises(ValueError,match='two independently visible'):validate_pair(r,[1,2])
    r.count_visible_bodies=2
    assert validate_pair(r,[1,2])==r


def test_one_reviewed_track_cannot_reclassify_unreviewed_siblings(tmp_path):
    result,shot,aliases=fixture(tmp_path)
    result.entities[0].kind='animal'
    extra=deepcopy(shot['tracks'][0]);extra['track_id']='z'
    for row in extra['observations']:row['bbox']=[175,0,185,10]
    shot['tracks'].append(extra);aliases['z']='C'
    result.track_assignments['C']='L01';result.track_roles['C']='whole_subject'
    result.entities[0].track_ids.append('C')
    out=audit_local_identity(result,{},shot,aliases,tmp_path,Backend('different'),{'generations':0})
    kinds={t:e.kind for e in out.entities for t in e.track_ids}
    assert kinds=={'A':'animal','B':'animal','C':'animal'}
    proposals=shot['local_identity_audit']['classification_proposals']
    assert {p['alias'] for p in proposals}=={'A','B'}
    assert all(p['proposed_kind']=='person' for p in proposals)


def test_same_requires_physical_basis_and_consistent_body_count():
    r=PairReview.model_validate(dict(decision='same',left=dict(role='whole_subject',kind='person',description='bald head'),
        right=dict(role='whole_subject',kind='person',description='red hair'),evidence_frames=[1,2],visible_evidence='They have similar clothing.'))
    with pytest.raises(ValueError,match='physical connection'):validate_pair(r,[1,2],[.1,.2])
    r.same_basis='duplicate_region';r.visible_connection='The regions show duplicate visible bodies.';r.count_visible_bodies=2
    with pytest.raises(ValueError,match='Two bodies'):validate_pair(r,[1,2],[.1,.2])
    r.count_visible_bodies=1
    with pytest.raises(ValueError,match='Low-overlap'):validate_pair(r,[1,2],[.1,.2])
    r.same_basis='visibly_connected_body_parts'
    with pytest.raises(ValueError,match='actual body-part'):validate_pair(r,[1,2],[.1,.2])
