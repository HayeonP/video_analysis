from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'01_low_freq_event')]
from local_classification_audit import (ClassificationReview,classification_candidates,
    apply_classification,audit_local_classification,validate_classification)
from story_semantics import ShotDecision,BudgetExhausted,validate_local


def fixture(tmp_path):
    from PIL import Image
    image=tmp_path/'frame.jpg';Image.new('RGB',(200,100),'gray').save(image)
    shot=dict(shot_id='unseen_film',width=200,height=100,fps=2,frame_count=8,tracks=[
        dict(track_id=tid,category='animal',category_evidence={'bird':999},observations=[dict(frame=f,
            bbox=box,confidence=.8,category=category,image=str(image),evidence='observed_detection') for f in range(8)])
        for tid,box,category in [('x',[0,0,70,80],'person'),('y',[80,0,160,80],'bird')]])
    result=ShotDecision.model_validate(dict(track_assignments={'A':'L01','B':'L01'},
        track_roles={'A':'body_part','B':'whole_subject'},entities=[dict(local_id='L01',kind='animal',label='animal',appearance='animal')],summary='animal'))
    return result,shot,{'x':'A','y':'B'}


def review(kind='person'):
    return ClassificationReview(kind=kind,role='whole_subject',description='Visible human face and torso',
        evidence_frames=[2,4,6],visible_evidence='A human face and clothed torso are directly visible in the target box.')


def test_segment_votes_only_propose_review_without_changing_identity(tmp_path):
    result,shot,aliases=fixture(tmp_path)
    rows=classification_candidates(result,shot,aliases)
    assert len(rows)==1 and rows[0]['alias']=='A' and result.entities[0].kind=='animal'
    for row in shot['tracks'][0]['observations']:row['confidence']=.3
    assert not classification_candidates(result,shot,aliases)


def test_visual_correction_splits_only_reviewed_track_and_is_idempotent(tmp_path):
    result,shot,aliases=fixture(tmp_path)
    corrected=apply_classification(result,'A',review())
    assert corrected.track_assignments['A']!=corrected.track_assignments['B']
    assert {t:e.kind for e in corrected.entities for t in e.track_ids}=={'A':'person','B':'animal'}
    assert validate_local(corrected,shot,aliases)
    assert apply_classification(corrected,'A',review())==corrected
    assert result.entities[0].track_ids==['A','B']
    # Detector disagreement never overrules a source-image animal decision.
    confirmed=apply_classification(result,'A',review('animal'))
    assert len(confirmed.entities)==1 and confirmed.entities[0].kind=='animal'


def test_animal_track_inside_a_person_group_is_reviewed_without_detector_override(tmp_path):
    result,shot,aliases=fixture(tmp_path)
    result.entities[0].kind='person'
    candidates=classification_candidates(result,shot,aliases)
    assert [c['alias'] for c in candidates]==['B']
    assert candidates[0]['detector_kind']=='animal'
    assert result.entities[0].kind=='person'
    corrected=apply_classification(result,'B',review('animal'))
    assert {t:e.kind for e in corrected.entities for t in e.track_ids}=={'A':'person','B':'animal'}
    assert validate_local(corrected,shot,aliases)
    # Bird-like detector shape alone cannot change an actual human classification.
    retained=apply_classification(result,'B',review('person'))
    assert len(retained.entities)==1 and retained.entities[0].kind=='person'


def test_paired_category_change_requires_independent_single_target_review(tmp_path):
    result,shot,aliases=fixture(tmp_path)
    shot['local_identity_audit']=dict(classification_proposals=[dict(alias='B',proposed_kind='person')])
    candidates=classification_candidates(result,shot,aliases)
    proposal=next(c for c in candidates if c['alias']=='B')
    assert proposal['trigger']=='paired_review_disagreement'
    assert result.entities[0].kind=='animal'


class Backend:
    def __init__(self):self.ledger={'generations':0};self.phase='local';self.reserve=2
    @property
    def remaining(self):return 20-self.ledger['generations']
    def request(self,prompt,schema,**kwargs):
        if self.remaining<=self.reserve:raise BudgetExhausted('capped')
        self.ledger['generations']+=1
        assert 'previous_kind' not in prompt
        assert len(kwargs['images'])==3
        return kwargs['validate'](review())


def test_bounded_source_review_provenance_and_pending_budget(tmp_path):
    result,shot,aliases=fixture(tmp_path);backend=Backend();state={'generations':0}
    corrected=audit_local_classification(result,{},shot,aliases,tmp_path,backend,state)
    assert state['generations']==1 and backend.reserve==2 and backend.phase=='local'
    assert len(corrected.entities)==2
    assert (tmp_path/'local_classification_audit/unseen_film/audit.json').is_file()
    backend=Backend()
    original=audit_local_classification(result,{},shot,aliases,tmp_path,backend,{'generations':4})
    assert backend.ledger['generations']==0 and original==result
    assert shot['local_classification_audit']['status']=='needs_review'


def test_review_requires_real_frame_citations():
    r=review();r.evidence_frames=[99,100]
    with pytest.raises(ValueError,match='supplied source'):validate_classification(r,[2,4,6])
    r.evidence_frames=[2]
    with pytest.raises(ValueError,match='two supplied'):validate_classification(r,[2,4,6])


def test_phrase_list_format_repair_retains_evidence_and_rejects_invented_coercions():
    raw=review().model_dump();raw['visible_evidence']=['Visible human face','Red hair and shirt']
    out=ClassificationReview.model_validate(raw)
    assert out.visible_evidence=='Visible human face; Red hair and shirt'
    assert out.kind=='person' and out.evidence_frames==[2,4,6]
    for value in ([],[''],['visible person',7],[{'body':'person'}]):
        raw['visible_evidence']=value
        with pytest.raises(ValueError):ClassificationReview.model_validate(raw)
