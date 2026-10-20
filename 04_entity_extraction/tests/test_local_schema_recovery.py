"""Recover malformed role/identity JSON through inference, never guessed merges."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_low_freq_event'))
from low_freq_event.backend import QwenBackend, ModelOutputError
from low_freq_event.config import Settings
from story_semantics import ShotDecision


def response(target):
    return dict(track_assignments={'T001':target},track_roles={'T001':'hand_part'},
                entities=[dict(local_id='L01',kind='person',label='subject',appearance='visible')],
                summary='observation')


def test_role_target_requires_model_correction_and_then_uses_cache(tmp_path):
    bad=response('body_part')
    with pytest.raises(ValueError,match='Allowed assignment targets'):
        ShotDecision.model_validate(bad)
    backend=QwenBackend(Settings(json_retries=1),tmp_path/'traces')
    with patch.object(backend,'_generate',side_effect=[json.dumps(bad),json.dumps(response('L01'))]) as generate:
        result=backend.request('review image',ShotDecision)
        assert result.entities[0].track_ids==['T001']
        assert result.track_roles['T001']=='body_part'
        correction=generate.call_args.args[0]
        assert 'track_assignments[T001]' in correction and 'Previous invalid JSON' in correction
        assert 'physical owner' in correction
        backend.request('review image',ShotDecision)
        assert generate.call_count==2


def test_unresolved_schema_error_is_not_silently_promoted(tmp_path):
    backend=QwenBackend(Settings(json_retries=1),tmp_path/'traces')
    with patch.object(backend,'_generate',return_value=json.dumps(response('body_part'))) as generate:
        with pytest.raises(ModelOutputError):backend.request('review image',ShotDecision)
        assert generate.call_count==2
