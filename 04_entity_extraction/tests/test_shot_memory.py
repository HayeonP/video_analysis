import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[2]/'01_low_freq_event')]
import pytest
from shot_memory import is_korean,korean_catalog_narrative

@pytest.mark.parametrize('text',["붉은 머리 여성이 걷는다.","검은 화면에 'COMING SOON' 문구가 나타난다.",'G0001이 "Sintel" 글자를 바라본다.'])
def test_korean_with_ids_and_quoted_screen_text(text):assert is_korean(text)

@pytest.mark.parametrize('text',['A woman walks in the desert.','붉은 머리 woman is walking.','G0001'])
def test_english_fallback_is_not_korean(text):assert not is_korean(text)


def test_korean_final_summary_keeps_raw_diagnostic_separate():
    bundle={'shots':[{'shot_id':'s','full_start_sec':2,'frame_count':24,'fps':24}]}
    local={'shots':[{'shot_id':'s','summary':'A man runs.','entities':[{'local_id':'l'}]}]}
    catalog={'mapping':{'l':'G0001'}}
    memory={'shots':[dict(shot_id='s',summary='인물이 달린다.',transition='continuous',transition_evidence='같은 동작이 이어진다.')]}
    result=korean_catalog_narrative(bundle,local,catalog,memory)
    assert result['shots'][0]['summary']==result['shots'][0]['initial_summary']=='인물이 달린다.'
    assert local['shots'][0]['source_summary']=='A man runs.'
    assert result['shots'][0]['entities']==['G0001']
    assert not result['shots'][0]['events'] # No invented event timings.


def test_single_root_brace_repair_is_explicit_and_changes_no_values():
    from low_freq_event.backend import parse_json_response
    text='{"F001":{"assignments":{"D1":"G0001"},"evidence":{"D1":"같은 몸"}}'
    with pytest.raises(ValueError):parse_json_response(text)
    log=[];value=parse_json_response(text,repair_root_closure=True,repair_log=log)
    assert value['F001']['assignments']=={'D1':'G0001'} and len(log)==1
    for truncated in ['{"value":1','{"F001":{"evidence":{"D1":"unfinished','{"F001":{"evidence":']:
        with pytest.raises(ValueError):parse_json_response(truncated,repair_root_closure=True)


def test_repaired_syntax_does_not_invent_missing_evidence():
    from low_freq_event.backend import parse_json_response
    from batched_identity import BatchFrames
    value=parse_json_response('{"F001":{"assignments":{"D1":"G0001"}}',repair_root_closure=True)
    with pytest.raises(ValueError):BatchFrames.model_validate(value)
