"""Timing must remain auditable on both completion and subprocess failure."""
import json
import os
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import benchmark_video as bench


def test_stage_failure_keeps_elapsed_and_child_log(tmp_path):
    report=dict(status='running',input='test',started_at=bench.utc_now(),stages=[])
    with pytest.raises(RuntimeError,match='exit 7'):
        bench.run_stage('failed_stage',[sys.executable,'-c','print("diagnostic",flush=True);raise SystemExit(7)'],tmp_path,report,os.environ.copy())
    saved=json.loads((tmp_path/'benchmark_timing.json').read_text())
    row=saved['stages'][0]
    assert row['status']=='failed' and row['returncode']==7 and row['elapsed_sec']>0
    assert 'diagnostic' in (tmp_path/'failed_stage.log').read_text()


def test_fresh_run_rejects_existing_output_before_model_import(tmp_path):
    video=tmp_path/'input.mp4';video.write_bytes(b'test')
    with pytest.raises(FileExistsError):bench.preflight(video,tmp_path,10)


def test_benchmark_plan_is_explicitly_sampled(tmp_path):
    cmd=bench.commands(tmp_path/'input.mp4',tmp_path/'run',320,96)[-1][1]
    assert cmd[cmd.index('--analysis-frames')+1]=='sampled'
    assert '--skip-mask-refinement' in cmd


def test_resume_preserves_stages_and_rejects_changed_input_or_budget(tmp_path):
    video=tmp_path/'input.mp4';video.write_bytes(b'original')
    report=dict(status='failed',input=str(video),input_sha256=bench.digest(video),
                policy=dict(analysis_frames='sampled',qwen_limit=320,global_review_limit=96),
                stages=[dict(name='02_shots',status='complete'),dict(name='03_sampling',status='complete'),
                        dict(name='04_entities',status='failed')],elapsed_sec=100)
    (tmp_path/'benchmark_timing.json').write_text(json.dumps(report))
    assert bench.resume_report(tmp_path,video,320,96)==report
    for relative in ('shots/shots.json','adaptive/summary.json'):
        path=tmp_path/relative;path.parent.mkdir();path.write_text('{}')
    plan=bench.resume_plan(bench.commands(video,tmp_path,320,96),report,tmp_path)
    assert [name for name,_ in plan]==['04_entities']
    assert len(report['stages'])==3 and report['elapsed_sec']==100
    with pytest.raises(ValueError,match='예산'):bench.resume_report(tmp_path,video,321,96)
    video.write_bytes(b'changed')
    with pytest.raises(ValueError,match='입력'):bench.resume_report(tmp_path,video,320,96)
    (tmp_path/'adaptive/summary.json').unlink()
    with pytest.raises(ValueError,match='중간 결과'):bench.resume_plan(plan,report,tmp_path)


def test_resumed_stage_retains_original_log_and_timing(tmp_path):
    (tmp_path/'04_entities.log').write_text('previous failure')
    report=dict(status='running',input='test',started_at=bench.utc_now(),resumed=True,
                stages=[dict(name='04_entities',status='failed',elapsed_sec=12)])
    bench.run_stage('04_entities',[sys.executable,'-c','print("resumed")'],tmp_path,report,os.environ.copy())
    assert (tmp_path/'04_entities.log').read_text()=='previous failure'
    assert report['stages'][0]['elapsed_sec']==12
    assert report['stages'][1]['status']=='complete'
    assert 'resumed' in (tmp_path/report['stages'][1]['log']).read_text()
    assert '중단 실행 이어하기' in (tmp_path/'benchmark_summary.txt').read_text()


def test_resume_shortcut_selects_only_matching_failed_video(tmp_path,monkeypatch):
    import video_shortcuts as shortcuts
    monkeypatch.setattr(shortcuts,'ROOT',tmp_path)
    monkeypatch.setattr(shortcuts,'VIDEOS',{'sintel':tmp_path/'sintel.mp4','frozen':tmp_path/'frozen.mp4'})
    cases=[('old','sintel','failed','1','sampled'),('selected','sintel','interrupted','2','sampled'),
           ('other','frozen','failed','9','sampled'),('done','sintel','complete','9','sampled'),
           ('active','sintel','running','9','sampled'),('dense','sintel','failed','9','dense')]
    for folder,movie,status,ended,mode in cases:
        path=tmp_path/'outputs'/folder;path.mkdir(parents=True)
        (path/'benchmark_timing.json').write_text(json.dumps(dict(input=str(shortcuts.VIDEOS[movie]),
            status=status,ended_at=ended,policy=dict(analysis_frames=mode))))
    assert shortcuts.latest_failed_result('sintel')==tmp_path/'outputs/selected'
    with pytest.raises(ValueError,match='실패/중단'):
        monkeypatch.setattr(shortcuts,'VIDEOS',{'none':tmp_path/'none.mp4'})
        shortcuts.latest_failed_result('none')


def test_benchmark_lock_prevents_two_resumes_overwriting_history(tmp_path):
    with bench.acquire_run_lock(tmp_path):
        with pytest.raises(ValueError,match='이미 실행 중'):
            bench.acquire_run_lock(tmp_path)
    with bench.acquire_run_lock(tmp_path):pass
