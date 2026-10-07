#!/usr/bin/env python3
"""Opt-in real GPU regression checks against the original Sintel failure fixture.

Uses a frozen v1 failure fixture and stage-03 adaptive input videos.
Outputs are isolated from production caches.
"""
import argparse
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent/'01_low_freq_event'), str(ROOT.parent/'03_adaptive_sampling')]
from low_freq_event.config import Settings
from low_freq_event.storage import atomic_json
from identity_review import ReviewBackend
from context_identity import adjudicate_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, default=ROOT/'tests/fixtures/sintel_context')
    parser.add_argument('--input', type=Path, default=ROOT/'inputs/sintel_adaptive')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    import torch
    torch.set_num_threads(4)
    import av
    import shutil
    fixture=json.loads((args.fixture/'fixture.json').read_text())
    groups,frames,history=fixture['groups'],fixture['frames'],fixture['history']
    source=args.output/'prepared_input';source.mkdir(exist_ok=True)
    shutil.copytree(args.fixture/'crops',source/'crops',dirs_exist_ok=True)
    (source/'frames').mkdir(exist_ok=True)
    summary=json.loads((args.input/'summary.json').read_text())
    for row in summary['videos']:
        meta_path=args.input/row['samples']
        metadata=json.loads(meta_path.read_text())
        shot_frames=[f for f in frames if f['shot_id']==row['id']]
        with av.open(str(meta_path.parent/metadata['output_video'])) as container:
            count=0
            for frame,record in zip(container.decode(video=0),shot_frames,strict=True):
                frame.to_image().convert('RGB').save(source/record['image'],quality=96)
                count+=1
            assert count==len(shot_frames)
    backend = ReviewBackend(Settings(perception_device='auto', max_pixels=196608), args.output/'review_traces')
    # Fixture IDs identify previously diagnosed observations, never runtime merge rules.
    cases = [('early_to_red_hair','S003_002','S005_001','same'),
             ('red_hair_to_middle','S005_001','S011_001','same'),
             ('red_hair_to_late','S005_001','S015_001','same'),
             ('older_man_vs_woman','S004_001','S005_001','different')]
    rows=[]
    for name,a,b,expected in cases:
        dest=args.output/name;dest.mkdir(exist_ok=True)
        start=time.perf_counter()
        decision=adjudicate_pair(a,b,groups,frames,history,source,backend,dest,24,
                                dict(forward_same=expected=='same',reverse_same=False))
        rows.append(dict(case=name,expected=expected,actual=decision.decision,
                         passed=decision.decision==expected,elapsed_sec=time.perf_counter()-start,
                         result=decision.model_dump()))
        atomic_json(args.output/'results.json',rows)
        print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
    raise SystemExit(0 if all(r['passed'] for r in rows) else 1)


if __name__=='__main__':main()
