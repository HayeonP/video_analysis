#!/usr/bin/env python3
"""Evaluation only: measure retained links and false merges at frozen source boxes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from evaluate_identity_spots import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True,type=Path);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--comparison',type=Path);p.add_argument('--labels',required=True,type=Path);p.add_argument('--observations',required=True,type=Path)
    a=p.parse_args();read=lambda path:json.loads(path.read_text())
    names={key:g['name'] for g in read(a.labels)['groups'] for key in g['ids']}
    annotations=[]
    for row in read(a.observations)['observations']:
        if row['entity_id'] not in names:continue
        w,h=row['source_size'];box=row.get('bbox_pixels') or [v*d/1000 for v,d in zip(row['bbox'],[w,h,w,h])]
        annotations.append(dict(id=row['detection_id'],identity=names[row['entity_id']],shot_id=row['shot_id'],frame=row['source_frame'],bbox=box))
    results={name:evaluate(path,annotations) for name,path in [('baseline',a.baseline),('repaired',a.run),*([('comparison',a.comparison)] if a.comparison else [])]}
    summary={name:dict(Counter(p['outcome'] for p in r['pairs'])) for name,r in results.items()}
    old={(p['left'],p['right']) for p in results['baseline']['pairs'] if p['outcome']=='correct_same'}
    new={(p['left'],p['right']) for p in results['repaired']['pairs'] if p['outcome']=='correct_same'}
    lost=sorted(old-new)
    gate=dict(passed=not lost and not summary['repaired'].get('false_merge') and not summary['repaired'].get('unmatched_or_ambiguous'),lost_previously_correct_links=lost,
              note='Gate preserves previously correct annotated associations, not every possible identity in the video.')
    report=dict(scope='Frozen partial source-box evaluation; pair counts include repeated appearances within shots. No labels enter inference.',
        annotation_count=len(annotations),annotation_shots=len({r['shot_id'] for r in annotations}),gate=gate,summary=summary,results=results,
        label_sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.labels,a.observations)})
    (a.run/'cross_shot_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(dict(gate=gate,summary=summary),ensure_ascii=False,indent=2))
    return 0 if gate['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
