"""Evaluation only: compare saved identities at manually inspected source boxes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shot_tracking import box_iou


def read(path):return json.loads(path.read_text())


def evaluate(root,annotations):
    bundle=read(root/'reviewed_tracking.json');local=read(root/'local_entities.json');catalog=read(root/'global_entities.json')
    groups={e['entity_id']:e for e in catalog['entities']}
    owners={tid:catalog['mapping'][e['local_id']] for s in local['shots'] for e in s['entities'] for tid in e['track_ids']}
    points=[]
    for target in annotations:
        shot=next(s for s in bundle['shots'] if s['shot_id']==target['shot_id'])
        scores=[(box_iou(o['bbox'],target['bbox']),owners[t['track_id']]) for t in shot['tracks'] if t['track_id'] in owners for o in t['observations'] if o['frame']==target['frame']]
        best=max((s for s,_ in scores),default=0)
        ids=sorted({eid for score,eid in scores if score>=max(.5,best-1e-6)})
        points.append(dict(id=target['id'],identity=target['identity'],global_ids=ids,
            labels=[groups[eid]['label'] for eid in ids],iou=best))
    pairs=[]
    for i,a in enumerate(points):
        for b in points[i+1:]:
            if len(a['global_ids'])!=1 or len(b['global_ids'])!=1:outcome='unmatched_or_ambiguous'
            elif a['identity']==b['identity']:
                outcome='correct_same' if a['global_ids']==b['global_ids'] else 'fragmented'
            else:outcome='false_merge' if a['global_ids']==b['global_ids'] else 'correct_separate'
            pairs.append(dict(left=a['id'],right=b['id'],outcome=outcome))
    artifacts={name:dict(path=str((root/name).resolve()),sha256=hashlib.sha256((root/name).read_bytes()).hexdigest()) for name in ('reviewed_tracking.json','local_entities.json','global_entities.json')}
    return dict(points=points,pairs=pairs,artifacts=artifacts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--labels',type=Path,required=True)
    args=parser.parse_args();annotations=read(args.labels)
    result=dict(scope='Partial source-box spot check only; not whole-video accuracy.',
        labels=dict(path=str(args.labels.resolve()),sha256=hashlib.sha256(args.labels.read_bytes()).hexdigest()),
        before=evaluate(args.reference,annotations['observations']),after=evaluate(args.run,annotations['observations']))
    (args.run/'identity_spot_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({key:{name:r[name] for name in ('points','pairs')} for key,r in result.items() if key in ('before','after')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
