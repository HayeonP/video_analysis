"""Evaluate manually annotated shadows and duplicate body crops, never infer IDs."""
import argparse
import hashlib
import json
from pathlib import Path


def evaluate(root, labels):
    read=lambda name:json.loads((root/name).read_text())
    bundle=read('reviewed_tracking.json');local=read('local_entities.json');catalog=read('global_entities.json')
    owners={tid:catalog['mapping'][e['local_id']] for s in local['shots'] for e in s['entities'] for tid in e['track_ids']}
    rejected={tid for s in local['shots'] for tid in s.get('rejected_tracks',[])}
    results=[]
    for case in json.loads(labels.read_text())['cases']:
        shot=next(s for s in bundle['shots'] if s['shot_id']==case['shot_id'])
        matches=[]
        for target in case['observations']:
            tids={t['track_id'] for t in shot['tracks'] for o in t['observations']
                  if o['frame']==case['frame'] and o['bbox']==target['bbox']}
            matches.append(tids)
        all_tracks=set().union(*matches)
        ids={owners[tid] for tid in all_tracks if tid in owners}
        if case['kind']=='excluded':
            passed=all(matches) and all_tracks<=rejected and not ids
            accounting='excluded' if passed else 'independent_or_unreviewed'
        elif case['kind']=='no_independent_shadow':
            reference=case['owner_bbox']
            owner_ids={owners[t['track_id']] for t in shot['tracks'] if t['track_id'] in owners
                       for o in t['observations'] if o['frame']==case['frame'] and o['bbox']==reference}
            discarded=all_tracks<=rejected and not ids
            attributed=all_tracks<=owners.keys() and len(owner_ids)==1 and ids==owner_ids
            passed=all(matches) and (discarded or attributed)
            accounting='excluded' if discarded else 'retained_with_owner' if attributed else 'independent_or_unreviewed'
        elif case['kind']=='same_body':
            expected={tuple(o['bbox']) for o in case['observations']}
            proof=any(p['shot_id']==case['shot_id'] and p['frame']==case['frame']
                      and expected<={tuple(m['bbox']) for m in p['members']}
                      for p in catalog.get('physical_groups',[]))
            passed=all(matches) and all_tracks<=owners.keys() and len(ids)==1 and proof
            accounting='consolidated_body' if passed else 'separate_or_unreviewed'
        else:raise ValueError('Unknown physical regression kind')
        results.append(dict(case=case,passed=bool(passed),track_ids=sorted(all_tracks),global_ids=sorted(ids),accounting=accounting))
    return dict(passed=all(r['passed'] for r in results),cases=results,
                artifacts={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                           for name in ['reviewed_tracking.json','local_entities.json','global_entities.json']},
                labels=dict(path=str(labels.resolve()),sha256=hashlib.sha256(labels.read_bytes()).hexdigest()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--labels',type=Path,required=True)
    args=parser.parse_args();result=evaluate(args.run,args.labels)
    (args.run/'physical_spot_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
