"""Chronological, grounded Korean shot memory shared with identity matching."""
import re
from typing import Literal
from pydantic import BaseModel

VERSION=1

class MemoryShot(BaseModel):
    shot_id:str
    summary:str
    transition:Literal['continuous','scene_change','montage','unknown']
    transition_evidence:str

def is_korean(text):
    # IDs, numeric values and quoted on-screen titles do not define prose language.
    cleaned=re.sub(r'\b(?:G\d+|[ITC]\d+|shot_\d+(?::\w+)?)\b','',text)
    cleaned=re.sub(r'"[^"\n]*"|\'[^\'\n]*\'|“[^”\n]*”|「[^」\n]*」' ,'',cleaned)
    return bool(re.search('[가-힣]',cleaned)) and not re.search(r'[A-Za-z]{3,}',cleaned)


def korean_catalog_narrative(bundle,local,catalog,memory):
    """Publish verified Korean memory with final IDs; do not reuse English fallback."""
    memory_shots={s['shot_id']:s for s in memory['shots']};shotmap={s['shot_id']:s for s in bundle['shots']}
    rows=[]
    for shot in local['shots']:
        sid=shot['shot_id'];m=memory_shots[sid];summary=m['summary']
        shot.setdefault('source_summary',shot['summary'])
        shot['summary']=summary
        if not is_korean(summary):raise ValueError('Korean summary invariant failed: '+sid)
        start=shotmap[sid]['full_start_sec'];duration=shotmap[sid]['frame_count']/float(shotmap[sid]['fps'])
        rows.append(dict(shot_id=sid,summary=summary,initial_summary=shot['summary'],events=[],
            entities=list(dict.fromkeys(catalog['mapping'][e['local_id']] for e in shot['entities'])),
            start_sec=start,end_sec=start+duration,reviewed=True,transition=m['transition'],
            transition_evidence=m['transition_evidence'],summary_scope='observed_scene_memory',
            correction_reason='Korean scene memory with final observation ID inventory; no unverified event timestamps'))
    catalog['shots']=rows;catalog['narrative_source']='chronological_korean_memory';catalog['narrative_failures']=[]
    return catalog
