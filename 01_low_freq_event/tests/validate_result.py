"""Check saved real-run cross references and time accounting; does not call a model."""
import json
import math
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
base=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else root/'outputs/sintel_10s'
data=json.loads((base/'timeline.json').read_text())
assert data['status']=='complete',data['status']
duration=data['video']['duration_sec']
settings=data['video']['settings']
assert duration>0
expected_windows=max(1,math.ceil((duration-settings['window_sec'])/settings['stride_sec']-1e-9)+1)
assert len(data['windows'])==expected_windows
entities={e['entity_id']:e for e in data['entities']}
assert len(data['events'])==sum(len(w['events']) for w in data['windows'])
assert len({e['event_id'] for e in data['events']})==len(data['events'])
observations={o['observation_id']:o for e in entities.values() for o in e['observations']}
shared_person_frames={}
group=lambda k:'subject' if k in {'person','character','animal'} else k
for entity in entities.values():
 assert {group(o['kind']) for o in entity['observations']}=={group(entity['kind'])},('incompatible kinds merged',entity['entity_id'])
assert len(observations)==sum(len(e['observations']) for e in entities.values())
for index,window in enumerate(data['windows']):
 assert abs(window['start_sec']-index*settings['stride_sec'])<1e-6
 assert abs(window['end_sec']-min(duration,window['start_sec']+settings['window_sec']))<1e-6
 assert len(window['frames'])>0
 if settings['sample_fps']<=data['video']['source_fps']:
  assert len(window['frames'])==math.ceil((window['end_sec']-window['start_sec'])*settings['sample_fps']-1e-9)
 assert len(window['observations'])==sum(len(f['boxes']) for f in window['frames'])
 for frame in window['frames']:
  ids=[o['entity_id'] for o in window['observations'] if o['frame_index']==frame['frame_index']]
  assert len(ids)==len(set(ids)),('co-visible entities merged',window['window_id'],frame['frame_index'])
  assert (base/frame['image']).is_file()
 for obs in window['observations']:
  assert obs['entity_id'] in entities
  assert observations[obs['observation_id']]['entity_id']==obs['entity_id']
  assert (base/obs['crop']).is_file()
  if settings.get('perception')=='person_reid' and obs['kind']=='person':
   assert obs['grounding_source']=='YOLO11s'
   assert obs['source_size']==[data['video']['width'],data['video']['height']]
   key=(obs['frame'],obs['detection_id'])
   value=(obs['entity_id'],tuple(obs['bbox']))
   assert shared_person_frames.setdefault(key,value)==value,('same source detection changed across windows',key)
 for event in window['events']:
  assert window['start_sec']<=event['start_sec']<=event['end_sec']<=window['end_sec']
  assert abs(event['duration_sec']-(event['end_sec']-event['start_sec']))<2e-6
  assert event['frame_indices']==list(range(event['start_frame'],event['end_frame']+1))
  assert len(event['evidence'])==len(event['frame_indices'])
  assert set(event['entity_ids'])<=entities.keys()
  if any(window['frames'][i].get('warnings') for i in event['frame_indices']):
   assert event['uncertain']
  for participant in event['participants']:
   assert set(participant['entity_ids'])<=set(event['entity_ids'])
   for oid in participant['observation_ids']:
    assert oid in observations
    assert observations[oid]['entity_id'] in participant['entity_ids']
    if observations[oid].get('identity_uncertain'):
     assert event['uncertain']
  for frame in event['evidence']:
   assert (base/frame['image']).is_file()
timing=json.loads((base/'timing.json').read_text())
assert timing['sessions'][-1]['status']=='complete'
assert abs(timing['total_active_sec']-sum(s['elapsed_sec'] for s in timing['sessions']))<.002
assert data['timing']['total_active_sec']==timing['total_active_sec']
print(json.dumps({'status':'passed','windows':len(data['windows']),'events':len(data['events']),
 'entities':len(entities),'bbox_observations':len(observations),'aliases':len(data['entity_aliases']),
 'frames_with_detection_warnings':sum(bool(f.get('warnings')) for w in data['windows'] for f in w['frames']),
 'total_active_sec':timing['total_active_sec'],'sessions':len(timing['sessions'])},ensure_ascii=False,indent=2))
