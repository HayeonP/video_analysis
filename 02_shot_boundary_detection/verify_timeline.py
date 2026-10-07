import sys,json,csv,subprocess,hashlib,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import numpy as np
parser=argparse.ArgumentParser(description='원본과 분할 MP4를 모두 디코딩해 프레임·시간축 누락을 검사합니다.')
parser.add_argument('result',type=Path)
root=parser.parse_args().result.expanduser().resolve()
data=json.loads((root/'shots.json').read_text());source=Path(data['source'])
def probe(path):
 return json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(path)],text=True))
def decode(path):
 raw=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-threads','2','-i',str(path),'-map','0:v:0','-vf','scale=160:90','-pix_fmt','rgb24','-vsync','0','-f','rawvideo','-'])
 return np.frombuffer(raw,np.uint8).reshape(-1,90,160,3)
def inspect_source():return probe(source),decode(source)
def inspect_shot(row):
 path=root/row['file'];return row,probe(path),decode(path)
with ThreadPoolExecutor(max_workers=2) as pool:
 original_future=pool.submit(inspect_source)
 futures=[pool.submit(inspect_shot,row) for row in data['shots']]
 original_info,original=original_future.result()
 source_video=next(s for s in original_info['streams'] if s['codec_type']=='video')
 fps=Fraction(source_video['avg_frame_rate']);duration=Fraction(len(original),1)/fps
 coverage=np.zeros(len(original),np.int16);details=[];decoded_total=0
 for future in futures:
  row,info,frames=future.result();video=next(s for s in info['streams'] if s['codec_type']=='video')
  start,end=row['start_frame'],row['end_frame_exclusive'];coverage[start:end]+=1
  expected=original[start:end];actual=len(frames);same_count=actual==len(expected)
  comparison={}
  if same_count:
   errors=np.abs(frames.astype(np.float32)-expected.astype(np.float32)).mean(axis=(1,2,3))
   comparison=dict(mean_pixel_error_0_to_255=float(errors.mean()),max_frame_mean_pixel_error=float(errors.max()))
  details.append(dict(shot_id=row['shot_id'],file=row['file'],source_start_frame=start,source_end_frame_exclusive=end,
    source_start_sec=float(Fraction(start,1)/fps),source_end_sec=float(Fraction(end,1)/fps),
    expected_frames=end-start,decoded_frames=actual,ffprobe_frames=int(video['nb_read_frames']),frame_count_matches=same_count,
    avg_frame_rate=video['avg_frame_rate'],video_start_time=video.get('start_time'),video_duration_sec=float(video['duration']),
    container_duration_sec=float(info['format']['duration']),audio_present=any(s['codec_type']=='audio' for s in info['streams']),**comparison))
  decoded_total+=actual
edges=np.diff(np.r_[False,coverage==0,False].astype(np.int8));gaps=[]
for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
 gaps.append(dict(start_frame=int(start),end_frame_exclusive=int(end),frame_count=int(end-start),
   start_sec=float(Fraction(int(start),1)/fps),end_sec=float(Fraction(int(end),1)/fps),duration_sec=float(Fraction(int(end-start),1)/fps)))
missing=sum(g['frame_count'] for g in gaps)
recorded={(r['start_frame'],r['end_frame_exclusive']) for r in data.get('excluded_ranges',[])}
actual_gaps={(g['start_frame'],g['end_frame_exclusive']) for g in gaps}
summary=dict(source=str(source),source_sha256_matches=hashlib.sha256(source.read_bytes()).hexdigest()==data['source_sha256'],
 source_fps=str(fps),source_decoded_frames=len(original),source_ffprobe_frames=int(source_video['nb_read_frames']),source_video_duration_sec=float(duration),
 clip_count=len(details),clips_decoded_frames=decoded_total,clips_video_duration_sec=float(Fraction(decoded_total,1)/fps),
 missing_frames=missing,missing_duration_sec=float(Fraction(missing,1)/fps),missing_percent=missing/len(original)*100,
 gap_count=len(gaps),overlapping_frames=int(np.count_nonzero(coverage>1)),
 all_clips_match_recorded_frame_counts=all(r['frame_count_matches'] and r['ffprobe_frames']==r['decoded_frames'] for r in details),
 all_clips_match_source_frame_rate=all(Fraction(r['avg_frame_rate'])==fps for r in details),
 computed_gaps_match_excluded_ranges=recorded==actual_gaps,
 mean_absolute_pixel_error_0_to_255=float(np.average([r['mean_pixel_error_0_to_255'] for r in details],weights=[r['decoded_frames'] for r in details])),
 largest_frame_mean_pixel_error=max(r['max_frame_mean_pixel_error'] for r in details),
 verdict='Pending',
 method='Independent full source/clip decoding and ffprobe frame counts; original interval coverage; 160x90 RGB frame comparisons allowing H.264 re-encoding differences.')
summary['passed'] = (summary['source_sha256_matches'] and missing == 0 and
    summary['overlapping_frames'] == 0 and decoded_total == len(original) and
    summary['all_clips_match_recorded_frame_counts'] and summary['all_clips_match_source_frame_rate'] and
    summary['largest_frame_mean_pixel_error'] < 3)
summary['verdict'] = 'PASS: all source frames retained exactly once with matching video timing.' if summary['passed'] else 'FAIL: inspect missing ranges, frame counts, timing, and pixel comparisons.'
(root/'timeline_audit.json').write_text(json.dumps(dict(summary=summary,missing_ranges=gaps,clips=details),ensure_ascii=False,indent=2)+'\n')
with (root/'timeline_gaps.csv').open('w',newline='',encoding='utf-8-sig') as file:
 writer=csv.DictWriter(file,fieldnames=['start_frame','end_frame_exclusive','frame_count','start_sec','end_sec','duration_sec']);writer.writeheader();writer.writerows(gaps)
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('GAPS:',[(round(g['start_sec'],3),round(g['end_sec'],3),g['frame_count']) for g in gaps])

sys.exit(0 if summary['passed'] else 1)
