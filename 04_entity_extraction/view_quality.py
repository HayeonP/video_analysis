"""Select unchanged observed crops by visibility and temporal diversity."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image


@lru_cache(maxsize=2)
def _face_detector(model_path,mtime_ns):
    import cv2
    return cv2.FaceDetectorYN.create(model_path,'',(320,320),.6,.3,5000,
                                    cv2.dnn.DNN_BACKEND_OPENCV,cv2.dnn.DNN_TARGET_CPU)


@lru_cache(maxsize=512)
def _face_detail(path,mtime_ns,size,model_path,model_mtime_ns):
    """Face localization supplies detail/quality only, never identity evidence."""
    import cv2
    pixels=cv2.imread(path)
    if pixels is None:return None
    height,width=pixels.shape[:2]
    if min(width,height)<32:return None
    try:
        detector=_face_detector(model_path,model_mtime_ns)
        detector.setInputSize((width,height))
        faces=detector.detect(pixels)[1]
    except cv2.error:return None
    candidates=[]
    for face in (() if faces is None else faces):
        x,y,w,h=map(float,face[:4]);confidence=float(face[14])
        if min(w,h)<24 or x<0 or y<0 or x+w>width or y+h>height:continue
        if y+h/2>height*.55 or not width*.2<=x+w/2<=width*.8:continue
        eye_distance=float(np.linalg.norm(face[4:6]-face[6:8]))
        if eye_distance/max(w,1)<.15:continue
        region=pixels[int(y):int(y+h),int(x):int(x+w)]
        if not region.size:continue
        sharpness=float(cv2.Laplacian(cv2.cvtColor(region,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
        box=[max(0,int(x-.25*w)),max(0,int(y-.4*h)),min(width,int(x+1.25*w)),min(height,int(y+1.15*h))]
        candidates.append(dict(box=box,detection=face.tolist(),confidence=confidence,sharpness=sharpness,
                               quality=confidence*float(np.log1p(sharpness))))
    return max(candidates,key=lambda r:r['quality']) if candidates else None


def face_detail(bundle,row,output):
    """Optional YuNet detail from the unchanged target crop; absence is allowed."""
    model=Path(__file__).resolve().parents[1]/'01_low_freq_event/models/yunet.onnx'
    if not model.is_file() or not row.get('crop'):return None
    path=Path(row['crop']);path=path if path.is_absolute() else Path(bundle.get('asset_root',output))/path
    try:
        stat=path.stat()
        return _face_detail(str(path.resolve()),stat.st_mtime_ns,stat.st_size,str(model),model.stat().st_mtime_ns)
    except (ImportError,OSError,ValueError):return None


@lru_cache(maxsize=2)
def _face_encoder(path,mtime_ns):
    import cv2
    return cv2.FaceRecognizerSF.create(path,'',cv2.dnn.DNN_BACKEND_OPENCV,cv2.dnn.DNN_TARGET_CPU)


@lru_cache(maxsize=512)
def _face_feature(path,mtime_ns,size,detection,model_path,model_mtime_ns):
    import cv2
    try:
        encoder=_face_encoder(model_path,model_mtime_ns)
        aligned=encoder.alignCrop(cv2.imread(path),np.asarray(detection,dtype=np.float32))
        vector=encoder.feature(aligned).reshape(-1)
        norm=float(np.linalg.norm(vector))
        return vector/norm if norm>0 and np.isfinite(vector).all() else None
    except cv2.error:return None


def face_retrieval(bundle,entries,tracks,output):
    """Multi-frame SFace similarities nominate comparisons; they NEVER join IDs."""
    from itertools import combinations
    model=Path(__file__).resolve().parents[1]/'01_low_freq_event/models/sface.onnx'
    if not model.is_file():return {}
    features={}
    for lid,entry in entries.items():
        if entry.get('kind')!='person' or entry.get('_part_or_worn'):continue
        rows={r['frame']:r for tid in entry['track_ids'] for r in tracks[tid]['observations'] if r.get('crop')}
        vectors=[]
        for row in select_identity_views(bundle,list(rows.values()),output,count=3,prefer_face=True):
            detail=face_detail(bundle,row,output)
            if not detail:continue
            path=Path(row['crop']);path=path if path.is_absolute() else Path(bundle.get('asset_root',output))/path
            try:
                stat=path.stat();vector=_face_feature(str(path.resolve()),stat.st_mtime_ns,stat.st_size,
                    tuple(detail['detection']),str(model),model.stat().st_mtime_ns)
            except (ImportError,OSError,ValueError):continue
            if vector is not None:vectors.append(vector)
        if len(vectors)>=2:features[lid]=np.stack(vectors)
    scores={}
    for left,right in combinations(features,2):
        similarities=np.sort((features[left]@features[right].T).reshape(-1))
        score=float(similarities[-min(3,len(similarities)):].mean())
        for a,b in ((left,right),(right,left)):
            scores.setdefault(a,[]).append(dict(local_id=b,face_similarity=score,
                left_face_views=len(features[a]),right_face_views=len(features[b])))
    return scores


@lru_cache(maxsize=512)
def _measure(path,mtime_ns,size):
    with Image.open(path) as image:
        pixels=np.asarray(image.convert('L').resize((128,128)),dtype=np.float32)/255.
    low,high=np.percentile(pixels,[5,95]);contrast=float(high-low)
    laplacian=(pixels[:-2,1:-1]+pixels[2:,1:-1]+pixels[1:-1,:-2]+pixels[1:-1,2:]-4*pixels[1:-1,1:-1])
    sharpness=float(np.log1p(float(laplacian.var())*10000))
    clipped=float(np.mean((pixels<.008)|(pixels>.992)))
    # Percentiles, rather than average brightness, allow dark garments around a
    # well-exposed face. Clipping is a soft score, never a black/white veto.
    exposure=float(high)*(1-.4*clipped)
    return exposure,contrast,sharpness


def select_identity_views(bundle,observations,output,count=3,*,prefer_face=False):
    """Pick the clearest actual crop first, then useful chronological variety.

    At most 64 files are inspected per call; measurements are cached by file
    revision. Unreadable crops never abort a run. If no crop can be measured,
    return deterministic temporal samples for the caller's fallback rendering.
    No pixels, boxes, labels, or observations are altered.
    """
    if count<=0 or not observations:return []
    rows=list(observations)
    def time(row):return float(row.get('original_time_sec',row.get('full_time_sec',row.get('time_sec',row.get('frame',0)))))
    rows=sorted(rows,key=lambda r:(time(r),int(r.get('frame',0)),str(r.get('crop',''))))
    if len(rows)>64:rows=[rows[i] for i in np.linspace(0,len(rows)-1,64,dtype=int)]
    measured=[]
    for row in rows:
        value=row.get('crop')
        if not value:continue
        path=Path(value);path=path if path.is_absolute() else Path(bundle.get('asset_root',output))/path
        try:
            stat=path.stat();values=_measure(str(path.resolve()),stat.st_mtime_ns,stat.st_size)
        except (OSError,ValueError):continue
        box=row.get('bbox',[0,0,0,0]);area=max(0,(box[2]-box[0])*(box[3]-box[1]))
        measured.append((row,np.asarray([*values,float(row.get('confidence',0)),float(np.sqrt(area))],dtype=float)))
    if not measured:
        return [rows[i] for i in np.linspace(0,len(rows)-1,min(count,len(rows)),dtype=int)]
    matrix=np.stack([m[1] for m in measured]);scale=np.maximum(matrix.max(axis=0),1e-9)
    quality=(matrix/scale)@np.asarray([.30,.30,.25,.10,.05])
    if prefer_face:
        facial=np.asarray([(face_detail(bundle,row,output) or {}).get('quality',0.) for row,_ in measured])
        if facial.max()>0:quality+=2*facial/facial.max()
    times=np.asarray([time(row) for row,_ in measured]);span=max(float(times.max()-times.min()),1e-9)
    chosen=[int(np.argmax(quality))]
    while len(chosen)<min(count,len(measured)):
        candidates=[i for i in range(len(measured)) if i not in chosen]
        # Severe fades do not win merely because they are far apart in time;
        # moderate exposure changes retain value as supporting views.
        best=max(candidates,key=lambda i:(float(quality[i])*(.75+.25*min(abs(times[i]-times[j]) for j in chosen)/span),-i))
        chosen.append(best)
    return [measured[i][0] for i in chosen]
