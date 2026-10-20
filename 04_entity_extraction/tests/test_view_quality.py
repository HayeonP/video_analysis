from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from view_quality import select_identity_views


def test_visible_anchor_beats_fade_and_retains_unchanged_temporal_views(tmp_path):
    pattern=((np.indices((128,128)).sum(axis=0)%2)*180+30).astype(np.uint8)
    rows=[]
    for frame,factor in [(0,.08),(1,1.),(2,1.),(20,.9),(40,.03)]:
        path=tmp_path/f'{frame}.png';Image.fromarray((pattern*factor).astype(np.uint8)).save(path)
        rows.append(dict(frame=frame,crop=path.name,bbox=[0,0,100,100],confidence=.8))
    chosen=select_identity_views({'asset_root':str(tmp_path)},rows,tmp_path)
    assert chosen[0]['frame']==1
    assert {r['frame'] for r in chosen}=={1,2,20}
    assert all(any(r is original for original in rows) for r in chosen)
    assert chosen==select_identity_views({'asset_root':str(tmp_path)},rows,tmp_path)


def test_black_or_white_clothing_is_not_an_absolute_exclusion(tmp_path):
    for value,name in [(0,'black'),(255,'white')]:
        array=np.full((128,128),value,dtype=np.uint8);array[20:70,40:90]=120
        path=tmp_path/(name+'.png');Image.fromarray(array).save(path)
        row=dict(frame=0,crop=str(path),bbox=[0,0,100,100],confidence=.9)
        assert select_identity_views({},[row],tmp_path)==[row]


def test_missing_crops_bounded_deterministic_fallback_and_zero_count(tmp_path):
    rows=[dict(frame=i,crop='absent.jpg') for i in range(1000)]
    selected=select_identity_views({},rows,tmp_path)
    assert len(selected)==3 and selected[0] is rows[0] and selected[-1] is rows[-1]
    assert select_identity_views({},rows,tmp_path,0)==[]
    assert select_identity_views({},[],tmp_path)==[]


def test_clear_face_can_outrank_sharp_background_without_changing_pixels(tmp_path,monkeypatch):
    import view_quality
    from PIL import ImageFilter
    pattern=((np.indices((128,128)).sum(axis=0)%2)*180+30).astype(np.uint8)
    rows=[]
    for frame in range(2):
        image=Image.fromarray(pattern)
        if frame:image=image.filter(ImageFilter.GaussianBlur(.7))
        path=tmp_path/f'{frame}.png';image.save(path)
        rows.append(dict(frame=frame,crop=str(path),bbox=[0,0,128,128],confidence=.8))
    before=[Path(r['crop']).read_bytes() for r in rows]
    monkeypatch.setattr(view_quality,'face_detail',lambda bundle,row,output:dict(quality=1 if row['frame']==1 else .05))
    assert select_identity_views({},rows,tmp_path,count=1)[0] is rows[0]
    assert select_identity_views({},rows,tmp_path,count=1,prefer_face=True)[0] is rows[1]
    assert [Path(r['crop']).read_bytes() for r in rows]==before
