"""Synchronized visual evidence for ambiguous relationships between tracks.

The detector's track hypotheses are not identities. These images let a reviewer
compare two hypotheses in the same coordinate system and at the same instant.
No semantic label, film-specific identity, or old identity annotation is used.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

LOG = logging.getLogger('story.relationships')
COLORS = ('#00c8ed', '#ff9e39')
PAGE_WIDTH = 1400
ROW_HEIGHT = 476


def _font(size):
    for path in (Path(__file__).parent/'assets/NanumGothic-Regular.ttf',
                 Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _asset(bundle, output, value):
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else Path(bundle.get('asset_root', output))/path


def _box(observation, size):
    raw = observation.get('bbox_pixels', observation.get('bbox'))
    if not raw or len(raw) != 4:
        return None
    try:
        values = [float(value) for value in raw]
    except (ValueError, TypeError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    width, height = size
    left, top, right, bottom = values
    result = (max(0, min(width, math.floor(left))),
              max(0, min(height, math.floor(top))),
              max(0, min(width, math.ceil(right))),
              max(0, min(height, math.ceil(bottom))))
    return result if result[2] > result[0] and result[3] > result[1] else None


def _has_mask(bundle, output, observation):
    path = _asset(bundle, output, observation.get('mask'))
    return bool(path and path.is_file())


def _select_pairs(bundle, shot, output, aliases, max_pairs):
    size = int(shot['width']), int(shot['height'])
    tracks = {track['track_id']: track for track in shot['tracks']}
    candidates, seen = [], set()
    for raw_pair in shot.get('mask_graph', {}).get('conditional_separation', []):
        if len(raw_pair) != 2:
            continue
        left, right = map(str, raw_pair)
        key = tuple(sorted((left, right)))
        if left == right or key in seen or any(tid not in tracks or tid not in aliases for tid in key):
            continue
        seen.add(key)
        left_rows = {int(row['frame']): row for row in tracks[left]['observations'] if _box(row, size)}
        right_rows = {int(row['frame']): row for row in tracks[right]['observations'] if _box(row, size)}
        common = sorted(set(left_rows) & set(right_rows))
        if not common:
            continue
        masked = {frame for frame in common if _has_mask(bundle, output, left_rows[frame])
                  and _has_mask(bundle, output, right_rows[frame])}
        midpoint = (common[0]+common[-1])/2

        def frame_rank(frame):
            boxes = [_box(rows[frame], size) for rows in (left_rows, right_rows)]
            smallest_area = min((box[2]-box[0])*(box[3]-box[1]) for box in boxes)
            # Actual simultaneous masks take priority; larger evidence is easier
            # to inspect, and the middle of a tie avoids unstable edge frames.
            return frame in masked, smallest_area, -abs(frame-midpoint), -frame

        frame = max(common, key=frame_rank)
        candidates.append(dict(left=left, right=right, frame=frame,
            left_observation=left_rows[frame], right_observation=right_rows[frame],
            shared_frames=len(common), shared_mask_frames=len(masked),
            rank=(len(masked), len(common), frame_rank(frame)[1])))
    candidates.sort(key=lambda item: item['rank'], reverse=True)
    return candidates[:max_pairs]


def _source_frames(bundle, shot, output, selected):
    wanted = {item['frame'] for item in selected}
    if shot.get('analysis_frames') == 'sampled':
        from sampled_frames import iter_shot_images
        return dict(iter_shot_images(shot, wanted))
    result = {}
    size = int(shot['width']), int(shot['height'])
    scene_images = {int(row['frame']): row.get('image') for row in shot.get('scene_frames', [])}
    for item in selected:
        frame = item['frame']
        if frame in result:
            continue
        for value in (item['left_observation'].get('image'), item['right_observation'].get('image'),
                      scene_images.get(frame)):
            path = _asset(bundle, output, value)
            if not path or not path.is_file():
                continue
            try:
                with Image.open(path) as image:
                    # A crop is not a full source frame even if a stale asset
                    # accidentally appears under the observation's image key.
                    if image.size == size:
                        result[frame] = image.convert('RGB')
                        break
            except (OSError, ValueError) as error:
                LOG.warning('Unreadable relationship frame %s: %s', path, error)
    missing = wanted-result.keys()
    if missing:
        import av
        source = _asset(bundle, output, shot.get('source'))
        if source is None:
            LOG.warning('%s lacks a source video for relationship frames %s', shot['shot_id'], sorted(missing))
            return result
        try:
            with av.open(str(source)) as container:
                for frame_number, frame in enumerate(container.decode(video=0)):
                    if frame_number in missing:
                        result[frame_number] = frame.to_image().convert('RGB')
                    if missing <= result.keys() or frame_number >= max(missing):
                        break
        except (OSError, ValueError) as error:
            LOG.warning('Cannot decode relationship evidence from %s: %s', source, error)
    return result


def _paste_fit(canvas, image, bounds):
    left, top, right, bottom = bounds
    thumb = ImageOps.contain(image, (right-left, bottom-top), Image.Resampling.LANCZOS)
    canvas.paste(thumb, (left+(right-left-thumb.width)//2, top+(bottom-top-thumb.height)//2))


def _panel(image, item, aliases, shot):
    panel = Image.new('RGB', (PAGE_WIDTH, ROW_HEIGHT), '#eef1f6')
    draw = ImageDraw.Draw(panel)
    labels = [aliases[item['left']], aliases[item['right']]]
    boxes = [_box(item[name], image.size) for name in ('left_observation', 'right_observation')]
    frame = item['frame']
    title = f'{shot["shot_id"]}  |  {labels[0]} + {labels[1]}  |  SAME FRAME f{frame}  ({frame/float(shot["fps"]):.3f}s)'
    draw.text((14, 10), title, font=_font(25), fill='#14233c')
    draw.text((14, 45), f'{item["shared_frames"]} co-visible frames; {item["shared_mask_frames"]} with both masks',
              font=_font(17), fill='#47566c')
    left = min(box[0] for box in boxes); top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes); bottom = max(box[3] for box in boxes)
    pad = max(12, round(max(right-left, bottom-top)*.18))
    union = (max(0, left-pad), max(0, top-pad),
             min(image.width, right+pad), min(image.height, bottom+pad))
    context = image.crop(union)
    pen = ImageDraw.Draw(context)
    # Paint each exact-frame geometry before scaling the joint context. Raw
    # individual crops remain untouched so color/text cannot hide anatomy.
    line = max(2, round(max(context.size)/220))
    font = _font(max(12, round(max(context.size)/33)))
    for label, color, box in zip(labels, COLORS, boxes):
        shifted = (box[0]-union[0], box[1]-union[1], box[2]-union[0]-1, box[3]-union[1]-1)
        pen.rectangle(shifted, outline=color, width=line)
        pen.text((shifted[0]+line, max(0, shifted[1]-font.size-2)), label,
                 font=font, fill=color, stroke_width=1, stroke_fill='black')
    draw.text((14, 78), 'Shared location context', font=_font(20), fill='#24344c')
    _paste_fit(panel, context, (12, 111, 748, 461))
    for index, (label, color, box) in enumerate(zip(labels, COLORS, boxes)):
        x = 760+index*320
        draw.text((x+5, 78), label+' — raw crop', font=_font(20), fill='#24344c')
        draw.rectangle((x, 108, x+307, 463), outline=color, width=4)
        _paste_fit(panel, image.crop(box), (x+7, 116, x+300, 455))
    metadata = dict(left=item['left'], right=item['right'], aliases=labels, frame=frame,
                    time_sec=frame/float(shot['fps']), union_box=list(union),
                    boxes=[list(box) for box in boxes], shared_frames=item['shared_frames'],
                    shared_mask_frames=item['shared_mask_frames'])
    return panel, metadata


def relationship_images(bundle, shot, output, aliases, max_pairs=6):
    """Return ``[(descriptive_label, PIL.Image), ...]``, two pairs per page.

    Each row compares exactly one common source frame: an expanded union of
    the two boxes and two raw crops from that same frame. Existing full-frame
    images are reused; all missing frames are decoded in one video pass.
    Unavailable frame evidence is omitted with a warning. The helper does not
    mutate the bundle, invoke models, or write files.
    """
    if max_pairs <= 0:
        return []
    selected = _select_pairs(bundle, shot, output, aliases, min(6, int(max_pairs)))
    if not selected:
        return []
    frames = _source_frames(bundle, shot, output, selected)
    panels = []
    for item in selected:
        image = frames.get(item['frame'])
        if image is None:
            LOG.warning('Skipping %s relationship %s/%s: exact source frame %s unavailable',
                        shot['shot_id'], item['left'], item['right'], item['frame'])
            continue
        panels.append(_panel(image, item, aliases, shot))
    pages = []
    for offset in range(0, len(panels), 2):
        batch = panels[offset:offset+2]
        page = Image.new('RGB', (PAGE_WIDTH, ROW_HEIGHT*len(batch)), '#d5dce7')
        rows = []
        for index, (panel, metadata) in enumerate(batch):
            page.paste(panel, (0, index*ROW_HEIGHT))
            rows.append(metadata)
        page.info['relationship_pairs'] = rows
        label = 'Synchronous relationships: '+ '; '.join(
            f'{row["aliases"][0]} + {row["aliases"][1]} at f{row["frame"]}' for row in rows)
        pages.append((label, page))
    return pages
