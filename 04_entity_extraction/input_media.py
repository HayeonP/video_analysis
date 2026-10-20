"""Recover portable shot inputs from an explicitly supplied original video.

Stage 03 metadata is never rewritten. Recovered media is frame-accurate and
lossless, with source hashes and frame intervals recorded next to the cache.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_media_path(raw, metadata, source):
    """Resolve relative media or an old absolute path after a repository move."""
    path = Path(raw).expanduser()
    metadata, source = Path(metadata), Path(source)
    candidates = [path] if path.is_absolute() else [metadata.parent / path, source / path, Path.cwd() / path]
    # Rebase only a matching repository directory suffix; never choose a video
    # solely because its basename happens to match in an unrelated folder.
    repo = Path(__file__).resolve().parent.parent
    if path.is_absolute():
        for index, part in enumerate(path.parts):
            if (repo / part).is_dir():
                candidates.append(repo.joinpath(*path.parts[index:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'Original shot video is unavailable: {raw}; metadata: {metadata}. '
                            'Supply --original-video to recover shots from their original frame intervals.')


def _portable_samples(value):
    if isinstance(value,list):return [_portable_samples(item) for item in value]
    if isinstance(value,dict):
        return {key:(Path(item).name if key in ('image','crop','path','file') and isinstance(item,str)
                     else _portable_samples(item)) for key,item in value.items()}
    return value


def _recorded_original_digest(shot):
    if shot.get('original_sha256'):return shot['original_sha256']
    # Older recovered bundles already have immutable original provenance in
    # the materializer sidecar, even before per-shot original hashes existed.
    source=Path(shot.get('source',''))
    record=source.with_suffix('.json') if source.name else None
    if record and record.is_file():
        try:
            digest=json.loads(record.read_text()).get('provenance',{}).get('original_sha256')
            if digest:return digest
        except (OSError,ValueError):pass
    if shot.get('original_video')==shot.get('source'):return shot.get('source_sha256')
    return None


def validate_tracking_sources(bundle,source,*,original_video=None,cache_dir=None):
    """Reject resume against replaced media or changed source-frame metadata.

    Paths may move when their content and timing remain identical. Rediscovery
    also validates/rebuilds materialized clips against their original bytes.
    """
    if bundle.get('analysis_frames') == 'sampled':
        from sampled_frames import validate_sampled_bundle
        validate_sampled_bundle(bundle, source)
        return dict(status='verified', shots=len(bundle['shots']), policy='Selected image bytes and timing bound')
    from shot_tracking import discover_shots
    previous=bundle.get('shots',[])
    if not previous:return dict(status='skipped_empty_bundle',shots=0)
    try:current=discover_shots(source,original_video=original_video,cache_dir=cache_dir)
    except (OSError,ValueError,RuntimeError) as exc:
        raise ValueError(f'Cannot verify tracking source inputs; rerun --stage all after restoring inputs: {exc}') from exc
    if [row['shot_id'] for row in current]!=[row['shot_id'] for row in previous]:
        raise ValueError('Source shot inventory/order changed; rerun --stage all.')
    keys=('source_sha256','fps','frame_count','width','height','original_start_frame','original_fps',
          'adaptive_frames','full_start_sec','duration_sec','source_metadata')
    for old,new in zip(previous,current):
        changed=[key for key in keys if key in old and old[key]!=new.get(key)]
        if _portable_samples(old.get('adaptive_samples',[]))!=_portable_samples(new.get('adaptive_samples',[])):
            changed.append('adaptive_samples')
        original_hash=_recorded_original_digest(old)
        if original_hash and new.get('original_sha256') and original_hash!=new['original_sha256']:
            changed.append('original_sha256')
        if changed:
            raise ValueError(f'{old["shot_id"]}: source content/metadata changed ({", ".join(changed)}); rerun --stage all.')
    return dict(status='verified',shots=len(current),policy='Content and timing bound; path relocation allowed.')


class ShotMaterializer:
    def __init__(self, original_video, cache_dir):
        import av
        self.video = Path(original_video).expanduser().resolve(strict=True)
        self.cache = Path(cache_dir)
        self.digest = file_sha256(self.video)
        with av.open(str(self.video)) as container:
            stream = container.streams.video[0]
            self.fps = Fraction(stream.average_rate or stream.base_rate)
            self.frames = int(stream.frames or sum(1 for _ in container.decode(stream)))

    def recover(self, meta):
        import av
        fps = Fraction(meta['source_fps'])
        original_fps = Fraction(meta.get('original_fps', meta['source_fps']))
        if self.fps != original_fps or fps != original_fps:
            raise ValueError('Frame-accurate shot recovery requires matching source/original FPS.')
        start, count = int(meta['original_start_frame']), int(meta['source_frames'])
        if start < 0 or count < 1 or start + count > self.frames:
            raise ValueError(f'Original frame interval {start}:{start + count} exceeds {self.frames} frames.')
        provenance = dict(version=2, original_video=str(self.video), original_sha256=self.digest,
                          video_stream=0,
                          original_start_frame=start, source_frames=count, fps=str(fps),
                          encoding='libx264 lossless; original frame range, no temporal resampling')
        key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        video = self.cache / f'{key}.mp4'
        record = video.with_suffix('.json')
        if video.is_file() and record.is_file():
            existing = json.loads(record.read_text())
            if existing.get('provenance') == provenance and existing.get('sha256') == file_sha256(video):
                return video.resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        partial = self.cache / f'{key}.partial.mp4'
        try:
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(self.video), '-map', '0:v:0',
                '-vf', f'trim=start_frame={start}:end_frame={start + count},setpts=PTS-STARTPTS',
                '-frames:v', str(count), '-an', '-c:v', 'libx264', '-crf', '0', '-preset', 'ultrafast',
                '-threads', '2', '-vsync', '0', str(partial)], check=True, capture_output=True)
            with av.open(str(partial)) as container:
                actual = sum(1 for _ in container.decode(video=0))
            if actual != count:
                raise ValueError(f'Recovered shot has {actual} frames; expected {count}.')
            partial.replace(video)
            record.write_text(json.dumps(dict(provenance=provenance, sha256=file_sha256(video)), indent=2) + '\n')
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'Frame-accurate media recovery failed: {exc.stderr.decode(errors="replace")[-2000:]}') from exc
        finally:
            partial.unlink(missing_ok=True)
        return video.resolve()
