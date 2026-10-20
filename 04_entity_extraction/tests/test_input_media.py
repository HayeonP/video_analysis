"""Portable frame-range reconstruction must retain source pixels and timing."""
from pathlib import Path
import sys
import tempfile
import unittest
import shutil

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from input_media import ShotMaterializer,validate_tracking_sources
from shot_tracking import discover_shots
from low_freq_event.storage import atomic_json


def write_video(path,*,frames=8,offset=0):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index * 28+offset, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class InputMediaTests(unittest.TestCase):
    def test_resume_rejects_same_path_two_frame_video_replacement_but_allows_move(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'clip.mp4';write_video(source,frames=2)
            bundle={'shots':discover_shots(source)}
            moved=root/'moved.mp4';shutil.copy2(source,moved)
            self.assertEqual(validate_tracking_sources(bundle,moved)['status'],'verified')
            write_video(source,frames=2,offset=70)
            with self.assertRaisesRegex(ValueError,'--stage all'):
                validate_tracking_sources(bundle,source)

    def test_resume_detects_original_replacement_behind_materialized_cache_and_metadata_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original=root/'original.mp4';write_video(original,frames=2)
            source=root/'adaptive';metadata=dict(source='/missing/clip.mp4',source_fps='24',source_frames=2,
                original_start_frame=0,original_fps='24',samples=[{'source_frame':0,'image':'frames/a.jpg'}])
            atomic_json(source/'summary.json',{'videos':[{'id':'clip','samples':'clip/samples.json'}]})
            atomic_json(source/'clip/samples.json',metadata)
            bundle={'shots':discover_shots(source,original_video=original,cache_dir=root/'cache')}
            # Backward compatibility: the older materializer sidecar is also a
            # valid source of the original digest when the bundle lacks it.
            bundle['shots'][0].pop('original_sha256')
            self.assertEqual(validate_tracking_sources(bundle,source,original_video=original,cache_dir=root/'cache')['status'],'verified')
            metadata['samples'][0]['source_frame']=1;atomic_json(source/'clip/samples.json',metadata)
            with self.assertRaisesRegex(ValueError,'adaptive'):
                validate_tracking_sources(bundle,source,original_video=original,cache_dir=root/'cache')
            metadata['samples'][0]['source_frame']=0;atomic_json(source/'clip/samples.json',metadata)
            write_video(original,frames=2,offset=80)
            with self.assertRaisesRegex(ValueError,'--stage all'):
                validate_tracking_sources(bundle,source,original_video=original,cache_dir=root/'cache')

    def test_recovery_uses_validated_first_video_stream_even_if_another_is_larger(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original=root/'multi.mp4'
            with av.open(str(original),'w') as container:
                streams=[container.add_stream('libx264',rate=24) for _ in range(2)]
                for stream,width in zip(streams,(64,96)):
                    stream.width,stream.height,stream.pix_fmt=width,48,'yuv420p'
                for index in range(6):
                    for number,stream in enumerate(streams):
                        pixels=np.full((48,stream.width,3),index*25+number*75,dtype=np.uint8)
                        frame=av.VideoFrame.from_ndarray(pixels,format='rgb24')
                        for packet in stream.encode(frame):container.mux(packet)
                for stream in streams:
                    for packet in stream.encode():container.mux(packet)
            recovered=ShotMaterializer(original,root/'cache').recover(dict(
                source_fps='24',original_start_frame=1,source_frames=3))
            with av.open(str(original)) as container:
                expected=[frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)][1:4]
            with av.open(str(recovered)) as container:
                actual=[frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
            self.assertEqual(len(actual),3)
            for left,right in zip(expected,actual):np.testing.assert_array_equal(left,right)

    def test_recovery_preserves_exact_frame_interval_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'video.mp4'
            write_video(original)
            source = root / 'adaptive'
            metadata = dict(source='/old-machine/missing.mp4', source_fps='24', source_frames=3,
                            original_start_frame=2, original_fps='24', samples=[{'source_frame': 0}])
            atomic_json(source / 'summary.json', {'videos': [{'id': 'clip_A', 'samples': 'clip_A/samples.json'}]})
            atomic_json(source / 'clip_A/samples.json', metadata)
            shots = discover_shots(source, original_video=original, cache_dir=root / 'cache')
            recovered = Path(shots[0]['source'])
            with av.open(str(original)) as container:
                expected = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)][2:5]
            with av.open(str(recovered)) as container:
                actual = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
            self.assertEqual(len(actual), 3)
            for left, right in zip(expected, actual):
                np.testing.assert_array_equal(left, right)
            self.assertEqual(shots[0]['original_start_frame'], 2)
            self.assertEqual(shots[0]['original_video'], str(original))
            stamp = recovered.stat().st_mtime_ns
            self.assertEqual(discover_shots(source, original_video=original, cache_dir=root / 'cache')[0]['source'], str(recovered))
            self.assertEqual(recovered.stat().st_mtime_ns, stamp)

    def test_invalid_range_and_fps_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'video.mp4'
            write_video(original)
            recover = ShotMaterializer(original, root / 'cache')
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                recover.recover(dict(source_fps='24', original_start_frame=7, source_frames=3))
            with self.assertRaisesRegex(ValueError, 'FPS'):
                recover.recover(dict(source_fps='30', original_start_frame=0, source_frames=3))


if __name__ == '__main__':
    unittest.main()
