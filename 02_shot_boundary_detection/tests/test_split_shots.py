"""Integration checks with known frame boundaries, optional audio, and no-cut video."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scenedetect import FrameTimecode
from gradual import contiguous_spans

import cv2

from split_shots import split_shots
from video_utils import probe


class ShotSplitTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_video(self, *, cuts, audio):
        source = self.root / ('three shots.mp4' if cuts else 'single.mp4')
        graph = ('color=red:s=160x90:r=24:d=1[r];'
                 'color=blue:s=160x90:r=24:d=1[b];'
                 'color=red:s=160x90:r=24:d=1[g];'
                 '[r][b][g]concat=n=3:v=1:a=0') if cuts else 'color=red:s=160x90:r=24:d=2'
        command = ['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i', graph]
        if audio:
            command += ['-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=3']
        command += ['-c:v', 'libx264', '-threads', '2', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(source)]
        subprocess.run(command, check=True)
        return source

    def test_known_boundaries_and_audio(self):
        source = self.make_video(cuts=True, audio=True)
        for detector in ('adaptive', 'content'):
            with self.subTest(detector=detector):
                output = self.root / detector
                result = split_shots(source, output, detector=detector, show_progress=False)
                self.assertEqual([(r['start_frame'], r['end_frame_exclusive']) for r in result['shots']],
                                 [(0, 24), (24, 48), (48, 72)])
                for row, dominant_channel in zip(result['shots'], (2, 0, 2)):
                    path = output / row['file']
                    streams = probe(path, count_frames=True)['streams']
                    video = next(s for s in streams if s['codec_type'] == 'video')
                    self.assertEqual(int(video['nb_read_frames']), 24)
                    self.assertTrue(any(s['codec_type'] == 'audio' for s in streams))
                    capture = cv2.VideoCapture(str(path))
                    try:
                        # Every exported frame belongs to the expected side of each cut.
                        count = 0
                        while True:
                            ok, frame = capture.read()
                            if not ok:
                                break
                            self.assertEqual(int(frame.mean(axis=(0, 1)).argmax()), dominant_channel)
                            count += 1
                        self.assertEqual(count, 24)
                    finally:
                        capture.release()

    def test_transnet_transition_frames_survive_real_mp4_export(self):
        source = self.root / 'fades.mp4'
        graph = ('color=red:s=160x90:r=24:d=1,fade=t=out:st=0.75:d=0.25[r];'
                 'color=black:s=160x90:r=24:d=0.25[b];'
                 'color=blue:s=160x90:r=24:d=1.25,fade=t=in:st=0:d=0.333333[g];'
                 '[r][b][g]concat=n=3:v=1:a=0')
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', graph,
                        '-c:v', 'libx264', '-threads', '2', '-pix_fmt', 'yuv420p', str(source)], check=True)
        many = np.zeros(60); many[18:24] = 1; many[30:38] = 1
        black = np.zeros(60); black[24:30] = 1
        spans, regions = contiguous_spans(np.zeros(60), many, black, 24)
        self.assertEqual(spans, [(0, 28), (28, 60)])
        prediction = ([(FrameTimecode(a, fps=24.0), FrameTimecode(b, fps=24.0)) for a,b in spans],
                      dict(total_frames=60, duration_sec=2.5, transition_ranges=regions))
        output = self.root / 'fades_result'
        with patch('gradual.detect_gradual', return_value=prediction):
            result = split_shots(source, output, show_progress=False)
        self.assertEqual(result['exported_frame_count'], 60)
        self.assertEqual(result['excluded_frame_count'], 0)
        self.assertEqual(result['extraction_mode'], 'contiguous')
        mapped = []
        for row in result['shots']:
            info = probe(output / row['file'], count_frames=True)
            count = int(info['streams'][0]['nb_read_frames'])
            self.assertEqual(count, row['frame_count'])
            mapped.extend(row['start_frame'] + local for local in range(count))
        self.assertEqual(mapped, list(range(60)))

    def test_no_cut_no_audio_and_existing_output(self):
        source = self.make_video(cuts=False, audio=False)
        output = self.root / 'result'
        result = split_shots(source, output, detector="adaptive", show_progress=False)
        self.assertEqual(result['shot_count'], 1)
        self.assertEqual(result['total_frames'], 48)
        streams = probe(output / result['shots'][0]['file'], count_frames=True)['streams']
        self.assertEqual(int(streams[0]['nb_read_frames']), 48)
        self.assertFalse(any(s['codec_type'] == 'audio' for s in streams))
        original = (output / 'shots.json').read_bytes()
        with self.assertRaises(FileExistsError):
            split_shots(source, output, show_progress=False)
        self.assertEqual((output / 'shots.json').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
