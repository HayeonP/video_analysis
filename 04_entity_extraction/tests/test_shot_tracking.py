"""Dense tracker contract checks using CPU-only synthetic frames/detections."""
from dataclasses import asdict, replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shot_tracking import (DenseTrackingConfig, DetectionBatch, _finish_tracks, _materialize_assets,
                           _tiles, build_dense_tracks, discover_shots, fuse_duplicate_boxes, make_tracker,
                           select_representatives, tracking_confidence, _compatible_detector_cache,
                           _digest, _file_digest, DETECTOR_CACHE_VERSION)


def row(box, category='person', confidence=.8):
    return dict(bbox=list(box), confidence=confidence, category=category, source='full')


def observation(frame, category='person', confidence=.8):
    return dict(frame=frame, source_frame=frame, shot_id='shot_0001', time_sec=frame / 24,
                original_time_sec=frame / 24, bbox=[10., 5., 35., 40.], confidence=confidence,
                category=category, image=None, crop=None, _feature=[1., 0., 0.])


class DenseTrackingTests(unittest.TestCase):
    def test_class_conflict_near_identical_box_is_one_detection(self):
        merged = fuse_duplicate_boxes([row((0, 0, 100, 100), 'person'),
                                       row((1, 1, 101, 101), 'bird', .6)])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['alternative_categories'], ['bird'])

    def test_nested_carried_creature_is_retained(self):
        merged = fuse_duplicate_boxes([row((0, 0, 100, 100), 'bird'),
                                       row((30, 40, 50, 70), 'person', .6)])
        self.assertEqual(len(merged), 2)

    def test_weak_tile_can_continue_but_cannot_birth_and_keeps_original_score(self):
        config = DenseTrackingConfig()
        proposal = dict(source='yolo_small_target_tile', confidence=.55)
        self.assertLess(tracking_confidence(proposal, config), config.high_confidence)
        self.assertGreater(tracking_confidence(proposal, config), config.detector_confidence)
        self.assertEqual(proposal['confidence'], .55)
        self.assertEqual(tracking_confidence(dict(source='yolo_full', confidence=.55), config), .55)

    def test_moving_revision_cache_requires_verified_matching_resolved_commit(self):
        from low_freq_event.storage import atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            weights = output / 'weights.pt'
            weights.write_bytes(b'test')
            config = asdict(DenseTrackingConfig())
            detector = {k: v for k, v in config.items() if k.startswith(('detector_', 'appearance', 'tile_'))}
            old_detector = dict(detector, appearance_revision='main')
            old_key = _digest(dict(version=DETECTOR_CACHE_VERSION, weights=_file_digest(weights), config=old_detector))
            atomic_json(output / 'dense_tracking/bundle.json', dict(detector_model=str(weights),
                        config=dict(config, appearance_revision='main')))
            frame = output / 'dense_tracking/cache' / old_key / 'source/f00000000.json'
            atomic_json(frame, dict(detections=[{'feature': [1., 0.]}], appearance_revision=config['appearance_revision']))
            self.assertEqual(_compatible_detector_cache(output, weights, detector, 'new-key'), old_key)
            atomic_json(frame, dict(detections=[{'feature': [1., 0.]}], appearance_revision='different-commit'))
            self.assertEqual(_compatible_detector_cache(output, weights, detector, 'new-key'), 'new-key')

    def test_tile_grid_covers_corners_with_bounded_calls(self):
        config = DenseTrackingConfig()
        rects = _tiles(3840, 2160, config)
        self.assertLessEqual(len(rects), config.tile_max_calls)
        for x, y in [(0, 0), (3839, 0), (0, 2159), (3839, 2159), (1920, 1080)]:
            self.assertTrue(any(a <= x < c and b <= y < d for a, b, c, d in rects))

    def test_singleton_and_label_conflict_remain_reviewable(self):
        groups = {4: [observation(2)], 8: [observation(5), observation(6, 'bird')]}
        tracks = _finish_tracks(groups, 24., DenseTrackingConfig())
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[0]['observed_frames'], 1)
        self.assertIn('single_observed_frame', tracks[0]['quality_flags'])
        self.assertIn('human_nonhuman_class_conflict', tracks[1]['quality_flags'])

    def test_representatives_are_bounded_time_ordered_with_endpoints(self):
        observations = [observation(i) for i in range(100)]
        observations[50]['_feature'] = [0., 1., 0.]
        reps = select_representatives(observations, 8)
        frames = [r['frame'] for r in reps]
        self.assertEqual(len(reps), 8)
        self.assertEqual(frames, sorted(set(frames)))
        self.assertEqual((frames[0], frames[-1]), (0, 99))
        self.assertIn(50, frames)

    def test_gaps_are_unknown_not_interpolated_observations(self):
        tracks = _finish_tracks({1: [observation(0), observation(4)]}, 24., DenseTrackingConfig())
        self.assertEqual(len(tracks[0]['observations']), 2)
        self.assertEqual(tracks[0]['occlusion_gaps'][0]['start_frame'], 1)
        self.assertEqual(tracks[0]['occlusion_gaps'][0]['end_frame_exclusive'], 4)

    def test_botsort_carries_original_index_and_low_confidence_continuity(self):
        from ultralytics.engine.results import Boxes
        config = DenseTrackingConfig(appearance=False)
        tracker = make_tracker(24, config)
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        first = DetectionBatch(Boxes(np.asarray([[5, 5, 15, 15, .1, 0],
                                                 [30, 20, 60, 70, .8, 0]], dtype=np.float32), (100, 100)))
        tracker.update(first, img=blank)
        visible = tracker.tracked_stracks
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0].idx, 1)
        track_id = visible[0].track_id
        second = DetectionBatch(Boxes(np.asarray([[31, 20, 61, 70, .12, 0]], dtype=np.float32), (100, 100)))
        tracker.update(second, img=blank)
        self.assertEqual(len(tracker.tracked_stracks), 1)
        self.assertEqual(tracker.tracked_stracks[0].track_id, track_id)
        self.assertEqual(tracker.tracked_stracks[0].idx, 0)

    def test_generic_appearance_path_uses_provided_features(self):
        from ultralytics.engine.results import Boxes
        tracker = make_tracker(24, DenseTrackingConfig())
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        for x in (20, 21, 22):
            batch = DetectionBatch(Boxes(np.asarray([[x, 20, x + 30, 70, .8, 0]], dtype=np.float32), (100, 100)))
            tracker.update(batch, img=blank, feats=np.asarray([[1., 0., 0.]], dtype=np.float32))
        self.assertEqual(len(tracker.tracked_stracks), 1)
        self.assertEqual(tracker.tracked_stracks[0].start_frame, 1)

    def test_original_media_and_representative_assets(self):
        import av
        from low_freq_event.storage import atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / 'original.mp4'
            with av.open(str(video), 'w') as container:
                stream = container.add_stream('libx264', rate=24)
                stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
                for i in range(6):
                    frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 30, dtype=np.uint8), format='rgb24')
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            source = root / 'adaptive'
            atomic_json(source / 'summary.json', {'videos': [{'id': 'shot_0001', 'samples': 'shot_0001/samples.json'}]})
            atomic_json(source / 'shot_0001/samples.json', dict(source=str(video), source_fps='24', source_frames=6,
                        original_start_frame=100, original_fps='24', samples=[{'source_frame': 0}, {'source_frame': 4}]))
            shots = discover_shots(source)
            self.assertEqual(shots[0]['source'], str(video))
            self.assertEqual(shots[0]['frame_count'], 6)
            self.assertEqual(shots[0]['original_start_frame'], 100)
            config = replace(DenseTrackingConfig(), representatives=2, scene_frames=3)
            shot = shots[0]
            shot['tracks'] = _finish_tracks({1: [observation(i) for i in range(6)]}, 24., config)
            _materialize_assets(shot, root / 'output', config)
            self.assertEqual(len(shot['scene_frames']), 3)
            for rep in shot['tracks'][0]['representatives']:
                self.assertTrue((root / 'output' / rep['crop']).is_file())
                self.assertTrue((root / 'output' / rep['image']).is_file())
                self.assertIn('feature', rep)
                self.assertNotIn('_feature', rep)
            self.assertIsNone(shot['tracks'][0]['observations'][1]['crop'])

    def test_complete_cpu_pipeline_cache_without_model_inference(self):
        import av
        from ultralytics.engine.results import Results

        class FakeDetector:
            names = {0: 'person'}
            calls = 0

            def __init__(self, path): pass

            def predict(self, image, **kwargs):
                import torch
                self.__class__.calls += 1
                box = torch.tensor([[10., 5., 35., 40., .9, 0.]], dtype=torch.float32)
                return [Results(np.asarray(image), '', self.names, boxes=box)]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / 'shot.mp4'
            weights = root / 'fake.pt'
            weights.write_bytes(b'unit-test-only')
            with av.open(str(video), 'w') as container:
                stream = container.add_stream('libx264', rate=24)
                stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
                for i in range(4):
                    frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 30, dtype=np.uint8), format='rgb24')
                    for packet in stream.encode(frame): container.mux(packet)
                for packet in stream.encode(): container.mux(packet)
            config = DenseTrackingConfig(appearance=False, tile_enabled=False)
            with patch('ultralytics.YOLO', FakeDetector), patch('strong_detector._weights_path', return_value=weights):
                bundle = build_dense_tracks(video, root / 'output', device='cpu', config=config)
                reused = build_dense_tracks(video, root / 'output', device='cpu', config=config)
            self.assertEqual(FakeDetector.calls, 4)
            self.assertEqual(bundle['stats']['dense_frames'], 4)
            self.assertEqual(bundle['stats']['tracks'], 1)
            self.assertEqual(bundle['asset_root'], str(root / 'output'))
            self.assertEqual(bundle['fingerprint'], reused['fingerprint'])
            self.assertEqual(len(bundle['shots'][0]['tracks'][0]['observations']), 4)


if __name__ == '__main__':
    unittest.main()
