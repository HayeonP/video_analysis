"""Timeline-conservation regressions for fades, hard cuts, and short fragments."""
import unittest
import numpy as np
from gradual import contiguous_spans


class GradualTest(unittest.TestCase):
    def assert_complete(self, spans, count):
        coverage = np.zeros(count, dtype=int)
        for start, end in spans:
            self.assertLess(start, end)
            coverage[start:end] += 1
        np.testing.assert_array_equal(coverage, np.ones(count, dtype=int))

    def test_dissolve_and_black_fade_split_once_without_omissions(self):
        single, many = np.zeros(120), np.zeros(120)
        black = np.zeros(120, dtype=bool)
        many[24:40] = 0.9
        black[72:78] = True
        single[71] = 0.9
        many[84:96] = 0.9
        spans, regions = contiguous_spans(single, many, black, 24)
        self.assertEqual(spans, [(0, 32), (32, 84), (84, 120)])
        self.assertEqual([r['cut_frame'] for r in regions], [32, 84])
        self.assert_complete(spans, 120)

    def test_short_real_shot_next_to_black_is_kept(self):
        single, many = np.zeros(96), np.zeros(96)
        black = np.zeros(96, dtype=bool)
        black[24:30] = True
        single[43] = 0.9
        spans, _ = contiguous_spans(single, many, black, 24)
        self.assertIn((27, 44), spans)
        self.assert_complete(spans, 96)

    def test_tiny_fragment_is_absorbed_without_deleting_frames(self):
        single, many = np.zeros(60), np.zeros(60)
        single[20:22] = 0.9
        single[25:27] = 0.9
        spans, _ = contiguous_spans(single, many, np.zeros(60), 24)
        self.assertEqual(spans, [(0, 24), (24, 60)])
        self.assert_complete(spans, 60)

    def test_initial_final_and_all_black_frames_are_kept(self):
        zeros = np.zeros(60)
        for black in (np.ones(60), zeros, np.r_[np.ones(8), np.zeros(44), np.ones(8)]):
            spans, _ = contiguous_spans(zeros, zeros, black, 24)
            self.assertEqual(spans, [(0, 60)])

    def test_single_frame_cut_assigns_last_old_frame_to_old_shot(self):
        single = np.zeros(48); single[23] = 1
        spans, _ = contiguous_spans(single, np.zeros(48), np.zeros(48), 24)
        self.assertEqual(spans, [(0, 24), (24, 48)])

    def test_arbitrary_transition_masks_always_cover_original(self):
        rng = np.random.default_rng(123)
        for size in (1, 5, 24, 100, 253):
            for _ in range(10):
                spans, _ = contiguous_spans(rng.random(size), rng.random(size), rng.random(size) > .85, 24)
                self.assert_complete(spans, size)
        self.assertEqual(contiguous_spans([], [], [], 24), ([], []))
