"""CPU-only checks for same-instant relationship evidence, never VLM calls."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from story_relationship_images import relationship_images, PAGE_WIDTH, ROW_HEIGHT


def fixture(folder, count=2, frames=(0, 1, 2)):
    folder = Path(folder)
    image = Image.new('RGB', (240, 160), '#123456')
    pen = ImageDraw.Draw(image)
    pen.rectangle((20, 30, 79, 119), fill='red')
    pen.rectangle((150, 40, 219, 129), fill='blue')
    for frame in frames:
        image.save(folder/f'frame_{frame}.png')
    tracks = []
    for number in range(count):
        box = [20, 30, 80, 120] if number % 2 == 0 else [150, 40, 220, 130]
        rows = [dict(frame=frame, bbox=box, image=f'frame_{frame}.png', crop='wrong_crop.png') for frame in frames]
        tracks.append(dict(track_id=f's:T{number+1}', observations=rows))
    shot = dict(shot_id='s', source=str(folder/'source.mp4'), fps=2, width=240, height=160,
                frame_count=max(frames)+1, tracks=tracks, scene_frames=[],
                mask_graph=dict(conditional_separation=[[tracks[0]['track_id'], tracks[1]['track_id']]]))
    aliases = {track['track_id']:f'T{i+1:03d}' for i,track in enumerate(tracks)}
    return dict(asset_root=str(folder), shots=[shot]), shot, aliases


class FakeFrame:
    def __init__(self, number):
        self.number = number

    def to_image(self):
        return Image.new('RGB', (240, 160), (self.number, 20, 30))


class FakeContainer:
    def __init__(self):
        self.visited = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def decode(self, **kwargs):
        for number in range(20):
            self.visited.append(number)
            yield FakeFrame(number)


class RelationshipImagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_same_frame_context_and_raw_crops_use_one_source_image(self):
        bundle, shot, aliases = fixture(self.folder)
        Image.new('RGB', (60, 90), 'green').save(self.folder/'wrong_crop.png')
        before = deepcopy(bundle)
        with patch('av.open') as decode:
            images = relationship_images(bundle, shot, self.folder, aliases)
        decode.assert_not_called()
        label, image = images[0]
        self.assertIn('T001 + T002 at f1', label)
        self.assertEqual(image.size, (PAGE_WIDTH, ROW_HEIGHT))
        evidence = image.info['relationship_pairs'][0]
        self.assertEqual(evidence['frame'], 1)
        self.assertEqual(evidence['time_sec'], .5)
        self.assertEqual(evidence['boxes'], [[20, 30, 80, 120], [150, 40, 220, 130]])
        # Both crops come from the exact source frame, never an old crop asset.
        self.assertEqual(image.getpixel((910, 280)), (255, 0, 0))
        self.assertEqual(image.getpixel((1230, 280)), (0, 0, 255))
        self.assertEqual(bundle, before)

    def test_actual_common_masks_choose_the_synchronized_frame(self):
        bundle, shot, aliases = fixture(self.folder)
        Image.new('L', (240, 160), 255).save(self.folder/'mask.png')
        for track in shot['tracks']:
            track['observations'][2]['mask'] = 'mask.png'
            track['observations'][0]['mask'] = 'nonexistent.png'
        image = relationship_images(bundle, shot, self.folder, aliases)[0][1]
        row = image.info['relationship_pairs'][0]
        self.assertEqual(row['frame'], 2)
        self.assertEqual(row['shared_mask_frames'], 1)

    def test_maximum_six_pairs_and_two_rows_per_page(self):
        bundle, shot, aliases = fixture(self.folder, count=9)
        shot['mask_graph']['conditional_separation'] = [['s:T1', f's:T{i}'] for i in range(2, 10)]
        pages = relationship_images(bundle, shot, self.folder, aliases, max_pairs=99)
        self.assertEqual(len(pages), 3)
        self.assertTrue(all(image.size == (PAGE_WIDTH, ROW_HEIGHT*2) for _,image in pages))
        self.assertEqual(sum(len(image.info['relationship_pairs']) for _,image in pages), 6)
        self.assertEqual(len(relationship_images(bundle, shot, self.folder, aliases, max_pairs=1)), 1)
        self.assertEqual(relationship_images(bundle, shot, self.folder, aliases, max_pairs=0), [])

    def test_pairs_prioritize_real_masks_then_repeated_coview(self):
        bundle, shot, aliases = fixture(self.folder, count=4)
        Image.new('L', (240, 160), 255).save(self.folder/'mask.png')
        shot['tracks'][2]['observations'] = [shot['tracks'][2]['observations'][0]]
        shot['tracks'][3]['observations'] = [shot['tracks'][3]['observations'][0]]
        for track in shot['tracks'][2:]:
            track['observations'][0]['mask'] = 'mask.png'
        shot['mask_graph']['conditional_separation'] = [['s:T1', 's:T2'], ['s:T3', 's:T4']]
        row = relationship_images(bundle, shot, self.folder, aliases, max_pairs=1)[0][1].info['relationship_pairs'][0]
        self.assertEqual(row['aliases'], ['T003', 'T004'])
        for track in shot['tracks'][2:]:
            track['observations'][0].pop('mask')
        row = relationship_images(bundle, shot, self.folder, aliases, max_pairs=1)[0][1].info['relationship_pairs'][0]
        self.assertEqual(row['aliases'], ['T001', 'T002'])

    def test_unknown_self_duplicate_and_nonconcurrent_pairs_are_skipped(self):
        bundle, shot, aliases = fixture(self.folder, count=3)
        shot['tracks'][2]['observations'][0]['frame'] = 20
        shot['tracks'][2]['observations'] = shot['tracks'][2]['observations'][:1]
        shot['mask_graph']['conditional_separation'] = [
            ['s:T1', 's:T2'], ['s:T2', 's:T1'], ['s:T1', 's:T1'],
            ['s:T1', 'missing'], ['s:T1', 's:T3']]
        pages = relationship_images(bundle, shot, self.folder, aliases)
        self.assertEqual(sum(len(image.info['relationship_pairs']) for _,image in pages), 1)
        shot['mask_graph']['conditional_separation'] = []
        with patch('av.open') as decode:
            self.assertEqual(relationship_images(bundle, shot, self.folder, aliases), [])
        decode.assert_not_called()

    def test_missing_images_decode_all_selected_frames_in_one_pass(self):
        bundle, shot, aliases = fixture(self.folder, count=4)
        for number, track in enumerate(shot['tracks']):
            track['observations'] = [dict(frame=3 if number < 2 else 7,
                bbox=[20, 30, 80, 120], image='missing.png')]
        shot['mask_graph']['conditional_separation'] = [['s:T1', 's:T2'], ['s:T3', 's:T4']]
        container = FakeContainer()
        with patch('av.open', return_value=container) as decode:
            pages = relationship_images(bundle, shot, self.folder, aliases)
        decode.assert_called_once_with(shot['source'])
        self.assertEqual(container.visited, list(range(8)))
        self.assertEqual([row['frame'] for row in pages[0][1].info['relationship_pairs']], [3, 7])

    def test_crop_sized_image_is_not_mistaken_for_source_frame(self):
        bundle, shot, aliases = fixture(self.folder, frames=(0,))
        Image.new('RGB', (30, 40), 'green').save(self.folder/'not_source.png')
        for track in shot['tracks']:
            track['observations'][0]['image'] = 'not_source.png'
        with patch('av.open', return_value=FakeContainer()) as decode:
            pages = relationship_images(bundle, shot, self.folder, aliases)
        decode.assert_called_once()
        self.assertEqual(len(pages), 1)

    def test_invalid_boxes_and_unavailable_video_do_not_create_evidence(self):
        bundle, shot, aliases = fixture(self.folder, frames=(0,))
        shot['tracks'][0]['observations'][0]['bbox'] = [3, 3, 3, 3]
        self.assertEqual(relationship_images(bundle, shot, self.folder, aliases), [])
        shot['tracks'][0]['observations'][0]['bbox'] = [float('nan'), 3, 30, 30]
        self.assertEqual(relationship_images(bundle, shot, self.folder, aliases), [])
        shot['tracks'][0]['observations'][0]['bbox'] = [3, 3, 30, 30]
        for track in shot['tracks']:
            track['observations'][0]['image'] = 'missing.png'
        with patch('av.open', side_effect=FileNotFoundError('test')), self.assertLogs('story.relationships', 'WARNING'):
            self.assertEqual(relationship_images(bundle, shot, self.folder, aliases), [])

    def test_scene_image_fallback_and_boxes_clip_to_source_extent(self):
        bundle, shot, aliases = fixture(self.folder, frames=(0,))
        for track in shot['tracks']:
            track['observations'][0].pop('image')
        shot['scene_frames'] = [dict(frame=0, image='frame_0.png')]
        shot['tracks'][0]['observations'][0]['bbox'] = [-10, -4, 300, 200]
        with patch('av.open') as decode:
            row = relationship_images(bundle, shot, self.folder, aliases)[0][1].info['relationship_pairs'][0]
        decode.assert_not_called()
        self.assertEqual(row['boxes'][0], [0, 0, 240, 160])
        self.assertEqual(row['union_box'], [0, 0, 240, 160])


if __name__ == '__main__':
    unittest.main()
