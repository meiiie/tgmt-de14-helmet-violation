"""Full-length export preserves real unconfirmed detections through audit reload."""
from pathlib import Path
import sys
import unittest
import cv2

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src'), str(Path(__file__).parent)]
import test_video_timeline as timeline_tests
from de14.video_io import probe_video, render_video


class RenderExportTest(unittest.TestCase):
    def test_json_reloaded_untracked_head_is_drawn_only_in_its_source_frame(self):
        helper = timeline_tests.VideoTimelineTest()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        source = helper.source(12, audio=True)
        result = {'timeline': {0: {'source_index': 0, 'timestamp_s': 0}}, 'per_frame': [(0, [])],
                  'frame_decisions': {0: {}}, 'tracks': {}, 'media': {'stride': 3},
                  'untracked_detections': {'0': [{'box': [15, 20, 75, 55],
                                                'heads': [([25, 22, 40, 40], .9)]}]}}
        output = render_video(source, probe_video(source), result, helper.root / 'result.mp4', preset='veryfast')
        cap = cv2.VideoCapture(output)
        ok1, first = cap.read()
        ok2, second = cap.read()
        cap.release()
        self.assertTrue(ok1 and ok2)
        self.assertGreater(float(first[22:25, 25:40, 2].mean()), 100)
        self.assertLess(float(second[22:25, 25:40, 2].mean()), 55)
        self.assertEqual(result['media']['output']['frame_count'], 12)
        self.assertEqual(result['media']['output']['fps'], 30)
        self.assertEqual(result['media']['output']['audio'][0]['codec_name'], 'aac')


if __name__ == '__main__':
    unittest.main()
