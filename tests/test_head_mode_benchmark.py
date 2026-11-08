"""Exercise benchmark export/metrics without downloading or running neural models."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'src'), str(Path(__file__).resolve().parents[1] / "scripts")]
import benchmark_head_modes as benchmark
from de14.runtime import RuntimeBundle, RuntimeConfig
from de14.detectors import HeadEnhancedDetector
from de14.classifiers import VerifiedHelmetCls


class HeadBenchmarkTest(unittest.TestCase):
    def test_report_exports_head_and_official_metrics_for_all_modes(self):
        config = RuntimeConfig(device='cpu', detect_heads=True)
        body = lambda _: {'motorcycle': [[5, 25, 60, 63]], 'motorcycle_score': [.9],
                          'person': [[10, 1, 58, 60]], 'person_score': [.9]}
        head = lambda _: [(.9, [20, 1, 55, 36])]
        base = lambda cs: [.1]*len(cs)
        strict = RuntimeBundle(HeadEnhancedDetector(body, head), VerifiedHelmetCls(base, lambda cs: [-3.]*len(cs)), config, 'cpu')
        def runtime(cfg, reuse=None):
            return strict if cfg.accurate else RuntimeBundle(strict.detector, VerifiedHelmetCls(base, None), cfg, 'cpu')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); image = root/'input.png'
            cv2.imwrite(str(image), np.zeros((64,64,3), np.uint8))
            manifest = root/'heads.json'; official = root/'official.json'; output = root/'result'
            manifest.write_text(json.dumps({'images':[{'id':'fixture','image':str(image), 'image_sha256':'fixturehash',
                                                      'gt_heads':[{'vehicle':'one','box':[20,1,55,36]}]}]}), encoding='utf-8')
            official.write_text(json.dumps({'source_resolution':[64,64], 'frames':[{'clip':'fixture','frame':0,
                'image':str(image), 'sha256':'fixturehash', 'vehicles':[{'box':[5,1,60,63], 'n_riders':1, 'violation':False}]}]}), encoding='utf-8')
            argv = ['benchmark', '--manifest',str(manifest),'--official-manifest',str(official), '--out',str(output),'--repeats','1']
            # Stub only external models and unrelated weight hashes. Real pipeline,
            # metric matching, image export, CLI and JSON writer run end to end.
            with patch.object(benchmark,'load_profile',return_value=config), patch.object(benchmark,'build_runtime',side_effect=runtime), \
                 patch.object(benchmark,'digest',return_value='fixturehash'), patch.object(sys,'argv',argv), redirect_stdout(io.StringIO()):
                benchmark.main()
            report = json.loads((output/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(set(report['official']['modes']), {'legacy_estimated','tight_dino','tight_clip'})
            self.assertEqual(report['official']['modes']['tight_dino']['metrics']['counts']['gt_vehicles'],1)
            self.assertEqual(report['modes']['tight_dino']['head_metrics']['recall'],1.)
            self.assertTrue((output/'fixture_comparison.png').is_file())


if __name__ == '__main__': unittest.main()
