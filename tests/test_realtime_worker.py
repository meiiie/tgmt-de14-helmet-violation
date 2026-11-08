"""Classifier errors must not kill the worker or make drain() hang. No dataset required."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from de14.realtime import RealtimePipeline

class WorkerTest(unittest.TestCase):
    def test_worker_recovers_after_classifier_error(self):
        class Boom:
            n = 0
            def __call__(self, crops):
                self.n += 1
                if self.n == 1: raise RuntimeError("simulated classifier failure")
                return [.9] * len(crops)
        # This worker regression does not invoke the detector or require its weights.
        with patch("ultralytics.YOLO"):
            pl = RealtimePipeline(classifier=Boom(), device="cpu")
        try:
            pl.q.put((1, [np.zeros((32, 32, 3), np.uint8)]))
            self.assertTrue(pl.drain(timeout=5))
            self.assertIsNotNone(pl.err)
            self.assertTrue(pl.w.is_alive())
            pl.q.put((2, [np.zeros((32, 32, 3), np.uint8)]))
            self.assertTrue(pl.drain(timeout=5))
            self.assertEqual(pl.votes[2], [.9])
        finally:
            pl.close()
        self.assertFalse(pl.w.is_alive())

if __name__ == "__main__":
    unittest.main()
