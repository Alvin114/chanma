from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

from aic_baseline.engine.metrics import compute_metrics, non_max_suppression, write_submission


class MetricsTests(unittest.TestCase):
    def test_perfect_prediction_has_perfect_class_ap(self):
        correct = np.ones((1, 10), dtype=bool)
        metrics = compute_metrics([(correct, np.array([0.9]), np.array([0.0]), np.array([0.0]))])
        self.assertAlmostEqual(metrics["per_class"]["person"]["ap50_95"], 1.0)
        self.assertAlmostEqual(metrics["map50_95"], 1.0 / 12.0)

    def test_nms_is_class_aware(self):
        prediction = torch.zeros((1, 3, 7))
        prediction[0, 0] = torch.tensor([50, 50, 20, 20, 0.9, 0.9, 0.0])
        prediction[0, 1] = torch.tensor([50, 50, 20, 20, 0.8, 0.8, 0.0])
        prediction[0, 2] = torch.tensor([50, 50, 20, 20, 0.7, 0.0, 0.9])
        output = non_max_suppression(prediction, 0.1, 0.5, 100)
        self.assertEqual(len(output[0]), 2)
        self.assertEqual(set(output[0][:, 5].tolist()), {0.0, 1.0})

    def test_submission_zip_has_root_level_txt_and_six_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = {
                "a": torch.tensor([[0.1, 0.2, 0.5, 0.6, 0.8, 2.0]]),
                "b": torch.zeros((0, 6)),
            }
            archive = write_submission(predictions, root / "txt", root / "submission.zip")
            line = (root / "txt" / "a.txt").read_text().strip().split()
            self.assertEqual(len(line), 6)
            self.assertEqual((root / "txt" / "b.txt").read_text(), "")
            with zipfile.ZipFile(archive) as zipped:
                self.assertEqual(sorted(zipped.namelist()), ["a.txt", "b.txt"])


if __name__ == "__main__":
    unittest.main()

