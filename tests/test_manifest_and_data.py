from __future__ import annotations

import unittest
from pathlib import Path

import torch

from aic_baseline.data import AICDataset, collate_batch
from aic_baseline.data.manifest import derive_group_id, read_jsonl


class ManifestTests(unittest.TestCase):
    def test_group_derivation(self):
        self.assertEqual(derive_group_id("000003_080_00000409", ".png"), "seq_000003_080")
        self.assertEqual(derive_group_id("hehe_146_000002_010_00000166", ".png"), "seq_000002_010")
        self.assertEqual(derive_group_id("00000104", ".jpg"), "numeric_jpg_000004")

    def test_prepared_split_has_no_group_overlap(self):
        root = Path("data/prepared/manifests")
        if not (root / "train.jsonl").exists():
            self.skipTest("Run prepare_data.py first")
        train = read_jsonl(root / "train.jsonl")
        validation = read_jsonl(root / "val.jsonl")
        self.assertFalse({item["group"] for item in train} & {item["group"] for item in validation})
        self.assertEqual(len(train) + len(validation), 2000)

    def test_real_archive_dataset(self):
        manifest = Path("data/prepared/manifests/train.jsonl")
        if not manifest.exists():
            self.skipTest("Run prepare_data.py first")
        dataset = AICDataset(manifest, "data", 320, "rgbtd", augment=False)
        try:
            images, targets, metadata = dataset[0]
            self.assertEqual(images["rgb"].shape, (3, 320, 320))
            self.assertEqual(images["infrared"].shape, (2, 320, 320))
            self.assertEqual(images["depth"].shape, (3, 320, 320))
            self.assertTrue(torch.isfinite(images["depth"]).all())
            self.assertEqual(targets.shape[1], 5)
            batch = collate_batch([dataset[0], dataset[1]])
            self.assertEqual(batch[0]["rgb"].shape, (2, 3, 320, 320))
            self.assertTrue((batch[1][:, 0] <= 1).all())
            self.assertIn("original_shape", metadata)
        finally:
            dataset.close()

    def test_real_archive_ir_only_dataset(self):
        manifest = Path("data/prepared/manifests/train.jsonl")
        if not manifest.exists():
            self.skipTest("Run prepare_data.py first")
        dataset = AICDataset(manifest, "data", 320, "ir", augment=False)
        try:
            images, targets, _ = dataset[0]
            self.assertEqual(set(images), {"infrared"})
            self.assertEqual(images["infrared"].shape, (2, 320, 320))
            self.assertEqual(targets.shape[1], 5)
        finally:
            dataset.close()


if __name__ == "__main__":
    unittest.main()

