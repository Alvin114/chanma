from __future__ import annotations

import unittest

import torch

from aic_baseline.models import M2DDistillationLoss, MultiModalYOLO, YoloLoss


class ModelTests(unittest.TestCase):
    def _images(self, mode: str):
        images = {"rgb": torch.rand(1, 3, 320, 320)}
        if mode == "ir":
            return {"infrared": torch.rand(1, 2, 320, 320)}
        if "t" in mode:
            images["infrared"] = torch.rand(1, 2, 320, 320)
        if "d" in mode:
            images["depth"] = torch.rand(1, 3, 320, 320)
        return images

    def test_all_ablation_modes_forward_backward(self):
        target = torch.tensor([[0, 0, 0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
        for mode in ("rgb", "ir", "rgbt", "rgbd", "rgbtd"):
            with self.subTest(mode=mode):
                model = MultiModalYOLO(mode, width=0.25, depth=0.33, fusion_heads=4)
                model.train()
                prediction = model(self._images(mode))
                self.assertEqual([tuple(item.shape) for item in prediction], [(1, 3, 40, 40, 17), (1, 3, 20, 20, 17), (1, 3, 10, 10, 17)])
                loss, components = YoloLoss(model, {"box": 0.05, "obj": 1.0, "cls": 0.5, "anchor_t": 4.0})(prediction, target)
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(components.shape, (3,))
                loss.backward()
                self.assertIsNotNone(next(model.parameters()).grad)
                model.eval()
                with torch.no_grad():
                    decoded, raw = model(self._images(mode))
                self.assertEqual(decoded.shape[0], 1)
                self.assertEqual(decoded.shape[2], 17)
                self.assertEqual(len(raw), 3)

    def test_lif_returns_modal_features_and_illumination(self):
        model = MultiModalYOLO("rgbt", width=0.25, depth=0.33, fusion_type="lif")
        model.train()
        predictions, auxiliary = model(self._images("rgbt"), return_aux=True)
        self.assertEqual(len(predictions), 3)
        self.assertEqual(tuple(auxiliary["illumination"].shape), (1, 1, 40, 40))
        self.assertEqual(set(auxiliary["modal_features"]), {"rgb", "ir"})
        depth_model = MultiModalYOLO("rgbtd", width=0.25, depth=0.33, fusion_type="lif")
        depth_model.train()
        depth_predictions, depth_auxiliary = depth_model(self._images("rgbtd"), return_aux=True)
        self.assertEqual(len(depth_predictions), 3)
        self.assertEqual(set(depth_auxiliary["modal_features"]), {"rgb", "ir", "depth"})

    def test_m2d_distillation_is_finite_and_backpropagates(self):
        student_rgb = tuple(
            torch.rand(2, channels, size, size, requires_grad=True)
            for channels, size in ((16, 20), (32, 10), (64, 5))
        )
        student_ir = tuple(
            torch.rand(2, channels, size, size, requires_grad=True)
            for channels, size in ((16, 20), (32, 10), (64, 5))
        )
        teacher_rgb = tuple(torch.rand_like(feature) for feature in student_rgb)
        teacher_ir = tuple(torch.rand_like(feature) for feature in student_ir)
        intra, cross = M2DDistillationLoss()(student_rgb, student_ir, teacher_rgb, teacher_ir)
        self.assertTrue(torch.isfinite(intra))
        self.assertTrue(torch.isfinite(cross))
        (intra + cross).backward()
        self.assertIsNotNone(student_rgb[0].grad)
        self.assertIsNotNone(student_ir[0].grad)


if __name__ == "__main__":
    unittest.main()

