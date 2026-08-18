from __future__ import annotations

import importlib.util
import random
import sys
import types
from pathlib import Path

import torch
from PIL import Image


DFINE_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "D-FINE"


def _package(name: str, path: Path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Avoid importing src/__init__.py, which eagerly imports every D-FINE zoo model.
_package("dfine_focus", DFINE_ROOT / "src")
_package("dfine_focus.data", DFINE_ROOT / "src" / "data")
_package("dfine_focus.data.transforms", DFINE_ROOT / "src" / "data" / "transforms")
_package("dfine_focus.nn", DFINE_ROOT / "src" / "nn")
_package("dfine_focus.nn.backbone", DFINE_ROOT / "src" / "nn" / "backbone")
core = types.ModuleType("dfine_focus.core")


def _register(name=None):
    if callable(name):
        return name

    def decorator(value):
        return value

    return decorator


core.register = _register
sys.modules["dfine_focus.core"] = core
misc = _load("dfine_focus.data._misc", DFINE_ROOT / "src" / "data" / "_misc.py")
copy_paste = _load(
    "dfine_focus.data.transforms.multimodal_copy_paste",
    DFINE_ROOT / "src" / "data" / "transforms" / "multimodal_copy_paste.py",
)
_load(
    "dfine_focus.nn.backbone.common",
    DFINE_ROOT / "src" / "nn" / "backbone" / "common.py",
)
hgnet = _load(
    "dfine_focus.nn.backbone.hgnetv2",
    DFINE_ROOT / "src" / "nn" / "backbone" / "hgnetv2.py",
)

convert_to_tv_tensor = misc.convert_to_tv_tensor
MultimodalTailCopyPaste = copy_paste.MultimodalTailCopyPaste
SanitizeMultimodalDepth = copy_paste.SanitizeMultimodalDepth
DepthValidityGatedResidualFusion = hgnet.DepthValidityGatedResidualFusion




class _DonorDataset:
    input_mode = "rgbtd"

    def indices_for_classes(self, classes):
        return [0]

    def load_item(self, index):
        images = [Image.new("RGB", (32, 32), color) for color in ((255, 0, 0), (64, 64, 64), (128, 255, 255))]
        target = {
            "boxes": convert_to_tv_tensor(
                torch.tensor([[8.0, 8.0, 16.0, 16.0]]),
                key="boxes",
                box_format="xyxy",
                spatial_size=[32, 32],
            ),
            "labels": torch.tensor([11]),
            "area": torch.tensor([64.0]),
            "iscrowd": torch.tensor([0]),
        }
        return images, target


def test_copy_paste_updates_all_modalities_and_target():
    random.seed(4)
    dataset = _DonorDataset()
    images = [Image.new("RGB", (32, 32), (0, 0, 0)) for _ in range(3)]
    target = {
        "boxes": convert_to_tv_tensor(
            torch.empty(0, 4), key="boxes", box_format="xyxy", spatial_size=[32, 32]
        ),
        "labels": torch.empty(0, dtype=torch.int64),
        "area": torch.empty(0),
        "iscrowd": torch.empty(0, dtype=torch.int64),
    }
    transform = MultimodalTailCopyPaste(p=1, max_paste=1, position_jitter=0)
    *output_images, output_target, _ = transform((*images, target, dataset))
    assert output_target["labels"].tolist() == [11]
    assert all(image.getbbox() is not None for image in output_images)


def test_depth_sanitizer_restores_binary_channels():
    dataset = _DonorDataset()
    depth = torch.tensor([[[0.8]], [[0.49]], [[0.51]]])
    sample = (torch.zeros(3, 1, 1), torch.zeros(3, 1, 1), depth, {}, dataset)
    *images, _, _ = SanitizeMultimodalDepth()(sample)
    assert images[-1][:, 0, 0].tolist() == [0.0, 0.0, 1.0]


def test_depth_fusion_is_identity_at_initialization_and_trains_output_layer():
    block = DepthValidityGatedResidualFusion(32, 16)
    rgb = torch.randn(2, 32, 9, 13, requires_grad=True)
    depth = torch.randn(2, 32, 9, 13, requires_grad=True)
    valid = torch.ones(2, 1, 18, 26)
    metric = torch.ones(2, 1, 18, 26)
    output = block(rgb, depth, valid, metric)
    torch.testing.assert_close(output, rgb, rtol=0, atol=0)
    output.square().mean().backward()
    gradient = block.residual[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


if __name__ == "__main__":
    test_copy_paste_updates_all_modalities_and_target()
    test_depth_sanitizer_restores_binary_channels()
    test_depth_fusion_is_identity_at_initialization_and_trains_output_layer()
    print("E13 D-FINE tests passed")
