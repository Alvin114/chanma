from pathlib import Path
import sys

import torch


DFINE_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "D-FINE"
sys.path.insert(0, str(DFINE_ROOT))

from src.data.dataloader import generate_scales  # noqa: E402
from src.nn.backbone.hgnetv2 import (  # noqa: E402
    WaveletGatedResidualFusion,
    _haar_dwt2,
    _haar_idwt2,
)


def test_haar_round_trip_supports_odd_shapes():
    for height, width in ((32, 48), (31, 47)):
        x = torch.randn(2, 8, height, width)
        bands, original_size = _haar_dwt2(x)
        restored = _haar_idwt2(bands, original_size)
        torch.testing.assert_close(restored, x, rtol=1e-5, atol=1e-6)


def test_wavelet_fusion_starts_as_rgb_identity_and_receives_gradients():
    block = WaveletGatedResidualFusion(channels=32, hidden_channels=16)
    rgb = torch.randn(2, 32, 17, 29, requires_grad=True)
    auxiliary = torch.randn(2, 32, 17, 29, requires_grad=True)

    output = block(rgb, auxiliary)
    torch.testing.assert_close(output, rgb, rtol=0, atol=0)

    output.square().mean().backward()
    assert block.high_scale.grad is not None
    assert torch.isfinite(block.high_scale.grad).all()
    assert block.high_scale.grad.abs().sum() > 0
    last_low_conv = block.low_fusion[-1]
    assert last_low_conv.weight.grad is not None
    assert torch.isfinite(last_low_conv.weight.grad).all()
    assert last_low_conv.weight.grad.abs().sum() > 0


def test_rectangular_scale_generation_preserves_aspect_ratio():
    scales = generate_scales([864, 1536], 5)
    assert scales.count((864, 1536)) == 5
    for height, width in scales:
        assert height % 32 == 0
        assert width % 32 == 0
        # Width is rounded to a stride-32 boundary.
        assert abs(width / height - 16 / 9) <= 16 / height


if __name__ == "__main__":
    test_haar_round_trip_supports_odd_shapes()
    test_wavelet_fusion_starts_as_rgb_identity_and_receives_gradients()
    test_rectangular_scale_generation_preserves_aspect_ratio()
    print("E12 focused regression tests passed")
