"""Numerical state replacement, tile ordering, and statistics regression tests."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from one2n_inference.data.statistics import Statistics, load_statistics
from one2n_inference.inference.engine import InferenceEngine
from one2n_inference.inference.packing import TilePacker


class EngineTests(unittest.TestCase):
    def test_tile_order_for_square_and_rectangular_layouts(self):
        for rows, columns in ((2, 2), (10, 10), (2, 3)):
            with self.subTest(layout=(rows, columns)):
                fine = (
                    torch.arange(5 * 3 * rows * 2 * columns * 4)
                    .reshape(1, 5, 3, rows * 2, columns * 4)
                    .float()
                )
                coarse = torch.zeros(1, 5, 3, 2, 4)
                packer = TilePacker(coarse.shape, fine.shape)
                packed = packer.model_input(coarse, fine)
                for variable in range(5):
                    for row in range(rows):
                        for column in range(columns):
                            index = 5 + variable * rows * columns + row * columns + column
                            torch.testing.assert_close(
                                packed[:, index],
                                fine[
                                    :,
                                    variable,
                                    :,
                                    row * 2 : (row + 1) * 2,
                                    column * 4 : (column + 1) * 4,
                                ],
                                atol=0,
                                rtol=0,
                            )
                torch.testing.assert_close(packer.unpack(packed), fine, atol=0, rtol=0)

    def test_statistics_preserve_precision_and_reference_scaling(self):
        with tempfile.TemporaryDirectory() as folder:
            for dtype in (torch.float32, torch.float64):
                means = torch.arange(15).reshape(5, 3).to(dtype) / 10
                variances = torch.full((5, 3), 2.0, dtype=dtype)
                path = Path(folder) / "statistics.pt"
                torch.save({"means": means, "variances": variances}, path)
                stats = load_statistics(path)
                values = torch.arange(1 * 5 * 3 * 2 * 4).reshape(1, 5, 3, 2, 4).float()
                expected = (
                    (values - means[None, :, :, None, None])
                    / torch.sqrt(variances[None, :, :, None, None] + 1e-12)
                ).float()
                normalized = stats.normalize(values, 1e-12)
                torch.testing.assert_close(normalized, expected, atol=0, rtol=0)
                physical = (
                    normalized * torch.sqrt(variances[None, :, :, None, None])
                    + means[None, :, :, None, None]
                ).float()
                torch.testing.assert_close(stats.denormalize(normalized), physical, atol=0, rtol=0)
                self.assertEqual(stats.means.dtype, dtype)

    def test_recurrence_replaces_state_and_does_not_leak_inference_mode(self):
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.patch_embed = SimpleNamespace(in_chans=25, img_size=(3, 2, 4))

            def forward(self, values):
                result = values.clone()
                result[:, :5] = 999
                result[:, 5:] = values[:, 5:] * 0.5 + 1
                return result

        initial = torch.full((1, 5, 3, 4, 8), 4.0)
        outputs = list(
            InferenceEngine(Network()).iter_predictions(torch.zeros(4, 5, 3, 2, 4), initial)
        )
        expected = initial
        for step, state in outputs:
            expected = expected * 0.5 + 1
            torch.testing.assert_close(state, expected, atol=0, rtol=0)
        self.assertEqual([step for step, _ in outputs], [1, 2, 3, 4])
        self.assertFalse(torch.is_inference_mode_enabled())

    def test_incompatible_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            TilePacker((1, 5, 3, 2, 4), (1, 5, 3, 5, 8))
