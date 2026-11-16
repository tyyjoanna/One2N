"""Small CPU architecture, strict loading, and original-reference regression."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from one2n_inference.config import load_config
from one2n_inference.constants import VARIABLES
from one2n_inference.models import create_model, load_model
from _fixtures import EPOCH

ROOT = Path(__file__).resolve().parents[1]


def small_model_config(**overrides):
    configured = load_config(ROOT / "configs/demo.yaml").model
    return replace(configured, in_chans=len(VARIABLES), **overrides)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = load_config(ROOT / "configs/demo.yaml")
        torch.set_num_threads(cls.demo.synthetic.torch_threads)

    def test_eval_forward_shapes_and_determinism(self):
        for ape, patch_norm in [(True, True), (False, True), (True, False)]:
            with self.subTest(ape=ape, patch_norm=patch_norm):
                torch.manual_seed(self.demo.synthetic.seed)
                settings = small_model_config(ape=ape, patch_norm=patch_norm)
                model = create_model(settings).eval()
                inputs = torch.randn(1, len(VARIABLES), *settings.img_size)
                with torch.inference_mode():
                    expected = model(inputs)
                    actual = model(inputs)
                self.assertEqual(actual.shape, inputs.shape)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_checkpoint_round_trip(self):
        torch.manual_seed(self.demo.synthetic.seed)
        settings = small_model_config()
        original = create_model(settings).eval()
        inputs = torch.randn(1, len(VARIABLES), *settings.img_size)
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / "checkpoint.pt"
            torch.save({"model_state_dict": original.state_dict()}, filename)
            loaded = load_model(filename, "cpu", settings)
            self.assertFalse(loaded.training)
            with torch.inference_mode():
                torch.testing.assert_close(loaded(inputs), original(inputs), atol=0, rtol=0)

    def test_original_model_reference(self):
        fixture = Path(__file__).parent / "fixtures/original_model_reference.npz"
        with np.load(fixture, allow_pickle=False) as reference:
            state = {
                name.removeprefix("state::"): torch.from_numpy(reference[name].copy())
                for name in reference.files
                if name.startswith("state::")
            }
            inputs = torch.from_numpy(reference["inputs"].copy())
            expected = torch.from_numpy(reference["expected"].copy())
        model = create_model(small_model_config()).eval()
        model.load_state_dict(state, strict=True)
        with torch.inference_mode():
            actual = model(inputs)
        torch.testing.assert_close(
            actual, expected, atol=self.demo.synthetic.atol, rtol=self.demo.synthetic.rtol
        )

    def test_checkpoint_rejects_missing_parameter(self):
        settings = small_model_config()
        state = create_model(settings).state_dict()
        state.pop("patch_embed.proj.weight")
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / "incomplete.pt"
            torch.save({"model_state_dict": state}, filename)
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                load_model(filename, "cpu", settings)

    def test_reference_checkpoint_geometry_without_allocating_weights(self):
        linspace = torch.linspace

        def cpu_linspace(*args, **kwargs):
            return linspace(*args, **(kwargs | {"device": "cpu"}))

        with patch.object(torch, "linspace", cpu_linspace), torch.device("meta"):
            model = create_model(
                load_config(ROOT / "configs/inference.yaml", start_time=EPOCH).model
            )
        state = model.state_dict()
        self.assertEqual(len(state), 197)
        self.assertEqual(tuple(state["absolute_pos_embed"].shape), (1, 2560, 128))
        self.assertEqual(tuple(state["patch_embed.proj.weight"].shape), (128, 505, 5, 16, 16))
        self.assertEqual(tuple(state["patch_recovery.proj.weight"].shape), (646400, 128, 1, 1, 1))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 174_016_176)


if __name__ == "__main__":
    unittest.main()
