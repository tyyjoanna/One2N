"""Stateful residual evolution shared by batch and interleaved inference."""

from collections.abc import Iterator
import torch
from .packing import TilePacker


class InferenceEngine:
    def __init__(self, model: torch.nn.Module):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.state = None
        self.packer = None

    def initialize(self, initial: torch.Tensor, coarse_shape=None) -> None:
        """Initialize once; step retains the predicted state thereafter."""
        if coarse_shape is None:
            coarse_shape = (*initial.shape[:2], *self.model.patch_embed.img_size)
        packer = TilePacker(coarse_shape, initial.shape)
        if self.model.patch_embed.in_chans != packer.input_channels:
            raise ValueError("Fine-grid tile count does not match the model's input channels")
        if tuple(coarse_shape[2:]) != tuple(self.model.patch_embed.img_size):
            raise ValueError(
                "Coarse-grid pressure/latitude/longitude shape does not match the model"
            )
        self.packer = packer
        self.state = initial.to(self.device)

    def step(self, coarse_increment: torch.Tensor) -> torch.Tensor:
        """Replace the normalized residual using one native increment."""
        if self.packer is None or self.state is None:
            raise RuntimeError("Initialize the residual state before advancing the model")
        with torch.inference_mode():
            model_input = self.packer.model_input(coarse_increment.to(self.device), self.state)
            output = self.model(model_input)
            self.state = self.packer.unpack(output)
        return self.state

    def iter_predictions(self, coarse, initial) -> Iterator[tuple[int, torch.Tensor]]:
        self.initialize(initial, coarse[:1].shape)
        del initial
        for index, increment in enumerate(coarse.split(1), start=1):
            yield index, self.step(increment)
