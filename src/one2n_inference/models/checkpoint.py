"""Configured architecture construction and strict author-checkpoint loading."""

from pathlib import Path

import torch

from ..config import ModelConfig
from .swin_unet import SwinUnet


def create_model(model_config: ModelConfig) -> SwinUnet:
    """Construct the network from its explicitly configured architecture."""
    return SwinUnet(**model_config.to_kwargs())


def load_model(
    checkpoint_path: str | Path, device: str | torch.device, model_config: ModelConfig
) -> SwinUnet:
    """Load trusted original-format weights and require every state to match."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()
