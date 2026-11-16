"""Normalization and physical output scaling used by the trained checkpoint."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Statistics:
    means: torch.Tensor
    variances: torch.Tensor

    def normalize(self, values: torch.Tensor, epsilon: float) -> torch.Tensor:
        return ((values - self.means) / torch.sqrt(self.variances + epsilon)).to(torch.float32)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        # The reference uses epsilon for normalization, but not physical scaling.
        return (values * torch.sqrt(self.variances) + self.means).to(torch.float32)


def load_statistics(filename: str | Path) -> Statistics:
    """Read trusted author statistics, preserving their stored arithmetic dtype."""
    raw = torch.load(filename, map_location="cpu", weights_only=False)
    means, variances = torch.as_tensor(raw["means"]), torch.as_tensor(raw["variances"])
    if means.shape != variances.shape:
        raise ValueError("Statistics means and variances must have matching shapes")
    if means.ndim == 2:
        means, variances = means[None, :, :, None, None], variances[None, :, :, None, None]
    return Statistics(means, variances)
