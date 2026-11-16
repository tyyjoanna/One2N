"""Tile layout defining the interface between coarse and fine model fields."""

import torch


class TilePacker:
    """Preserve variable, latitude-tile, longitude-tile channel ordering."""

    def __init__(self, coarse_shape, fine_shape):
        self.batch, self.channels, self.levels, self.height, self.width = coarse_shape
        if tuple(coarse_shape[:3]) != tuple(fine_shape[:3]):
            raise ValueError(
                "Coarse and fine states must share batch, variables, and pressure levels"
            )
        fine_height, fine_width = fine_shape[-2:]
        if fine_height % self.height or fine_width % self.width:
            raise ValueError("Fine-grid dimensions must be divisible by the coarse-grid dimensions")
        self.latitude_tiles = fine_height // self.height
        self.longitude_tiles = fine_width // self.width

    @property
    def input_channels(self) -> int:
        return self.channels * (1 + self.latitude_tiles * self.longitude_tiles)

    def pack(self, fine: torch.Tensor) -> torch.Tensor:
        patches = fine.unfold(3, self.height, self.height).unfold(4, self.width, self.width)
        return patches.permute(0, 1, 3, 4, 2, 5, 6).reshape(
            self.batch, -1, self.levels, self.height, self.width
        )

    def unpack(self, output: torch.Tensor) -> torch.Tensor:
        patches = output[:, self.channels :].reshape(
            self.batch,
            self.channels,
            self.latitude_tiles,
            self.longitude_tiles,
            self.levels,
            self.height,
            self.width,
        )
        return patches.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            self.batch,
            self.channels,
            self.levels,
            self.latitude_tiles * self.height,
            self.longitude_tiles * self.width,
        )

    def model_input(self, coarse_increment: torch.Tensor, fine: torch.Tensor) -> torch.Tensor:
        return torch.cat([coarse_increment, self.pack(fine)], dim=1)
