"""Isotropic tensor-product Haar wavelet dictionary on ``[0, 1]^d``."""

import torch

from .base import InfiDictionary


class HaarWaveletDictionary(InfiDictionary):
    """Infinite Haar basis indexed by ``[level, shifts..., orientation, channel]``.

    ``level=-1`` denotes the per-channel constant scaling function.  At level
    ``j >= 0``, ``shifts`` lies in ``[0, 2**j)^d`` and ``orientation`` is a
    non-zero bit mask: bit ``i`` selects the 1-D Haar wavelet on axis ``i``.
    The PMF gives total mass ``scaling_mass`` to scaling atoms and distributes
    the remaining mass geometrically over wavelet levels.
    """

    def __init__(
        self,
        domain_dim: int,
        num_channels: int,
        scaling_mass: float = 0.5,
        level_decay: float = 0.5,
    ):
        super().__init__()
        if domain_dim < 1 or num_channels < 1:
            raise ValueError("domain_dim and num_channels must both be >= 1")
        if not 0.0 < scaling_mass < 1.0:
            raise ValueError("scaling_mass must be in (0, 1)")
        if not 0.0 < level_decay < 1.0:
            raise ValueError("level_decay must be in (0, 1)")
        self.domain_dim = int(domain_dim)
        self.num_channels = int(num_channels)
        self.scaling_mass = float(scaling_mass)
        self.level_decay = float(level_decay)

    @property
    def _index_width(self) -> int:
        return self.domain_dim + 3

    def _wavelet_pmf(self, level: torch.Tensor) -> torch.Tensor:
        count = self.num_channels * (2 ** self.domain_dim - 1) * (2 ** (self.domain_dim * level))
        return ((1.0 - self.scaling_mass) * (1.0 - self.level_decay) * self.level_decay ** level) / count

    def _atoms(self, coords: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.to(coords.device)
        if idx.ndim != 2 or idx.shape[1] != self._index_width:
            raise ValueError(f"idx must have shape (A, {self._index_width})")
        level, shifts = idx[:, 0].long(), idx[:, 1:1 + self.domain_dim].long()
        orientation, channel = idx[:, -2].long(), idx[:, -1].long()
        valid_channel = (channel >= 0) & (channel < self.num_channels)
        scaling = (level == -1) & (shifts == 0).all(dim=1) & (orientation == 0) & valid_channel
        # All atoms are evaluated in one batch (no per-atom Python loop or host
        # syncs). Scaling atoms are the constant 1; a wavelet at level j is
        # 2^(j d/2) times a product over axes of the indicator of its cell and,
        # on the axes selected by the orientation bits, a ±1 Haar sign.
        cells = torch.exp2(level.clamp(min=0).to(coords.dtype))                       # (A,)
        local = coords[None] * cells[:, None, None] - shifts[:, None].to(coords.dtype)  # (A, N, d)
        inside = ((local >= 0) & (local < 1)).all(dim=-1)                              # (A, N)
        axis_bits = (orientation[:, None] >> torch.arange(self.domain_dim, device=idx.device)) & 1
        signs = torch.where((local < 0.5) | (axis_bits[:, None, :] == 0), 1.0, -1.0).prod(dim=-1)
        wavelet = (
            (level >= 0) & valid_channel
            & (orientation >= 1) & (orientation < 2 ** self.domain_dim)
            & (shifts >= 0).all(dim=1) & (shifts < cells[:, None]).all(dim=1)
        )
        amplitude = torch.where(wavelet, cells ** (self.domain_dim / 2), scaling.to(coords.dtype))
        factor = torch.where(wavelet[:, None], signs * inside, 1.0) * amplitude[:, None]  # (A, N)
        channel_mask = channel[:, None] == torch.arange(self.num_channels, device=idx.device)
        return factor[:, :, None] * channel_mask[:, None, :].to(coords.dtype)

    def _index_pmfs(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.to(torch.long)
        out = torch.zeros(idx.shape[0], dtype=torch.get_default_dtype(), device=idx.device)
        if idx.ndim != 2 or idx.shape[1] != self._index_width:
            raise ValueError(f"idx must have shape (A, {self._index_width})")
        level, shifts = idx[:, 0], idx[:, 1:1 + self.domain_dim]
        orientation, channel = idx[:, -2], idx[:, -1]
        valid_channel = (channel >= 0) & (channel < self.num_channels)
        scaling = (level == -1) & (shifts == 0).all(dim=1) & (orientation == 0) & valid_channel
        out[scaling] = self.scaling_mass / self.num_channels
        wavelet = (level >= 0) & (orientation >= 1) & (orientation < 2 ** self.domain_dim) & valid_channel
        if wavelet.any():
            rows = wavelet.nonzero(as_tuple=False).squeeze(-1)
            cells = 2 ** level[rows, None]
            valid_shift = (shifts[rows] >= 0).all(dim=1) & (shifts[rows] < cells).all(dim=1)
            rows = rows[valid_shift]
            out[rows] = self._wavelet_pmf(level[rows])
        return out

    def _level_indices(self, level: int) -> torch.Tensor:
        cells = 2 ** level
        axes = torch.arange(cells)
        shifts = torch.stack(torch.meshgrid(*([axes] * self.domain_dim), indexing="ij"), dim=-1).reshape(-1, self.domain_dim)
        orientations = torch.arange(1, 2 ** self.domain_dim)
        channels = torch.arange(self.num_channels)
        base = shifts[:, None, None, :].expand(-1, len(orientations), self.num_channels, -1).reshape(-1, self.domain_dim)
        orient = orientations[None, :, None].expand(shifts.shape[0], -1, self.num_channels).reshape(-1, 1)
        channel = channels[None, None, :].expand(shifts.shape[0], len(orientations), -1).reshape(-1, 1)
        return torch.cat([torch.full((base.shape[0], 1), level), base, orient, channel], dim=1)

    def _high_probability_indices(self, tail_probability: float) -> torch.Tensor:
        if tail_probability <= 0.0:
            raise ValueError("tail_probability must be positive for the infinite Haar prior")
        rows = []
        if self.scaling_mass / self.num_channels >= tail_probability:
            channels = torch.arange(self.num_channels)[:, None]
            rows.append(torch.cat([torch.full((self.num_channels, 1), -1), torch.zeros(self.num_channels, self.domain_dim + 1, dtype=torch.long), channels], dim=1))
        j = 0
        while float(self._wavelet_pmf(torch.tensor(j))) >= tail_probability:
            rows.append(self._level_indices(j))
            j += 1
        return torch.cat(rows, dim=0) if rows else torch.empty((0, self._index_width), dtype=torch.long)

    def save(self, path: str) -> None:
        """Save the constructor args; the dictionary has no learned state."""
        torch.save({"domain_dim": self.domain_dim, "num_channels": self.num_channels, "scaling_mass": self.scaling_mass, "level_decay": self.level_decay}, path)

    @classmethod
    def load(cls, path: str, map_location=None) -> "HaarWaveletDictionary":
        return cls(**torch.load(path, map_location=map_location, weights_only=False))
