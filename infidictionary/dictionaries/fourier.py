import math
import warnings

import torch

from .base import InfiDictionary


class FourierDictionary(InfiDictionary):
    """Infinite real Fourier dictionary.

    Atoms are indexed by ``(k_1, ..., k_d, c)`` where each signed spatial
    frequency contributes a real 1-D factor:

    * ``k = 0``: constant ``1``.
    * ``k > 0``: ``sqrt(2) cos(2 pi k x)``.
    * ``k < 0``: ``sqrt(2) sin(2 pi |k| x)``.

    The spatial atom is the tensor product of those 1-D factors, placed in the
    selected output channel ``c``. The prior is the infinite isotropic power law
    ``P(k, c) proportional to (1 + ||k||_2^2)^(-steepness)``.
    """

    def __init__(
        self,
        domain_dim: int,
        num_channels: int,
        steepness: float = 2.0,
        m_max: int = 1024,
    ):
        super().__init__()
        if domain_dim < 1:
            raise ValueError(f"domain_dim must be >= 1; got {domain_dim}")
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1; got {num_channels}")
        if m_max < 0:
            raise ValueError(f"m_max must be non-negative; got {m_max}")
        self.domain_dim = int(domain_dim)
        self.num_channels = int(num_channels)
        self.steepness = float(steepness)
        self.m_max = int(m_max)

        if self.steepness <= 0.5 * self.domain_dim:
            warnings.warn(
                f"steepness={steepness} <= d/2 = {0.5 * self.domain_dim}; the "
                "power-law normalizer would diverge without the finite m_max "
                "approximation.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._Z_L2 = self._precompute_l2_shell_weights().sum().item()

    def save(self, path: str) -> None:
        """Save the constructor args; the dictionary has no learned state."""
        torch.save(
            {
                "domain_dim": self.domain_dim,
                "num_channels": self.num_channels,
                "steepness": self.steepness,
                "m_max": self.m_max,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location=None) -> "FourierDictionary":
        return cls(**torch.load(path, map_location=map_location, weights_only=False))

    # -- Shell helpers -----------------------------------------------------

    def _precompute_l2_shell_weights(self) -> torch.Tensor:
        """Shell weights for the approximate power-law normalizer."""
        d = self.domain_dim
        alpha = self.steepness
        W = torch.zeros(self.m_max + 1)
        W[0] = 1.0

        for m in range(1, self.m_max + 1):
            if d == 1:
                W[m] = 2.0 * float((1.0 + m * m) ** (-alpha))
            elif d == 2:
                k2 = torch.arange(-m, m + 1).float()
                w_rows = 2.0 * (1.0 + m * m + k2 * k2).pow(-alpha).sum().item()
                k1 = torch.arange(-(m - 1), m).float()
                w_cols = 2.0 * (1.0 + k1 * k1 + m * m).pow(-alpha).sum().item()
                W[m] = w_rows + w_cols
            else:
                shell_sz = float((2 * m + 1) ** d - (2 * m - 1) ** d)
                W[m] = shell_sz * float((1.0 + m * m) ** (-alpha))

        return W

    # -- Coordinate and atom helpers --------------------------------------

    def _spatial_atoms(
        self,
        coords: torch.Tensor,
        spatial_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate tensor-product real Fourier spatial atoms."""
        spatial_idx = spatial_idx.to(coords.device)
        vals = torch.ones(
            (spatial_idx.shape[0], coords.shape[0]),
            device=coords.device,
            dtype=coords.dtype,
        )

        scale = math.sqrt(2.0)
        for dim in range(self.domain_dim):
            freq = spatial_idx[:, dim]
            x = coords[:, dim][None, :]
            factor = torch.ones_like(vals)

            pos = freq > 0
            if pos.any():
                k = freq[pos].to(coords.dtype)[:, None]
                factor[pos] = scale * torch.cos(2.0 * math.pi * k * x)

            neg = freq < 0
            if neg.any():
                k = (-freq[neg]).to(coords.dtype)[:, None]
                factor[neg] = scale * torch.sin(2.0 * math.pi * k * x)

            vals = vals * factor

        return vals

    def _atoms(self, coords: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Evaluate real Fourier atoms."""
        idx = idx.to(coords.device)
        spatial_idx = idx[:, :-1]
        channel_idx = idx[:, -1].long()
        A, N = idx.shape[0], coords.shape[0]
        C = self.num_channels

        phi = self._spatial_atoms(coords, spatial_idx)
        vals = torch.zeros((A, N, C), device=coords.device, dtype=phi.dtype)
        valid = (channel_idx >= 0) & (channel_idx < C)
        if valid.any():
            rows = valid.nonzero(as_tuple=False).squeeze(-1)
            vals[rows, :, channel_idx[valid]] = phi[rows]
        return vals

    # -- Index-level primitives -------------------------------------------

    def _index_pmfs(self, idx: torch.Tensor) -> torch.Tensor:
        spatial_idx = idx[:, :-1]
        channel_idx = idx[:, -1]
        valid = (channel_idx >= 0) & (channel_idx < self.num_channels)
        l2sq = spatial_idx.pow(2).sum(dim=-1).float()
        base = (1.0 + l2sq).pow(-self.steepness) / self._Z_L2 / self.num_channels
        out = torch.zeros(idx.shape[0], dtype=base.dtype, device=idx.device)
        out[valid] = base[valid]
        return out

    def _compute_M_bound(self, tail_probability: float) -> int:
        scale = 1.0 / self.num_channels / self._Z_L2
        last_m = 0
        for m in range(self.m_max + 1):
            max_l2sq = float(self.domain_dim) * float(m) ** 2
            atom_pmf = float((1.0 + max_l2sq) ** (-self.steepness)) * scale
            if atom_pmf < tail_probability:
                return last_m
            last_m = m
        return self.m_max

    def _box_indices(self, max_freq: int) -> torch.Tensor:
        """All ``(k_1, ..., k_d, c)`` with ``|k_i| <= max_freq`` for every axis."""
        vals = torch.arange(-max_freq, max_freq + 1)
        grids = torch.meshgrid(*[vals] * self.domain_dim, indexing="ij")
        spatial_idx = torch.stack(grids, dim=-1).view(-1, self.domain_dim)

        C = self.num_channels
        A_s = spatial_idx.shape[0]
        channels = torch.arange(C).unsqueeze(0).expand(A_s, -1).reshape(-1)
        spatial_rep = (
            spatial_idx.unsqueeze(1)
            .expand(-1, C, -1)
            .reshape(-1, self.domain_dim)
        )
        return torch.cat([spatial_rep, channels.unsqueeze(-1)], dim=-1)

    def _high_probability_indices(self, tail_probability: float) -> torch.Tensor:
        """Return all indices with prior PMF at least ``tail_probability``.

        The selected set is an isotropic ``||k||`` ball in frequency space (the
        PMF depends only on ``||k||``), hence rotation-closed.
        """
        M = self._compute_M_bound(float(tail_probability))
        idx = self._box_indices(M)
        return idx[self._index_pmfs(idx) >= tail_probability]
