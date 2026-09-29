from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal

from .base import InfiDictionary


class VoronoiPWC(InfiDictionary):
    """Piecewise-constant Voronoi dictionary with an optional synthesis.

    The domain is partitioned into ``L`` Voronoi cells around fixed
    ``centroids``.  For a regular block partition of ``[0, 1]^d``, pass
    ``resolution`` and ``dimension`` instead: this constructs a regular grid
    of cell centers and unit-norm block indicators automatically. The *base*
    atoms are indexed by ``(l, c)`` (cell ``l``,
    channel ``c``) and are piecewise-constant indicators:

        base_{l,c}(x)[c'] = amplitude[l] · 1[x ∈ cell l] · 1[c' = c],

    i.e. the constant ``amplitude[l]`` on Voronoi cell ``l`` in channel ``c``,
    zero elsewhere.  Flattening the multi-index as ``m = l·C + c`` gives
    ``M_in = L·C`` base atoms; cells have disjoint support and channels are
    distinct, so they are pairwise orthogonal.

    When ``learn_synthesis=True``, a semi-orthogonal synthesis
    ``W ∈ C^{num_atoms × M_in}`` mixes the base down to ``num_atoms`` output
    atoms:

        atom_a = Σ_m W[a, m] · base_m ,   a = 0 … num_atoms-1 .

    ``W`` has orthonormal *rows* (``W Wᴴ = I``), so the ``num_atoms`` outputs are
    orthonormal under the same (empirical ν) measure that makes the base atoms
    orthonormal.  ``num_atoms == M_in`` recovers the full-rank square orthogonal map.

    :meth:`get_atoms` evaluates the requested *synthesized* atoms directly: at a
    point in cell ``l`` only the base atoms of that cell are non-zero, so

        atom_a(x)[c'] = amplitude[cell(x)] · W[a, cell(x)·C + c'] ,

    which is computed with a single gather (no ``(num_atoms, N, C)`` blow-up).

    **Index set & PMF.**  Finite: ``num_atoms`` output atoms, indexed by a single
    integer ``a`` in ``idx[:, 0]`` (``idx[:, 1]`` is retained for the ``(A, 2)``
    convention shared with other dictionaries but ignored).  The per-atom PMF is
    ``probs[a]``.

    Args:
        centroids:    Explicit cell centers, shape ``(L, d)``. Must be supplied
                      together with ``amplitudes``, unless using the regular
                      grid constructor.
        amplitudes:   Explicit per-cell amplitudes, shape ``(L,)``.
        resolution:   Number of equal-width blocks per axis in the regular
                      grid constructor. Requires ``dimension`` and cannot be
                      combined with explicit ``centroids`` or ``amplitudes``.
        dimension:    Dimension of the regular grid constructor. Requires
                      ``resolution``.
        num_channels: Number of output channels ``C``.
        num_atoms:    Number of synthesized output atoms.  ``None`` (default) ⇒
                      ``M_in = L·C`` (full-rank square synthesis).  Must satisfy
                      ``1 <= num_atoms <= L·C``.
        probs:        Per-output-atom prior weights, shape ``(num_atoms,)``.
                      Non-negative; normalized to sum to 1.  ``None`` (default) ⇒
                      harmonic ``p_i ∝ 1/(i+1)`` over synthesized atom indices.
        learn_synthesis: Whether the semi-orthogonal synthesis is trainable.
                      ``False`` gives the fixed, plain tessellated PWC basis
                      and requires ``num_atoms == L*C``.
    """

    def __init__(
        self,
        centroids: Optional[torch.Tensor] = None,  # (L, d)
        amplitudes: Optional[torch.Tensor] = None, # (L,)
        num_channels: int = 1,
        num_atoms: Optional[int] = None,
        probs: Optional[torch.Tensor] = None,   # (num_atoms,)
        learn_synthesis: bool = True,
        resolution: Optional[int] = None,
        dimension: Optional[int] = None,
    ):
        super().__init__()

        using_regular_grid = resolution is not None or dimension is not None
        if using_regular_grid:
            if resolution is None or dimension is None:
                raise ValueError("resolution and dimension must be specified together")
            if centroids is not None or amplitudes is not None:
                raise ValueError(
                    "resolution/dimension cannot be combined with explicit "
                    "centroids or amplitudes"
                )
            resolution = int(resolution)
            dimension = int(dimension)
            if resolution < 1:
                raise ValueError(f"resolution must be >= 1; got {resolution}")
            if dimension < 1:
                raise ValueError(f"dimension must be >= 1; got {dimension}")

            axis = (
                torch.arange(resolution, dtype=torch.get_default_dtype()) + 0.5
            ) / resolution
            grid = torch.meshgrid(*([axis] * dimension), indexing="ij")
            centroids = torch.stack([coordinate.reshape(-1) for coordinate in grid], dim=-1)
            # Each block has volume resolution**(-dimension), so this scaling
            # makes every PWC indicator have unit L2 norm on [0, 1]^dimension.
            amplitudes = torch.full(
                (centroids.shape[0],),
                resolution ** (dimension / 2),
                dtype=centroids.dtype,
            )
        elif centroids is None or amplitudes is None:
            raise ValueError(
                "provide both centroids and amplitudes, or provide resolution and dimension"
            )

        centroids = torch.as_tensor(centroids)
        if not torch.is_floating_point(centroids):
            centroids = centroids.float()
        if centroids.ndim != 2:
            raise ValueError(f"centroids must be (L, d); got shape {tuple(centroids.shape)}")
        L, d = centroids.shape

        amplitudes = torch.as_tensor(amplitudes)
        if amplitudes.shape != (L,):
            raise ValueError(f"amplitudes must be ({L},); got shape {tuple(amplitudes.shape)}")

        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1; got {num_channels}")
        num_channels = int(num_channels)

        M_in = L * num_channels
        if num_atoms is None:
            num_atoms = M_in
        num_atoms = int(num_atoms)
        if not (1 <= num_atoms <= M_in):
            raise ValueError(f"num_atoms must be in [1, L*C={M_in}]; got {num_atoms}")

        if probs is None:
            # Atom i (one-indexed) has probability proportional to 1 / i.
            probs = 1.0 / torch.arange(1, num_atoms + 1, dtype=centroids.dtype)
            probs = probs / probs.sum()
        else:
            probs = torch.as_tensor(probs)
            if probs.shape != (num_atoms,):
                raise ValueError(f"probs must be ({num_atoms},); got shape {tuple(probs.shape)}")
            if (probs < 0).any():
                raise ValueError("probs must be non-negative")
            total = probs.sum()
            if total <= 0:
                raise ValueError("probs must have a positive sum")
            probs = probs / total

        self.domain_dim   = d
        self.num_channels = num_channels
        self._num_atoms   = num_atoms
        self.learn_synthesis = bool(learn_synthesis)

        if not self.learn_synthesis and num_atoms != M_in:
            raise ValueError(
                "learn_synthesis=False requires num_atoms == L*C "
                f"({M_in}); got {num_atoms}"
            )

        # Fixed (non-trainable) tensors.
        self._centroids  = centroids                 # (L, d)
        self._amplitudes = amplitudes                # (L,)
        self._probs      = probs                     # (num_atoms,)
        self._device     = centroids.device

        # Semi-orthogonal synthesis W (num_atoms, M_in) over the L*C base atoms.
        # Its rows are orthonormal, so the synthesized outputs stay orthonormal.
        synth = nn.Linear(M_in, num_atoms, bias=False)
        with torch.no_grad():
            if self.learn_synthesis:
                # A Haar-distributed Stiefel sample: QR gives orthonormal columns,
                # whose transpose supplies the required orthonormal synthesis rows.
                gaussian = torch.randn(M_in, num_atoms, dtype=synth.weight.dtype)
                columns, upper = torch.linalg.qr(gaussian, mode="reduced")
                signs = torch.sign(torch.diagonal(upper))
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                synth.weight.copy_((columns * signs[None, :]).transpose(0, 1))
            else:
                synth.weight.copy_(torch.eye(M_in, dtype=synth.weight.dtype))  # plain PWC basis
        self._synthesis = orthogonal(synth)
        if not self.learn_synthesis:
            for parameter in self._synthesis.parameters():
                parameter.requires_grad_(False)

    # ── Device handling ─────────────────────────────────────────────────────────

    def _move_to(self, device: torch.device) -> None:
        """Lazily migrate the fixed tensors and the synthesis matrix onto ``device``."""
        if self._device != device:
            self._centroids  = self._centroids.to(device)
            self._amplitudes = self._amplitudes.to(device)
            self._probs      = self._probs.to(device)
            self._synthesis.to(device)
            self._device     = device

    # ── Geometry ────────────────────────────────────────────────────────────────

    def _assign_cells(self, coords: torch.Tensor) -> torch.Tensor:
        """Nearest-centroid (Euclidean) cell index for each coordinate.

        Returns ``(N,)`` long.  Uses the ``‖x‖² − 2⟨x,c⟩ + ‖c‖²`` expansion so
        only an ``(N, L)`` matrix is materialized (no ``(N, L, d)`` blow-up),
        keeping the assignment a single GPU matmul.
        """
        c2 = self._centroids.to(coords.dtype)                 # (L, d); align dtype with coords
        sq_x = coords.pow(2).sum(dim=1, keepdim=True)         # (N, 1)
        sq_c = c2.pow(2).sum(dim=1)                           # (L,)
        d2 = sq_x - 2.0 * (coords @ c2.t()) + sq_c            # (N, L)
        return d2.argmin(dim=1)                               # (N,)

    # ── Core dictionary methods ──────────────────────────────────────────────────

    def get_atoms(
        self,
        coords: torch.Tensor,  # (N, d)
        idx: torch.Tensor,     # (A, 2)
        synthesis: bool = True,
    ) -> torch.Tensor:         # (A, N, C)
        """Evaluate synthesized atoms, or raw ``(cell, channel)`` PWC atoms.

        With ``synthesis=True``, ``idx[:, 0]`` selects an output row of the
        synthesis matrix and column 1 is ignored. With ``synthesis=False``,
        the two columns select a raw base atom's cell and output channel.
        """
        device = coords.device
        self._move_to(device)
        C = self.num_channels
        A, N = idx.shape[0], coords.shape[0]

        cell = self._assign_cells(coords)                     # (N,)
        amp  = self._amplitudes[cell].to(coords.dtype)        # (N,) amplitude of each point's cell

        if not synthesis:
            cell_ids = idx[:, 0].long()
            channels = idx[:, 1].long()
            valid = (
                (cell_ids >= 0)
                & (cell_ids < self._centroids.shape[0])
                & (channels >= 0)
                & (channels < C)
            )
            active = (cell_ids[:, None] == cell[None, :]) & valid[:, None]
            vals = torch.zeros(A, N, C, dtype=coords.dtype, device=device)
            safe_channels = channels.clamp(0, C - 1)
            vals.scatter_(
                2,
                safe_channels[:, None, None].expand(A, N, 1),
                (active.to(coords.dtype) * amp[None, :]).unsqueeze(-1),
            )
            return vals

        a = idx[:, 0].long().clamp(0, self._num_atoms - 1)    # (A,) output-atom rows of W

        W_rows = self._synthesis.weight.to(coords.dtype)[a]   # (A, M_in) rows of the synthesis
        # Only cell(x)'s base atoms are non-zero at x, so atom_a(x)[c'] picks
        # the column cell(x)*C + c' of W, scaled by amplitude[cell(x)].
        col = (cell[:, None] * C + torch.arange(C, device=device)[None, :]).reshape(-1)  # (N*C,)
        vals = W_rows[:, col].reshape(A, N, C)                # (A, N, C)
        return vals * amp[None, :, None]

    def sample_indices(
        self,
        num_samples: int,
        with_replacement: bool = True,
    ) -> torch.Tensor:
        """Sample output-atom indices ``a`` ∝ the per-atom PMF ``probs[a]``.

        The finite ``num_atoms`` support lets a single :func:`torch.multinomial`
        draw handle both modes; ``with_replacement=False`` requires
        ``num_samples <= num_atoms``.  Returns ``(num_samples, 2)`` with the
        atom index in column 0 and zeros in column 1 (channel slot, unused).
        """
        a = torch.multinomial(self._probs, num_samples, replacement=with_replacement)
        return torch.stack([a, torch.zeros_like(a)], dim=-1)  # (num_samples, 2)

    def get_index_pmfs(self, idx: torch.Tensor) -> torch.Tensor:
        """Per-atom PMF ``probs[a]``; ``0`` for indices outside ``[0, num_atoms)``."""
        a = idx[:, 0]
        valid = (a >= 0) & (a < self._num_atoms)
        probs = self._probs.to(idx.device)
        out = torch.zeros(idx.shape[0], dtype=probs.dtype, device=idx.device)
        out[valid] = probs[a[valid].clamp(0, self._num_atoms - 1)]
        return out

    def get_high_probability_indices(self, tail_probability: float) -> torch.Tensor:
        """Return synthesized atom indices whose finite PMF exceeds the threshold."""
        atom_ids = torch.arange(self._num_atoms, device=self._probs.device)
        indices = torch.stack([atom_ids, torch.zeros_like(atom_ids)], dim=-1)
        return indices[self._probs >= tail_probability]

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Trainable parameters backing the synthesis matrix ``W``."""
        if not self.learn_synthesis:
            return iter(())
        return super().parameters(recurse=recurse)

    def save(self, path: str) -> None:
        """Save the cells, amplitudes, PMF and synthesis weights to ``path``.

        The synthesis is a parametrized module, which cannot be pickled directly,
        so its ``state_dict`` is stored and reloaded in :meth:`load`.
        """
        torch.save(
            {
                "centroids": self._centroids.detach().cpu(),
                "amplitudes": self._amplitudes.detach().cpu(),
                "probs": self._probs.detach().cpu(),
                "num_channels": self.num_channels,
                "num_atoms": self._num_atoms,
                "learn_synthesis": self.learn_synthesis,
                "synthesis": self._synthesis.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location=None) -> "VoronoiPWC":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        # Older payloads predate `num_atoms`; their (square) synthesis had one row
        # per output atom, so `len(probs)` recovers it.
        num_atoms = int(payload.get("num_atoms", payload["probs"].shape[0]))
        # Build on CPU so the lazy `_move_to` later migrates every tensor (incl.
        # the synthesis params) onto the eval device together — constructing with
        # on-device centroids would orphan the CPU-created synthesis.
        obj = cls(
            centroids=payload["centroids"].cpu(),
            amplitudes=payload["amplitudes"].cpu(),
            num_channels=int(payload["num_channels"]),
            num_atoms=num_atoms,
            probs=payload["probs"].cpu(),
            learn_synthesis=bool(payload.get("learn_synthesis", True)),
        )
        obj._synthesis.load_state_dict({k: v.cpu() for k, v in payload["synthesis"].items()})
        return obj
