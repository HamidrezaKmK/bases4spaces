import math
from typing import Callable, Dict, Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .base import NeuralIsometry
from infidictionary.networks import ConditionalField, RMSNorm


def _index_encoding(num_tokens: int, dim: int) -> torch.Tensor:
    """Standard transformer sinusoidal encoding of the indices ``0..num_tokens-1``."""
    position = torch.arange(num_tokens, dtype=torch.float32)[:, None]
    freqs = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    enc = torch.zeros(num_tokens, dim)
    enc[:, 0::2] = torch.sin(position * freqs)
    enc[:, 1::2] = torch.cos(position * freqs)[:, : dim // 2]
    return enc


class _MixingLayer(nn.Module):
    """One transformer layer over the mixing tokens ``q`` and function tokens ``p``.

    The causal cross-attention (row ``r`` attends to ``k <= r``) is PyTorch's
    ``nn.MultiheadAttention``; its head-averaged weights are the matrix ``A``
    used by the Cayley step. Both residual streams advance with that same
    ``A``: ``q ← q + MHA(q, p, p)`` and ``p ← p + Aᵀ·V_p(q)``, each followed by
    a residual feed-forward block.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int):
        super().__init__()
        self.norm_q = RMSNorm()
        self.norm_p = RMSNorm()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, bias=False, batch_first=True)
        self.V_p = nn.Linear(d_model, d_model, bias=False)
        self.ffn_q = nn.Sequential(
            RMSNorm(), nn.Linear(d_model, ffn_hidden), nn.SiLU(), nn.Linear(ffn_hidden, d_model)
        )
        self.ffn_p = nn.Sequential(
            RMSNorm(), nn.Linear(d_model, ffn_hidden), nn.SiLU(), nn.Linear(ffn_hidden, d_model)
        )

    def _cross_attend(self, q: torch.Tensor, p: torch.Tensor):
        R = q.shape[0]
        # True marks a masked position: token r may not see k > r.
        future = torch.ones(R, R, dtype=torch.bool, device=q.device).triu(1)
        kv = self.norm_p(p)[None]
        out, A = self.cross_attn(
            self.norm_q(q)[None], kv, kv,
            attn_mask=future, need_weights=True, average_attn_weights=True,
        )
        return out[0], A[0]

    def attention(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Lower-triangular ``(R, R)`` attention; row ``r`` is a softmax over ``k <= r``."""
        return self._cross_attend(q, p)[1]

    def forward(self, q: torch.Tensor, p: torch.Tensor):
        attended, A = self._cross_attend(q, p)
        q_new = q + attended
        p_new = p + A.T @ self.V_p(self.norm_q(q))
        q_new = q_new + self.ffn_q(q_new)
        p_new = p_new + self.ffn_p(p_new)
        return A, q_new, p_new


class EulerianIsometry(NeuralIsometry):
    """Learned isometry of ``L²`` as a stack of causal sequence-mixing layers.

    See ``.knowledge/sequence-models.md`` and *Learning Orthonormal Bases for
    Function Spaces* (https://arxiv.org/abs/2605.19959).

    The input to :meth:`pushforward` is an *ordered prefix* of functions
    ``e_1, …, e_K`` (``K >= R``), e.g. the top-``K`` atoms of a dictionary. Each
    of the ``L`` layers is one exact rotation ``Q_ℓ`` applied to all ``K``
    functions:

    1. ``R`` mixing tokens ``q_r`` condition a shared field,
       ``u_r(x) = MLP(x, q_r)``.
    2. The mixing tokens attend causally to the function tokens ``p_k``,
       giving a lower-triangular ``A``; ``a_r = Σ_{k<=r} A_rk e_k``.
    3. ``L = Σ_r u_r⟨a_r, ·⟩ - a_r⟨u_r, ·⟩ = Φ J Φ*`` with ``Φ = [u; a]`` and
       ``J = [[0, I], [-I, 0]]`` is skew-adjoint, and
       ``Q = (I - ½L)⁻¹(I + ½L)`` is its Cayley transform.

    ``L`` is skew-adjoint in the weighted empirical inner product (a
    Monte-Carlo average over the sampled coordinates), so ``Q`` is orthogonal
    there and the Gram matrix of the functions is preserved *exactly*. See :meth:`_cayley_apply`
    for how the inverse collapses to a ``2R×2R`` solve.

    The tokens are standard transformer residual streams: ``q^(0)`` is learned,
    ``p^(0)_k`` is a sinusoidal encoding of ``k`` (it never reads ``e_k``), and
    both are advanced by each layer. Every quantity a layer uses depends only on
    ``e_1, …, e_R`` and indices ``<= R``, so the first ``K`` outputs do not
    depend on how many further functions are appended (prefix faithfulness).

    With ``gradient_checkpointing`` each layer is recomputed in the backward
    pass (``torch.utils.checkpoint``), so activation memory does not grow with
    the number of layers.
    """

    def __init__(
        self,
        coords_dim: int,
        channels_dim: int,
        rank: int,
        num_layers: int,
        scalar_field_partial: Callable[[Dict[str, Any]], ConditionalField],
        d_model: int = 64,
        n_heads: int = 1,
        ffn_hidden: int = 128,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.coords_dim = coords_dim
        self.channels_dim = channels_dim
        self.rank = rank
        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing

        self.function_field = scalar_field_partial(
            coords_dim=coords_dim,
            output_dim=channels_dim,
            rank=1,
            cond_dim=d_model,
        )
        self.q0 = nn.Parameter(torch.randn(rank, d_model) / math.sqrt(d_model))
        self.register_buffer("index_encoding", _index_encoding(rank, d_model), persistent=False)
        self.p_embed = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([_MixingLayer(d_model, n_heads, ffn_hidden) for _ in range(num_layers)])

        J = torch.zeros(2 * rank, 2 * rank)
        J[:rank, rank:] = torch.eye(rank)
        J[rank:, :rank] = -torch.eye(rank)
        self.register_buffer("J", J, persistent=False)

        self._collect_diagnostics = False
        self._diagnostics: dict[str, float] = {}

    def _initial_tokens(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q0, self.p_embed(self.index_encoding)

    def _mixing_functions(self, q: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Evaluate ``u_r = MLP(·, q_r)`` for every token at every coordinate: ``(N, R, C)``."""
        N, R = coords.shape[0], q.shape[0]
        cond = q.repeat_interleave(N, dim=0)            # (R·N, d_model)
        x = coords.repeat(R, 1)                         # (R·N, d)
        u = self.function_field(cond, x).view(R, N, self.channels_dim)
        return u.transpose(0, 1)

    def _layer_factors(self, layer, q, p, coords, frame):
        """Build ``Φ = [u; a]`` of shape ``(N, 2R, C)`` from the current tokens and frame."""
        A, q_new, p_new = layer(q, p)
        u = self._mixing_functions(q, coords).to(frame.dtype)
        a = torch.einsum("rk,knc->nrc", A.to(frame.dtype), frame)
        return torch.cat([u, a], dim=1), q_new, p_new

    def _cayley_apply(
        self,
        Phi: torch.Tensor,
        B: torch.Tensor,
        logabsdet: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Apply ``(I - M)⁻¹(I + M)`` with ``M = ½ Φ B Φ*`` and ``B`` skew.

        Shapes — ``Phi: (N, S, C)``, ``B: (S, S)``, ``logabsdet: (N,)``,
        ``values: (V, N, C)``; returns ``(V, N, C)``. ``Φ`` synthesizes a function
        from ``S`` coefficients and ``Φ*`` is its adjoint in the weighted ``L²``
        inner product (an ``exp(logabsdet)``-weighted average over the samples).

        ``(I + M) y`` is formed explicitly. The inverse uses the push-through
        identity

            ``(I - Φ B' Φ*)⁻¹ = I + Φ B' (I_S - G B')⁻¹ Φ*``,   ``G = Φ*Φ``, ``B' = ½B``,

        so an operator on the sampled function space reduces to an ``S×S`` solve.
        ``I_S - G B'`` is always invertible: ``G`` is PSD and ``B'`` skew, so
        ``G B'`` has purely imaginary spectrum. Passing ``-B`` gives the inverse
        rotation.
        """
        N, S, _ = Phi.shape
        w = logabsdet.exp().to(Phi.dtype)
        B = 0.5 * B.to(Phi.dtype)
        G = torch.einsum("nrc,nsc,n->rs", Phi, Phi, w) / N
        I_S = torch.eye(S, device=Phi.device, dtype=Phi.dtype)

        c = torch.einsum("nrc,vnc,n->vr", Phi, values, w) / N
        y_pre = values + torch.einsum("nrc,vr->vnc", Phi, c @ B.T)

        tilde_c = torch.einsum("nrc,vnc,n->vr", Phi, y_pre, w) / N
        Msys = I_S - G @ B
        z = torch.linalg.solve(Msys, tilde_c.T).T @ B.T
        y = y_pre + torch.einsum("nrc,vr->vnc", Phi, z)

        if self._collect_diagnostics:
            with torch.no_grad():
                sv = torch.linalg.svdvals(Msys.float())
                self._diagnostics["cond_msys"] = max(
                    self._diagnostics.get("cond_msys", 0.0),
                    (sv.amax() / sv.amin().clamp(min=1e-30)).item(),
                )
                self._diagnostics["min_eig_gram"] = min(
                    self._diagnostics.get("min_eig_gram", float("inf")),
                    torch.linalg.eigvalsh(G.float()).amin().item(),
                )
        return y

    def _maybe_checkpoint(self, fn, *args):
        if self.gradient_checkpointing and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _check_prefix(self, functions: torch.Tensor, name: str) -> None:
        assert functions.shape[0] >= self.rank, (
            f"{name} must hold at least rank={self.rank} ordered functions (K >= R); "
            f"got K={functions.shape[0]}"
        )

    def layer_states(self, coords: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer attention ``A: (R, R)`` and mixing functions ``u: (N, R, C)``.

        The tokens never read the frame, so both depend on the parameters only;
        this is for inspection and plotting.
        """
        q, p = self._initial_tokens()
        states = []
        for layer in self.layers:
            A, q_next, p_next = layer(q, p)
            states.append((A, self._mixing_functions(q, coords)))
            q, p = q_next, p_next
        return states

    def pop_diagnostics(self) -> dict[str, float]:
        """Return and clear the numerical diagnostics gathered since the last call.

        Enables collection on first use, so a caller that never asks pays nothing.
        """
        self._collect_diagnostics = True
        out, self._diagnostics = self._diagnostics, {}
        return out

    def pushforward(
        self,
        src_coords: torch.Tensor,     # (N, d)
        src_logabsdet: torch.Tensor,  # (N,)
        src_field: torch.Tensor,      # (K, N, C) ordered prefix, or (B, N, C) with ``frame``
        frame: torch.Tensor | None = None,  # (K, N, C) ordered prefix the map is built from
        num_layers_to_apply: int | None = None,
    ):
        """Map the ordered prefix ``e_1..e_K`` to ``T(e_1..e_K)``; requires ``K >= R``.

        Without ``frame``, ``src_field`` *is* the ordered prefix and its first
        ``R`` rows build the rotations. With ``frame``, the rotations are built
        from ``frame`` and applied to the arbitrary functions in ``src_field``;
        by prefix faithfulness this is exactly the map ``frame`` is sent through.
        ``num_layers_to_apply`` stops after that many layers, for inspecting
        intermediate frames.
        """
        if frame is not None:
            self._check_prefix(frame, "frame")
            stacked = torch.cat([frame[: self.rank], src_field], dim=0)
            _, _, out = self.pushforward(
                src_coords, src_logabsdet, stacked, num_layers_to_apply=num_layers_to_apply
            )
            return src_coords, src_logabsdet, out[self.rank:]

        self._check_prefix(src_field, "src_field")
        q, p = self._initial_tokens()
        y = src_field

        for layer in self.layers[:num_layers_to_apply]:
            def step(y, q, p, _layer=layer):
                Phi, q, p = self._layer_factors(_layer, q, p, src_coords, y[: self.rank])
                return self._cayley_apply(Phi, self.J, src_logabsdet, y), q, p

            y, q, p = self._maybe_checkpoint(step, y, q, p)

        return src_coords, src_logabsdet, y

    def pullback(
        self,
        tgt_coords: torch.Tensor,     # (N, d)
        tgt_logabsdet: torch.Tensor,  # (N,)
        tgt_field: torch.Tensor,      # (B, N, C) arbitrary functions
        frame: torch.Tensor,          # (K, N, C) the ordered prefix the map was built from
        num_layers_to_apply: int | None = None,
    ):
        """Apply the inverse map ``Q_1⁻¹ ∘ … ∘ Q_L⁻¹`` to arbitrary functions.

        The rotations depend on the frame, so the ordered prefix used by
        :meth:`pushforward` must be supplied; only its first ``R`` rows matter.
        """
        self._check_prefix(frame, "frame")
        q, p = self._initial_tokens()
        e = frame[: self.rank]
        factors = []

        for layer in self.layers[:num_layers_to_apply]:
            def frame_step(e, q, p, _layer=layer):
                Phi, q, p = self._layer_factors(_layer, q, p, tgt_coords, e)
                return self._cayley_apply(Phi, self.J, tgt_logabsdet, e), Phi, q, p

            e, Phi, q, p = self._maybe_checkpoint(frame_step, e, q, p)
            factors.append(Phi)

        y = tgt_field
        for Phi in reversed(factors):
            y = self._maybe_checkpoint(
                lambda y, Phi: self._cayley_apply(Phi, -self.J, tgt_logabsdet, y), y, Phi
            )
        return tgt_coords, tgt_logabsdet, y
