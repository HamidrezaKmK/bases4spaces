import torch
from torch import nn

from .base import NeuralIsometry
from infidictionary.networks.base import _build_mlp


class ChannelOrthogonalIsometry(NeuralIsometry):
    """Spatially varying orthogonal mixing of channels.

    At each coordinate the channel vector is multiplied by a point-dependent
    orthogonal matrix ``Q(x) = Q₀ exp(A(x))``, where ``Q₀`` is a fixed
    Haar-random orthogonal matrix and ``A(x)`` is skew-symmetric.  The learned
    perturbation starts at zero, so ``Q(x) = Q₀`` exactly at initialization for
    every coordinate. Coordinates and log-abs-det weights pass through
    unchanged.
    """

    def __init__(
        self,
        coords_dim: int,
        channels_dim: int,
        hidden_dims: tuple = (64, 64),
        activation=nn.SiLU,
    ):
        super().__init__()
        self.coords_dim = coords_dim
        self.channels_dim = channels_dim

        self.gen_mlp = _build_mlp(
            in_dim=coords_dim,
            out_dim=channels_dim * channels_dim,
            hidden_dims=tuple(hidden_dims),
            activation=activation,
            use_batchnorm=False,
            use_rmsnorm=False,
            bias=True,
        )

        # Zero-init the last layer so Q(x) == base_orthogonal exactly at init.
        final_linear = next(
            layer for layer in reversed(self.gen_mlp) if isinstance(layer, nn.Linear)
        )
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

        # QR with sign-corrected R diagonal samples from the Haar measure on O(C).
        gaussian = torch.randn(channels_dim, channels_dim)
        base_orthogonal, upper = torch.linalg.qr(gaussian)
        signs = torch.sign(torch.diagonal(upper))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        base_orthogonal = base_orthogonal * signs[None, :]
        self.register_buffer("base_orthogonal", base_orthogonal)

    def _orthogonal(self, coords: torch.Tensor) -> torch.Tensor:
        """Per-point orthogonal matrix ``Q(x)``, shape ``(N, c, c)``."""
        N = coords.shape[0]
        c = self.channels_dim
        p_dtype = next(self.gen_mlp.parameters()).dtype
        x = coords.to(p_dtype)

        K = self.gen_mlp(x).view(N, c, c)
        A = K - K.transpose(-1, -2)
        learned_rotation = torch.linalg.matrix_exp(A)
        return self.base_orthogonal.to(dtype=x.dtype, device=x.device)[None] @ learned_rotation

    def pushforward(
        self,
        src_coords: torch.Tensor,
        src_logabsdet: torch.Tensor,
        src_field: torch.Tensor,
    ):
        Q = self._orthogonal(src_coords)
        out = torch.einsum("nij,bnj->bni", Q.to(src_field.dtype), src_field)
        return src_coords, src_logabsdet, out
