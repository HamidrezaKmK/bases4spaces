import torch

from abc import ABC, abstractmethod

class NeuralIsometry(ABC, torch.nn.Module):

    """
    Implements a general abstract Neural Isometry that maps arbitrary
    functions to other functions while preserving inner products.

    The pushforward and pullback methods take in coordinates, along their field values
    at those coordinates, then the function returns the transformed coordinates, and
    transformed function values at those coordinates.
    """
    def __init__(
        self,
    ):
        super().__init__()
        
    @abstractmethod
    def pushforward(
        self,
        src_coords: torch.Tensor, # (N, d)
        src_logabsdet: torch.Tensor, # (N, )
        src_field: torch.Tensor, # (B, N, c),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the coordinates and values of the pushforward operation
        """
        pass

    @abstractmethod
    def pullback(
        self,
        tgt_coords: torch.Tensor, # (N, d)
        tgt_logabsdet: torch.Tensor, # (N, )
        tgt_field: torch.Tensor, # (B, N, c),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the coordinates and values of the pullback operation
        Due to the isometry property, the pullback should be the inverse of the pushforward.
        """
        pass

    def pop_diagnostics(self) -> dict[str, float]:
        """Numerical diagnostics since the last call. Empty unless overridden."""
        return {}

class IdentityIsometry(NeuralIsometry):
    """
    A toy neural isometry that does nothing, but is still an isometry. Useful for debugging and testing.

    Accepts and ignores any extra keyword arguments. Each concrete isometry takes
    its own: ``EulerianIsometry.pullback`` requires the ``frame`` it was built
    from, ``ChannelOrthogonalIsometry`` takes none. Swallowing the extras is what
    keeps the identity a drop-in stand-in for any of them. (``ChainedIsometry``
    solves the same problem per-link with ``_filter_kwargs``.)
    """

    def pushforward(
        self,
        src_coords: torch.Tensor, # (N, d)
        src_logabsdet: torch.Tensor, # (N, )
        src_field: torch.Tensor, # (B, N, c),
        **_isometry_kwargs,
    ):
        return src_coords, src_logabsdet, src_field

    def pullback(
        self,
        tgt_coords: torch.Tensor, # (N, d)
        tgt_logabsdet: torch.Tensor, # (N, )
        tgt_field: torch.Tensor, # (B, N, c),
        **_isometry_kwargs,
    ):
        return tgt_coords, tgt_logabsdet, tgt_field
