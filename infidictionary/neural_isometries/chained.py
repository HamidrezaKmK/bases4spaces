import inspect
from typing import Sequence, Callable

import torch
from torch import nn

from .base import NeuralIsometry


class ChainedIsometry(NeuralIsometry):
    """Composition of several neural isometries applied in sequence.

    Given isometries ``[Q_1, Q_2, ..., Q_n]``, the pushforward is the
    composition ``Q_n ∘ ... ∘ Q_1`` (``Q_1`` applied first), and the pullback
    is the exact inverse ``Q_1^{-1} ∘ ... ∘ Q_n^{-1}`` (reverse order).  Since
    each link preserves the ``L²`` inner product, so does the chain.

    The links may have heterogeneous signatures — e.g.
    :meth:`EulerianIsometry.pullback` needs the ``frame`` it was built from while a
    :class:`ChannelOrthogonalIsometry` does not.  Extra keyword arguments passed to
    :meth:`pushforward`/:meth:`pullback` are forwarded to each link, but
    **filtered to the arguments that link actually accepts**, so a single call
    drives the whole chain.

    Args:
        isometries: Ordered sequence of :class:`NeuralIsometry` links.
    """

    def __init__(
        self, 
        coords_dim: int, 
        channels_dim: int,
        isometries_partial: Sequence[Callable[..., NeuralIsometry]],
    ):
        super().__init__()

        if len(isometries_partial) == 0:
            raise ValueError("ChainedIsometry requires at least one isometry")
        
        self.isometries = nn.ModuleList(
            [iso(coords_dim=coords_dim, channels_dim=channels_dim) for iso in isometries_partial]
        )

    @staticmethod
    def _filter_kwargs(fn, kwargs: dict) -> dict:
        """Keep only the kwargs that ``fn`` accepts (all of them if it has **kwargs)."""
        params = inspect.signature(fn).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(kwargs)
        return {k: v for k, v in kwargs.items() if k in params}

    def pushforward(
        self,
        src_coords: torch.Tensor,    # (N, d)
        src_logabsdet: torch.Tensor, # (N,)
        src_field: torch.Tensor,     # (B, N, C)
        frame: torch.Tensor | None = None,  # (K, N, C) ordered prefix, if any link needs it
        **kwargs,
    ):
        """Apply the chain; with ``frame``, each link sees the frame as it arrives there."""
        coords, logabsdet, field = src_coords, src_logabsdet, src_field
        for iso in self.isometries:
            link_kwargs = dict(kwargs) if frame is None else {**kwargs, "frame": frame}
            if frame is not None:
                _, _, next_frame = iso.pushforward(
                    coords, logabsdet, frame, **self._filter_kwargs(iso.pushforward, kwargs)
                )
            coords, logabsdet, field = iso.pushforward(
                coords, logabsdet, field, **self._filter_kwargs(iso.pushforward, link_kwargs)
            )
            if frame is not None:
                frame = next_frame
        return coords, logabsdet, field

    def pullback(
        self,
        tgt_coords: torch.Tensor,    # (N, d)
        tgt_logabsdet: torch.Tensor, # (N,)
        tgt_field: torch.Tensor,     # (B, N, C)
        frame: torch.Tensor | None = None,  # (K, N, C) ordered prefix, if any link needs it
        **kwargs,
    ):
        """Invert the chain; each link is handed the frame as it arrives at that link.

        A frame-dependent link (``EulerianIsometry``) is built from its *input*
        frame, which is the original frame pushed through the preceding links.
        """
        # Every current link keeps coordinates fixed, so the frame can be pushed
        # through the links on the same coordinates the field is pulled back on.
        link_frames = []
        for iso in self.isometries:
            link_frames.append(frame)
            if frame is not None:
                _, _, frame = iso.pushforward(
                    tgt_coords, tgt_logabsdet, frame,
                    **self._filter_kwargs(iso.pushforward, kwargs),
                )

        coords, logabsdet, field = tgt_coords, tgt_logabsdet, tgt_field
        for iso, link_frame in zip(reversed(self.isometries), reversed(link_frames)):
            link_kwargs = dict(kwargs) if link_frame is None else {**kwargs, "frame": link_frame}
            coords, logabsdet, field = iso.pullback(
                coords, logabsdet, field, **self._filter_kwargs(iso.pullback, link_kwargs)
            )
        return coords, logabsdet, field

    def pop_diagnostics(self) -> dict[str, float]:
        """Worst-case diagnostics across links — one bad link is one bad chain."""
        merged: dict[str, float] = {}
        for iso in self.isometries:
            for key, value in iso.pop_diagnostics().items():
                pick = min if key.startswith("min_") else max
                merged[key] = pick(merged[key], value) if key in merged else value
        return merged
