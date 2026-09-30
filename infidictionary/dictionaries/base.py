from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class InfiDictionary(nn.Module, ABC):
    """Abstract base class for infinite dictionaries over a continuous domain.

    A dictionary is an infinite orthonormal family of *atoms* — scalar- or
    vector-valued functions on a continuous domain — equipped with a
    probability mass function (PMF) over its index set, e.g. the real Fourier
    basis on ``[0,1]^d`` with a summable power-law prior over frequencies, or
    the Haar wavelet basis with a geometric prior over levels. Dictionaries
    are fixed: they own no learned parameters.

    A dictionary is only ever probed through its **ordered prefix**: the first
    ``K`` atoms ``e_1, …, e_K`` by descending PMF, via :meth:`get_prefix` and
    :meth:`get_prefix_pmfs`. Prefixes are nested — the first ``K`` atoms of the
    ``K+1`` prefix are the ``K`` prefix — and ties in the PMF are broken by
    the subclass's fixed enumeration order, so the ordering is deterministic.

    Subclasses implement the index-level primitives (:meth:`_atoms`,
    :meth:`_index_pmfs`, :meth:`_high_probability_indices`); their index
    layout is private to the subclass.

    Shape conventions used throughout:
        N  — number of quadrature / sample points
        d  — spatial dimension of the domain
        C  — number of channels (output dimension of each function)
        K  — prefix length
    """

    def __init__(self):
        super().__init__()
        self._prefix_cache: dict[int, torch.Tensor] = {}

    @abstractmethod
    def _atoms(self, coords: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Evaluate the atoms selected by ``idx`` at ``coords``: ``(A, N, C)``."""
        raise NotImplementedError

    @abstractmethod
    def _index_pmfs(self, idx: torch.Tensor) -> torch.Tensor:
        """Prior probability of each atom index: ``(A,)``."""
        raise NotImplementedError

    @abstractmethod
    def _high_probability_indices(self, tail_probability: float) -> torch.Tensor:
        """Every atom index with PMF at least ``tail_probability``."""
        raise NotImplementedError

    def _prefix_indices(self, num_atoms: int) -> torch.Tensor:
        """Indices of the ``num_atoms`` highest-PMF atoms, by descending PMF (cached)."""
        if num_atoms < 1:
            raise ValueError(f"num_atoms must be >= 1; got {num_atoms}")
        if num_atoms not in self._prefix_cache:
            # Every atom at or above the threshold is enumerated, so once the
            # set holds K atoms its top K is the global top K.
            tail_probability = 1e-2
            idx = self._high_probability_indices(tail_probability)
            while idx.shape[0] < num_atoms:
                if tail_probability < 1e-30:
                    raise ValueError(
                        f"dictionary has fewer than num_atoms={num_atoms} atoms with non-zero PMF"
                    )
                tail_probability /= 4
                idx = self._high_probability_indices(tail_probability)
            order = torch.sort(self._index_pmfs(idx), descending=True, stable=True).indices
            self._prefix_cache[num_atoms] = idx[order[:num_atoms].to(idx.device)]
        return self._prefix_cache[num_atoms]

    def get_prefix(self, coords: torch.Tensor, num_atoms: int) -> torch.Tensor:
        """The first ``num_atoms`` atoms ``e_1..e_K`` evaluated at ``coords``: ``(K, N, C)``."""
        return self._atoms(coords, self._prefix_indices(num_atoms).to(coords.device))

    def get_prefix_pmfs(self, num_atoms: int) -> torch.Tensor:
        """PMF weights ``p_1 >= … >= p_K`` of the first ``num_atoms`` atoms: ``(K,)``."""
        return self._index_pmfs(self._prefix_indices(num_atoms))

    def save(self, path: str) -> None:
        """Persist the state needed to rebuild this dictionary later."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, map_location=None) -> "InfiDictionary":
        """Reconstruct a dictionary previously written by :meth:`save`."""
        raise NotImplementedError
