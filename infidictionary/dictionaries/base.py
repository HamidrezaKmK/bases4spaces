from abc import ABC, abstractmethod
from typing import Iterator

import torch
import torch.nn as nn

from infidictionary.utils import pairwise_inner_product


class InfiDictionary(nn.Module, ABC):
    """Abstract base class for dictionaries over a continuous domain.

    A dictionary is a collection of *atoms* — scalar- or vector-valued
    functions on a continuous domain — equipped with a probability mass
    function (PMF) over its index set.  The index set may be either

    * **infinite** (e.g. the real Fourier basis on ``[0,1]^d``, with a
      summable power-law prior over frequencies), or
    * **finite** (e.g. a set of fixed resolution hat functions or Voronoi cell 
      step functions).

    Subclasses implement specific atom families and decide how the index set is
    laid out, how atoms are evaluated, and what probability each index carries.
    Everything beyond these core operations is dictionary-specific and lives on
    the subclass.

    Atoms are evaluated at a finite set of *coordinates* (quadrature points) and
    are identified by integer *indices* whose meaning is dictionary-specific.

    Shape conventions used throughout:
        B  — batch size (number of functions)
        N  — number of quadrature / sample points
        d  — spatial dimension of the domain
        C  — number of channels (output dimension of each function)
        A  — number of atoms
    """

    @abstractmethod
    def get_atoms(
        self,
        coords: torch.Tensor, # (N, d)
        idx: torch.Tensor, # (A, ...)
        synthesis: bool = True,
    ) -> torch.Tensor: # (A, N, C)
        """Evaluate dictionary atoms at the given coordinates.

        Args:
            coords: Quadrature / sample points, shape ``(N, d)``.
            idx: Integer indices selecting which atoms to evaluate,
                shape ``(A, ...)``.  The inner dimensions are
                dictionary-specific (e.g. ``(A, d)`` for multi-index atoms).
            synthesis: Whether to apply the dictionary's optional synthesis
                transform. ``False`` requests its raw/base atoms. Dictionaries
                without a synthesis transform return the same atoms either way.

        Returns:
            Atom values at each coordinate, shape ``(A, N, C)``; entry
            ``(i, j, c)`` is the ``j``-th evaluation of the ``i``-th atom on
            channel ``c``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_index_pmfs(self, idx: torch.Tensor) -> torch.Tensor:
        """Return the prior probability ``p(k)`` for each atom index.

        Args:
            idx: Integer indices, shape ``(A, ...)`` matching :meth:`get_atoms`.

        Returns:
            Probability tensor of shape ``(A,)``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_high_probability_indices(self, tail_probability: float) -> torch.Tensor:
        """Return every atom index with PMF at least ``tail_probability``."""
        raise NotImplementedError

    def get_top_indices(self, num_atoms: int) -> torch.Tensor:
        """Return the ``num_atoms`` highest-PMF indices, ordered by descending PMF.

        This is the ordered prefix ``e_1, …, e_K`` that is fed through the
        isometry. Ties keep the order of :meth:`get_high_probability_indices`
        (the sort is stable), so the prefix is deterministic.
        """
        if num_atoms < 1:
            raise ValueError(f"num_atoms must be >= 1; got {num_atoms}")
        tail_probability = 1e-2
        idx = self.get_high_probability_indices(tail_probability)
        while idx.shape[0] < num_atoms:
            if tail_probability < 1e-30:
                raise ValueError(
                    f"dictionary has fewer than num_atoms={num_atoms} atoms with non-zero PMF"
                )
            tail_probability /= 4
            idx = self.get_high_probability_indices(tail_probability)
        pmfs = self.get_index_pmfs(idx)
        order = torch.sort(pmfs, descending=True, stable=True).indices
        return idx[order[:num_atoms].to(idx.device)]

    def get_reconstructions(
        self,
        coords: torch.Tensor,
        functions: torch.Tensor,
        atom_indices: torch.Tensor,
        synthesis: bool = True,
    ) -> torch.Tensor:
        """Project ``functions`` onto selected synthesized or raw dictionary atoms."""
        atoms = self.get_atoms(
            coords,
            atom_indices.to(coords.device),
            synthesis=synthesis,
        )
        coefficients = pairwise_inner_product(functions, atoms)
        return (coefficients @ atoms.reshape(atoms.shape[0], -1)).view_as(functions)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Trainable parameters owned by the dictionary itself.

        A dictionary with trainable overrides this to yield those 
        parameters, so the training loop can hand them to the optimizer 
        alongside the isometry's — letting the initial dictionary itself 
        be learned jointly. This is mostly useful for linear synthesis
        style mixing that can be learned, e.g. instead of using a fixed
        Fourier basis, allow for a quadratic mixing of the Fourier bases.
        """
        return (
            parameter
            for parameter in super().parameters(recurse=recurse)
            if parameter.requires_grad
        )

    def save(self, path: str) -> None:
        """Persist the state needed to rebuild this dictionary later."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, map_location=None) -> "InfiDictionary":
        """Reconstruct a dictionary previously written by :meth:`save`."""
        raise NotImplementedError
