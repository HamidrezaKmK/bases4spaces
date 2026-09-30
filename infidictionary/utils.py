import torch


def pairwise_inner_product(
    f1: torch.Tensor,                       # (A, N, C)
    f2: torch.Tensor,                       # (B, N, C)
    logabsdet: torch.Tensor | None = None,  # (N,)
) -> torch.Tensor:                          # (A, B)
    """Weighted L² inner product."""
    if logabsdet is None:
        logabsdet = torch.zeros(f1.shape[1], device=f1.device, dtype=f1.dtype)
    w = torch.exp(logabsdet).to(f1.dtype)
    ret = torch.einsum("anc, n, bnc -> ab", f1, w, f2)
    return ret / f1.shape[1]


def norm2(
    f: torch.Tensor,                        # (B, N, C)
    logabsdet: torch.Tensor | None = None,  # (N,)
) -> torch.Tensor:                          # (B,) — non-negative
    """Squared weighted L² norm."""
    if logabsdet is None:
        logabsdet = torch.zeros(f.shape[1], device=f.device, dtype=f.dtype)
    w = torch.exp(logabsdet).to(f.dtype)
    ret = torch.einsum("bnc, n, bnc -> b", f, w, f)
    return ret / f.shape[1]


def prefix_weighted_energy(
    values: torch.Tensor,                   # (B, N, C)
    atoms: torch.Tensor,                    # (K, N, C) the learned prefix
    pmfs: torch.Tensor,                     # (K,)
    logabsdet: torch.Tensor | None = None,  # (N,)
) -> torch.Tensor:                          # (B,)
    """PMF-weighted analysis energy ``Σ_k p_k ⟨f, e_k⟩²`` over a fixed prefix.

    Well defined for any prefix, orthonormal or not. The isometry preserves the
    prefix's Gram matrix, so the objective cannot be raised by inflating the
    atoms; for an orthonormal prefix it is the PMF-weighted captured energy.
    """
    coeffs = pairwise_inner_product(values, atoms, logabsdet)
    return (coeffs.square() * pmfs.to(coeffs)[None, :]).sum(dim=-1)


def project_onto_span(
    values: torch.Tensor,                   # (B, N, C)
    atoms: torch.Tensor,                    # (K, N, C)
    logabsdet: torch.Tensor | None = None,  # (N,)
) -> tuple[torch.Tensor, torch.Tensor]:     # (B, N, C), (B,)
    """Orthogonal projection of ``values`` onto ``span(atoms)``, and its energy.

    Uses the Gram matrix ``G = ⟨e_i, e_j⟩``, so the atoms need not be
    orthonormal (e.g. overlapping Gaussian bumps): the coefficients are
    ``G⁺ ⟨e, f⟩`` and the captured energy is ``‖P f‖² = ⟨f, P f⟩``. For an
    orthonormal prefix this reduces to ``Σ_k ⟨e_k, f⟩ e_k`` and ``Σ_k ⟨f, e_k⟩²``.
    """
    gram = pairwise_inner_product(atoms, atoms, logabsdet)       # (K, K)
    analysis = pairwise_inner_product(values, atoms, logabsdet)  # (B, K)
    coeffs = analysis @ torch.linalg.pinv(gram, hermitian=True)  # G symmetric
    projection = (coeffs @ atoms.reshape(atoms.shape[0], -1)).view_as(values)
    return projection, (coeffs * analysis).sum(dim=-1)


def nested_projection_energies(
    values: torch.Tensor,                   # (B, N, C)
    atoms: torch.Tensor,                    # (K, N, C) an ordered prefix
    logabsdet: torch.Tensor | None = None,  # (N,)
) -> torch.Tensor:                          # (B, K)
    """``‖P_k f‖²`` for every prefix length ``k = 1..K`` at once.

    Gram–Schmidt in prefix order is the Cholesky factor ``G = L Lᵀ``, so the
    coefficients of ``f`` in the orthonormalized prefix are ``L⁻¹ ⟨e, f⟩`` and
    their cumulative squared sums are the energies captured by each nested
    span. For an orthonormal prefix this is ``cumsum_k ⟨f, e_k⟩²``.
    """
    gram = pairwise_inner_product(atoms, atoms, logabsdet)       # (K, K)
    analysis = pairwise_inner_product(values, atoms, logabsdet)  # (B, K)
    chol = torch.linalg.cholesky(gram)
    ortho = torch.linalg.solve_triangular(chol, analysis.T, upper=False).T
    return ortho.square().cumsum(dim=-1)


def isometry_defect(
    src: torch.Tensor,                      # (B, N, C) before the map
    tgt: torch.Tensor,                      # (B, N, C) after the map
    logabsdet: torch.Tensor | None = None,  # (N,)
) -> float:
    """``‖Qf‖² / ‖f‖²`` in the weighted L² norm — 1.0 for a true isometry.

    A tripwire, not a loss: an isometry that stops preserving norms lets the
    energy objective win by amplifying instead of by learning a better basis,
    and that is invisible in the energy curve until it has run away.
    """
    n_src = norm2(src, logabsdet).mean()
    n_tgt = norm2(tgt, logabsdet).mean()
    return (n_tgt / n_src.clamp(min=1e-30)).item()
