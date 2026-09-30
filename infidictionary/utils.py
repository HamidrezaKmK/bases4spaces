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


def prefix_captured_energy(
    values: torch.Tensor,                   # (B, N, C)
    atoms: torch.Tensor,                    # (K, N, C) the learned prefix basis
    pmfs: torch.Tensor,                     # (K,)
    logabsdet: torch.Tensor | None = None,  # (N,)
) -> torch.Tensor:                          # (B,)
    """PMF-weighted captured energy ``Σ_k p_k ⟨f, e_k⟩²`` over a fixed prefix."""
    coeffs = pairwise_inner_product(values, atoms, logabsdet)
    return (coeffs.square() * pmfs.to(coeffs)[None, :]).sum(dim=-1)


def isometry_defect(
    src: torch.Tensor,                      # (B, N, C) before the map
    tgt: torch.Tensor,                      # (B, N, C) after the map
    logabsdet: torch.Tensor | None = None,  # (N,)
) -> float:
    """``‖Qf‖² / ‖f‖²`` in the weighted L² norm — 1.0 for a true isometry.

    A tripwire, not a loss: an isometry that stops preserving norms lets the
    captured-energy objective win by amplifying instead of by learning a better
    basis, and that is invisible in the energy curve until it has run away.
    """
    n_src = norm2(src, logabsdet).mean()
    n_tgt = norm2(tgt, logabsdet).mean()
    return (n_tgt / n_src.clamp(min=1e-30)).item()
