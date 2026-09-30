"""Empirical check of prefix faithfulness for ``EulerianIsometry``.

See ``.knowledge/sequence-models.md`` (section *Prefix faithfulness*). With the
rank ``R`` fixed, pushing the Fourier prefix ``e_1..e_K`` through a randomly
initialised isometry must give the same first ``K`` outputs for every
``K >= R``. Changing ``R`` builds different rotations, so the outputs should
change. For the ``R' > R`` model every parameter the two models share is copied
over from the rank-``R`` model (``q0[:R]``, the field MLP, all layers), so the
only difference is the ``R' - R`` extra mixing tokens.

Writes ``functions.png``, ``differences.png``, ``prefix_table.md`` and
``prefix_table.csv`` to ``--out`` and prints ``metric`` lines.

    python scripts/prefix_faithfulness.py --out outputs/prefix_check
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import torch
from torch import nn

from infidictionary.dictionaries import FourierDictionary
from infidictionary.networks import NerfConditionalField
from infidictionary.neural_isometries import EulerianIsometry


def build_isometry(rank: int, num_layers: int, seed: int) -> EulerianIsometry:
    torch.manual_seed(seed)
    field = lambda **kw: NerfConditionalField(activation=nn.SiLU, **kw)
    return EulerianIsometry(
        coords_dim=1, channels_dim=1, rank=rank, num_layers=num_layers,
        scalar_field_partial=field,
    ).double().eval()


def copy_shared_parameters(src: EulerianIsometry, dst: EulerianIsometry) -> None:
    """Copy every parameter of ``src`` into ``dst``; ``q0`` fills the first ``src.rank`` rows."""
    dst_state = dst.state_dict()
    for name, value in src.state_dict().items():
        if name == "q0":
            dst_state[name][: src.rank] = value
        else:
            dst_state[name] = value
    dst.load_state_dict(dst_state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--rank_prime", type=int, default=7)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R, Rp = args.rank, args.rank_prime

    coords = ((torch.arange(args.num_points, dtype=torch.float64) + 0.5) / args.num_points)[:, None]
    logabsdet = torch.zeros(args.num_points, dtype=torch.float64)
    fourier = FourierDictionary(domain_dim=1, num_channels=1, learn_synthesis=False)
    K_max = Rp + 1
    prefix = fourier.get_atoms(coords, fourier.get_top_indices(K_max)).double()  # (K_max, N, 1)

    model_R = build_isometry(R, args.num_layers, args.seed)
    model_Rp = build_isometry(Rp, args.num_layers, args.seed)
    copy_shared_parameters(model_R, model_Rp)

    configs = [(R, R), (R, R + 1), (R, R + 2), (Rp, Rp), (Rp, Rp + 1)]
    outputs = {}
    with torch.no_grad():
        for rank, K in configs:
            model = model_R if rank == R else model_Rp
            _, _, out = model.pushforward(coords, logabsdet, prefix[:K])
            outputs[(rank, K)] = out[..., 0]  # (K, N)

    def l2(f):  # empirical L² norm on the uniform grid, per function
        return f.pow(2).mean(-1).sqrt()

    ref = outputs[(R, R)]
    rows = []
    for rank, K in configs:
        out = outputs[(rank, K)]
        gram = out @ out.T / args.num_points
        gram0 = prefix[:K, :, 0] @ prefix[:K, :, 0].T / args.num_points
        diff = l2(out[:R] - ref)  # relative == absolute: every atom has unit norm
        same_rank_ref = outputs[(rank, rank)]
        diff_same_rank = l2(out[:rank] - same_rank_ref)
        rows.append({
            "R": rank,
            "K": K,
            "max_rel_L2_vs_R5K5_first5": diff.max().item(),
            "mean_rel_L2_vs_R5K5_first5": diff.mean().item(),
            f"max_rel_L2_vs_(R,K=R)_first_R": diff_same_rank.max().item(),
            "max_abs_vs_R5K5_first5": (out[:R] - ref).abs().max().item(),
            "gram_defect_max": (gram - gram0).abs().max().item(),
            "moved_from_fourier_first5": l2(out[:R] - prefix[:R, :, 0]).mean().item(),
        })
    keys = list(rows[0].keys())
    # Replace the literal "5" in the column names with the actual R.
    names = [k.replace("R5K5", f"R{R}K{R}").replace("first5", f"first{R}") for k in keys]

    with open(os.path.join(args.out, "prefix_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        for row in rows:
            w.writerow([row[k] for k in keys])
    fmt = lambda v: str(v) if isinstance(v, int) else f"{v:.3e}"
    lines = ["| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
    lines += ["| " + " | ".join(fmt(row[k]) for k in keys) + " |" for row in rows]
    table = "\n".join(lines)
    with open(os.path.join(args.out, "prefix_table.md"), "w") as f:
        f.write(table + "\n")
    print(table)

    # Per-function table: ‖e_i^{(R,K)} − e_i^{(R,R)}‖ for every i <= R.
    per_fn = ["| run | " + " | ".join(f"e_{i+1}" for i in range(R)) + " |", "|---|" + "---|" * R]
    for rank, K in configs:
        d = l2(outputs[(rank, K)][:R] - ref)
        per_fn.append(f"| R={rank}, K={K} | " + " | ".join(f"{v:.2e}" for v in d.tolist()) + " |")
    per_fn = "\n".join(per_fn)
    with open(os.path.join(args.out, "prefix_table.md"), "a") as f:
        f.write(f"\nPer-function L² distance to R={R}, K={R}:\n\n" + per_fn + "\n")
    print(per_fn)

    # Figure 1: one row per run, one column per output function.
    x = coords[:, 0].numpy()
    ncols = max(K for _, K in configs)
    fig, axes = plt.subplots(len(configs), ncols, figsize=(2.0 * ncols, 1.6 * len(configs)),
                             sharex=True, sharey=True)
    for row_i, (rank, K) in enumerate(configs):
        out = outputs[(rank, K)]
        for col in range(ncols):
            ax = axes[row_i, col]
            ax.set_xticks([]); ax.set_yticks([])
            if col >= K:
                ax.axis("off")
                continue
            if col < R:
                ax.plot(x, ref[col].numpy(), color="0.7", lw=3, label=f"R={R}, K={R}")
            ax.plot(x, out[col].numpy(), color="C0" if rank == R else "C3", lw=1.0)
            if row_i == 0:
                ax.set_title(f"$e_{{{col + 1}}}$", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"R={rank}\nK={K}", fontsize=9)
    fig.suptitle(f"Random EulerianIsometry on a Fourier prefix (L={args.num_layers}); "
                 f"grey = R={R}, K={R} reference", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "functions.png"), dpi=150)

    # Figure 2: pointwise difference to the reference, first R functions.
    fig, axes = plt.subplots(1, len(configs) - 1, figsize=(3.2 * (len(configs) - 1), 2.6), sharey=False)
    for ax, (rank, K) in zip(axes, configs[1:]):
        d = (outputs[(rank, K)][:R] - ref).numpy()
        for i in range(R):
            ax.plot(x, d[i], lw=1, label=f"$e_{{{i + 1}}}$")
        ax.set_title(f"R={rank}, K={K}  minus  R={R}, K={R}", fontsize=9, pad=14)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "differences.png"), dpi=150)

    same_R = [r for r in rows if r["R"] == R]
    other_R = [r for r in rows if r["R"] == Rp]
    print(f"metric fixed_R_max_rel_diff={max(r['max_rel_L2_vs_R5K5_first5'] for r in same_R):.3e}")
    print(f"metric changed_R_min_rel_diff={min(r['max_rel_L2_vs_R5K5_first5'] for r in other_R):.3e}")
    print(f"metric changed_R_prefix_max_rel_diff={max(r['max_rel_L2_vs_(R,K=R)_first_R'] for r in other_R):.3e}")
    print(f"metric gram_defect_max={max(r['gram_defect_max'] for r in rows):.3e}")


if __name__ == "__main__":
    main()
