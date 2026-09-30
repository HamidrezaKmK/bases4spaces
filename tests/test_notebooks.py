"""Opt-in notebook execution and commit-scoped figure extraction."""
import re
import subprocess
from pathlib import Path

import pytest

from notebook_figures import OUTPUT_ROOT, extract_figures, reset_log, write_run_timestamp

REPO_ROOT = Path(__file__).parent.parent
NB_DIR = REPO_ROOT / "notebooks"
NOTEBOOKS = sorted(path.name for path in NB_DIR.glob("*.ipynb"))
EXPERIMENTS = {
    "eulerian-fourier-nerf": (
        "eulerian_transform.ipynb",
        {
            "BASE_DICTIONARY_1D": "Fourier",
            "BASE_DICTIONARY_2D": "Fourier",
            "BASE_DICTIONARY_3CH": "Fourier",
            "MIXING_FIELD_ARCHITECTURE_1D": "nerf",
            "MIXING_FIELD_ARCHITECTURE_2D": "nerf",
            "MIXING_FIELD_ARCHITECTURE_3CH": "nerf",
            "COMPILER_MODE": "latex",
        },
    ),
    "eulerian-haar-nerf": (
        "eulerian_transform.ipynb",
        {
            "BASE_DICTIONARY_1D": "Haar",
            "BASE_DICTIONARY_2D": "Haar",
            "BASE_DICTIONARY_3CH": "Haar",
            "MIXING_FIELD_ARCHITECTURE_1D": "nerf",
            "MIXING_FIELD_ARCHITECTURE_2D": "nerf",
            "MIXING_FIELD_ARCHITECTURE_3CH": "nerf",
            "COMPILER_MODE": "latex",
        },
    ),
    "eulerian-with-haar": (
        "eulerian_transform.ipynb",
        {
            "BASE_DICTIONARY_1D": "Haar",
            "MIXING_FIELD_ARCHITECTURE_1D": "nerf",
            "NUM_LAYERS_1D": 8,
            "BASE_DICTIONARY_2D": "Fourier",
            "MIXING_FIELD_ARCHITECTURE_2D": "nerf",
            "NUM_LAYERS_2D": 8,
            "BASE_DICTIONARY_3CH": "Haar",
            "MIXING_FIELD_ARCHITECTURE_3CH": "nerf",
            "COMPILER_MODE": "latex",
        },
    ),
    "fpca-1d-wavelet": (
        "fpca_1d.ipynb",
        {
            # Fill in from your own run; original: "../outputs/checkpoints/wandb-zn5gsiqm"
            "CHECKPOINT_DIR": "../outputs/checkpoints/XXX",
            # Fill in from your own run; original: "step_4000.pt"
            "CHECKPOINT_FILE": "step_XXX.pt",
            "BASIS_VIS_ATOMS": 10,
            "B_VIS": 128,
            "SEED0": 1000,
            "N_MEAN_VIS": 1024,
            "M_MEAN_EST": 2000,
            "N_VIS_ATOM": 1024,
            "M_CORPUS": 2048,
            "N_COARSE": 16,
            "N_FINE": 1024,
            "K_PC_SHOW": 8,
            "K_LIST_RECON": [2, 4, 16, 32],
            "N_RECON_SHOW": 4,
            "SEED_BASE": 5000,
            "SIG_IDX": 2,
            "K_LIST": [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128],
            "SNAPSHOT_LAYERS": [0, 1, 2, 4, 8],
            "MIXING_FUNCTIONS_SHOW": 3,
            "ATOM_RANK": 0,
            "N_TEST": 1024,
            "PANEL_YMAX_FACTOR": 2.0,
            "N_ATOMS_GRAM": 8,
            "COMPILER_MODE": "latex",
        },
    ),
    "fpca-1d-fourier": (
        "fpca_1d.ipynb",
        {
            # Fill in from your own run; original: "../outputs/checkpoints/wandb-ynf4w0b4"
            "CHECKPOINT_DIR": "../outputs/checkpoints/XXX",
            # Fill in from your own run; original: "step_4000.pt"
            "CHECKPOINT_FILE": "step_XXX.pt",
            "BASIS_VIS_ATOMS": 10,
            "B_VIS": 128,
            "SEED0": 1000,
            "N_MEAN_VIS": 1024,
            "M_MEAN_EST": 2000,
            "N_VIS_ATOM": 1024,
            "M_CORPUS": 2048,
            "N_COARSE": 16,
            "N_FINE": 1024,
            "K_PC_SHOW": 8,
            "K_LIST_RECON": [2, 4, 16, 32],
            "N_RECON_SHOW": 4,
            "SEED_BASE": 5000,
            "SIG_IDX": 2,
            "K_LIST": [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128],
            "SNAPSHOT_LAYERS": [0, 1, 2, 4, 8],
            "MIXING_FUNCTIONS_SHOW": 3,
            "ATOM_RANK": 0,
            "N_TEST": 1024,
            "PANEL_YMAX_FACTOR": 2.0,
            "N_ATOMS_GRAM": 8,
            "COMPILER_MODE": "latex",
        },
    ),
    "fpca-celeba": (
        "fpca_celeba.ipynb",
        {
            # Fill in from your own run; original: "../outputs/checkpoints/wandb-46bx095i"
            "CHECKPOINT_DIR": "../outputs/checkpoints/XXX",
            # Fill in from your own run; original: "step_6000.pt"
            "CHECKPOINT_FILE": "step_XXX.pt",
            "N_SHOW_LINEAR": 12,
            "N_VIS": 64,
            "N_PEEK": 6,
            "MOSAIC_SIDE": 11,
            "N_VIS_ATOM": 128,
            "TILE_SIZE": 0.55,
            "ZOOM_ATOM_IDX": 11,
            "ZOOM_LEFT": 1 / 2.0 - 0.1,
            "ZOOM_RIGHT": 5 / 6.0,
            "ZOOM_DOWN": 1 / 6.0 + 0.05,
            "ZOOM_UP": 1 / 2.0 + 0.15,
            "ZOOM_FULL_RES": 228,
            "ZOOM_RES": 6,
            "MIXING_LAYERS": None,
            "N_VIS_U": 64,
            "RECON_ATOM_BUDGETS": [3, 27, 75, 243, 1083, 2523],
            "N_VAL": 4,
            "K_SHOW": 12,
            "N_OOD_SHOW": 4,
            "N_SPEC": 300,
            "SPECTRAL_ATOMS": 243,
            "N_VIOLIN": 10,
            "COMPILER_MODE": "latex",
        },
    ),
    "fpca-inr-mnist": (
        "inr_fpca.ipynb",
        {
            # Fill in from your own run; original: "../outputs/checkpoints/wandb-28370exp"
            "CHECKPOINT_DIR": "../outputs/checkpoints/XXX",
            # Fill in from your own run; original: "step_1000.pt"
            "CHECKPOINT_FILE": "step_XXX.pt",
            "EXP_NAME": "MNIST",
            "CMAP": "Greys_r",
            "N_MEAN": 64,
            "N_BATCH_SHOW": 12,
            "N_VIS_ATOMS": 32,
            "N_VAL": 4,
            "N_RECON": 32,
            "N_SHOW_LINEAR": 12,
            "ATOM_TILE_SIZE": 0.55,
            "RECON_ATOM_BUDGETS": [1, 4, 16, 64, 128],
            "N_SPEC": 200,
            "SPECTRAL_ATOMS": 256,
            "N_VIS_SPEC": 32,
            "N_VIOLIN": 10,
            "MOSAIC_SIDE": 7,
            "COMPILER_MODE": "latex",
        },
    ),
    "fpca-inr-cifar10": (
        "inr_fpca.ipynb",
        {
            # Fill in from your own run; original: "../outputs/checkpoints/wandb-6cs71pox"
            "CHECKPOINT_DIR": "../outputs/checkpoints/XXX",
            # Fill in from your own run; original: "step_1000.pt"
            "CHECKPOINT_FILE": "step_XXX.pt",
            "EXP_NAME": "CIFAR10",
            "CMAP": "Greys_r",
            "N_MEAN": 64,
            "N_BATCH_SHOW": 12,
            "N_VIS_ATOMS": 32,
            "N_VAL": 4,
            "N_RECON": 32,
            "N_SHOW_LINEAR": 12,
            "ATOM_TILE_SIZE": 0.55,
            "RECON_ATOM_BUDGETS": [1, 4, 16, 128, 256],
            "MOSAIC_SIDE": 5,
            "N_SPEC": 200,
            "SPECTRAL_ATOMS": 256,
            "N_VIS_SPEC": 32,
            "N_VIOLIN": 10,
            "COMPILER_MODE": "latex",
        },
    ),
}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def commit_scoped_output_root(skip_git_check: bool = False) -> Path:
    if skip_git_check:
        return OUTPUT_ROOT / "temp"
    try:
        dirty, commit = _git("status", "--porcelain"), _git("rev-parse", "--short", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.fail(f"could not query git state for notebook run: {exc}", pytrace=False)
    if dirty:
        pytest.fail("full notebook runs require a clean working tree; use --git-no-check while developing.\n" + dirty, pytrace=False)
    return OUTPUT_ROOT / commit


@pytest.fixture(scope="session")
def output_root(pytestconfig):
    root = commit_scoped_output_root(pytestconfig.getoption("--git-no-check"))
    reset_log()
    write_run_timestamp(root)
    return root


def _checkpoint_dir_from_notebook(nb):
    """The CHECKPOINT_DIR default declared in a notebook's `parameters` cell.

    Takes a notebook already parsed by nbformat — find_parameters_cell uses
    attribute access, so a plain json.loads dict will not do.
    """
    from notebook_figures import find_parameters_cell
    idx = find_parameters_cell(nb)
    if idx is None:
        return None
    source = "".join(nb.cells[idx].source)
    match = re.search(r'^\s*CHECKPOINT_DIR\s*=\s*["\'](.+?)["\']', source, re.M)
    return match.group(1) if match else None


def skip_without_checkpoint(checkpoint_dir, label):
    """Skip when the run directory is absent — checkpoints are not shipped.

    Notebooks and presets carry `XXX` placeholders rather than run ids nobody
    else can reproduce, so a fresh clone should report skips, not failures.
    """
    if checkpoint_dir is None:
        return
    resolved = (NB_DIR / checkpoint_dir).resolve()
    if not resolved.is_dir():
        pytest.skip(
            f"{label} needs a checkpoint at {resolved}. Run the matching experiment "
            f"(see the notebook's parameters cell), then set CHECKPOINT_DIR/CHECKPOINT_FILE."
        )


@pytest.mark.notebook
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=[Path(name).stem for name in NOTEBOOKS])
def test_notebook_runs(notebook):
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbclient")
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
    path = NB_DIR / notebook
    nb = nbformat.read(str(path), as_version=4)
    skip_without_checkpoint(_checkpoint_dir_from_notebook(nb), notebook)
    try:
        NotebookClient(nb, timeout=-1, kernel_name=nb.metadata.get("kernelspec", {}).get("name", "python3"), resources={"metadata": {"path": str(NB_DIR)}}).execute()
    except CellExecutionError as exc:
        pytest.fail(f"{notebook} failed:\n{exc}", pytrace=False)


@pytest.mark.notebook_full
@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_notebook_runs_end_to_end(experiment, output_root):
    pytest.importorskip("nbclient")
    notebook, params = EXPERIMENTS[experiment]
    params = dict(params)
    skip_without_checkpoint(params.get("CHECKPOINT_DIR"), f"{notebook} [{experiment}]")
    compiler_mode = params.pop("COMPILER_MODE", "normal")
    ok, error, _, _ = extract_figures(NB_DIR / notebook, experiment, params, compiler_mode, output_root=output_root)
    assert ok, f"{notebook} [{experiment}] failed: {error}"
