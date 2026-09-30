from functools import partial
import math

import pytest
import torch

from infidictionary.checkpointing import Checkpointer
from infidictionary.datasets import ImplicitZooCIFARDataset, OneDimDiscontinuousGenerator
from infidictionary.domain_samplers import SquareSampler
from infidictionary.dictionaries import FourierDictionary, HaarWaveletDictionary, VoronoiPWC
from infidictionary.neural_isometries import (
    ChainedIsometry,
    ChannelOrthogonalIsometry,
    EulerianIsometry,
)
from infidictionary.networks import NerfConditionalField, NerfNeuralField
from infidictionary.utils import isometry_defect, pairwise_inner_product, prefix_captured_energy


def _eulerian(rank=4, num_layers=3):
    torch.manual_seed(0)
    iso = EulerianIsometry(
        1, 1, rank, num_layers, lambda **kw: NerfConditionalField(hidden_dims=(32,), **kw)
    ).double()
    iso.eval()
    return iso


def _prefix(num_atoms, n_points=64):
    torch.manual_seed(1)
    dictionary = FourierDictionary(domain_dim=1, num_channels=1)
    coords = torch.rand(n_points, 1)
    atoms = dictionary.get_atoms(coords, dictionary.get_top_indices(num_atoms))
    return coords.double(), torch.zeros(n_points, dtype=torch.float64), atoms.double()


def test_eulerian_pushforward_preserves_the_gram_matrix():
    # The defining property: every layer is an exact Cayley rotation, so the
    # Gram matrix of the prefix is unchanged, not merely penalized toward it.
    iso = _eulerian()
    coords, lad, atoms = _prefix(9)
    with torch.no_grad():
        _, _, out = iso.pushforward(coords, lad, atoms)
    assert not torch.allclose(out, atoms, atol=1e-2)  # it actually moved
    assert torch.allclose(
        pairwise_inner_product(out, out, lad), pairwise_inner_product(atoms, atoms, lad), atol=1e-10
    )
    assert isometry_defect(atoms, out, lad) == pytest.approx(1.0, abs=1e-10)


def test_pullback_inverts_pushforward_on_arbitrary_functions():
    iso = _eulerian()
    coords, lad, atoms = _prefix(6)
    f = torch.randn(3, coords.shape[0], 1, dtype=torch.float64)
    with torch.no_grad():
        _, _, pushed_atoms = iso.pushforward(coords, lad, atoms)
        _, _, pulled_atoms = iso.pullback(coords, lad, pushed_atoms, frame=atoms)
        _, _, pulled = iso.pullback(coords, lad, f, frame=atoms)
    assert torch.allclose(pulled_atoms, atoms, atol=1e-10)
    with torch.no_grad():
        _, _, pushed_back = iso.pushforward(coords, lad, pulled, frame=atoms)
        _, _, pushed_tail = iso.pushforward(coords, lad, atoms[3:], frame=atoms)
    assert torch.allclose(pushed_back, f, atol=1e-10)
    assert torch.allclose(pushed_tail, pushed_atoms[3:], atol=1e-12)
    # <Q e_k, f> = <e_k, Q^{-1} f>: pulling data back is the adjoint of pushing atoms.
    assert torch.allclose(
        pairwise_inner_product(f, pushed_atoms, lad), pairwise_inner_product(pulled, atoms, lad), atol=1e-10
    )


def test_eulerian_is_prefix_faithful():
    # Appending further functions to the prefix must not change the first K outputs.
    iso = _eulerian()
    coords, lad, atoms = _prefix(9)
    with torch.no_grad():
        _, _, short = iso.pushforward(coords, lad, atoms[:5])
        _, _, long = iso.pushforward(coords, lad, atoms)
    assert torch.allclose(long[:5], short, atol=1e-12)


def test_eulerian_requires_at_least_rank_functions():
    iso = _eulerian(rank=4)
    coords, lad, atoms = _prefix(3)
    with pytest.raises(AssertionError, match="K >= R"):
        iso.pushforward(coords, lad, atoms)
    with pytest.raises(AssertionError, match="K >= R"):
        iso.pullback(coords, lad, atoms, frame=atoms)


def test_mixing_attention_is_causal():
    iso = _eulerian(rank=5)
    q, p = iso._initial_tokens()
    A = iso.layers[0].attention(q, p)
    assert torch.equal(A, A.tril())
    assert torch.allclose(A[0], torch.eye(5, dtype=A.dtype)[0])
    assert torch.allclose(A.sum(dim=-1), torch.ones(5, dtype=A.dtype))


def test_partial_pushforward_applies_only_the_requested_layers():
    iso = _eulerian(num_layers=3)
    coords, lad, atoms = _prefix(6)
    with torch.no_grad():
        _, _, none = iso.pushforward(coords, lad, atoms, num_layers_to_apply=0)
        _, _, all_layers = iso.pushforward(coords, lad, atoms, num_layers_to_apply=3)
        _, _, full = iso.pushforward(coords, lad, atoms)
    assert torch.equal(none, atoms)
    assert torch.equal(all_layers, full)


def test_diagnostics_are_opt_in_and_reported_after_first_pop():
    iso = _eulerian()
    coords, lad, atoms = _prefix(6)
    with torch.no_grad():
        iso.pushforward(coords, lad, atoms)
    assert iso.pop_diagnostics() == {}          # collection off until first ask
    with torch.no_grad():
        iso.pushforward(coords, lad, atoms)
    diag = iso.pop_diagnostics()
    assert {"cond_msys", "min_eig_gram"} <= diag.keys()
    assert iso.pop_diagnostics() == {}          # cleared by the previous pop


@pytest.mark.parametrize(
    "dictionary",
    [
        FourierDictionary(domain_dim=2, num_channels=3, steepness=2.0),
        HaarWaveletDictionary(domain_dim=1, num_channels=1),
        VoronoiPWC(resolution=10, dimension=1, num_channels=1, learn_synthesis=False),
    ],
    ids=["fourier", "haar", "voronoi"],
)
def test_top_indices_are_the_highest_pmf_atoms_in_order(dictionary):
    idx = dictionary.get_top_indices(8)
    pmfs = dictionary.get_index_pmfs(idx)
    assert idx.shape[0] == 8
    assert torch.all(pmfs[:-1] >= pmfs[1:])
    # Nothing outside the prefix beats its smallest PMF.
    candidates = dictionary.get_high_probability_indices(float(pmfs[-1]) * 0.999)
    assert int((dictionary.get_index_pmfs(candidates) > pmfs[-1]).sum()) <= 8
    assert torch.equal(dictionary.get_top_indices(8), idx)  # deterministic


def test_prefix_captured_energy_is_pmf_weighted_squared_coefficients():
    atoms = torch.eye(3)[:, :, None] * math.sqrt(3.0)   # orthonormal on 3 equally weighted points
    values = torch.tensor([[1.0, 2.0, 0.0]])[:, :, None] * math.sqrt(3.0)
    pmfs = torch.tensor([0.5, 0.25, 0.25])
    energy = prefix_captured_energy(values, atoms, pmfs)
    assert energy.shape == (1,)
    assert energy.item() == pytest.approx(0.5 * 1.0 + 0.25 * 4.0)


def test_real_multichannel_fourier_atoms_and_energy():
    coords = torch.rand(64, 2)
    dictionary = FourierDictionary(domain_dim=2, num_channels=3, steepness=2.0)
    atoms = dictionary.get_atoms(coords, dictionary.get_lowpass_indices(1))
    assert atoms.shape[-1] == 3
    assert not atoms.is_complex()
    values = torch.rand(4, 64, 3)
    pmfs = dictionary.get_index_pmfs(dictionary.get_lowpass_indices(1))
    energy = prefix_captured_energy(values, atoms, pmfs)
    assert energy.shape == (4,)
    assert not energy.is_complex()
    reconstruction = dictionary.get_reconstructions(coords, values, dictionary.get_lowpass_indices(1))
    assert reconstruction.shape == values.shape


def test_real_orthogonal_channel_mixing_preserves_gram():
    coords, field = torch.rand(48, 2), torch.rand(3, 48, 2)
    isometry = ChannelOrthogonalIsometry(coords_dim=2, channels_dim=2)
    _, _, pushed = isometry.pushforward(coords, torch.zeros(48), field)
    assert not pushed.is_complex()
    assert torch.allclose(pairwise_inner_product(field, field), pairwise_inner_product(pushed, pushed), atol=1e-5)


def test_shared_nerf_fields_are_real_and_differentiable():
    coords = torch.rand(12, 2, requires_grad=True)
    mean_field = NerfNeuralField(
        coords_dim=2, output_dim=3, hidden_dims=(8,), nerf_n_levels=2, nerf_n_base=4
    )
    conditional_field = NerfConditionalField(
        coords_dim=2,
        output_dim=3,
        rank=2,
        cond_dim=5,
        hidden_dims=(8,),
        nerf_n_levels=2,
        nerf_n_base=4,
    )

    mean = mean_field(coords)
    generated = conditional_field(torch.rand(12, 5), coords)
    (mean.square().mean() + generated.square().mean()).backward()

    assert mean.shape == (12, 3) and not mean.is_complex()
    assert generated.shape == (12, 2, 3) and not generated.is_complex()
    assert coords.grad is not None


def test_chained_isometry_forwards_the_frame_only_where_it_is_needed():
    torch.manual_seed(0)
    chain = ChainedIsometry(
        coords_dim=1,
        channels_dim=2,
        isometries_partial=(
            partial(ChannelOrthogonalIsometry),
            partial(
                EulerianIsometry,
                rank=2,
                num_layers=2,
                scalar_field_partial=lambda **kwargs: NerfConditionalField(hidden_dims=(16,), **kwargs),
            ),
        ),
    )
    coords, lad = torch.rand(16, 1), torch.zeros(16)
    frame = FourierDictionary(domain_dim=1, num_channels=2).get_atoms(
        coords, FourierDictionary(domain_dim=1, num_channels=2).get_top_indices(4)
    )
    with torch.no_grad():
        _, _, pushed = chain.pushforward(coords, lad, frame)
        _, _, pulled = chain.pullback(coords, lad, pushed, frame=frame)
        f = torch.randn(3, 16, 2)
        _, _, f_pushed = chain.pushforward(coords, lad, f, frame=frame)
        _, _, f_back = chain.pullback(coords, lad, f_pushed, frame=frame)
    assert pushed.shape == frame.shape
    assert torch.allclose(f_back, f, atol=1e-4)
    assert torch.allclose(pulled, frame, atol=1e-4)


def test_channel_mixer_starts_at_a_coordinate_independent_haar_rotation():
    torch.manual_seed(7)
    isometry = ChannelOrthogonalIsometry(coords_dim=2, channels_dim=3)
    rotations = isometry._orthogonal(torch.rand(8, 2))
    expected = isometry.base_orthogonal.expand_as(rotations)
    assert torch.allclose(rotations, expected)
    assert torch.allclose(
        isometry.base_orthogonal.T @ isometry.base_orthogonal,
        torch.eye(3),
        atol=1e-6,
    )


def test_real_eulerian_isometry_and_synthetic_dataset_batch():
    dataset = OneDimDiscontinuousGenerator(domain_sample_size=32, n_functions=16)
    coords, values = dataset.get_batch(3)
    assert coords.shape == (32, 1) and values.shape == (3, 32, 1)
    isometry = EulerianIsometry(1, 1, 2, 2, lambda **kwargs: NerfConditionalField(**kwargs))
    dictionary = FourierDictionary(domain_dim=1, num_channels=1)
    frame = dictionary.get_atoms(coords, dictionary.get_top_indices(4))
    _, _, moved = isometry.pullback(coords, torch.zeros(32), values, frame=frame)
    assert moved.shape == values.shape and not moved.is_complex()


def test_one_dimensional_synthetic_dataset_uses_shared_sampler():
    dataset = OneDimDiscontinuousGenerator(domain_sample_size=8)
    assert isinstance(dataset.domain_sampler, SquareSampler)
    assert dataset.domain_sampler.domain_dim == 1


def test_cifar_inr_dataset_prefers_extracted_checkpoints(tmp_path):
    checkpoint_dir = tmp_path / "cifar_inrs_dataset" / "train" / "ckpts" / "00"
    checkpoint_dir.mkdir(parents=True)
    torch.save(
        {
            "params": {
                "seq.0.weight": torch.ones(3, 2),
                "seq.0.bias": torch.zeros(3),
            }
        },
        checkpoint_dir / "000000_0000042.ckpt",
    )

    dataset = ImplicitZooCIFARDataset(
        extracted_dir=str(tmp_path / "cifar_inrs_dataset"),
        n_inrs=1,
        domain_sample_size=2,
    )

    coords, values = dataset.get_batch(1)
    assert coords.shape == (4, 2) and values.shape == (1, 4, 3)


def test_real_voronoi_atoms():
    dictionary = VoronoiPWC(
        torch.tensor([[0.0], [1.0]]), torch.ones(2), num_channels=2,
        learn_synthesis=False,
    )
    coords = torch.rand(12, 1)
    top = dictionary.get_top_indices(2)
    atoms = dictionary.get_atoms(coords, top)
    assert atoms.shape == (2, 12, 2)
    assert not atoms.is_complex()
    indices = dictionary.get_high_probability_indices(0.0)
    pmfs = dictionary.get_index_pmfs(indices)
    assert torch.isclose(pmfs.sum(), torch.tensor(1.0))
    assert torch.all(pmfs[:-1] > pmfs[1:])
    values = torch.rand(3, 12, 2)
    energy = prefix_captured_energy(values, atoms, dictionary.get_index_pmfs(top))
    assert energy.shape == (3,)


def test_fixed_voronoi_has_no_trainable_synthesis():
    dictionary = VoronoiPWC(
        torch.tensor([[0.0], [1.0]]), torch.ones(2), num_channels=1,
        learn_synthesis=False,
    )
    assert list(dictionary.parameters()) == []


def test_regular_grid_voronoi_builds_unit_norm_blocks_and_harmonic_prior():
    dictionary = VoronoiPWC(
        resolution=2,
        dimension=2,
        num_channels=1,
        learn_synthesis=False,
    )

    assert torch.equal(
        dictionary._centroids,
        torch.tensor([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]),
    )
    assert torch.equal(dictionary._amplitudes, torch.full((4,), 2.0))
    pmfs = dictionary.get_index_pmfs(dictionary.get_high_probability_indices(0.0))
    assert torch.all(pmfs[:-1] > pmfs[1:])
    assert torch.isclose(pmfs.sum(), torch.tensor(1.0))


def test_haar_1d_atoms_and_pmf():
    dictionary = HaarWaveletDictionary(domain_dim=1, num_channels=1, learn_synthesis=False)
    coords = torch.tensor([[0.125], [0.375], [0.625], [0.875]])
    idx = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0]])
    atoms = dictionary.get_atoms(coords, idx, synthesis=False).squeeze(-1)
    assert torch.equal(atoms[0], torch.ones(4))
    assert torch.equal(atoms[1], torch.tensor([1.0, 1.0, -1.0, -1.0]))
    assert torch.equal(atoms[2], torch.tensor([0.0, 0.0, math.sqrt(2), -math.sqrt(2)]))
    pmfs = dictionary.get_index_pmfs(idx)
    assert torch.allclose(pmfs, torch.tensor([0.5, 0.25, 0.0625]))


def test_haar_2d_orientation_and_threshold_indices():
    dictionary = HaarWaveletDictionary(domain_dim=2, num_channels=1, learn_synthesis=False)
    coords = torch.tensor([[0.25, 0.25], [0.75, 0.25]])
    # Orientation 1 applies the wavelet on x only; orientation 2 on y only.
    atoms = dictionary.get_atoms(
        coords,
        torch.tensor([[0, 0, 0, 1, 0], [0, 0, 0, 2, 0]]),
        synthesis=False,
    ).squeeze(-1)
    assert torch.equal(atoms[0], torch.tensor([1.0, -1.0]))
    assert torch.equal(atoms[1], torch.tensor([1.0, 1.0]))
    indices = dictionary.get_high_probability_indices(1e-2)
    assert indices.shape[1] == 5
    assert torch.all(dictionary.get_index_pmfs(indices) >= 1e-2)


def test_learnable_synthesis_starts_as_orthogonal_mixing():
    torch.manual_seed(7)
    dictionary = FourierDictionary(domain_dim=1, num_channels=2, synthesis_tail_probability=1e-2)
    weight = dictionary._synthesis.weight.detach()
    assert not torch.allclose(weight, torch.eye(weight.shape[0]))
    assert torch.allclose(weight @ weight.T, torch.eye(weight.shape[0]), atol=1e-6)


def test_fourier_can_bypass_synthesis_and_reconstruct_with_raw_atoms():
    torch.manual_seed(3)
    dictionary = FourierDictionary(
        domain_dim=1,
        num_channels=1,
        synthesis_tail_probability=1e-2,
    )
    coords = torch.linspace(0.0, 1.0, 64).unsqueeze(-1)
    indices = dictionary._synthesis_indices[:4]

    raw_atoms = dictionary.get_atoms(coords, indices, synthesis=False)
    synthesized_atoms = dictionary.get_atoms(coords, indices)

    assert torch.allclose(raw_atoms, dictionary._get_base_atoms(coords, indices))
    assert not torch.allclose(raw_atoms, synthesized_atoms)

    functions = raw_atoms[:1]
    reconstruction = dictionary.get_reconstructions(
        coords,
        functions,
        indices[:1],
        synthesis=False,
    )
    coefficient = pairwise_inner_product(functions, raw_atoms[:1])
    expected = (coefficient @ raw_atoms[:1].reshape(1, -1)).view_as(functions)
    assert torch.allclose(reconstruction, expected)


def test_voronoi_can_bypass_synthesis_with_cell_channel_indices():
    dictionary = VoronoiPWC(
        centroids=torch.tensor([[0.0], [1.0]]),
        amplitudes=torch.ones(2),
        num_channels=2,
        learn_synthesis=True,
    )
    coords = torch.tensor([[0.0], [0.2], [0.8], [1.0]])
    raw_indices = torch.tensor([[0, 1], [1, 0]])

    raw_atoms = dictionary.get_atoms(coords, raw_indices, synthesis=False)

    expected = torch.tensor(
        [
            [[0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        ]
    )
    assert torch.equal(raw_atoms, expected)


@pytest.mark.parametrize(
    "make_dictionary",
    [
        lambda: FourierDictionary(domain_dim=1, num_channels=1, synthesis_tail_probability=1e-2),
        lambda: HaarWaveletDictionary(domain_dim=1, num_channels=1, synthesis_tail_probability=1e-2),
        lambda: VoronoiPWC(
            centroids=torch.tensor([[0.0], [1.0]]),
            amplitudes=torch.ones(2),
            num_channels=1,
            learn_synthesis=True,
        ),
    ],
    ids=["fourier", "haar", "voronoi"],
)
def test_trainable_dictionary_synthesis_is_checkpointed(tmp_path, make_dictionary):
    dictionary = make_dictionary()
    optimizer = torch.optim.Adam(dictionary.parameters(), lr=1e-2)
    dictionary._synthesis.weight.sum().backward()
    optimizer.step()
    expected_state = {name: value.detach().clone() for name, value in dictionary.state_dict().items()}

    checkpoint = Checkpointer(
        checkpoint_dir=str(tmp_path / "source"),
        models={"initial_dictionary": dictionary},
        optimizers={"dictionary": optimizer},
        schedulers={"dictionary": None},
    )._build_checkpoint(epoch=0, metric=0.0)

    restored_dictionary = make_dictionary()
    restored_optimizer = torch.optim.Adam(restored_dictionary.parameters(), lr=1e-2)
    Checkpointer(
        checkpoint_dir=str(tmp_path / "restored"),
        models={"initial_dictionary": restored_dictionary},
        optimizers={"dictionary": restored_optimizer},
        schedulers={"dictionary": None},
    ).restore(checkpoint)

    actual_state = restored_dictionary.state_dict()
    assert all(torch.equal(actual_state[name], value) for name, value in expected_state.items())
