from functools import partial
import math

import pytest
import torch

from infidictionary.datasets import ImplicitZooCIFARDataset, OneDimDiscontinuousGenerator
from infidictionary.domain_samplers import SquareSampler
from infidictionary.dictionaries import FourierDictionary, HaarWaveletDictionary
from infidictionary.neural_isometries import (
    ChainedIsometry,
    ChannelOrthogonalIsometry,
    EulerianIsometry,
)
from infidictionary.networks import NerfConditionalField, NerfNeuralField
from infidictionary.utils import isometry_defect, pairwise_inner_product, prefix_captured_energy


R = 4  # default number of mixing tokens used below


def _eulerian(num_layers=3):
    torch.manual_seed(0)
    iso = EulerianIsometry(
        1, 1, num_layers, lambda **kw: NerfConditionalField(hidden_dims=(32,), **kw)
    ).double()
    iso.eval()
    return iso


def _prefix(num_atoms, n_points=64):
    torch.manual_seed(1)
    dictionary = FourierDictionary(domain_dim=1, num_channels=1)
    coords = torch.rand(n_points, 1)
    atoms = dictionary.get_prefix(coords, num_atoms)
    return coords.double(), torch.zeros(n_points, dtype=torch.float64), atoms.double()


def test_eulerian_pushforward_preserves_the_gram_matrix():
    # The defining property: every layer is an exact Cayley rotation, so the
    # Gram matrix of the prefix is unchanged, not merely penalized toward it.
    iso = _eulerian()
    coords, lad, atoms = _prefix(9)
    with torch.no_grad():
        _, _, out = iso.pushforward(coords, lad, atoms, rank=R)
    assert not torch.allclose(out, atoms, atol=1e-2)  # it actually moved
    assert torch.allclose(
        pairwise_inner_product(out, out, lad), pairwise_inner_product(atoms, atoms, lad), atol=1e-10
    )
    assert isometry_defect(atoms, out, lad) == pytest.approx(1.0, abs=1e-10)


def test_eulerian_is_prefix_faithful():
    # Appending further functions to the prefix must not change the first K outputs.
    iso = _eulerian()
    coords, lad, atoms = _prefix(9)
    with torch.no_grad():
        _, _, short = iso.pushforward(coords, lad, atoms[:5], rank=R)
        _, _, long = iso.pushforward(coords, lad, atoms, rank=R)
    assert torch.allclose(long[:5], short, atol=1e-12)


def test_eulerian_requires_at_least_rank_functions():
    iso = _eulerian()
    coords, lad, atoms = _prefix(3)
    with pytest.raises(AssertionError, match="K >= R"):
        iso.pushforward(coords, lad, atoms, rank=R)


def test_mixing_attention_is_causal():
    iso = _eulerian()
    q, p = iso._initial_tokens(5)
    A = iso.layers[0].attention(q, p)
    assert torch.equal(A, A.tril())
    assert torch.allclose(A[0], torch.eye(5, dtype=A.dtype)[0])
    assert torch.allclose(A.sum(dim=-1), torch.ones(5, dtype=A.dtype))


def test_rank_is_a_call_argument_of_one_model():
    # One set of parameters defines a map for every R: each is an exact
    # isometry and prefix-faithful for its own R, and different R give
    # different rotations.
    iso = _eulerian()
    coords, lad, atoms = _prefix(9)
    gram = pairwise_inner_product(atoms, atoms, lad)
    outputs = {}
    with torch.no_grad():
        for rank in (2, 4, 7):
            _, _, out = iso.pushforward(coords, lad, atoms, rank=rank)
            _, _, short = iso.pushforward(coords, lad, atoms[:rank], rank=rank)
            assert torch.allclose(pairwise_inner_product(out, out, lad), gram, atol=1e-10)
            assert torch.allclose(out[:rank], short, atol=1e-12)
            outputs[rank] = out
    assert not torch.allclose(outputs[2], outputs[4], atol=1e-3)
    assert not torch.allclose(outputs[4], outputs[7], atol=1e-3)


def test_partial_pushforward_applies_only_the_requested_layers():
    iso = _eulerian(num_layers=3)
    coords, lad, atoms = _prefix(6)
    with torch.no_grad():
        _, _, none = iso.pushforward(coords, lad, atoms, num_layers_to_apply=0, rank=R)
        _, _, all_layers = iso.pushforward(coords, lad, atoms, num_layers_to_apply=3, rank=R)
        _, _, full = iso.pushforward(coords, lad, atoms, rank=R)
    assert torch.equal(none, atoms)
    assert torch.equal(all_layers, full)


@pytest.mark.parametrize(
    "dictionary",
    [
        FourierDictionary(domain_dim=2, num_channels=3, steepness=2.0),
        HaarWaveletDictionary(domain_dim=1, num_channels=1),
    ],
    ids=["fourier", "haar"],
)
def test_prefix_is_ordered_nested_and_orthonormal(dictionary):
    coords = torch.rand(4096, dictionary.domain_dim, dtype=torch.float64)
    pmfs = dictionary.get_prefix_pmfs(12)
    assert pmfs.shape == (12,)
    assert torch.all(pmfs[:-1] >= pmfs[1:])                 # descending PMF
    # Nothing outside the prefix beats its smallest PMF.
    candidates = dictionary._high_probability_indices(float(pmfs[-1]) * 0.999)
    assert int((dictionary._index_pmfs(candidates) > pmfs[-1]).sum()) <= 12
    # Nested: the K prefix is the first K atoms of the K+1 prefix.
    short, long = dictionary.get_prefix(coords, 8), dictionary.get_prefix(coords, 12)
    assert torch.equal(long[:8], short)
    assert torch.equal(dictionary.get_prefix_pmfs(8), pmfs[:8])
    # Orthonormal up to Monte-Carlo error of the uniform sample.
    gram = pairwise_inner_product(long, long)
    assert torch.allclose(gram, torch.eye(12, dtype=gram.dtype), atol=0.1)


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
    atoms = dictionary.get_prefix(coords, 27)
    assert atoms.shape == (27, 64, 3)
    assert not atoms.is_complex()
    values = torch.rand(4, 64, 3)
    energy = prefix_captured_energy(values, atoms, dictionary.get_prefix_pmfs(27))
    assert energy.shape == (4,)
    assert not energy.is_complex()


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


def test_chained_isometry_forwards_rank_only_where_it_is_needed():
    torch.manual_seed(0)
    chain = ChainedIsometry(
        coords_dim=1,
        channels_dim=2,
        isometries_partial=(
            partial(ChannelOrthogonalIsometry),
            partial(
                EulerianIsometry,
                num_layers=2,
                scalar_field_partial=lambda **kwargs: NerfConditionalField(hidden_dims=(16,), **kwargs),
            ),
        ),
    )
    coords, lad = torch.rand(16, 1), torch.zeros(16)
    prefix = FourierDictionary(domain_dim=1, num_channels=2).get_prefix(coords, 4)
    with torch.no_grad():
        _, _, pushed = chain.pushforward(coords, lad, prefix, rank=2)
    assert pushed.shape == prefix.shape
    assert torch.allclose(
        pairwise_inner_product(pushed, pushed, lad), pairwise_inner_product(prefix, prefix, lad), atol=1e-4
    )


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
    isometry = EulerianIsometry(1, 1, 2, lambda **kwargs: NerfConditionalField(**kwargs))
    dictionary = FourierDictionary(domain_dim=1, num_channels=1)
    _, _, learned = isometry.pushforward(coords, torch.zeros(32), dictionary.get_prefix(coords, 4), rank=2)
    energy = prefix_captured_energy(values, learned, dictionary.get_prefix_pmfs(4))
    assert learned.shape == (4, 32, 1) and energy.shape == (3,) and not learned.is_complex()


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


def test_haar_1d_prefix_atoms_and_pmf():
    dictionary = HaarWaveletDictionary(domain_dim=1, num_channels=1)
    coords = torch.tensor([[0.125], [0.375], [0.625], [0.875]])
    atoms = dictionary.get_prefix(coords, 4).squeeze(-1)
    r2 = math.sqrt(2)
    assert torch.equal(atoms[0], torch.ones(4))                          # scaling
    assert torch.equal(atoms[1], torch.tensor([1.0, 1.0, -1.0, -1.0]))    # level 0
    assert torch.equal(atoms[2], torch.tensor([r2, -r2, 0.0, 0.0]))       # level 1, shift 0
    assert torch.equal(atoms[3], torch.tensor([0.0, 0.0, r2, -r2]))       # level 1, shift 1
    assert torch.allclose(dictionary.get_prefix_pmfs(4), torch.tensor([0.5, 0.25, 0.0625, 0.0625]))


def test_haar_2d_prefix_orientations():
    dictionary = HaarWaveletDictionary(domain_dim=2, num_channels=1)
    coords = torch.tensor([[0.25, 0.25], [0.75, 0.25]])
    # After the scaling atom come the three level-0 wavelets: orientation 1
    # applies the wavelet on x only, orientation 2 on y only.
    atoms = dictionary.get_prefix(coords, 4).squeeze(-1)
    assert torch.equal(atoms[1], torch.tensor([1.0, -1.0]))
    assert torch.equal(atoms[2], torch.tensor([1.0, 1.0]))


def test_infinite_dictionaries_have_no_learned_state(tmp_path):
    # Fourier and Haar are fixed infinite bases: nothing to optimize, and a
    # save/load round trip reproduces the atoms from the constructor args alone.
    coords = torch.rand(32, 1)
    for dictionary in (
        FourierDictionary(domain_dim=1, num_channels=1),
        HaarWaveletDictionary(domain_dim=1, num_channels=1),
    ):
        assert list(dictionary.parameters()) == []
        assert dictionary.state_dict() == {}
        dictionary.save(str(tmp_path / "d.pt"))
        restored = type(dictionary).load(str(tmp_path / "d.pt"))
        assert torch.equal(restored.get_prefix(coords, 8), dictionary.get_prefix(coords, 8))
