import torch

from diag_sqrtD import _seeded_randn, active_for_target


def test_target_stopping_is_per_seed_and_monotone():
    active = torch.tensor([True, True, True])
    active = active_for_target(torch.tensor([0.1, 0.5, 0.2]), active, 0.3)
    assert active.tolist() == [False, True, False]
    active = active_for_target(torch.tensor([0.9, 0.4, 0.9]), active, 0.3)
    assert active.tolist() == [False, True, False]


def _draw(seed_ids):
    generators = [torch.Generator().manual_seed(seed) for seed in seed_ids]
    return (_seeded_randn(generators, (2, 3), "cpu"),
            _seeded_randn(generators, (4,), "cpu"))


def test_seed_initialisation_is_chunk_invariant():
    full_a, full_b = _draw([11, 12, 13, 14])
    left_a, left_b = _draw([11, 12])
    right_a, right_b = _draw([13, 14])
    assert torch.equal(full_a, torch.cat([left_a, right_a]))
    assert torch.equal(full_b, torch.cat([left_b, right_b]))
