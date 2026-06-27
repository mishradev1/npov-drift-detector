import numpy as np

from npov_drift.embedding import FakeEncoder


def test_deterministic_and_unit_norm():
    e = FakeEncoder(dim=64)
    a = e.encode(["sodium chloride salt"])
    b = e.encode(["sodium chloride salt"])
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-9


def test_shared_tokens_are_closer():
    e = FakeEncoder()
    v = e.encode(["apple banana cherry", "apple banana date", "x y z"])
    sim_shared = float(v[0] @ v[1])  # share apple, banana
    sim_disjoint = float(v[0] @ v[2])
    assert sim_shared > sim_disjoint


def test_empty():
    assert FakeEncoder().encode([]).shape[0] == 0
