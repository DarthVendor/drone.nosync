"""The coordinate the learned term reads its beams in.

`linear` divides range by `obs_scale`, which spends input resolution in
proportion to distance -- over 0.2-6 m it gives the far half (2.5-6 m) 0.875 of
the input and the near half (0.2-1.2 m) only 0.05.  That is backwards for
collision avoidance, and it shows up in the trained controller: the
obstacle-induced force varies 9% between 0.2 m and 1.2 m, where a barrier would
vary by orders of magnitude.

`proximity` reads s / (d + s) instead: 1 at contact, falling toward 0 at range.
It is the coordinate a barrier is simple in, and it costs no extra parameters,
which matters because added dimension has measurably hurt more than added
expressiveness here (the gyroscopic head lost its A/B at +28% dimension).
"""
import pytest
import torch

from lagrangian_es.trainables.learned import LearnedShaping


def _term(transform, **kw):
    return LearnedShaping(3, n_obs=24, hidden=16, obs_transform=transform, **kw)


def test_the_transform_costs_no_parameters():
    assert _term("linear").dim == _term("proximity").dim


def test_proximity_is_one_at_contact_and_falls_to_zero_at_range():
    t = _term("proximity", prox_scale=1.0)
    z = t._read({"range": torch.tensor([[0.0, 1.0, 1e6]], dtype=torch.float64)})
    assert float(z[0, 0]) == pytest.approx(1.0)
    assert float(z[0, 1]) == pytest.approx(0.5)
    assert float(z[0, 2]) == pytest.approx(0.0, abs=1e-5)


def test_proximity_is_strictly_decreasing_in_distance():
    t = _term("proximity")
    d = torch.tensor([[0.2, 0.6, 1.2, 2.5, 6.0]], dtype=torch.float64)
    z = t._read({"range": d})[0]
    assert torch.all(z[:-1] > z[1:])


def test_proximity_spends_its_resolution_on_the_near_field():
    """The whole point: invert which half of the range gets the input span."""
    d = torch.tensor([[0.2, 1.2, 2.5, 6.0]], dtype=torch.float64)
    spans = {}
    for name in ("linear", "proximity"):
        z = _term(name)._read({"range": d})[0]
        spans[name] = (abs(float(z[0] - z[1])), abs(float(z[2] - z[3])))
    near_lin, far_lin = spans["linear"]
    near_pro, far_pro = spans["proximity"]
    assert near_lin < far_lin, "linear favours the far field"
    assert near_pro > far_pro, "proximity must favour the near field"


def test_proximity_never_goes_negative_on_a_bad_reading():
    """A no-return reads as max_range; nothing should produce a negative range,
    but the clamp means a stray one cannot flip the coordinate's sign."""
    t = _term("proximity")
    z = t._read({"range": torch.tensor([[-1.0, 0.0]], dtype=torch.float64)})
    assert torch.all(z > 0) and torch.all(z <= 1.0)


def test_linear_is_unchanged():
    t = _term("linear")
    d = torch.tensor([[0.4, 4.0, 40.0]], dtype=torch.float64)
    z = t._read({"range": d})
    assert torch.allclose(z, torch.tensor([[0.1, 1.0, 4.0]], dtype=torch.float64))


def test_an_unknown_transform_is_refused():
    with pytest.raises(ValueError, match="obs_transform"):
        _term("sigmoid")


def test_missing_observations_still_return_none():
    t = _term("proximity")
    assert t._read(None) is None
    assert t._read({"other": torch.zeros(1, 24)}) is None
