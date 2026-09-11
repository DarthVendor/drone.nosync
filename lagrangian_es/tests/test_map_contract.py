"""Three files must agree on one 8-slot layout, and nothing makes them.

`MapPrior.observe` (the carried survey) and `BuiltMap.read` (what the vehicle's
own beams accumulated) both emit entries that `Tokenizer._map_rows` unpacks as

    cx, cy, hw, hd, height, yaw, conf, age

Every shared convention of this kind that was checked this session turned out
to be broken somewhere -- the tokenizer's beam bearings disagreed with the rays
the sensor cast, the chain's "last entry" was a padded slot, and the built map
picked its fan by config order rather than geometry.  This one happens to be
right; the test is so it stays that way.
"""
import torch

from lagrangian_es.composer.tokens import Tokenizer
from lagrangian_es.mapping import FEATS as BUILT_FEATS
from lagrangian_es.sensors.map_view import FEATS as PRIOR_FEATS

SLOTS = ("cx", "cy", "hw", "hd", "height", "yaw", "conf", "age")


def test_both_maps_declare_the_same_width():
    assert BUILT_FEATS == PRIOR_FEATS == len(SLOTS)


def _decode(entry):
    """Run one entry through the tokenizer's map path, at the origin facing +x."""
    tok = Tokenizer(scale=30.0, reach=10.0, sensors=())
    o = torch.tensor(entry, dtype=torch.float64).reshape(1, 1, len(SLOTS))
    x = torch.zeros(1, 3, dtype=torch.float64)
    psi = torch.zeros(1, dtype=torch.float64)
    return tok._map_rows(o, 1, torch.float64, torch.device("cpu"), x, psi, 1)[0, 0]


def test_the_tokenizer_reads_each_slot_as_its_name_says():
    scale = 30.0
    # a box 6 m ahead, 3 m to the left, half-extents 2 x 4, height 9, no yaw,
    # fully confident, seen 100 steps ago
    row = _decode([6.0, 3.0, 2.0, 4.0, 9.0, 0.0, 1.0, 100.0])
    assert abs(float(row[0]) - 6.0 / scale) < 1e-9, "slot 0 is not cx"
    assert abs(float(row[1]) - 3.0 / scale) < 1e-9, "slot 1 is not cy"
    assert abs(float(row[2]) - (6.0 ** 2 + 3.0 ** 2) ** 0.5 / scale) < 1e-9, "slot 2 is not the distance"
    assert abs(float(row[3]) - 2.0 / scale) < 1e-9, "slot 3 is not the half-width"
    assert abs(float(row[4]) - 4.0 / scale) < 1e-9, "slot 4 is not the half-depth"
    assert abs(float(row[5]) - 9.0 / scale) < 1e-9, "slot 5 is not the height"
    assert abs(float(row[6]) - 1.0) < 1e-9, "slot 6 is not the confidence"
    assert abs(float(row[7]) - 100.0 / 200.0) < 1e-9, "slot 7 is not the age"


def test_an_unconfident_entry_reads_as_nothing_at_all():
    """An empty viewport and an empty built map must look identical, or the
    composer learns to tell them apart by an artefact."""
    row = _decode([6.0, 3.0, 2.0, 4.0, 9.0, 0.0, 0.0, 100.0])      # conf = 0
    assert float(row.abs().max()) == 0.0, f"a zero-confidence entry leaked {row.tolist()}"


def test_a_yawed_footprint_becomes_a_conservative_bounding_box():
    """A rotated box is presented as its ego-frame bounding box: never smaller
    than the true footprint, so the composer is never told an obstacle is
    narrower than it is."""
    import math
    square = _decode([10.0, 0.0, 2.0, 2.0, 5.0, math.pi / 4, 1.0, 0.0])
    axis = _decode([10.0, 0.0, 2.0, 2.0, 5.0, 0.0, 1.0, 0.0])
    assert float(square[3]) >= float(axis[3]) - 1e-9, "a rotated box shrank in width"
    assert float(square[4]) >= float(axis[4]) - 1e-9, "a rotated box shrank in depth"


def test_the_built_map_emits_what_the_tokenizer_expects():
    """End to end: a beam return becomes a remembered cell that decodes back to
    roughly where the beam ended."""
    from lagrangian_es.mapping import BuiltMap
    m = BuiltMap()
    m.reset(1, torch.device("cpu"), torch.float32)
    p = torch.zeros(1, 3)
    fwd = torch.tensor([[[1.0, 0.0, 0.0]]])                 # one beam, straight ahead
    m.update(p, fwd, torch.tensor([[5.0]]), max_range=6.0,
             live=torch.ones(1, dtype=torch.bool), t=0.0)
    out = m.read(p, now=10.0)
    assert out.shape[-1] % BUILT_FEATS == 0
    first = out.reshape(1, -1, BUILT_FEATS)[0, 0]
    assert float(first[6]) > 0.0, "the cell was recorded but reads as unconfident"
    # it should sit near 5 m ahead, within one cell
    assert abs(float(first[0]) - 5.0) <= m.cell, (float(first[0]), m.cell)
    assert abs(float(first[1])) <= m.cell
